# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import datetime
import functools
import logging
import math
import pickle
import re
import warnings
from collections import Counter, OrderedDict, defaultdict
from timeit import default_timer
from typing import List

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.infer.autoguide import (
    AutoDelta,
    AutoGuideList,
    AutoLowRankMultivariateNormal,
    AutoNormal,
)
from pyro.infer.reparam import LocScaleReparam
from pyro.ops.streaming import CountMeanVarianceStats, StatsOfDict
from pyro.optim import ClippedAdam
from pyro.poutine.util import site_is_subsample

import pyrocov.geo

from . import pangolin, sarscov2
from .util import pearson_correlation

logger = logging.getLogger(__name__)

# Reasonable values might be week (7), fortnight (14), or month (28)
TIMESTEP = 14  # in days
GENERATION_TIME = 5.5  # in days
START_DATE = "2019-12-01"


def date_range(stop):
    start = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    step = datetime.timedelta(days=TIMESTEP)
    return np.array([start + step * t for t in range(stop)])


def get_fine_regions(columns, min_samples=50):
    """
    Select regions that have at least ``min_samples`` samples.
    Remaining regions will be coarsely aggregated up to country level.
    """
    # Count number of samples in each subregion.
    counts = Counter()
    for location in columns["location"]:
        parts = location.split("/")
        if len(parts) < 2:
            continue
        parts = tuple(p.strip() for p in parts[:3])
        counts[parts] += 1

    # Select fine countries.
    return frozenset(parts for parts, count in counts.items() if count >= min_samples)


def rank_loo_lineages(
    full_dataset: dict,
    full_result: dict,
    min_samples: int = 100,
) -> List[str]:
    """
    Compute a list of lineages ranked in descending order of how much their
    growth rate differs from their parents' growth rate. This is used in growth
    rate leave-one-out prediction experiments.
    """
    # Decompress lineage names before computing parents.
    lineage_id_inv = [
        pangolin.decompress(name) for name in full_dataset["lineage_id_inv"]
    ]
    lineage_id = {name: i for i, name in enumerate(lineage_id_inv)}
    ancestors = set(lineage_id)

    # Filter to often-observed lineages.
    weekly_strains = full_dataset["weekly_strains"]  # [T, P, S]
    lineage_counts = weekly_strains.sum([0, 1])  # [S]
    lineages = []
    for c, child in enumerate(lineage_id_inv):
        if child in ("A", "B", "B.1"):
            continue  # ignore very early lineages
        if lineage_counts[c] < min_samples:
            continue  # ignore rare lineages
        lineages.append(child)

    # Sort leaf nodes by their distance from parent.
    rate_loc = full_result["median"]["rate_loc"]
    ranked_lineages = []
    for child in lineages:
        # Allow grandparent to adopt orphan, since e.g. B.1.617 is missing from
        # lineage_id, but B.1.617.2 is very important.
        parent = pangolin.get_most_recent_ancestor(child, ancestors)
        assert parent is not None
        c = lineage_id[child]
        p = lineage_id[parent]
        gap = (rate_loc[c] - rate_loc[p]).abs().item()
        ranked_lineages.append((gap, child))
    ranked_lineages.sort(reverse=True)

    # Compress lineage names before returning.
    return [pangolin.compress(name) for gap, name in ranked_lineages]


def load_gisaid_data(
    *,
    device="cpu",
    include={},
    exclude={},
    end_day=None,
    gisaid_columns_filename="results/gisaid.columns.pkl",
    nextclade_features_filename="results/nextclade.features.pt",
) -> dict:
    """
    Loads the two files gisaid_columns_filename and nextclade_features_filename,
    converts teh input to PyTorch tensors and truncates the data according to
    ``include`` and ``exclude``.

    Keyword arguments:
    device -- torch device to use
    include --
    exclude --
    end_day -- last day to include
    gisaid_columns_filename --
    nextclade_features_filename --
    """
    logger.info("Loading data")
    include = include.copy()
    exclude = exclude.copy()

    if end_day:
        logger.info(f"Load gisaid data end_day: {end_day}")

    # Load ``gisaid_columns_filename``
    with open(gisaid_columns_filename, "rb") as f:
        columns = pickle.load(f)

    logger.info("Training on {} rows with columns:".format(len(columns["day"])))
    logger.info(", ".join(columns.keys()))

    # Filter regions to at least 50 sample and aggregate rest to country level
    fine_regions = get_fine_regions(columns)

    # Filter features into numbers of mutations and possibly genes.
    aa_features = torch.load(nextclade_features_filename)
    mutations = aa_features["mutations"]
    features = aa_features["features"].to(
        device=device, dtype=torch.get_default_dtype()
    )
    keep = [m.count(",") == 0 for m in mutations]  # restrict to single mutations
    if include.get("gene"):
        re_gene = re.compile(include.pop("gene"))
        keep = [k and bool(re_gene.search(m)) for k, m in zip(keep, mutations)]
    if exclude.get("gene"):
        re_gene = re.compile(exclude.pop("gene"))
        keep = [k and not re_gene.search(m) for k, m in zip(keep, mutations)]
    if include.get("region"):
        gene, region = include.pop("region")
        lb, ub = sarscov2.GENE_STRUCTURE[gene][region]
        for i, m in enumerate(mutations):
            g, m = m.split(":")
            if g != gene:
                keep[i] = False
                continue
            match = re.search("[0-9]+", m)
            assert match is not None
            pos = int(match.group())
            if not (lb < pos <= ub):
                keep[i] = False
    mutations = [m for k, m in zip(keep, mutations) if k]
    if mutations:
        features = features[:, keep]
    else:
        warnings.warn("No mutations selected; using empty features")
        mutations = ["S:D614G"]  # bogus
        features = features[:, :1] * 0
    logger.info("Loaded {} feature matrix".format(" x ".join(map(str, features.shape))))

    # Aggregate regions

    # Get lineages
    lineages = list(map(pangolin.compress, columns["lineage"]))
    lineage_id_inv = list(map(pangolin.compress, aa_features["lineages"]))
    lineage_id = {k: i for i, k in enumerate(lineage_id_inv)}

    sparse_data: dict = Counter()
    location_id: dict = OrderedDict()

    # Set of lineages that are skipped
    skipped = set()

    # Generate sparse_data
    for virus_name, day, location, lineage in zip(
        columns["virus_name"], columns["day"], columns["location"], lineages
    ):
        if lineage not in lineage_id:
            if lineage not in skipped:
                skipped.add(lineage)
                logger.warning(f"WARNING skipping unsampled lineage {lineage}")
            continue

        # Filter by include/exclude
        row = {
            "virus_name": virus_name,
            "location": location,
            "day": day,
            "lineage": pangolin.compress(lineage),
        }
        if not all(v.search(row[k]) for k, v in include.items()):
            continue
        if any(v.search(row[k]) for k, v in exclude.items()):
            continue

        # Filter by day
        if end_day is not None:
            if day > end_day:
                continue

        # preprocess parts
        parts = location.split("/")
        if len(parts) < 2:
            continue
        parts = tuple(p.strip() for p in parts[:3])
        if len(parts) == 3 and parts not in fine_regions:
            parts = parts[:2]
        location = " / ".join(parts)

        p = location_id.setdefault(location, len(location_id))
        s = lineage_id[lineage]
        t = day // TIMESTEP
        sparse_data[t, p, s] += 1

    # Generate weekly_strains tensor from sparse_data
    if end_day is not None:
        T = 1 + end_day // TIMESTEP
    else:
        T = 1 + max(columns["day"]) // TIMESTEP

    P = len(location_id)
    S = len(lineage_id)
    weekly_strains = torch.zeros(T, P, S)
    for (t, p, s), n in sparse_data.items():
        weekly_strains[t, p, s] = n

    logger.info(f"Dataset size [T x P x S] {T} x {P} x {S}")

    logger.info(
        f"Keeping {int(weekly_strains.sum())}/{len(lineages)} rows "
        f"(dropped {len(lineages) - int(weekly_strains.sum())})"
    )

    # Filter regions.
    num_times_observed = (weekly_strains > 0).max(2).values.sum(0)
    ok_regions = (num_times_observed >= 2).nonzero(as_tuple=True)[0]
    ok_region_set = set(ok_regions.tolist())
    logger.info(f"Keeping {len(ok_regions)}/{weekly_strains.size(1)} regions")
    weekly_strains = weekly_strains.index_select(1, ok_regions)
    locations = [k for k, v in location_id.items() if v in ok_region_set]
    location_id = OrderedDict(zip(locations, range(len(locations))))

    # Construct region-local time scales centered around observations.
    num_obs = weekly_strains.sum(-1)
    local_time = torch.arange(float(len(num_obs))) * TIMESTEP / GENERATION_TIME
    local_time = local_time[:, None]
    local_time = local_time - (local_time * num_obs).sum(0) / num_obs.sum(0)

    return {
        "location_id": location_id,
        "mutations": mutations,
        "weekly_strains": weekly_strains,
        "features": features,
        "lineage_id": lineage_id,
        "lineage_id_inv": lineage_id_inv,
        "local_time": local_time,
    }


def subset_gisaid_data(
    gisaid_dataset: dict,
    location_queries=None,
    max_strains=math.inf,
) -> dict:
    """
    Selects a small subset of data for exploratory fitting of a small model.
    This is not used in the final published results.
    """
    old = gisaid_dataset
    new = old.copy()

    # Select locations.
    if location_queries is not None:
        locations = sorted(
            {
                location
                for location in new["location_id"]
                if any(q in location for q in location_queries)
            }
        )
        ids = torch.tensor([old["location_id"][location] for location in locations])
        new["location_id"] = {name: i for i, name in enumerate(locations)}
        new["weekly_strains"] = new["weekly_strains"].index_select(1, ids)
        new["local_time"] = new["local_time"].index_select(1, ids)

    # Select strains.
    if new["weekly_strains"].size(-1) > max_strains:
        ids = (
            new["weekly_strains"]
            .sum([0, 1])
            .sort(0, descending=True)
            .indices[:max_strains]
        )
        new["weekly_strains"] = new["weekly_strains"].index_select(-1, ids)
        new["features"] = new["features"].index_select(0, ids)
        new["lineage_id_inv"] = [new["lineage_id_inv"][i] for i in ids.tolist()]
        new["lineage_id"] = {name: i for i, name in enumerate(new["lineage_id_inv"])}

    # Select mutations.
    gaps = new["features"].max(0).values - new["features"].min(0).values
    ids = (gaps >= 0.5).nonzero(as_tuple=True)[0]
    new["mutations"] = [new["mutations"][i] for i in ids.tolist()]
    new["features"] = new["features"].index_select(-1, ids)

    logger.info(
        "Selected {}/{} places, {}/{} strains, {}/{} mutations, {}/{} samples".format(
            len(new["location_id"]),
            len(old["location_id"]),
            len(new["lineage_id"]),
            len(old["lineage_id"]),
            len(new["mutations"]),
            len(old["mutations"]),
            int(new["weekly_strains"].sum()),
            int(old["weekly_strains"].sum()),
        )
    )

    return new


def load_jhu_data(gisaid_data: dict) -> dict:
    """
    Load case count time series.

    This is used for plotting but is not used for fitting a model.
    """
    # Load raw JHU case count data.
    us_cases_df = pyrocov.geo.read_csv("time_series_covid19_confirmed_US.csv")
    global_cases_df = pyrocov.geo.read_csv("time_series_covid19_confirmed_global.csv")
    daily_cases = torch.cat(
        [
            pyrocov.geo.pd_to_torch(us_cases_df, columns=slice(11, None)),
            pyrocov.geo.pd_to_torch(global_cases_df, columns=slice(4, None)),
        ]
    ).T
    logger.info(
        "Loaded {} x {} daily case data, totaling {}".format(
            *daily_cases.shape, daily_cases[-1].sum().item()
        )
    )

    # Convert JHU locations to GISAID locations.
    locations = list(gisaid_data["location_id"])
    matrix = pyrocov.geo.gisaid_to_jhu_location(locations, us_cases_df, global_cases_df)
    assert matrix.shape == (len(locations), daily_cases.shape[-1])
    daily_cases = daily_cases @ matrix.T
    daily_cases[1:] -= daily_cases[:-1].clone()  # cumulative -> density
    daily_cases.clamp_(min=0)
    assert daily_cases.shape[1] == len(gisaid_data["location_id"])

    # Convert daily counts to TIMESTEP counts (e.g. weekly).
    start_date = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    jhu_start_date = pyrocov.geo.parse_date(us_cases_df.columns[11])
    assert start_date < jhu_start_date
    dt = (jhu_start_date - start_date).days
    T = len(gisaid_data["weekly_strains"])
    weekly_cases = daily_cases.new_zeros(T, len(locations))
    for w in range(TIMESTEP):
        t0 = (w + dt) // TIMESTEP
        source = daily_cases[w::TIMESTEP]
        destin = weekly_cases[t0 : t0 + len(source)]
        destin[:] += source[: len(destin)]
    assert weekly_cases.sum() > 0

    return {
        "daily_cases": daily_cases.clamp(min=0),
        "weekly_cases": weekly_cases.clamp(min=0),
    }


def model(dataset, *, forecast_steps=None):
    """
    Bayesian regression model of lineage portions as a function of mutation features.

    This function can be run in two different modes:
    - During training, ``forecast_steps=None`` and the model is conditioned on
      observed data.
    - During prediction (after training), the likelihood statement is omitted
      and instead a ``probs`` tensor is recorded; this is the predicted lineage
      portions in each (time, regin) bin.
    """
    # Tensor shapes are commented at at the end of some lines.
    features = dataset["features"]
    local_time = dataset["local_time"][..., None]  # [T, P, 1]
    T, P, _ = local_time.shape
    S, F = features.shape
    if forecast_steps is None:  # During inference.
        weekly_strains = dataset["weekly_strains"]
        assert weekly_strains.shape == (T, P, S)
    else:  # During prediction.
        T = T + forecast_steps
        t0 = local_time[0]
        dt = local_time[1] - local_time[0]
        local_time = t0 + dt * torch.arange(float(T))[:, None, None]
        assert local_time.shape == (T, P, 1)

    # Configure reparametrization (which does not affect model density).
    reparam = {}
    local_time = local_time + pyro.param(
        "local_time", lambda: torch.zeros(P, S)
    )  # [T, P, S]
    reparam["coef"] = LocScaleReparam()
    reparam["rate"] = LocScaleReparam()
    reparam["init_loc"] = LocScaleReparam()
    reparam["init"] = LocScaleReparam()
    with poutine.reparam(config=reparam):

        # Sample global random variables.
        coef_scale = pyro.sample("coef_scale", dist.InverseGamma(5e3, 1e2))[..., None]
        rate_scale = pyro.sample("rate_scale", dist.LogNormal(-4, 2))[..., None]
        init_loc_scale = pyro.sample("init_loc_scale", dist.LogNormal(0, 2))[..., None]
        init_scale = pyro.sample("init_scale", dist.LogNormal(0, 2))[..., None]

        # Assume relative growth rate depends strongly on mutations and weakly on place.
        coef_loc = torch.zeros(F)
        coef = pyro.sample(
            "coef", dist.Logistic(coef_loc, coef_scale).to_event(1)
        )  # [F]
        rate_loc = pyro.deterministic(
            "rate_loc", 0.01 * coef @ features.T, event_dim=1
        )  # [S]

        # Assume initial infections depend strongly on strain and place.
        init_loc = pyro.sample(
            "init_loc", dist.Normal(torch.zeros(S), init_loc_scale).to_event(1)
        )  # [S]
        with pyro.plate("place", P, dim=-1):
            rate = pyro.sample(
                "rate", dist.Normal(rate_loc, rate_scale).to_event(1)
            )  # [P, S]
            init = pyro.sample(
                "init", dist.Normal(init_loc, init_scale).to_event(1)
            )  # [P, S]

            # Finally observe counts.
            with pyro.plate("time", T, dim=-2):
                logits = init + rate * local_time  # [T, P, S]

                if forecast_steps is None:  # During inference.
                    pyro.sample(
                        "obs",
                        dist.Multinomial(logits=logits, validate_args=False),
                        obs=weekly_strains,
                    )
                else:  # During prediction.
                    pyro.deterministic("probs", logits.softmax(-1), event_dim=1)


class InitLocFn:
    """
    Initializer for latent variables.

    This is passed as the ``init_loc_fn`` to guides.
    """

    def __init__(self, dataset):
        # Initialize init.
        init = dataset["weekly_strains"].sum(0)
        init.add_(1 / init.size(-1)).div_(init.sum(-1, True))
        init.log_().sub_(init.median(-1, True).values)
        self.init = init  # [P, S]
        self.init_decentered = init / 2
        self.init_loc = init.mean(0)  # [S]
        self.init_loc_decentered = self.init_loc / 2
        assert not torch.isnan(self.init).any()
        logger.info(f"init stddev = {self.init.std():0.3g}")

    def __call__(self, site):
        name = site["name"]
        shape = site["fn"].shape()
        if hasattr(self, name):
            result = getattr(self, name)
            assert result.shape == shape
            return result
        if name in ("coef_scale", "init_scale", "init_loc_scale"):
            return torch.ones(shape)
        if name == "logits_scale":
            return torch.full(shape, 0.002)
        if name in ("rate_scale", "place_scale", "strain_scale"):
            return torch.full(shape, 0.01)
        if name in ("coef", "coef_decentered", "rate", "rate_decentered"):
            return torch.rand(shape).sub_(0.5).mul_(0.01)
        if name == "coef_loc":
            return torch.rand(shape).sub_(0.5).mul_(0.01).add_(1.0)
        raise ValueError(f"InitLocFn found unhandled site {repr(name)}; please update.")


class Guide(AutoGuideList):
    """
    Custom guide for large-scale inference.

    This combines a low-rank multivariate normal guide over mutation
    coefficients with a mean field guide over remaining latent variables.
    """

    def __init__(self, model, init_loc_fn, init_scale, rank):
        super().__init__(model)

        # Jointly estimate mutation coefficients and shrinkage.
        mvn = ["coef_scale", "coef", "coef_decentered"]
        self.append(
            AutoLowRankMultivariateNormal(
                poutine.block(model, expose=mvn),
                init_loc_fn=init_loc_fn,
                init_scale=init_scale,
                rank=rank,
            )
        )
        model = poutine.block(model, hide=mvn)

        # Mean-field estimate all remaining latent variables.
        self.append(AutoNormal(model, init_loc_fn=init_loc_fn, init_scale=init_scale))


@torch.no_grad()
@poutine.mask(mask=False)
def predict(
    guide,
    dataset,
    *,
    num_samples=1000,
    vectorize=None,
    save_params=("rate", "init", "probs"),
    forecast_steps=0,
) -> dict:
    model = guide.model

    def get_conditionals(data):
        trace = poutine.trace(poutine.condition(model, data)).get_trace(
            dataset, forecast_steps=forecast_steps
        )
        return {
            name: site["value"].detach()
            for name, site in trace.nodes.items()
            if site["type"] == "sample" and not site_is_subsample(site)
            if name != "obs"
        }

    # Compute median point estimate.
    result: dict = defaultdict(dict)
    for name, value in get_conditionals(guide.median(dataset)).items():
        if value.numel() < 1e5 or name in save_params:
            result["median"][name] = value

    # Compute moments.
    save_params = {
        k for k, v in result["median"].items() if v.numel() < 1e5 or k in save_params
    }
    if vectorize is None:
        vectorize = result["median"]["probs"].numel() < 1e6
    if vectorize:
        with pyro.plate("particles", num_samples, dim=-3):
            samples = get_conditionals(guide())
        for k, v in samples.items():
            if k in save_params:
                result["mean"][k] = v.mean(0).squeeze()
                result["std"][k] = v.std(0).squeeze()
    else:
        stats = StatsOfDict({k: CountMeanVarianceStats for k in save_params})
        for _ in range(num_samples):
            stats.update(get_conditionals(guide()))
            print(".", end="", flush=True)
        for name, stats_ in stats.get().items():
            if "mean" in stats_:
                result["mean"][name] = stats_["mean"]
            if "variance" in stats_:
                result["std"][name] = stats_["variance"].sqrt()
    return dict(result)


def fit_svi(
    dataset: dict,
    *,
    guide_type: str,
    cond_data={},
    forecast_steps=0,
    learning_rate=0.05,
    learning_rate_decay=0.1,
    num_steps=3001,
    num_samples=1000,
    clip_norm=10.0,
    rank=200,
    jit=True,
    log_every=50,
    seed=20210319,
    check_loss=False,
) -> dict:
    """
    Fits a variational posterior using stochastic variational inference (SVI).
    """
    start_time = default_timer()

    logger.info(f"Fitting {guide_type} guide via SVI")
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    param_store = pyro.get_param_store()

    # Initialize guide so we can count parameters and register hooks.
    cond_data = {k: torch.as_tensor(v) for k, v in cond_data.items()}
    model_ = poutine.condition(model, cond_data)
    init_loc_fn = InitLocFn(dataset)
    if guide_type == "map":
        guide = AutoDelta(model_, init_loc_fn=init_loc_fn)
    elif guide_type == "normal":
        guide = AutoNormal(model_, init_loc_fn=init_loc_fn, init_scale=0.01)
    elif guide_type == "full":
        guide = AutoLowRankMultivariateNormal(
            model_, init_loc_fn=init_loc_fn, init_scale=0.01, rank=rank
        )
    else:
        guide = Guide(model_, init_loc_fn=init_loc_fn, init_scale=0.01, rank=rank)
    # This initializes the guide:
    latent_shapes = {k: v.shape for k, v in guide(dataset).items()}
    latent_numel = {k: v.numel() for k, v in latent_shapes.items()}
    logger.info(
        "\n".join(
            [f"Model has {sum(latent_numel.values())} latent variables of shapes:"]
            + [f" {k} {tuple(v)}" for k, v in latent_shapes.items()]
        )
    )
    param_shapes = {k: v.shape for k, v in pyro.get_param_store().named_parameters()}
    param_numel = {k: v.numel() for k, v in param_shapes.items()}
    logger.info(
        "\n".join(
            [f"Guide has {sum(param_numel.values())} parameters of shapes:"]
            + [f" {k} {tuple(v)}" for k, v in param_shapes.items()]
        )
    )

    # Log gradient norms during inference.
    series: dict = defaultdict(list)

    def hook(g, series):
        series.append(torch.linalg.norm(g.reshape(-1), math.inf).item())

    for name, value in pyro.get_param_store().named_parameters():
        value.register_hook(functools.partial(hook, series=series[name]))

    def optim_config(param_name):
        config: dict = {
            "lr": learning_rate,
            "lrd": learning_rate_decay ** (1 / num_steps),
            "clip_norm": clip_norm,
        }
        scalars = [k for k, v in latent_numel.items() if v == 1]
        if any("locs." + s in name for s in scalars):
            config["lr"] *= 0.2
        elif "scales" in param_name:
            config["lr"] *= 0.1
        elif "scale_tril" in param_name:
            config["lr"] *= 0.05
        elif "weight" in param_name:
            config["lr"] *= 0.05
        elif "_centered" in param_name:
            config["lr"] *= 0.1
        return config

    optim = ClippedAdam(optim_config)
    Elbo = JitTrace_ELBO if jit else Trace_ELBO
    elbo = Elbo(max_plate_nesting=2, ignore_jit_warnings=True)
    svi = SVI(model_, guide, optim, elbo)
    losses = []
    num_obs = dataset["weekly_strains"].count_nonzero()
    for step in range(num_steps):
        loss = svi.step(dataset=dataset)
        assert not math.isnan(loss)
        losses.append(loss)
        median = guide.median()
        for name, value in median.items():
            if value.numel() == 1:
                series[name].append(float(value))
        if log_every and step % log_every == 0:
            logger.info(
                " ".join(
                    [f"step {step: >4d} L={loss / num_obs:0.6g}"]
                    + [
                        "{}={:0.3g}".format(
                            "".join(p[0] for p in k.split("_")).upper(), v.item()
                        )
                        for k, v in median.items()
                        if v.numel() == 1
                    ]
                )
            )
        if check_loss and step >= 50:
            prev = torch.tensor(losses[-50:-25], device="cpu").median().item()
            curr = torch.tensor(losses[-25:], device="cpu").median().item()
            assert (curr - prev) < num_obs, "loss is increasing"

    result = predict(
        guide, dataset, num_samples=num_samples, forecast_steps=forecast_steps
    )
    result["losses"] = losses
    series["loss"] = losses
    result["series"] = dict(series)
    result["params"] = {
        k: v.detach().float().cpu().clone()
        for k, v in param_store.items()
        if v.numel() < 1e7
    }
    result["walltime"] = default_timer() - start_time
    return result


@torch.no_grad()
def log_stats(dataset: dict, result: dict) -> dict:
    """
    Logs statistics of predictions and model fit in the ``result`` of
    ``fit_svi()``.

    :param dict dataset: The dataset dictionary.
    :param dict result: The output of :func:`fit_svi`.
    :returns: A dictionary of statistics.
    """
    stats = {k: float(v) for k, v in result["median"].items() if v.numel() == 1}
    stats["loss"] = float(np.median(result["losses"][-100:]))
    mutations = dataset["mutations"]
    mean = result["mean"]["coef"].cpu()
    if not mean.shape:
        return stats  # Work around error in map estimation.

    # Statistical significance.
    std = result["std"]["coef"].cpu()
    sig = mean.abs() / std
    logger.info(f"|μ|/σ [median,max] = [{sig.median():0.3g},{sig.max():0.3g}]")
    stats["|μ|/σ median"] = sig.median()
    stats["|μ|/σ max"] = sig.max()

    # Effects of individual mutations.
    for name in ["S:D614G", "S:N501Y", "S:E484K", "S:L452R"]:
        if name not in mutations:
            continue
        i = mutations.index(name)
        m = mean[i] * 0.01
        s = std[i] * 0.01
        logger.info(f"ΔlogR({name}) = {m:0.3g} ± {s:0.2f}")
        stats[f"ΔlogR({name}) mean"] = m
        stats[f"ΔlogR({name}) std"] = s

    # Growth rates of individual lineages.
    try:
        i = dataset["lineage_id"]["A"]
        rate_A = result["mean"]["rate"][..., i].mean(0)
    except KeyError:
        rate_A = result["mean"]["rate"].median()
    for s in ["B.1.1.7", "B.1.617.2"]:
        i = dataset["lineage_id"][s]
        rate = result["median"]["rate"][..., i].mean()
        R_RA = (rate - rate_A).exp()
        logger.info(f"R({s})/R(A) = {R_RA:0.3g}")
        stats[f"R({s})/R(A)"] = R_RA

    # Accuracy of mutation-only model, ie without region-local effects.
    true = dataset["weekly_strains"] + 1e-20  # avoid nans
    counts = true.sum(-1, True)
    true_probs = true / counts
    local_time = dataset["local_time"][..., None]
    local_time = local_time + result["params"]["local_time"].to(local_time.device)
    rate = 0.01 * result["median"]["coef"] @ dataset["features"].T
    pred = result["median"]["init"] + rate * local_time
    pred -= pred.logsumexp(-1, True)  # apply log sigmoid function
    kl = true.mul(true_probs.log() - pred).sum(-1)
    kl = stats["naive KL"] = kl.sum() / counts.sum()  # in units of nats / observation
    error = (pred.exp() - true_probs) * counts ** 0.5  # scaled by Poisson stddev
    mae = stats["naive MAE"] = error.abs().sum(-1).mean()
    rmse = stats["naive RMSE"] = error.square().sum(-1).mean().sqrt()
    logger.info(f"naive KL = {kl:0.4g}, MAE = {mae:0.4g}, RMSE = {rmse:0.4g}")

    # Posterior predictive error.
    pred = result["median"]["probs"][: len(true)] + 1e-20  # truncate, avoid nans
    kl = true.mul(true_probs.log() - pred.log()).sum([0, -1])
    error = (pred - true_probs) * counts ** 0.5  # scaled by Poisson stddev
    mae = error.abs().mean(0)  # average over time
    mse = error.square().mean(0)  # average over time
    stats["MAE"] = mae.sum(-1).mean()  # average over region
    stats["RMSE"] = mse.sum(-1).mean().sqrt()  # root average over region
    stats["KL"] = kl.sum() / counts.sum()  # in units of nats / observation
    logger.info("KL = {KL:0.4g}, MAE = {MAE:0.4g}, RMSE = {RMSE:0.4g}".format(**stats))

    # Examine the MSE and RMSE over a few regions of interest.
    queries = {
        "England": ["B.1.1.7"],
        # "England": ["B.1.1.7", "B.1.177", "B.1.1", "B.1"],
        # "USA / California": ["B.1.1.7", "B.1.429", "B.1.427", "B.1.2", "B.1", "P.1"],
    }
    for place, strains in queries.items():
        matches = [p for name, p in dataset["location_id"].items() if place in name]
        if not matches:
            continue
        assert len(matches) == 1, matches
        p = matches[0]
        stats[f"{place} KL"] = kl[p].sum() / true[:, p].sum()
        stats[f"{place} MAE"] = mae[p].sum()
        stats[f"{place} RMSE"] = mse[p].sum().sqrt()
        logger.info(
            "{}\tKL = {:0.3g}, MAE = {:0.3g}, RMSE = {:0.3g}".format(
                place,
                stats[f"{place} KL"],
                stats[f"{place} MAE"],
                stats[f"{place} RMSE"],
            )
        )

        for strain in strains:
            s = dataset["lineage_id"][strain]
            stats[f"{place} {strain} MAE"] = mae[p, s]
            stats[f"{place} {strain} RMSE"] = mse[p, s].sqrt()
            logger.info(
                "{} {}\tMAE = {:0.3g}, RMSE = {:0.3g}".format(
                    place,
                    strain,
                    stats[f"{place} {strain} MAE"],
                    stats[f"{place} {strain} RMSE"],
                )
            )

    return {k: float(v) for k, v in stats.items()}


@torch.no_grad()
def log_holdout_stats(fits: dict) -> dict:
    """
    Logs statistics comparing multiple results from ``fit_svi``.
    """
    assert len(fits) > 1
    fits = list(fits.items())
    stats = {}
    for i, (name1, fit1) in enumerate(fits[:-1]):
        for name2, fit2 in fits[i + 1 :]:
            # Compute mutation similarity.
            mutations = sorted(set(fit1["mutations"]) & set(fit2["mutations"]))
            medians = []
            for fit in (fit1, fit2):
                mutation_id = {m: i for i, m in enumerate(fit["mutations"])}
                idx = torch.tensor([mutation_id[m] for m in mutations])
                medians.append(fit["median"]["coef"][idx] * 0.01)
            error = medians[0] - medians[1]
            mutation_std = torch.cat(medians).std().item()
            mutation_rmse = error.square().mean().sqrt().item()
            mutation_mae = error.abs().mean().item()
            mutation_correlation = pearson_correlation(medians[0], medians[1]).item()

            # Compute lineage similarity.
            means = []
            for fit in (fit1, fit2):
                rate = fit["mean"]["rate"]
                if rate.dim() == 2:
                    rate = rate.mean(0)
                means.append(rate)
            error = means[0] - means[1]
            lineage_std = torch.cat(means).std().item()
            lineage_rmse = error.square().mean().sqrt().item()
            lineage_mae = error.abs().mean().item()
            lineage_correlation = pearson_correlation(means[0], means[1]).item()

            # Print stats.
            logger.info(
                f"{name1} vs {name2} mutations: "
                f"ρ = {mutation_correlation:0.3g}, "
                f"RMSE = {mutation_rmse:0.3g}, "
                f"MAE = {mutation_mae:0.3g}"
            )
            logger.info(
                f"{name1} vs {name2} lineages: "
                f"ρ = {lineage_correlation:0.3g}, "
                f"RMSE = {lineage_rmse:0.3g}, "
                f"MAE = {lineage_mae:0.3g}"
            )

            # Save stats.
            stats["mutation_corr"] = mutation_correlation
            stats["mutation_rmse"] = mutation_rmse
            stats["mutation_mae"] = mutation_mae
            stats["mutation_stddev"] = mutation_std
            stats["lineage_corr"] = lineage_correlation
            stats["lineage_rmse"] = lineage_rmse
            stats["lineage_mae"] = lineage_mae
            stats["lineage_stdev"] = lineage_std

    return {k: float(v) for k, v in stats.items()}
