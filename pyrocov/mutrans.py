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
import tqdm
from pyro import poutine
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.infer.autoguide import (
    AutoDelta,
    AutoGuideList,
    AutoLowRankMultivariateNormal,
    AutoNormal,
    AutoStructured,
)
from pyro.infer.autoguide.initialization import InitMessenger
from pyro.infer.reparam import LocScaleReparam
from pyro.nn.module import PyroModule, PyroParam
from pyro.ops.streaming import CountMeanVarianceStats, StatsOfDict
from pyro.optim import ClippedAdam
from pyro.poutine.util import site_is_subsample
from torch.distributions import constraints

import pyrocov.geo

from . import pangolin, sarscov2
from .ops import sparse_multinomial_likelihood
from .util import pearson_correlation, quotient_central_moments

# Requires https://github.com/pyro-ppl/pyro/pull/2953
try:
    from pyro.infer.autoguide.effect import AutoRegressiveMessenger
except ImportError:
    AutoRegressiveMessenger = object

logger = logging.getLogger(__name__)

# Reasonable values might be week (7), fortnight (14), or month (28)
TIMESTEP = 14  # in days
GENERATION_TIME = 5.5  # in days
START_DATE = "2019-12-01"


def date_range(stop):
    start = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    step = datetime.timedelta(days=TIMESTEP)
    return np.array([start + step * t for t in range(stop)])


def get_fine_regions(columns, min_samples):
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


def rank_loo_lineages(full_dataset: dict, min_samples: int = 50) -> List[str]:
    """
    Compute a list of lineages ranked in descending order of cut size.
    This is used in growth rate leave-one-out prediction experiments.
    """

    def get_parent(lineage):
        lineage = pangolin.decompress(lineage)
        parent = pangolin.get_parent(lineage)
        if parent is not None:
            parent = pangolin.compress(parent)
        return parent

    # Compute sample counts.
    lineage_id_inv = full_dataset["lineage_id_inv"]
    lineage_id = full_dataset["lineage_id"]
    clade_counts = full_dataset["weekly_clades"].sum([0, 1])
    lineage_counts = clade_counts.new_zeros(len(lineage_id)).scatter_add_(
        0, full_dataset["clade_id_to_lineage_id"], clade_counts
    )
    weekly_clades = full_dataset["weekly_clades"]  # [T, P, C]
    lineage_counts = weekly_clades.sum([0, 1])  # [C]
    descendent_counts = lineage_counts.clone()
    for c, lineage in enumerate(lineage_id_inv):
        ancestor = get_parent(lineage)
        while ancestor is not None:
            a = lineage_id.get(ancestor)
            if a is not None:
                descendent_counts[a] += lineage_counts[c]
            ancestor = get_parent(ancestor)
    total = lineage_counts.sum().item()
    cut_size = torch.min(descendent_counts, total - descendent_counts)

    # Filter and sort lineages by cut size.
    ranked_lineages = [
        (size, lineage)
        for size, lineage in zip(cut_size.tolist(), lineage_id_inv)
        if lineage not in ("A", "B", "B.1")
        if size >= min_samples
    ]
    ranked_lineages.sort(reverse=True)
    return [name for gap, name in ranked_lineages]


def dense_to_sparse(x):
    index = x.nonzero(as_tuple=False).T.contiguous()
    value = x[tuple(index)]
    total = x.sum(-1)
    return {"index": index, "value": value, "total": total}


def load_gisaid_data(
    *,
    device="cpu",
    min_region_size=50,
    include={},
    exclude={},
    end_day=None,
    columns_filename="results/usher.columns.pkl",
    features_filename="results/usher.features.pt",
    feature_type="aa",
) -> dict:
    """
    Loads the two files columns_filename and features_filename,
    converts the input to PyTorch tensors and truncates the data according to
    ``include`` and ``exclude``.

    :param str device: torch device to use
    :param dict include: filters of data to include
    :param dict exclude: filters of data to exclude
    :param end_day: last day to include
    :param str columns_filename:
    :param str features_filename:
    :param str feature_type: Either "aa" for amino acid features or "nuc" for
        nucleotide features.
    :returns: A dataset dict
    :rtype: dict
    """
    logger.info("Loading data")
    include = include.copy()
    exclude = exclude.copy()

    if end_day:
        logger.info(f"Load gisaid data end_day: {end_day}")

    # Load column data.
    with open(columns_filename, "rb") as f:
        columns = pickle.load(f)
    logger.info(f"Training on {len(columns['day'])} rows with columns:")
    logger.info(", ".join(columns.keys()))

    # Aggregate regions smaller than min_region_size to country level.
    fine_regions = get_fine_regions(columns, min_region_size)

    # Filter features into numbers of mutations and possibly genes.
    usher_features = torch.load(features_filename)
    mutations = usher_features[f"{feature_type}_mutations"]
    features = usher_features[f"{feature_type}_features"].to(
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

    # Construct the list of clades.
    clade_id_inv = usher_features["clades"]
    clade_id = {k: i for i, k in enumerate(clade_id_inv)}
    clades = columns["clade"]

    # Generate sparse_data.
    sparse_data: dict = Counter()
    countries = set()
    states = set()
    state_to_country_dict = {}
    location_id: dict = OrderedDict()
    skipped_clades = set()
    num_obs = 0
    for day, location, clade in zip(columns["day"], columns["location"], clades):
        if clade not in clade_id:
            if clade not in skipped_clades:
                skipped_clades.add(clade)
                if not clade.startswith("fine"):
                    logger.warning(f"WARNING skipping unsampled clade {clade}")
            continue

        # Filter by include/exclude
        row = {
            "location": location,
            "day": day,
            "clade": clade,
        }
        if not all(re.search(v, row[k]) for k, v in include.items()):
            continue
        if any(re.search(v, row[k]) for k, v in exclude.items()):
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
        # Populate countries on the left and states on the right.
        if len(parts) == 2:  # country only
            countries.add(location)
            p = location_id.setdefault(location, len(countries) - 1)
        else:  # state and country
            country = " / ".join(parts[:2])
            countries.add(country)
            c = location_id.setdefault(country, len(countries) - 1)
            states.add(location)
            p = location_id.setdefault(location, -len(states))
            state_to_country_dict[p] = c

        # Save sparse data.
        num_obs += 1
        t = day // TIMESTEP
        c = clade_id[clade]
        sparse_data[t, p, c] += 1
    logger.warning(f"WARNING skipped {len(skipped_clades)} unsampled clades")
    state_to_country = torch.full((len(states),), 999999, dtype=torch.long)
    for s, c in state_to_country_dict.items():
        state_to_country[s] = c
    logger.info(f"Found {len(states)} states in {len(countries)} countries")
    location_id_inv = [None] * len(location_id)
    for k, i in location_id.items():
        location_id_inv[i] = k
    assert all(location_id_inv)

    # Generate weekly_clades tensor from sparse_data.
    if end_day is not None:
        T = 1 + end_day // TIMESTEP
    else:
        T = 1 + max(columns["day"]) // TIMESTEP
    P = len(location_id)
    C = len(clade_id)
    weekly_clades = torch.zeros(T, P, C)
    for tps, n in sparse_data.items():
        weekly_clades[tps] = n
    logger.info(f"Dataset size [T x P x C] {T} x {P} x {C}")

    logger.info(
        f"Keeping {num_obs}/{len(clades)} rows "
        f"(dropped {len(clades) - int(num_obs)})"
    )

    # Construct sparse representation.
    pc_index = weekly_clades.ne(0).any(0).reshape(-1).nonzero(as_tuple=True)[0]
    sparse_counts = dense_to_sparse(weekly_clades)

    # Construct time scales centered around observations.
    time = torch.arange(float(T)) * TIMESTEP / GENERATION_TIME
    time -= time.mean()

    # Construct lineage <-> clade mappings.
    lineage_to_clade = usher_features["lineage_to_clade"]
    clade_to_lineage = usher_features["clade_to_lineage"]
    lineage_id_inv = sorted(lineage_to_clade)
    lineage_id = {k: i for i, k in enumerate(lineage_id_inv)}
    clade_id_to_lineage_id = torch.zeros(len(clade_to_lineage), dtype=torch.long)
    for c, l in clade_to_lineage.items():
        clade_id_to_lineage_id[clade_id[c]] = lineage_id[l]
    lineage_id_to_clade_id = torch.zeros(len(lineage_to_clade), dtype=torch.long)
    for l, c in lineage_to_clade.items():
        lineage_id_to_clade_id[lineage_id[l]] = clade_id[c]

    dataset = {
        "clade_id": clade_id,
        "clade_id_inv": clade_id_inv,
        "clade_id_to_lineage_id": clade_id_to_lineage_id,
        "clade_to_lineage": usher_features["clade_to_lineage"],
        "features": features,
        "lineage_id": lineage_id,
        "lineage_id_inv": lineage_id_inv,
        "lineage_id_to_clade_id": lineage_id_to_clade_id,
        "lineage_to_clade": usher_features["lineage_to_clade"],
        "location_id": location_id,
        "location_id_inv": location_id_inv,
        "mutations": mutations,
        "pc_index": pc_index,
        "sparse_counts": sparse_counts,
        "state_to_country": state_to_country,
        "time": time,
        "weekly_clades": weekly_clades,
    }
    return dataset


def subset_gisaid_data(
    gisaid_dataset: dict,
    location_queries=None,
    max_clades=math.inf,
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
        new["weekly_clades"] = new["weekly_clades"].index_select(1, ids)

    # Select clades.
    if new["weekly_clades"].size(-1) > max_clades:
        ids = (
            new["weekly_clades"]
            .sum([0, 1])
            .sort(0, descending=True)
            .indices[:max_clades]
        )
        new["weekly_clades"] = new["weekly_clades"].index_select(-1, ids)
        new["features"] = new["features"].index_select(0, ids)
        new["clade_id_inv"] = [new["clade_id_inv"][i] for i in ids.tolist()]
        new["clade_id"] = {name: i for i, name in enumerate(new["clade_id_inv"])}
        new["sparse_counts"] = dense_to_sparse(new["weekly_clades"])

    # Select mutations.
    gaps = new["features"].max(0).values - new["features"].min(0).values
    ids = (gaps >= 0.5).nonzero(as_tuple=True)[0]
    new["mutations"] = [new["mutations"][i] for i in ids.tolist()]
    new["features"] = new["features"].index_select(-1, ids)

    logger.info(
        "Selected {}/{} places, {}/{} clades, {}/{} mutations, {}/{} samples".format(
            len(new["location_id"]),
            len(old["location_id"]),
            len(new["clade_id"]),
            len(old["clade_id"]),
            len(new["mutations"]),
            len(old["mutations"]),
            int(new["weekly_clades"].sum()),
            int(old["weekly_clades"].sum()),
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
    T = len(gisaid_data["weekly_clades"])
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


def model(dataset, model_type, *, forecast_steps=None):
    """
    Bayesian regression model of clade portions as a function of mutation features.

    This function can be run in two different modes:
    - During training, ``forecast_steps=None`` and the model is conditioned on
      observed data.
    - During prediction (after training), the likelihood statement is omitted
      and instead a ``probs`` tensor is recorded; this is the predicted clade
      portions in each (time, regin) bin.
    """
    # Tensor shapes are commented at at the end of some lines.
    features = dataset["features"]
    time = dataset["time"]  # [T]
    weekly_clades = dataset["weekly_clades"]
    sparse_counts = dataset["sparse_counts"]
    clade_id_to_lineage_id = dataset["clade_id_to_lineage_id"]
    pc_index = dataset["pc_index"]
    T, P, C = weekly_clades.shape
    C, F = features.shape
    L = len(dataset["lineage_id"])
    PC = len(pc_index)
    assert PC <= P * C
    assert time.shape == (T,)
    assert clade_id_to_lineage_id.shape == (C,)

    # Optionally extend time axis.
    if forecast_steps is not None:  # During prediction.
        T = T + forecast_steps
        t0 = time[0]
        dt = time[1] - time[0]
        time = t0 + dt * torch.arange(float(T))
        assert time.shape == (T,)

    clade_plate = pyro.plate("clade", C, dim=-1)
    place_plate = pyro.plate("place", P, dim=-2)
    time_plate = pyro.plate("time", T, dim=-3)
    pc_plate = pyro.plate("place_clade", PC, dim=-1)

    # Configure reparametrization (which does not affect model density).
    reparam = {}
    if "reparam" in model_type:
        if "nofeatures" not in model_type:
            reparam["coef"] = LocScaleReparam()
        if "localrate" in model_type or "nofeatures" in model_type:
            reparam["rate_loc"] = LocScaleReparam()
        if "localinit" in model_type:
            reparam["init_loc"] = LocScaleReparam()
        reparam["pc_rate"] = LocScaleReparam()
        reparam["pc_init"] = LocScaleReparam()
    with poutine.reparam(config=reparam):

        # Sample global random variables.
        if "nofeatures" not in model_type:
            coef_scale = pyro.sample("coef_scale", dist.LogNormal(-4, 2))
        rate_scale = pyro.sample("rate_scale", dist.LogNormal(-4, 2))
        init_scale = pyro.sample("init_scale", dist.LogNormal(0, 2))
        if "localrate" or "nofeatures" in model_type:
            rate_loc_scale = pyro.sample("rate_loc_scale", dist.LogNormal(-4, 2))
        if "localinit" in model_type:
            init_loc_scale = pyro.sample("init_loc_scale", dist.LogNormal(0, 2))

        # Assume relative growth rate depends strongly on mutations and weakly
        # on clade and place. Assume initial infections depend strongly on
        # clade and place.
        if "nofeatures" not in model_type:
            coef = pyro.sample(
                "coef", dist.Laplace(torch.zeros(F), coef_scale).to_event(1)
            )  # [F]
        with clade_plate:
            if "localrate" in model_type:
                rate_loc = pyro.sample(
                    "rate_loc", dist.Normal(0.01 * coef @ features.T, rate_loc_scale)
                )  # [C]
            elif "nofeatures" in model_type:
                rate_loc = pyro.sample(
                    "rate_loc", dist.Normal(torch.zeros((C,)), rate_loc_scale.expand(C))
                )  # [C]
            else:
                rate_loc = pyro.deterministic(
                    "rate_loc", 0.01 * coef @ features.T
                )  # [C]
            if "localinit" in model_type:
                init_loc = pyro.sample(
                    "init_loc", dist.Normal(0, init_loc_scale)
                )  # [C]
            else:
                init_loc = rate_loc.new_zeros(())
        with pc_plate:
            pc_rate_loc = rate_loc.expand(P, C).reshape(-1)
            pc_init_loc = init_loc.expand(P, C).reshape(-1)
            pc_rate = pyro.sample(
                "pc_rate", dist.Normal(pc_rate_loc[pc_index], rate_scale)
            )  # [PC]
            pc_init = pyro.sample(
                "pc_init", dist.Normal(pc_init_loc[pc_index], init_scale)
            )  # [PC]
        with place_plate, clade_plate:
            rate = pyro.deterministic(
                "rate",
                pc_rate_loc.scatter(0, pc_index, pc_rate).reshape(P, C),
            )  # [P, C]
            init = pyro.deterministic(
                "init",
                torch.full((P * C,), -1e2).scatter(0, pc_index, pc_init).reshape(P, C),
            )  # [P, C]
        logits = init + rate * time[:, None, None]  # [T, P, C]

        # Optionally predict probabilities (during prediction).
        if forecast_steps is not None:
            probs = logits.new_zeros(logits.shape[:2] + (L,)).scatter_add_(
                -1, clade_id_to_lineage_id.expand_as(logits), logits.softmax(-1)
            )
            with time_plate, place_plate, pyro.plate("lineage", L, dim=-1):
                pyro.deterministic("probs", probs)
            return

        # Finally observe counts (during inference).
        if "dense" in model_type:  # equivalent either way
            # Compute a dense likelihood.
            with time_plate, place_plate:
                pyro.sample(
                    "obs",
                    dist.Multinomial(logits=logits.unsqueeze(-2), validate_args=False),
                    obs=weekly_clades.unsqueeze(-2),
                )  # [T, P, 1, C]
            return
        # Compromise between sparse and dense.
        logits = logits.log_softmax(-1)
        t, p, c = sparse_counts["index"]
        pyro.factor(
            "obs",
            sparse_multinomial_likelihood(
                sparse_counts["total"], logits[t, p, c], sparse_counts["value"]
            ),
        )


class InitLocFn:
    """
    Initializer for latent variables.

    This is passed as the ``init_loc_fn`` to guides.
    """

    def __init__(self, dataset):
        # Initialize init.
        init = dataset["weekly_clades"].sum(0)  # [P, C]
        init.add_(1 / init.size(-1)).div_(init.sum(-1, True))
        init.log_().sub_(init.median(-1, True).values).add_(torch.randn(init.shape))
        self.init = init  # [P, C]
        self.init_decentered = init / 2
        self.init_loc = init.mean(0)  # [C]
        self.init_loc_decentered = self.init_loc / 2
        self.pc_init = self.init.reshape(-1)[dataset["pc_index"]]
        self.pc_init = self.pc_init / 2
        assert not torch.isnan(self.init).any()
        logger.info(f"init stddev = {self.init.std():0.3g}")

    def __call__(self, site):
        name = site["name"]
        shape = site["fn"].shape()
        if hasattr(self, name):
            result = getattr(self, name)
            assert result.shape == shape
            return result
        if name in (
            "coef_scale",
            "init_scale",
            "init_loc_scale",
        ):
            return torch.ones(shape)
        if name == "logits_scale":
            return torch.full(shape, 0.002)
        if name in (
            "rate_scale",
            "rate_loc_scale",
            "place_scale",
            "clade_scale",
        ):
            return torch.full(shape, 0.01)
        if name in (
            "rate_loc",
            "rate_loc_decentered",
            "coef",
            "coef_decentered",
            "rate",
            "rate_decentered",
            "pc_rate",
            "pc_rate_decentered",
        ):
            return torch.rand(shape).sub_(0.5).mul_(0.01)
        if name == "coef_loc":
            return torch.rand(shape).sub_(0.5).mul_(0.01).add_(1.0)
        raise ValueError(f"InitLocFn found unhandled site {repr(name)}; please update.")


class Guide(AutoGuideList):
    """
    Custom guide for large-scale inference.

    This combines a low-rank multivariate normal guide over small variables
    with a mean field guide over remaining latent variables.
    """

    def __init__(self, model, init_loc_fn, init_scale, rank):
        super().__init__(InitMessenger(init_loc_fn)(model))

        # Jointly estimate globals, mutation coefficients, and clade coefficients.
        mvn = [
            "coef_scale",
            "rate_loc_scale",
            "init_loc_scale",
            "rate_scale",
            "init_scale",
            "coef",
            "coef_decentered",
            "rate_loc",
            "rate_loc_decentered",
            "init_loc",
            "init_loc_decentered",
        ]
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


class RegressiveGuide(AutoRegressiveMessenger):
    def get_posterior(self, name, prior):
        if name == "coef":
            if not hasattr(self, "coef"):
                # Initialize.
                self.coef = PyroModule()
                n = prior.shape()[-1]
                rank = 100
                assert n > 1
                init_loc = self.init_loc_fn({"name": name, "fn": prior})
                self.coef.loc = PyroParam(init_loc, event_dim=1)
                self.coef.scale = PyroParam(
                    torch.full((n,), self._init_scale),
                    event_dim=1,
                    constraint=constraints.positive,
                )
                self.coef.cov_factor = PyroParam(
                    torch.empty(n, rank).normal_(0, 1 / rank ** 0.5),
                    event_dim=2,
                )
            scale = self.coef.scale
            cov_factor = self.coef.cov_factor * scale.unsqueeze(-1)
            cov_diag = scale * scale
            return dist.LowRankMultivariateNormal(self.coef.loc, cov_factor, cov_diag)

        return super().get_posterior(name, prior)


@torch.no_grad()
@poutine.mask(mask=False)
def predict(
    model,
    guide,
    dataset,
    model_type,
    *,
    num_samples=1000,
    vectorize=None,
    save_params=("rate", "init", "probs"),
    forecast_steps=0,
) -> dict:
    def get_conditionals(data):
        trace = poutine.trace(poutine.condition(model, data)).get_trace(
            dataset, model_type, forecast_steps=forecast_steps
        )
        return {
            name: site["value"].detach()
            for name, site in trace.nodes.items()
            if site["type"] == "sample" and not site_is_subsample(site)
            if not name.startswith("obs")
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
        with pyro.plate("particles", num_samples, dim=-4):
            samples = get_conditionals(guide())
        for k, v in samples.items():
            if k in save_params:
                result["mean"][k] = v.mean(0).squeeze()
                result["std"][k] = v.std(0).squeeze()
    else:
        stats = StatsOfDict({k: CountMeanVarianceStats for k in save_params})
        for _ in tqdm.tqdm(range(num_samples)):
            stats.update(get_conditionals(guide()))
        for name, stats_ in stats.get().items():
            if "mean" in stats_:
                result["mean"][name] = stats_["mean"]
            if "variance" in stats_:
                result["std"][name] = stats_["variance"].sqrt()
    return dict(result)


def fit_svi(
    dataset: dict,
    *,
    model_type: str,
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
    num_ell_particles=256,
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
    Elbo = JitTrace_ELBO if jit else Trace_ELBO
    if guide_type == "map":
        guide = AutoDelta(model_, init_loc_fn=init_loc_fn)
    elif guide_type == "normal":
        guide = AutoNormal(model_, init_loc_fn=init_loc_fn, init_scale=0.01)
    elif guide_type == "full":
        guide = AutoLowRankMultivariateNormal(
            model_, init_loc_fn=init_loc_fn, init_scale=0.01, rank=rank
        )
    elif guide_type == "structured":
        guide = AutoStructured(
            model_,
            init_loc_fn=init_loc_fn,
            init_scale=0.01,
            conditionals=defaultdict(
                lambda: "normal",
                rate_scale="delta",
                init_loc_scale="delta",
                init_scale="delta",
                coef="mvn",
                coef_decentered="mvn",
            ),
        )
    elif guide_type == "regressive":
        guide = RegressiveGuide(model_, init_loc_fn=init_loc_fn, init_scale=0.01)
    else:
        guide = Guide(model_, init_loc_fn=init_loc_fn, init_scale=0.01, rank=rank)
    # This initializes the guide:
    latent_shapes = {k: v.shape for k, v in guide(dataset, model_type).items()}
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
        elif "factors" in param_name or "prec_sqrts" in param_name:
            config["lr"] *= 0.05
        elif "weight_" in param_name:
            config["lr"] *= 0.01
        elif "weight" in param_name:
            config["lr"] *= 0.03
        elif "_centered" in param_name:
            config["lr"] *= 0.1
        return config

    optim = ClippedAdam(optim_config)
    elbo = Elbo(max_plate_nesting=3, ignore_jit_warnings=True)
    svi = SVI(model_, guide, optim, elbo)
    losses = []
    num_obs = dataset["weekly_clades"].count_nonzero()
    for step in range(num_steps):
        loss = svi.step(dataset=dataset, model_type=model_type)
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

    # compute expected log probability
    ell = 0.0
    with torch.no_grad(), poutine.block():
        for _ in range(num_ell_particles):
            guide_trace = poutine.trace(guide).get_trace(
                dataset=dataset, model_type=model_type
            )
            replayed_model = poutine.replay(model_, trace=guide_trace)
            model_trace = poutine.trace(replayed_model).get_trace(
                dataset=dataset, model_type=model_type
            )
            model_trace.compute_log_prob()
            ell += model_trace.nodes["obs"]["unscaled_log_prob"].item() / float(
                num_ell_particles
            )

    result = predict(
        model_,
        guide,
        dataset,
        model_type,
        num_samples=num_samples,
        forecast_steps=forecast_steps,
    )
    result["ELL"] = ell
    result["losses"] = losses
    series["loss"] = losses
    result["series"] = dict(series)
    result["params"] = {
        k: v.detach().float().cpu().clone()
        for k, v in param_store.items()
        if v.numel() < 1e8
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

    if "coef" in result["mean"]:
        mean = result["mean"]["coef"].cpu()
        if not mean.shape:
            return stats  # Work around error in map estimation.
        logger.info(
            "Dense data has shape {} totaling {} sequences".format(
                " x ".join(map(str, dataset["weekly_clades"].shape)),
                int(dataset["weekly_clades"].sum()),
            )
        )

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

    # Growth rates of individual clades.
    rate = quotient_central_moments(
        result["mean"]["rate"].mean(0), dataset["clade_id_to_lineage_id"]
    )[1]
    rate = rate - rate[dataset["lineage_id"]["A"]]
    for lineage in ["B.1.1.7", "B.1.617.2", "AY.23.1"]:
        R_RA = float(rate[dataset["lineage_id"][lineage]].exp())
        logger.info(f"R({lineage})/R(A) = {R_RA:0.3g}")
        stats[f"R({lineage})/R(A)"] = R_RA

    # Posterior predictive error.
    L = len(dataset["lineage_id"])
    weekly_clades = dataset["weekly_clades"]
    weekly_lineages = torch.zeros(weekly_clades.shape[:-1] + (L,)).scatter_add_(
        -1, dataset["clade_id_to_lineage_id"].expand_as(weekly_clades), weekly_clades
    )
    true = weekly_lineages + 1e-20  # avoid nans
    counts = true.sum(-1, True)
    true_probs = true / counts
    pred = result["median"]["probs"][: len(true)] + 1e-20  # truncate, avoid nans
    kl = true.mul(true_probs.log() - pred.log()).sum([0, -1])
    error = (pred - true_probs) * counts ** 0.5  # scaled by Poisson stddev
    mae = error.abs().mean(0)  # average over time
    mse = error.square().mean(0)  # average over time
    stats["MAE"] = float(mae.sum(-1).mean())  # average over region
    stats["RMSE"] = float(mse.sum(-1).mean().sqrt())  # root average over region
    stats["KL"] = float(kl.sum() / counts.sum())  # in units of nats / observation
    if "ELL" in result:
        stats["ELL"] = result["ELL"]

    logger.info("KL = {KL:0.4g}, MAE = {MAE:0.4g}, RMSE = {RMSE:0.4g}".format(**stats))

    # Examine the MSE and RMSE over a few regions of interest.
    queries = {
        "England": ["B.1.1.7"],
        # "England": ["B.1.1.7", "B.1.177", "B.1.1", "B.1"],
        # "USA / California": ["B.1.1.7", "B.1.429", "B.1.427", "B.1.2", "B.1", "P.1"],
    }
    for place, lineages in queries.items():
        matches = [p for name, p in dataset["location_id"].items() if place in name]
        if not matches:
            continue
        assert len(matches) == 1, matches
        p = matches[0]
        stats[f"{place} KL"] = float(kl[p].sum() / true[:, p].sum())
        stats[f"{place} MAE"] = float(mae[p].sum())
        stats[f"{place} RMSE"] = float(mse[p].sum().sqrt())
        logger.info(
            "{}\tKL = {:0.3g}, MAE = {:0.3g}, RMSE = {:0.3g}".format(
                place,
                stats[f"{place} KL"],
                stats[f"{place} MAE"],
                stats[f"{place} RMSE"],
            )
        )

        for lineage in lineages:
            i = dataset["lineage_id"][lineage]
            stats[f"{place} {lineage} MAE"] = mae[p, i]
            stats[f"{place} {lineage} RMSE"] = mse[p, i].sqrt()
            logger.info(
                "{} {}\tMAE = {:0.3g}, RMSE = {:0.3g}".format(
                    place,
                    lineage,
                    stats[f"{place} {lineage} MAE"],
                    stats[f"{place} {lineage} RMSE"],
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
