import datetime
import functools
import logging
import math
import pickle
import re
from collections import Counter, OrderedDict, defaultdict

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoStructured, init_to_feasible
from pyro.infer.reparam import StructuredReparam
from pyro.optim import ClippedAdam
from pyro.poutine.util import site_is_subsample

import pyrocov.geo
from pyrocov import pangolin

logger = logging.getLogger(__name__)

# Reasonable values might be week (7), fortnight (14), or month (28)
TIMESTEP = 7  # in days
GENERATION_TIME = 5.5  # in days
START_DATE = "2019-12-01"


def get_fine_countries(columns, min_samples=1000):
    """
    Select countries have at least two subregions with at least ``min_samples``
    samples. These will be finely partitioned into subregions. Remaining
    countries will be coarsely aggregated at country level.
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
    fine_countries = Counter()
    for parts, count in counts.items():
        if count >= min_samples:
            fine_countries[parts[1]] += 1
    fine_countries = frozenset(
        name for name, count in fine_countries.items() if count >= 2
    )
    logger.info(
        "Partitioning the following countries into subregions: {}".format(
            ", ".join(sorted(fine_countries))
        )
    )
    return fine_countries


def load_gisaid_data(
    *,
    device="cpu",
    include={},
    exclude={},
):
    logger.info("Loading data")
    include = {k: re.compile(v) for k, v in include.items()}
    exclude = {k: re.compile(v) for k, v in exclude.items()}
    with open("results/gisaid.columns.pkl", "rb") as f:
        columns = pickle.load(f)
    logger.info("Training on {} rows with columns:".format(len(columns["day"])))
    logger.info(", ".join(columns.keys()))
    fine_countries = get_fine_countries(columns)

    # Filter features into numbers of mutations.
    aa_features = torch.load("results/nextclade.features.pt")
    mutations = aa_features["mutations"]
    features = aa_features["features"].to(
        device=device, dtype=torch.get_default_dtype()
    )
    keep = [m.count(",") == 0 for m in mutations]
    mutations = [m for k, m in zip(keep, mutations) if k]
    features = features[:, keep]
    logger.info("Loaded {} feature matrix".format(" x ".join(map(str, features.shape))))

    # Aggregate regions.
    lineages = list(map(pangolin.compress, columns["lineage"]))
    lineage_id_inv = list(map(pangolin.compress, aa_features["lineages"]))
    lineage_id = {k: i for i, k in enumerate(lineage_id_inv)}
    sparse_data = Counter()
    location_id = OrderedDict()
    skipped = set()
    for virus_name, day, location, lineage in zip(
        columns["virus_name"], columns["day"], columns["location"], lineages
    ):
        row = {"virus_name": virus_name, "location": location}
        if lineage not in lineage_id:
            if lineage not in skipped:
                skipped.add(lineage)
                logger.warning(f"WARNING skipping unsampled lineage {lineage}")
            continue
        if not all(v.search(row[k]) for k, v in include.items()):
            continue
        if any(v.search(row[k]) for k, v in exclude.items()):
            continue
        parts = location.split("/")
        if len(parts) < 2:
            continue
        parts = [p.strip() for p in parts[:3]]
        if parts[1] not in fine_countries:
            parts = parts[:2]
        location = " / ".join(parts)
        p = location_id.setdefault(location, len(location_id))
        s = lineage_id[lineage]
        t = day // TIMESTEP
        sparse_data[t, p, s] += 1

    T = 1 + max(columns["day"]) // TIMESTEP
    P = len(location_id)
    S = len(lineage_id)
    weekly_strains = torch.zeros(T, P, S)
    for (t, p, s), n in sparse_data.items():
        weekly_strains[t, p, s] = n
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


def load_jhu_data(gisaid_data):
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


def model(dataset, *, obs=True):
    local_time = dataset["local_time"]
    weekly_strains = dataset["weekly_strains"]
    features = dataset["features"]
    assert weekly_strains.shape[-1] == features.shape[0]
    assert local_time.shape == weekly_strains.shape[:2]
    T, P, S = weekly_strains.shape
    S, F = features.shape

    # Sample global parameters.
    feature_scale = pyro.sample("feature_scale", dist.LogNormal(0, 1))
    place_scale = pyro.sample("place_scale", dist.LogNormal(-4, 1))
    concentration = pyro.sample("concentration", dist.LogNormal(2, 4))

    # Assume relative growth rate depends strongly on mutations and weakly on place.
    rate_coef = pyro.sample(
        "rate_coef", dist.SoftLaplace(torch.zeros(F), feature_scale).to_event(1)
    )
    if not obs:
        return
    with pyro.plate("place", P, dim=-1):
        rate_bias = pyro.sample(
            "rate_bias", dist.Normal(torch.zeros(P, S), place_scale).to_event(1)
        )
        rate = pyro.deterministic("rate", 0.01 * (rate_bias + rate_coef @ features.T))

        # Finally observe overdispersed counts.
        init = pyro.sample("init", dist.SoftLaplace(torch.zeros(S), 10).to_event(1))
        strain_probs = (init + rate * local_time[:, :, None]).softmax(-1)
        with pyro.plate("time", T, dim=-2):
            pyro.sample(
                "obs",
                dist.DirichletMultinomial(
                    total_count=weekly_strains.sum(-1).max(),
                    concentration=concentration * strain_probs + 1e-20,
                    is_sparse=True,  # uses a faster algorithm
                ),
                obs=weekly_strains,
            )


class InitLocFn:
    def __init__(self, dataset):
        # Initialize init to mean.
        init = dataset["weekly_strains"].sum(0).add_(1)
        init.div_(init.sum(-1, True)).log_()
        init.sub_(init.mean(-1, True))
        self.init = init
        logger.info(f"init stddev = {init.std():0.3g}")

    def __call__(self, site):
        name = site["name"]
        shape = site["fn"].shape()
        if name == "feature_scale":
            return torch.ones(shape)
        if name == "place_scale":
            return torch.full(shape, 0.02)
        if name == "concentration":
            return torch.full(shape, 10.0)
        if name in ("rate_coef", "rate_bias"):
            return torch.randn(shape) * 0.01
        elif name == "init":
            return self.init
        raise ValueError(site["name"])


class RateBiasRateCoefLinear(torch.nn.Module):
    def __init__(self, P, S, F):
        super().__init__()
        self.PSF = P, S, F
        self.weight = torch.nn.Parameter(torch.zeros(F, S))

    def forward(self, x):
        P, S, F = self.PSF
        batch_shape = x.shape[:-1]
        y = x @ self.weight
        y = y.reshape(batch_shape + (1, S))
        y = y.expand(batch_shape + (P, S))
        y = y.reshape(batch_shape + (-1,))
        return y


class InitRateBiasLinear(torch.nn.Module):
    def __init__(self, P, S):
        super().__init__()
        self.PS = P, S
        self.weight_ps = torch.nn.Parameter(torch.zeros(P, S))
        self.weight_ss = torch.nn.Parameter(torch.zeros(S, S))

    def forward(self, x):
        P, S = self.PS
        batch_shape = x.shape[:-1]
        x = x.reshape(batch_shape + (P, S))
        y = x * self.weight_ps + x @ self.weight_ss
        y = y.reshape(batch_shape + (P * S,))
        return y


class Guide(AutoStructured):
    def __init__(self, dataset, guide_type="mvn_dependent"):
        self.guide_type = guide_type
        conditionals = {}
        dependencies = defaultdict(dict)
        conditionals["concentration"] = "delta"
        conditionals["feature_scale"] = "delta"
        conditionals["place_scale"] = "delta"
        if guide_type == "map":
            conditionals["rate_coef"] = "delta"
            conditionals["rate_bias"] = "delta"
            conditionals["init"] = "delta"
        elif guide_type.startswith("normal_delta"):
            conditionals["rate_coef"] = "normal"
            conditionals["rate_bias"] = "delta"
            conditionals["init"] = "delta"
        elif guide_type.startswith("normal"):
            conditionals["rate_coef"] = "normal"
            conditionals["rate_bias"] = "normal"
            conditionals["init"] = "normal"
        elif guide_type.startswith("mvn_delta"):
            conditionals["rate_coef"] = "mvn"
            conditionals["rate_bias"] = "delta"
            conditionals["init"] = "delta"
        elif guide_type.startswith("mvn_normal"):
            conditionals["rate_coef"] = "mvn"
            conditionals["rate_bias"] = "normal"
            conditionals["init"] = "normal"
        else:
            raise ValueError(f"Unsupported guide type: {guide_type}")

        if guide_type.endswith("_dependent"):
            T, P, S = dataset["weekly_strains"].shape
            S, F = dataset["features"].shape
            dependencies["rate_bias"]["rate_coef"] = RateBiasRateCoefLinear(P, S, F)
            dependencies["init"]["rate_bias"] = InitRateBiasLinear(P, S)

        super().__init__(
            model,
            conditionals=conditionals,
            dependencies=dependencies,
            init_loc_fn=InitLocFn(dataset),
            init_scale=0.01,
        )

    @torch.no_grad()
    @poutine.mask(mask=False)
    def stats(self, dataset, *, num_samples=1000):
        # Compute median point estimate.
        # FIXME this silently fails for map inference.
        result = {"median": self.median(dataset)}
        trace = poutine.trace(poutine.condition(model, result["median"])).get_trace(
            dataset, obs=False
        )
        for name, site in trace.nodes.items():
            if site["type"] == "sample" and not site_is_subsample(site):
                result["median"][name] = site["value"]

        # Compute moments.
        save_params = ["concentration", "feature_scale", "place_scale", "rate_coef"]
        with pyro.plate("particles", num_samples, dim=-3):
            samples = {k: v.v for k, v in self.get_deltas(save_params).items()}
            trace = poutine.trace(poutine.condition(model, samples)).get_trace(
                dataset, obs=False
            )
        samples = {
            name: site["value"]
            for name, site in trace.nodes.items()
            if site["type"] == "sample" and not site_is_subsample(site)
        }
        result["mean"] = {k: v.mean(0).squeeze() for k, v in samples.items()}
        result["std"] = {k: v.std(0).squeeze() for k, v in samples.items()}
        return result


def fit_svi(
    dataset,
    guide_type,
    learning_rate=0.02,
    learning_rate_decay=0.1,
    num_steps=3001,
    log_every=50,
    seed=20210319,
    check_loss=False,
):
    logger.info(f"Fitting {guide_type} guide via SVI")
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    param_store = pyro.get_param_store()

    # Initialize guide so we can count parameters.
    guide = Guide(dataset, guide_type)
    guide(dataset)
    num_params = sum(p.numel() for p in guide.parameters())
    logger.info(f"Training guide with {num_params} parameters:")

    def optim_config(param_name):
        config = {"lr": learning_rate, "lrd": learning_rate_decay ** (1 / num_steps)}
        if "scales" in param_name:
            config["lr"] *= 0.5
        elif "scale_tril" in param_name:
            config["lr"] *= 0.1
        elif "weight" in param_name:
            config["lr"] *= 0.05
        return config

    optim = ClippedAdam(optim_config)
    elbo = Trace_ELBO(max_plate_nesting=2)
    svi = SVI(model, guide, optim, elbo)
    losses = []
    num_obs = dataset["weekly_strains"].count_nonzero()
    for step in range(num_steps):
        loss = svi.step(dataset)
        assert not math.isnan(loss)
        losses.append(loss)
        if step % log_every == 0:
            median = guide.median()
            concentration = median["concentration"].item()
            feature_scale = median["feature_scale"].tolist()
            place_scale = median["place_scale"].tolist()
            assert (
                median["feature_scale"].ge(0.02).all()
            ), "implausibly small feature_scale"
            assert median["place_scale"].ge(0.001).all(), "implausibly small bias_scale"
            assert (
                median["concentration"].ge(1).all()
            ), "implausible small concentration"
            logger.info(
                f"step {step: >4d} loss = {loss / num_obs:0.6g}\t"
                f"conc. = {concentration:0.3g}\t"
                f"f.scale = {feature_scale:0.3g}\t"
                f"p.scale = {place_scale:0.3g}"
            )
        if check_loss and step >= 50:
            prev = torch.tensor(losses[-50:-25], device="cpu").median().item()
            curr = torch.tensor(losses[-25:], device="cpu").median().item()
            assert (curr - prev) < num_obs, "loss is increasing"

    result = guide.stats(dataset)
    result["losses"] = losses
    result["params"] = {k: v.detach().clone() for k, v in param_store.items()}
    result["guide"] = guide.float()

    log_stats(dataset, result)
    return result


def log_stats(dataset, result):
    mutations = dataset["mutations"]
    mean = result["mean"]["rate_coef"].cpu()
    if not mean.shape:
        return  # Work around error in map estimation.
    std = result["std"]["rate_coef"].cpu()
    sig = mean.abs() / std
    logger.info(f"|μ|/σ [median,max] = [{sig.median():0.3g},{sig.max():0.3g}]")
    for m in ["S:D614G", "S:N501Y", "S:E484K", "S:L452R"]:
        i = mutations.index(m)
        logger.info("ΔlogR({}) = {:0.3g} ± {:0.2f}".format(m, mean[i], std[i]))


def hook_fn(log_every, losses, num_obs, kernel, params, stage, i):
    loss = float(kernel._potential_energy_last)
    if log_every and len(losses) % log_every == 0:
        logger.info(f"loss = {loss / num_obs:0.6g}")
    losses.append(loss)


def fit_mcmc(
    dataset,
    guide=None,
    num_warmup=1000,
    num_samples=1000,
    num_chains=1,
    max_tree_depth=10,
    log_every=50,
    seed=20210319,
):
    pyro.set_rng_seed(seed)

    # Configure a kernel.
    model_ = model if guide is None else StructuredReparam(guide).reparam(model)
    with torch.no_grad(), pyro.validation_enabled(False):
        numel = {
            name: site["value"].numel()
            for name, site in poutine.trace(model_)
            .get_trace(dataset)
            .iter_stochastic_nodes()
            if "Subsample" not in type(site["fn"]).__name__
        }
    num_params = sum(numel.values())
    save_params = [name for name, n in numel.items() if n <= 10000]
    logger.info(f"Fitting via MCMC over {num_params} parameters")
    kernel = NUTS(
        model_,
        init_strategy=init_to_feasible,
        max_tree_depth=max_tree_depth,
        max_plate_nesting=2,
        jit_compile=True,
        ignore_jit_warnings=True,
    )

    # Run MCMC.
    losses = []
    num_obs = dataset["weekly_strains"].count_nonzero()
    hook_fn_ = functools.partial(hook_fn, log_every, losses, num_obs)
    mcmc = MCMC(
        kernel,
        warmup_steps=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        mp_context=(None if torch.zeros(()).device.type == "cpu" else "spawn"),
        hook_fn=hook_fn_,
        save_params=save_params,
    )
    mcmc.run(dataset)
    predict = Predictive(
        model_,
        mcmc.get_samples(),
        return_sites=["feature_scale", "concentration", "rate_coef", "rate"],
    )
    samples = predict(dataset, obs=False)

    result = {}
    result["losses"] = losses
    result["diagnostics"] = mcmc.diagnostics()
    result["median"] = median = {} if guide is None else guide.median()
    for k, v in samples.items():
        median[k] = v.median(0).values.squeeze()
    result["mean"] = {k: v.mean(0).squeeze() for k, v in samples.items()}
    result["std"] = {k: v.std(0).squeeze() for k, v in samples.items()}
    # Save only a subset of samples, since data can be large.
    result["samples"] = {k: samples[k].squeeze() for k in ["rate_coef", "rate"]}

    log_stats(dataset, result)
    return result
