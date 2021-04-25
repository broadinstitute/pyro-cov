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

import pyrocov.geo
from pyrocov import pangolin

logger = logging.getLogger(__name__)

# Reasonable values might be week (7), fortnight (14), or month (28)
TIMESTEP = 14
START_DATE = "2019-12-01"

# The following countries had at least one subregion with at least 5000 samples
# as of 2021-04-05, and will be finely partitioned into subregions. Remaining
# countries will be coarsely aggregated to country level.
FINE_COUNTRIES = {
    "United Kingdom",
    "Denmark",
    "Australia",
    "USA",
    "Canada",
    "Germany",
    "Sweden",
}


def load_gisaid_data(
    *,
    max_feature_order=0,
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

    # Filter features into numbers of mutations.
    aa_features = torch.load("results/nextclade.features.pt")
    mutations = aa_features["mutations"]
    features = aa_features["features"].to(
        device=device, dtype=torch.get_default_dtype()
    )
    keep = [m.count(",") <= max_feature_order for m in mutations]
    mutations = [m for k, m in zip(keep, mutations) if k]
    features = features[:, keep]
    feature_order = torch.tensor([m.count(",") for m in mutations])
    logger.info("Loaded {} feature matrix".format(features.shape))

    # Aggregate regions.
    lineages = list(map(pangolin.compress, columns["lineage"]))
    lineage_id_inv = list(map(pangolin.compress, aa_features["lineages"]))
    lineage_id = {k: i for i, k in enumerate(lineage_id_inv)}
    sparse_data = Counter()
    location_id = OrderedDict()
    for virus_name, day, location, lineage in zip(
        columns["virus_name"], columns["day"], columns["location"], lineages
    ):
        row = {"virus_name": virus_name, "location": location}
        if lineage not in lineage_id:
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
        if parts[1] not in FINE_COUNTRIES:
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
    logger.info(f"Keeping {int(weekly_strains.sum())}/{len(lineages)} rows")

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
    local_time = torch.arange(float(len(num_obs))) * TIMESTEP / 365.25  # in years
    local_time = local_time[:, None]
    local_time = local_time - (local_time * num_obs).sum(0) / num_obs.sum(0)

    return {
        "location_id": location_id,
        "mutations": mutations,
        "weekly_strains": weekly_strains,
        "features": features,
        "feature_order": feature_order,
        "feature_order_max": feature_order.max().item(),
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
            *daily_cases.shape, daily_cases.sum().item()
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


def model(dataset):
    local_time = dataset["local_time"]
    weekly_strains = dataset["weekly_strains"]
    features = dataset["features"]
    feature_order = dataset["feature_order"]
    feature_order_max = dataset["feature_order_max"]
    assert weekly_strains.shape[-1] == features.shape[0]
    assert local_time.shape == weekly_strains.shape[:2]
    assert feature_order.size(0) == features.size(-1)
    T, P, S = weekly_strains.shape
    S, F = features.shape
    time_plate = pyro.plate("time", T, dim=-2)
    place_plate = pyro.plate("place", P, dim=-1)

    # Assume relative growth rate depends on mutation features but not time or place.
    feature_scale = pyro.sample(
        "feature_scale",
        dist.LogNormal(0, 1).expand([feature_order_max + 1]).to_event(1),
    )
    rate_coef = pyro.sample(
        "rate_coef", dist.SoftLaplace(0, feature_scale[feature_order]).to_event(1)
    )
    rate = pyro.deterministic("rate", rate_coef @ features.T, event_dim=1)

    # Assume places differ only in their initial infection count.
    with place_plate:
        init = pyro.sample("init", dist.SoftLaplace(0, 10).expand([S]).to_event(1))

    # Finally observe overdispersed counts.
    strain_probs = (init + rate * local_time[:, :, None]).softmax(-1)
    concentration = pyro.sample("concentration", dist.LogNormal(2, 4))
    with time_plate, place_plate:
        pyro.sample(
            "obs",
            dist.DirichletMultinomial(
                total_count=weekly_strains.sum(-1).max(),
                concentration=concentration * strain_probs,
                is_sparse=True,  # uses a faster algorithm
            ),
            obs=weekly_strains,
        )


def init_loc_fn(site):
    name = site["name"]
    shape = site["fn"].shape()
    if name == "feature_scale":
        return torch.ones(shape)
    if name == "concentration":
        return torch.full(shape, 10.0)
    if name in ("rate_coef", "init"):
        return torch.zeros(shape)
    raise ValueError(site["name"])


class SparseLinear(torch.nn.Module):
    """
    Factorized linear funcction representing dependency of init on rate_coef.
    """

    def __init__(self, P, S, F):
        super().__init__()
        self.weight_s = torch.nn.Parameter(torch.zeros(S, F))
        self.weight_p = torch.nn.Parameter(torch.zeros(P, 1, F))
        assert self.weight_p.requires_grad

    def forward(self, delta):
        return (self.weight_s @ delta + self.weight_p @ delta).reshape(-1)


# This requires https://github.com/pyro-ppl/pyro/pull/2812
class Guide(AutoStructured):
    def __init__(self, dataset, guide_type="mvn_dependent"):
        self.guide_type = guide_type
        conditionals = {}
        dependencies = defaultdict(dict)
        if guide_type == "map":
            conditionals["feature_scale"] = "delta"
            conditionals["concentration"] = "delta"
            conditionals["rate_coef"] = "delta"
            conditionals["init"] = "delta"
        elif guide_type.startswith("normal_delta"):
            conditionals["feature_scale"] = "normal"
            conditionals["concentration"] = "normal"
            conditionals["rate_coef"] = "normal"
            conditionals["init"] = "delta"
        elif guide_type.startswith("normal"):
            conditionals["feature_scale"] = "normal"
            conditionals["concentration"] = "normal"
            conditionals["rate_coef"] = "normal"
            conditionals["init"] = "normal"
        elif guide_type.startswith("mvn_delta"):
            conditionals["feature_scale"] = "normal"
            conditionals["concentration"] = "normal"
            conditionals["rate_coef"] = "mvn"
            conditionals["init"] = "delta"
        elif guide_type.startswith("mvn_normal"):
            conditionals["feature_scale"] = "normal"
            conditionals["concentration"] = "normal"
            conditionals["rate_coef"] = "mvn"
            conditionals["init"] = "normal"
        else:
            raise ValueError(f"Unsupported guide type: {guide_type}")

        if guide_type.endswith("_dependent"):
            T, P, S = dataset["weekly_strains"].shape
            S, F = dataset["features"].shape
            sparse_linear = SparseLinear(P, S, F)
            dependencies["feature_scale"]["concentration"] = "linear"
            dependencies["rate_coef"]["concentration"] = "linear"
            dependencies["rate_coef"]["feature_scale"] = "linear"
            dependencies["init"]["concentration"] = "linear"
            dependencies["init"]["feature_scale"] = "linear"
            dependencies["init"]["rate_coef"] = sparse_linear

        super().__init__(
            model,
            conditionals=conditionals,
            dependencies=dependencies,
            init_loc_fn=init_loc_fn,
        )

    @torch.no_grad()
    def stats(self, dataset):
        result = {
            "median": self.median(dataset),
            "mean": {"rate_coef": self.locs.rate_coef.detach()},
        }
        if self.guide_type == "normal":
            result["std"] = {"rate_coef": self.scales.rate_coef.detach()}
        elif "mvn" in self.guide_type:
            scale = self.scales.rate_coef.detach()
            scale_tril = self.scale_trils.rate_coef.detach()
            scale_tril = scale[:, None] * scale_tril
            result["cov"] = {"rate_coef": scale_tril @ scale_tril.T}
            scale_tril = dataset["features"] @ scale_tril
            result["cov"]["rate"] = scale_tril @ scale_tril.T
            result["var"] = {k: v.diag() for k, v in result["cov"].items()}
            result["std"] = {k: v.sqrt() for k, v in result["var"].items()}
        return result


def fit_svi(
    dataset,
    guide_type,
    learning_rate=0.01,
    learning_rate_decay=0.01,
    num_steps=3001,
    log_every=50,
    seed=20210319,
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
        if "scale_tril" in param_name or "weight" in param_name:
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
            assert median["feature_scale"].ge(0.02).all(), "implausible"
            feature_scale = "[{}]".format(", ".join(f"{f:0.3g}" for f in feature_scale))
            logger.info(
                f"step {step: >4d} loss = {loss / num_obs:0.6g}\t"
                f"conc. = {concentration:0.3g}\t"
                f"f.scale = {feature_scale}"
            )

    result = guide.stats(dataset)
    result["losses"] = losses
    result["params"] = {k: v.detach().clone() for k, v in param_store.items()}
    result["guide"] = guide.float()
    return result


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
        return_sites=["feature_scale", "concentration", "rate_coef", "rate", "init"],
    )
    samples = predict(dataset)

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
    return result
