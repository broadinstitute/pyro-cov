import datetime
import logging
import math
import pickle
import re
from collections import Counter, OrderedDict

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.distributions import constraints
from pyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import init_to_value
from pyro.optim import ClippedAdam
from pyro.poutine.messenger import Messenger

import pyrocov.geo
from pyrocov import pangolin
from pyrocov.distributions import SoftLaplace

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
        "rate_coef", SoftLaplace(0, feature_scale[feature_order]).to_event(1)
    )
    rate = pyro.deterministic("rate", rate_coef @ features.T, event_dim=1)

    # Assume places differ only in their initial infection count.
    with place_plate:
        init = pyro.sample("init", SoftLaplace(0, 10).expand([S]).to_event(1))

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


def map_estimate(name, init, constraint=constraints.real):
    value = pyro.param(name + "_loc", init, constraint=constraint)
    pyro.sample(name, dist.Delta(value, event_dim=constraint.event_dim))


class Guide:
    def __init__(self, guide_type="mvn_dependent"):
        super().__init__()
        self.guide_type = guide_type

    def __call__(self, dataset):
        weekly_strains = dataset["weekly_strains"]
        features = dataset["features"]
        feature_order_max = dataset["feature_order_max"]
        assert weekly_strains.shape[-1] == features.shape[0]
        T, P, S = weekly_strains.shape
        S, F = features.shape

        # Map estimate global parameters.
        map_estimate(
            "feature_scale",
            lambda: torch.ones(feature_order_max + 1),
            constraints.independent(constraints.positive, 1),
        )
        map_estimate("concentration", lambda: torch.tensor(10.0), constraints.positive)

        # Sample rate_coef from a full-rank multivariate normal distribution.
        loc = pyro.param("rate_coef_loc", lambda: torch.zeros(F), event_dim=1)
        if self.guide_type == "map":
            rate_coef = pyro.sample("rate_coef", dist.Delta(loc, event_dim=1))
        elif self.guide_type == "normal":
            scale = pyro.param(
                "rate_coef_scale", lambda: torch.ones(F) * 0.01, constraints.positive
            )
            rate_coef = pyro.sample("rate_coef", dist.Normal(loc, scale).to_event(1))
        elif "mvn" in self.guide_type:
            scale = pyro.param(
                "rate_coef_scale", lambda: torch.ones(F) * 0.01, constraints.positive
            )
            scale_tril = pyro.param(
                "rate_coef_scale_tril", lambda: torch.eye(F), constraints.lower_cholesky
            )
            scale_tril = scale[:, None] * scale_tril

            rate_coef = pyro.sample(
                "rate_coef", dist.MultivariateNormal(loc, scale_tril=scale_tril)
            )
        else:
            raise ValueError(f"Unknown guide type: {self.guide_type}")

        # MAP estimate init, but depending on rate_coef.
        init_loc = pyro.param("init_loc", lambda: torch.zeros(P, S))
        if "dependent" in self.guide_type:
            weight_s = pyro.param("init_weight_s", lambda: torch.zeros(S, F))
            weight_p = pyro.param("init_weight_p", lambda: torch.zeros(P, 1, F))
            init = init_loc + weight_s @ rate_coef + weight_p @ rate_coef
        else:
            init = init_loc
        with pyro.plate("place", P, dim=-1):
            pyro.sample("init", dist.Delta(init, event_dim=1))

    @torch.no_grad()
    def median(self, dataset):
        rate_coef = pyro.param("rate_coef_loc").detach()
        result = {
            "feature_scale": pyro.param("feature_scale_loc").detach(),
            "concentration": pyro.param("concentration_loc").detach(),
            "rate_coef": rate_coef,
            "rate": rate_coef @ dataset["features"].T,
        }

        init_loc = pyro.param("init_loc").detach()
        if "dependent" in self.guide_type:
            weight_s = pyro.param("init_weight_s").detach()
            weight_p = pyro.param("init_weight_p").detach()
            result["init"] = init_loc + weight_s @ rate_coef + weight_p @ rate_coef
        else:
            result["init"] = init_loc

        return result

    @torch.no_grad()
    def stats(self, dataset):
        result = {
            "median": self.median(dataset),
            "mean": {"rate_coef": pyro.param("rate_coef_loc").detach()},
        }
        if self.guide_type == "normal":
            result["std"] = {"rate_coef": pyro.param("rate_coef_scale").detach()}
        elif "mvn" in self.guide_type:
            scale = pyro.param("rate_coef_scale").detach()
            scale_tril = pyro.param("rate_coef_scale_tril").detach()
            scale_tril = scale[:, None] * scale_tril
            result["cov"] = {"rate_coef": scale_tril @ scale_tril.T}
            scale_tril = dataset["features"] @ scale_tril
            result["cov"]["rate"] = scale_tril @ scale_tril.T
            result["var"] = {k: v.diag() for k, v in result["cov"].items()}
            result["std"] = {k: v.sqrt() for k, v in result["var"].items()}
        return result


class DeterministicMessenger(Messenger):
    """
    Condition all but the "rate_coef" variable on parameters of an
    mvn_dependent guide.
    """

    def __init__(self, params):
        self.feature_scale = params["feature_scale_loc"]
        self.concentration = params["concentration_loc"]

        self.rate_coef_loc = params["rate_coef_loc"]
        scale = params["rate_coef_scale"]
        scale_tril = params["rate_coef_scale_tril"]
        self.rate_coef_scale_tril = scale[:, None] * scale_tril

        self.init_loc = params["init_loc"]
        self.init_weight_s = params["init_weight_s"]
        self.init_weight_p = params["init_weight_p"]

        self.rate_coef = None
        super().__init__()

    def _pyro_sample(self, msg):
        if msg["name"] == "feature_scale":
            msg["value"] = self.feature_scale
            msg["is_observed"] = True
        elif msg["name"] == "concentration":
            msg["value"] = self.concentration
            msg["is_observed"] = True
        elif msg["name"] == "rate_coef":
            assert self.rate_coef is None
            fn = msg["fn"]
            decentered_value = pyro.sample(
                "rate_coef_centered",
                dist.Normal(0, 1).expand(fn.shape()).to_event(fn.event_dim).mask(False),
            )
            msg["value"] = (
                self.rate_coef_loc + self.rate_coef_scale_tril @ decentered_value
            )
            msg["is_observed"] = True
            self.rate_coef = msg["value"]
        elif msg["name"] == "init":
            assert self.rate_coef is not None
            msg["value"] = (
                self.init_loc
                + self.init_weight_s @ self.rate_coef
                + self.init_weight_p @ self.rate_coef
            )
            msg["is_observed"] = True
            self.rate_coef = None


class PreconditionMessenger(Messenger):
    """
    Condition on fixed globals,
    """

    def __init__(self, params):
        self.feature_scale = params["feature_scale_loc"]
        self.concentration = params["concentration_loc"]

        self.rate_coef_loc = params["rate_coef_loc"]
        scale = params["rate_coef_scale"]
        scale_tril = params["rate_coef_scale_tril"]
        self.rate_coef_scale_tril = scale[:, None] * scale_tril

        self.init_loc = params["init_loc"]
        self.init_weight_s = params["init_weight_s"]
        self.init_weight_p = params["init_weight_p"]

        self.rate_coef = None
        super().__init__()

    def _pyro_sample(self, msg):
        if msg["name"] == "feature_scale":
            msg["value"] = self.feature_scale
            msg["is_observed"] = True
        elif msg["name"] == "concentration":
            msg["value"] = self.concentration
            msg["is_observed"] = True
        elif msg["name"] == "rate_coef":
            assert self.rate_coef is None
            fn = msg["fn"]
            decentered_value = pyro.sample(
                "rate_coef_centered",
                dist.Normal(0, 1).expand(fn.shape()).to_event(fn.event_dim).mask(False),
            )
            msg["value"] = (
                self.rate_coef_loc + self.rate_coef_scale_tril @ decentered_value
            )
            msg["is_observed"] = True
            self.rate_coef = msg["value"]
        elif msg["name"] == "init":
            assert self.rate_coef is not None
            fn = msg["fn"]
            decentered_value = pyro.sample(
                "init_centered",
                dist.Normal(0, 1).expand(fn.shape()).to_event(fn.event_dim).mask(False),
            )
            msg["value"] = (
                decentered_value
                + self.init_loc
                + self.init_weight_s @ self.rate_coef
                + self.init_weight_p @ self.rate_coef
            )
            msg["is_observed"] = True
            self.rate_coef = None


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
    guide = Guide(guide_type)
    guide(dataset)
    num_params = sum(p.unconstrained().numel() for p in param_store.values())
    logger.info(f"Training guide with {num_params} parameters:")

    def optim_config(module_name, param_name):
        config = {"lr": learning_rate, "lrd": learning_rate_decay ** (1 / num_steps)}
        if param_name in ["init_weight_s", "init_weight_p", "rate_coef_scale_tril"]:
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
            concentration = param_store["concentration_loc"].item()
            feature_scale = param_store["feature_scale_loc"].tolist()
            feature_scale = "[{}]".format(", ".join(f"{f:0.3g}" for f in feature_scale))
            logger.info(
                f"step {step: >4d} loss = {loss / num_obs:0.6g}\t"
                f"conc. = {concentration:0.3g}\t"
                f"f.scale = {feature_scale}"
            )

    result = guide.stats(dataset)
    result["losses"] = losses
    result["params"] = {k: v.detach().clone() for k, v in param_store.items()}
    return result


def fit_mcmc(
    dataset,
    svi_params,
    model_type="dependent",
    num_warmup=1000,
    num_samples=1000,
    max_tree_depth=10,
    log_every=50,
    seed=20210319,
):
    logger.info("Fitting via MCMC over {} parameters".format(len(dataset["mutations"])))
    pyro.set_rng_seed(seed)

    # Configure a kernel.
    if model_type == "conditioned":
        partial_model = poutine.condition(
            model,
            data={
                "feature_scale": svi_params["feature_scale_loc"],
                "concentration": svi_params["concentration_loc"],
            },
        )
    elif model_type == "dependent":
        partial_model = DeterministicMessenger(svi_params)(model)
    elif model_type == "preconditioned":
        partial_model = PreconditionMessenger(svi_params)(model)
    else:
        raise ValueError(model_type)
    init_values = {
        "init": svi_params["median"]["init"],
        "rate_coef": svi_params["median"]["rate_coef"],
        "init_decentered": torch.zeros_like(svi_params["median"]["init"]),
        "rate_coef_decentered": torch.zeros_like(svi_params["median"]["rate_coef"]),
    }
    kernel = NUTS(
        partial_model,
        init_strategy=init_to_value(values=init_values),
        max_tree_depth=max_tree_depth,
        max_plate_nesting=2,
        jit_compile=True,
        ignore_jit_warnings=True,
    )

    # Run MCMC.
    num_obs = dataset["weekly_strains"].count_nonzero()
    losses = []

    def hook_fn(kernel, params, stage, i):
        assert set(params) == {"rate_coef"}
        loss = float(kernel._potential_energy_last)
        if log_every and len(losses) % log_every == 0:
            logger.info(f"loss = {loss / num_obs:0.6g}")
        losses.append(loss)

    mcmc = MCMC(
        kernel,
        warmup_steps=num_warmup,
        num_samples=num_samples,
        hook_fn=hook_fn,
    )
    mcmc.run(dataset)
    predict = Predictive(
        partial_model,
        mcmc.get_samples(),
        return_sites=["feature_scale", "concentration", "rate_coef", "rate", "init"],
    )
    samples = predict(dataset)

    result = {}
    result["losses"] = losses
    result["diagnostics"] = mcmc.diagnostics()
    result["median"] = median = svi_params.copy()
    for k, v in samples.items():
        median[k] = v.median(0).values.squeeze()
    result["mean"] = {k: v.mean(0).squeeze() for k, v in samples.items()}
    result["std"] = {k: v.std(0).squeeze() for k, v in samples.items()}
    # Save only a subset of samples, since data can be large.
    result["samples"] = {k: samples[k].squeeze() for k in ["rate_coef", "rate"]}
    return result
