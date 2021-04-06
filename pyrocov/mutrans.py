import logging
import math
import pickle
import re
from collections import Counter

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from pyro.infer.autoguide import init_to_value
from pyro.optim import ClippedAdam
from pyro.poutine.messenger import Messenger

from pyrocov import pangolin
from pyrocov.distributions import SoftLaplace

logger = logging.getLogger(__name__)

# Reasonable values might be week (7), fortnight (14), or month (28)
TIMESTEP = 14


def load_data(
    *,
    device="cpu",
    virus_name_pattern=None,
    location_pattern=None,
):
    logger.info("Loading data")
    if isinstance(virus_name_pattern, str):
        virus_name_pattern = re.compile(virus_name_pattern)
    if isinstance(location_pattern, str):
        location_pattern = re.compile(location_pattern)
    with open("results/gisaid.columns.pkl", "rb") as f:
        columns = pickle.load(f)
    logger.info("Training on {} rows with columns:".format(len(columns["day"])))
    logger.info(", ".join(columns.keys()))
    aa_features = torch.load("results/nextclade.features.pt")
    logger.info("Loaded {} feature matrix".format(aa_features["features"].shape))

    # Aggregate regions.
    features = aa_features["features"].to(
        device=device, dtype=torch.get_default_dtype()
    )
    lineages = list(map(pangolin.compress, columns["lineage"]))
    lineage_id_inv = list(map(pangolin.compress, aa_features["lineages"]))
    lineage_id = {k: i for i, k in enumerate(lineage_id_inv)}

    sparse_data = Counter()
    location_id = {}
    for virus_name, day, location, lineage in zip(
        columns["virus_name"], columns["day"], columns["location"], lineages
    ):
        if lineage not in lineage_id:
            logger.warning(f"WARNING skipping unsampled lineage {lineage}")
            continue
        if virus_name_pattern and not virus_name_pattern.search(virus_name):
            continue
        if location_pattern and not location_pattern.search(location):
            continue
        parts = location.split("/")
        if len(parts) < 2:
            continue
        parts = [p.strip() for p in parts[:3]]
        if parts[1] not in ("USA", "United Kingdom"):
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
    location_id = dict(zip(locations, range(len(ok_regions))))

    # Filter mutations.
    mutations = aa_features["mutations"]
    num_strains_with_mutation = (features >= 0.5).sum(0)
    ok_mutations = (num_strains_with_mutation >= 1).nonzero(as_tuple=True)[0]
    logger.info(f"Keeping {len(ok_mutations)}/{len(mutations)} mutations")
    mutations = [mutations[i] for i in ok_mutations.tolist()]
    features = features.index_select(1, ok_mutations)

    return {
        "location_id": location_id,
        "mutations": mutations,
        "weekly_strains": weekly_strains,
        "features": features,
        "lineage_id": lineage_id,
        "lineage_id_inv": lineage_id_inv,
    }


def model(weekly_strains, features):
    assert weekly_strains.shape[-1] == features.shape[0]
    T, P, S = weekly_strains.shape
    S, F = features.shape
    time_plate = pyro.plate("time", T, dim=-2)
    place_plate = pyro.plate("place", P, dim=-1)
    time = torch.arange(float(T)) * TIMESTEP / 365.25  # in years
    time -= time.max()

    # Assume relative growth rate depends on mutation features but not time or place.
    feature_scale = pyro.sample("feature_scale", dist.LogNormal(0, 1))
    rate_coef = pyro.sample(
        "rate_coef", SoftLaplace(0, feature_scale).expand([F]).to_event(1)
    )
    rate = pyro.deterministic("rate", rate_coef @ features.T, event_dim=1)

    # Assume places differ only in their initial infection count.
    with place_plate:
        init = pyro.sample("init", SoftLaplace(0, 10).expand([S]).to_event(1))

    # Finally observe overdispersed counts.
    strain_probs = (init + rate * time[:, None, None]).softmax(-1)
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

    def __call__(self, weekly_strains, features):
        assert weekly_strains.shape[-1] == features.shape[0]
        T, P, S = weekly_strains.shape
        S, F = features.shape

        # Map estimate global parameters.
        map_estimate("feature_scale", lambda: torch.tensor(1.0), constraints.positive)
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
            weight = pyro.param("init_weight", lambda: torch.zeros(S, F))
            init = init_loc + weight @ rate_coef
        else:
            init = init_loc
        with pyro.plate("place", P, dim=-1):
            pyro.sample("init", dist.Delta(init, event_dim=1))

    @torch.no_grad()
    def median(self, weekly_strains, features):
        result = {
            "feature_scale": pyro.param("feature_scale_loc").detach(),
            "concentration": pyro.param("concentration_loc").detach(),
            "rate_coef": pyro.param("rate_coef_loc").detach(),
        }

        init_loc = pyro.param("init_loc").detach()
        if "dependent" in self.guide_type:
            weight = pyro.param("init_weight").detach()
            result["init"] = init_loc + weight @ result["rate_coef"]
        else:
            result["init"] = init_loc

        result["rate"] = result["rate_coef"] @ features.T
        return result

    @torch.no_grad()
    def stats(self, weekly_strains, features):
        result = {
            "median": self.median(weekly_strains, features),
            "mean": pyro.param("rate_coef_loc").detach(),
        }
        if self.guide_type == "normal":
            result["std"] = pyro.param("rate_coef_scale").detach()
        elif "mvn" in self.guide_type:
            scale = pyro.param("rate_coef_scale").detach()
            scale_tril = pyro.param("rate_coef_scale_tril").detach()
            scale_tril = scale[:, None] * scale_tril
            result["cov"] = scale_tril @ scale_tril.T
            result["var"] = result["cov"].diag()
            result["std"] = result["var"].sqrt()
        return result


class DeterministicMessenger(Messenger):
    """
    Condition all but the "rate_coef" variable on parameters of an
    mvn_dependent guide.
    """

    def __init__(self, params):
        self.feature_scale = params["feature_scale_loc"]
        self.concentration = params["concentration_loc"]
        self.init_loc = params["init_loc"]
        self.init_weight = params["init_weight"]
        self.rate_coef = None
        super().__init__()

    def _pyro_post_sample(self, msg):
        if msg["name"] == "rate_coef":
            assert self.rate_coef is None
            self.rate_coef = msg["value"]
            assert self.rate_coef is not None

    def _pyro_sample(self, msg):
        if msg["name"] == "init":
            assert self.rate_coef is not None
            msg["value"] = self.init_loc + self.init_weight @ self.rate_coef
            msg["is_observed"] = True
            self.rate_coef = None
        elif msg["name"] == "feature_scale":
            msg["value"] = self.feature_scale
            msg["is_observed"] = True
        elif msg["name"] == "concentration":
            msg["value"] = self.concentration
            msg["is_observed"] = True


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
    weekly_strains = dataset["weekly_strains"]
    features = dataset["features"]

    # Initialize guide so we can count parameters.
    guide = Guide(guide_type)
    guide(weekly_strains, features)
    num_params = sum(p.unconstrained().numel() for p in param_store.values())
    logger.info(f"Training guide with {num_params} parameters:")

    def optim_config(module_name, param_name):
        config = {"lr": learning_rate, "lrd": learning_rate_decay ** (1 / num_steps)}
        if param_name in ["init_weight", "rate_coef_scale_tril"]:
            config["lr"] *= 0.02
        return config

    optim = ClippedAdam(optim_config)
    svi = SVI(model, guide, optim, Trace_ELBO())
    losses = []
    num_obs = dataset["weekly_strains"].count_nonzero()
    for step in range(num_steps):
        loss = svi.step(weekly_strains, features)
        assert not math.isnan(loss)
        losses.append(loss)
        if step % log_every == 0:
            concentration = param_store["concentration_loc"].item()
            feature_scale = param_store["feature_scale_loc"].item()
            logger.info(
                f"step {step: >4d} loss = {loss / num_obs:0.6g}\t"
                f"conc. = {concentration:0.3g}\t"
                f"f.scale = {feature_scale:0.3g}"
            )

    result = guide.stats(weekly_strains, features)
    result["losses"] = losses
    result["params"] = {k: v.detach().clone() for k, v in param_store.items()}
    return result


def fit_mcmc(
    dataset,
    svi_params,
    num_warmup=1000,
    num_samples=1000,
    max_tree_depth=10,
    log_every=50,
    seed=20210319,
):
    logger.info("Fitting via MCMC")
    pyro.set_rng_seed(seed)

    # Configure a kernel.
    kernel = NUTS(
        DeterministicMessenger(svi_params)(model),
        init_strategy=init_to_value(values=svi_params),
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
    mcmc.run(dataset["weekly_strains"], dataset["features"])

    result = {}
    result["losses"] = losses
    result["diagnostics"] = mcmc.diagnostics()
    result["samples"] = mcmc.get_samples()
    return result
