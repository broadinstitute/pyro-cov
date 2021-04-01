import copy
import logging
import math
import pickle
import re
from collections import Counter

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoGuideList, AutoNormal, init_to_median
from pyro.infer.autoguide.initialization import InitMessenger
from pyro.optim import ClippedAdam
from pyro.poutine.util import prune_subsample_sites

from pyrocov import pangolin
from pyrocov.distributions import SmoothLaplace

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


def model(weekly_strains, features, *, mask=None):
    assert weekly_strains.shape[-1] == features.shape[0]
    T, P, S = weekly_strains.shape
    S, F = features.shape
    time_plate = pyro.plate("time", T, dim=-2)
    place_plate = pyro.plate("place", P, dim=-1)
    time = torch.arange(float(T)) * TIMESTEP / 365.25  # in years
    time -= time.max()

    # Assume relative growth rate depends on mutation features but not time or place.
    feature_scale = pyro.sample("feature_scale", dist.LogNormal(0, 1))
    log_rate_coef = pyro.sample(
        "log_rate_coef", SmoothLaplace(0, feature_scale).expand([F]).to_event(1)
    )
    if mask:
        log_rate_coef = log_rate_coef.masked_fill(mask, 0)
    log_rate = pyro.deterministic("log_rate", log_rate_coef @ features.T, event_dim=1)

    # Assume places differ only in their initial infection count.
    with place_plate:
        log_init = pyro.sample("log_init", dist.Normal(0, 10).expand([S]).to_event(1))

    # Finally observe overdispersed counts.
    strain_probs = (log_init + log_rate * time[:, None, None]).softmax(-1)
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


def dropout_model(weekly_strains, features):
    S, F = features.shape
    mask = torch.eye(F, dtype=torch.bool).reshape(F, 1, 1, F)
    with pyro.plate("features", dim=-3):
        model(weekly_strains, features, mask=mask)


def init_loc_fn(site):
    if site["name"] in ("log_rate_coef", "log_rate", "log_init", "noise", "noise_haar"):
        return torch.zeros(site["fn"].shape())
    if site["name"] == "feature_scale":
        return torch.ones(site["fn"].shape())
    if site["name"].endswith("noise_scale"):
        return torch.full(site["fn"].shape(), 3.0)
    if site["name"] == "concentration":
        return torch.full(site["fn"].shape(), 5.0)
    return init_to_median(site)


@torch.no_grad()
def eval_loss_terms(model, guide, *args, vectorized=False):
    guide_trace = poutine.trace(guide).get_trace(*args)
    model_trace = poutine.trace(poutine.replay(model, guide_trace)).get_trace(*args)
    traces = {
        "model": prune_subsample_sites(model_trace),
        "guide": prune_subsample_sites(guide_trace),
    }
    result = {}
    for trace_name, trace in traces.items():
        trace.compute_log_prob()
        result[trace_name] = {
            name: site["log_prob"] if vectorized else site["log_prob_sum"].item()
            for name, site in trace.nodes.items()
            if site["type"] == "sample"
        }
    return result


def fit_map(
    dataset,
    model,
    guide=None,
    *,
    vectorized=False,
    learning_rate=0.05,
    num_steps=301,
    log_every=50,
    seed=20210319,
):
    logger.info("Fitting via MAP")
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    weekly_strains = dataset["weekly_strains"]
    features = dataset["features"]

    if guide is None:
        guide = AutoDelta(model, init_loc_fn=init_loc_fn)
        # Initialize guide so we can count parameters.
        guide(weekly_strains, features)
    else:
        guide = copy.deepcopy(guide)
    num_params = sum(p.numel() for p in guide.parameters())
    logger.info(f"Training guide with {num_params} parameters:")

    optim = ClippedAdam({"lr": learning_rate, "betas": (0.8, 0.99)})
    svi = SVI(model, guide, optim, Trace_ELBO())
    num_obs = weekly_strains.count_nonzero()
    losses = []
    for step in range(num_steps):
        loss = svi.step(weekly_strains, features)
        assert not math.isnan(loss)
        losses.append(loss)
        if step % log_every == 0:
            logger.info(f"step {step: >4d} loss = {loss / num_obs:0.6g}")

    median = guide.median()
    median["log_rate"] = median["log_rate_coef"] @ dataset["features"].T

    return {
        "guide": guide,
        "losses": losses,
        "median": median,
        "mode": median["log_rate_coef"],
        "loss_terms": eval_loss_terms(
            model,
            guide,
            weekly_strains,
            features,
            vectorized=vectorized,
        ),
    }


def fit_svi(
    dataset,
    model=model,
    learning_rate=0.05,
    num_steps=1001,
    log_every=50,
    seed=20210319,
):
    logger.info("Fitting via SVI")
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    guide = AutoGuideList(InitMessenger(init_loc_fn)(model))
    guide.append(
        AutoDelta(
            poutine.block(model, hide=["log_rate_coef"]),
            init_loc_fn=init_loc_fn,
        )
    )
    guide.append(
        AutoNormal(
            poutine.block(model, expose=["log_rate_coef"]),
            init_loc_fn=init_loc_fn,
            init_scale=0.01,
        )
    )
    # Initialize guide so we can count parameters.
    guide(dataset["weekly_strains"], dataset["features"])
    num_params = sum(p.numel() for p in guide.parameters())
    logger.info(f"Training guide with {num_params} parameters:")

    optim = ClippedAdam({"lr": learning_rate, "lrd": 0.1 ** (1 / num_steps)})
    svi = SVI(model, guide, optim, Trace_ELBO())
    losses = []
    num_obs = dataset["weekly_strains"].count_nonzero()
    for step in range(num_steps):
        loss = svi.step(dataset["weekly_strains"], dataset["features"])
        assert not math.isnan(loss)
        losses.append(loss)
        if step % log_every == 0:
            median = guide.median()
            concentration = median["concentration"].item()
            feature_scale = median["feature_scale"].item()
            logger.info(
                f"step {step: >4d} loss = {loss / num_obs:0.6g}\t"
                f"conc. = {concentration:0.3g}\t"
                f"feat.scale = {feature_scale:0.3g}"
            )

    median = guide.median()
    median["log_rate"] = median["log_rate_coef"] @ dataset["features"].T

    guide.to(torch.double)
    sigma_points = dist.Normal(0, 1).cdf(torch.tensor([-1.0, 1.0])).double()
    pos = guide[1].quantiles(sigma_points[1].item())["log_rate_coef"]
    neg = guide[1].quantiles(sigma_points[0].item())["log_rate_coef"]
    mean = (pos + neg) / 2
    std = (pos - neg) / 2

    return {
        "guide": guide,
        "losses": losses,
        "mean": mean,
        "std": std,
        "median": median,
    }
