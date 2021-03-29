import argparse
import copy
import functools
import logging
import math
import os
import pickle
from collections import Counter

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoGuideList, AutoNormal, init_to_median
from pyro.infer.autoguide.initialization import InitMessenger
from pyro.optim import ClippedAdam

from pyrocov import pangolin

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def cached(filename):
    def decorator(fn):
        @functools.wraps(fn)
        def cached_fn(*args, **kwargs):
            f = filename(*args, **kwargs) if callable(filename) else filename
            if args[0].force or not os.path.exists(f):
                result = fn(*args, **kwargs)
                torch.save(result, f)
            else:
                result = torch.load(f)
            return result

        return cached_fn

    return decorator


def load_data(args):
    logger.info("Loading data")
    with open("results/gisaid.columns.pkl", "rb") as f:
        columns = pickle.load(f)
    logger.info("Training on {} rows with columns:".format(len(columns["day"])))
    logger.info(", ".join(columns.keys()))
    aa_features = torch.load("results/nextclade.features.pt")
    logger.info("Loaded {} feature matrix".format(aa_features["features"].shape))

    # Aggregate regions.
    features = aa_features["features"].to(device=args.device)
    lineages = list(map(pangolin.compress, columns["lineage"]))
    lineage_id_inv = list(map(pangolin.compress, aa_features["lineages"]))
    lineage_id = {k: i for i, k in enumerate(lineage_id_inv)}

    sparse_data = Counter()
    location_id = {}
    quotient = {}
    for day, location, lineage in zip(columns["day"], columns["location"], lineages):
        if lineage not in lineage_id:
            logger.warning(f"WARNING skipping unsampled lineage {lineage}")
            continue
        parts = location.split("/")
        if len(parts) < 2:
            continue
        parts = [p.strip() for p in parts[:3]]
        if parts[1] not in ("USA", "United Kingdom"):
            parts = parts[:2]
        quotient[location] = " / ".join(parts)
        p = location_id.setdefault(quotient[location], len(location_id))
        s = lineage_id[lineage]
        t = day // args.timestep
        sparse_data[t, p, s] += 1

    T = 1 + max(columns["day"]) // args.timestep
    P = len(location_id)
    S = len(lineage_id)
    weekly_strains = torch.zeros(T, P, S)
    for (t, p, s), n in sparse_data.items():
        weekly_strains[t, p, s] = n

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
        "args": args,
        "location_id": location_id,
        "mutations": mutations,
        "weekly_strains": weekly_strains,
        "features": features,
    }


def model(weekly_strains, features):
    assert weekly_strains.shape[-1] == features.shape[0]
    T, P, S = weekly_strains.shape
    S, F = features.shape
    time_plate = pyro.plate("time", T, dim=-2)
    place_plate = pyro.plate("place", P, dim=-1)
    time = torch.arange(float(T)) * args.timestep / 365.25  # in years
    time -= time.max()

    # Assume relative growth rate depends on mutation features but not time or place.
    feature_scale = pyro.sample("feature_scale", dist.LogNormal(0, 1))
    log_rate_coef = pyro.sample(
        "log_rate_coef", dist.Laplace(0, feature_scale).expand([F]).to_event(1)
    )
    log_rate = pyro.deterministic("log_rate", log_rate_coef @ features.T)

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


def _fit_map_filename(args, dataset, cond_data, guide=None, without_feature=None):
    return f"results/rank_mutations.{guide is None}.{without_feature}.pt"


@cached(_fit_map_filename)
def fit_map(args, dataset, cond_data, guide=None, without_feature=None):
    logger.info(f"Fitting via MAP (without_feature = {without_feature})")
    pyro.clear_param_store()
    pyro.set_rng_seed(20210319)

    if without_feature is not None:
        # Drop feature.
        dataset = dataset.copy()
        dataset["features"] = dataset["features"].clone()
        dataset["features"][:, without_feature] = 0

    cond_data = {k: torch.as_tensor(v) for k, v in cond_data.items()}
    cond_model = poutine.condition(model, cond_data)
    if guide is None:
        guide = AutoDelta(cond_model, init_loc_fn=init_loc_fn)
        # Initialize guide so we can count parameters.
        guide(dataset["weekly_strains"], dataset["features"])
    else:
        guide = copy.deepcopy(guide)
    num_params = sum(p.numel() for p in guide.parameters())
    logger.info(f"Training guide with {num_params} parameters:")

    optim = ClippedAdam({"lr": args.map_learning_rate, "betas": (0.8, 0.99)})
    svi = SVI(cond_model, guide, optim, Trace_ELBO())
    num_obs = dataset["weekly_strains"].count_nonzero()
    losses = []
    for step in range(args.map_num_steps):
        loss = svi.step(dataset["weekly_strains"], dataset["features"])
        assert not math.isnan(loss)
        losses.append(loss)
        if step % args.log_every == 0:
            logger.info(f"step {step: >4d} loss = {loss / num_obs:0.6g}")

    result = {"args": args, "mutation": None, "guide": guide, "losses": losses}
    if without_feature is not None:
        result["mutation"] = dataset["mutations"][without_feature]
        result["guide"] = None
    return result


def fit_svi(args, dataset):
    logger.info("Fitting via SVI")
    pyro.clear_param_store()
    pyro.set_rng_seed(20210319)

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

    optim = ClippedAdam(
        {"lr": args.svi_learning_rate, "lrd": 0.1 ** (1 / args.svi_num_steps)}
    )
    svi = SVI(model, guide, optim, Trace_ELBO())
    losses = []
    num_obs = dataset["weekly_strains"].count_nonzero()
    for step in range(args.svi_num_steps):
        loss = svi.step(dataset["weekly_strains"], dataset["features"])
        assert not math.isnan(loss)
        losses.append(loss)
        if step % args.log_every == 0:
            median = guide.median()
            concentration = median["concentration"].item()
            feature_scale = median["feature_scale"].item()
            logger.info(
                f"step {step: >4d} loss = {loss / num_obs:0.6g}\t"
                f"conc. = {concentration:0.3g}\t"
                f"feat.scale = {feature_scale:0.3g}"
            )
    return {"args": args, "guide": guide, "losses": losses}


@cached("results/rank_mutations.rank_svi.pt")
def rank_svi(args, dataset):
    guide = fit_svi(args, dataset)["guide"]

    guide.to(torch.double)
    sigma_points = dist.Normal(0, 1).cdf(torch.tensor([-1.0, 1.0])).double()
    pos = guide[1].quantiles(sigma_points[1].item())["log_rate_coef"].cpu()
    neg = guide[1].quantiles(sigma_points[0].item())["log_rate_coef"].cpu()
    mean = (pos + neg) / 2
    std = (pos - neg) / 2
    median = guide.median()

    result = {
        "args": args,
        "mean": mean,
        "std": std,
        "ranks": (mean / std).sort(0, descending=True).indices,
        "cond_data": {
            "feature_scale": median["feature_scale"].item(),
            "concentration": median["concentration"].item(),
        },
    }
    return result


def rank_map(args, dataset, initial_ranks):
    """
    Given an initial approximate ranking of features, compute MAP log
    likelihood ratios of the most siginficant features.
    """
    # Fit an initial model for warm-starting.
    cond_data = initial_ranks["cond_data"]
    guide = fit_map(args, dataset, cond_data)["guide"]

    # Evaluate on the null hypothesis + the most positive features.
    losses = {}
    for feature in [None] + initial_ranks["ranks"][: args.num_features].tolist():
        fit = fit_map(args, dataset, cond_data, guide, feature)
        losses[feature] = fit["losses"][-1]
        del fit

    # Compute log likelihood ratios.
    log_likelihood = {
        f: losses[None] - score for f, score in losses.items() if f is not None
    }
    result = {
        "args": args,
        "mutations": dataset["mutations"],
        "initial_ranks": initial_ranks,
        "losses": losses,
        "log_likelihood": log_likelihood,
    }
    torch.save(result, "results/rank_mutations.pt")


def main(args):
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    dataset = load_data(args)
    initial_ranks = rank_svi(args, dataset)
    if args.map_num_steps:
        rank_map(args, dataset, initial_ranks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank mutations via SVI and leave-feature-out MAP"
    )
    parser.add_argument(
        "--timestep",
        default=14,
        type=int,
        help="Reasonable values might be week (7), fortnight (14), or month (28)",
    )
    parser.add_argument("--svi-learning-rate", default=0.05, type=float)
    parser.add_argument("--svi-num-steps", default=1001, type=int)
    parser.add_argument("--map-learning-rate", default=0.05, type=float)
    parser.add_argument("--map-num-steps", default=301, type=int)
    parser.add_argument("--num-features", type=int)
    parser.add_argument(
        "--cuda", action="store_true", default=torch.cuda.is_available()
    )
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-l", "--log-every", default=50, type=int)
    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    main(args)
