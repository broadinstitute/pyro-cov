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
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoNormal, init_to_median
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
    logger.info("Training on {} feature matrix".format(aa_features["features"].shape))

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
            print(f"WARNING skipping unsampled lineage {lineage}")
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

    return {"args": args, "weekly_strains": weekly_strains, "features": features}


def model(weekly_strains, features, feature_scale=1.0):
    assert weekly_strains.shape[-1] == features.shape[0]
    T, P, S = weekly_strains.shape
    S, F = features.shape
    time_plate = pyro.plate("time", T, dim=-2)
    place_plate = pyro.plate("place", P, dim=-1)
    time = torch.arange(float(T)) * args.timestep / 365.25  # in years
    time -= time.max()

    # Assume relative growth rate depends on mutation features but not time or place.
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
    if site["name"].endswith("noise_scale"):
        return torch.full(site["fn"].shape(), 3.0)
    if site["name"] == "concentration":
        return torch.full(site["fn"].shape(), 5.0)
    return init_to_median(site)


def _fit_map_filename(args, dataset, guide=None, without_feature=None):
    return f"results/rank_mutations.{guide is None}.{without_feature}.pt"


#@cached(_fit_map_filename)
def fit_map(args, dataset, guide=None):
    logger.info(f"Fitting via MAP")
    pyro.clear_param_store()
    pyro.set_rng_seed(20210319)

    if guide is None:
        guide = AutoDelta(model, init_loc_fn=init_loc_fn)
        # Initialize guide so we can count parameters.
        guide(dataset["weekly_strains"], dataset["features"])
    else:
        guide = copy.deepcopy(guide)
    num_params = sum(p.numel() for p in guide.parameters())
    logger.info(f"Training guide with {num_params} parameters:")

    optim = ClippedAdam({"lr": args.map_learning_rate, "betas": (0.8, 0.99)})
    svi = SVI(model, guide, optim, Trace_ELBO())
    num_obs = dataset["weekly_strains"].count_nonzero()
    losses = []
    for step in range(args.map_num_steps):
        loss = svi.step(dataset["weekly_strains"], dataset["features"]) / num_obs
        assert not math.isnan(loss)
        losses.append(loss)
        if step % args.log_every == 0:
            median = guide.median()
            concentration = median["concentration"].item()
            logger.info(
                f"step {step: >4d} loss = {loss:0.6g}\tconc. = {concentration:0.3g}\t"
            )
    return {
        "args": args,
        "guide": guide,
        "losses": losses,
    }


def main(args):
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    dataset = load_data(args)

    fit_map(args, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=torch.cuda.is_available()
    )
    parser.add_argument(
        "--timestep",
        default=14,
        type=int,
        help="Reasonable values might be week, fortnight, or month",
    )
    parser.add_argument("--map-learning-rate", default=0.05, type=float)
    parser.add_argument("--map-num-steps", default=301, type=int)
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-l", "--log-every", default=50, type=int)
    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    main(args)
