import argparse
import copy
import functools
import logging
import math
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, init_to_median
from pyro.optim import ClippedAdam
from pyro.poutine import mask, trace, replay, scale

from pyrocov import pangolin

from collections import defaultdict


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def load_data(args):
    logger.info("Loading data")
    with open("results/gisaid.columns.pkl", "rb") as f:
        columns = pickle.load(f)
    logger.info("Training on {} rows with columns:".format(len(columns["day"])))
    logger.info(", ".join(columns.keys()))
    aa_features = torch.load("results/nextclade.features.pt")

    mutations = aa_features['mutations']

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
            # print(f"WARNING skipping unsampled lineage {lineage}")
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
    num_strains_with_mutation = (features >= args.mutation_cutoff).sum(0)
    ok_mutations = (num_strains_with_mutation >= 1).nonzero(as_tuple=True)[0]
    logger.info(f"Keeping {len(ok_mutations)}/{len(mutations)} mutations")
    mutations = [mutations[i] for i in ok_mutations.tolist()]
    features = features.index_select(1, ok_mutations)

    T, P, S = weekly_strains.shape

    return {"args": args, "weekly_strains": weekly_strains, "features": features, "T": T, "P": P, "S": S,
            "mutations": mutations}


def model(weekly_strains, features, num_particles=3, normalizer=1.0, weights=torch.tensor(1.0)):
    assert weekly_strains.shape[-1] == features.shape[0]

    T, P, S = weekly_strains.shape
    S, F = features.shape
    particle_plate = pyro.plate("particles", num_particles, dim=-3)
    time_plate = pyro.plate("time", T, dim=-2)
    place_plate = pyro.plate("place", P, dim=-1)
    time = torch.arange(float(T)) * args.timestep / 365.25  # in years
    time -= time.max()

    with particle_plate, scale(scale=1.0 / normalizer):
        log_rate_coef = pyro.sample(
            "log_rate_coef", dist.Laplace(0, features.new_ones(F)).to_event(1).mask(False)
        )

        log_rate = pyro.deterministic("log_rate", log_rate_coef @ features.T)  # NP S

        # Assume places differ only in their initial infection count.
        with place_plate:
            log_init = pyro.sample("log_init", dist.Normal(0, 10).expand([S]).mask(False).to_event(1))

        # Finally observe overdispersed counts.
        strain_probs = (log_init + log_rate * time[:, None, None]).softmax(-1)
        concentration = pyro.sample("concentration", dist.LogNormal(2, 4).mask(False))[:, None]

        obs_dist = dist.DirichletMultinomial(
                    total_count=weekly_strains.sum(-1).max(),
                    concentration=concentration * strain_probs,
                    is_sparse=True)  # uses a faster algorithm

        with time_plate, place_plate, scale(scale=weights):
            pyro.sample("obs", obs_dist, obs=weekly_strains)


def init_loc_fn(site):
    if site["name"] in ("log_rate_coef", "log_rate", "log_init"):
        return torch.zeros(site["fn"].shape())
    if site["name"] == "concentration":
        return torch.full(site["fn"].shape(), 5.0)
    return init_to_median(site)


def fit_mle(args, dataset):
    logger.info("Fitting via MLE")
    pyro.clear_param_store()
    pyro.set_rng_seed(20210319)

    guide = AutoDelta(model, init_loc_fn=init_loc_fn)
    guide(dataset["weekly_strains"], dataset["features"], num_particles=args.num_particles)

    num_params = sum(p.numel() for p in guide.parameters())
    logger.info(f"Training guide with {num_params} parameters:")

    optim = ClippedAdam({"lr": args.learning_rate, "betas": (0.8, 0.99),
                         "lrd": args.lrd ** (1 / args.num_steps)})
    svi = SVI(model, guide, optim, Trace_ELBO())
    normalizer = dataset["weekly_strains"].count_nonzero() * args.num_particles
    losses = []

    weights = dist.Dirichlet(dataset['T'] * torch.ones(dataset['T'] * dataset['P']))
    weights = weights.sample(sample_shape=(args.num_particles,))
    weights = dataset['T'] * dataset['P'] * weights.reshape(args.num_particles, dataset['T'], dataset['P'])
    print("weights", weights.min(), weights.max())

    for step in range(args.num_steps):
        loss = svi.step(dataset["weekly_strains"], dataset["features"],
                        num_particles=args.num_particles, normalizer=normalizer, weights=weights)
        assert not math.isnan(loss)
        losses.append(loss)
        if step % args.log_every == 0:
            concentration = guide.median()["concentration"].median().item()
            logger.info(
                f"step {step: >4d} loss = {loss:0.6g}\tconc. = {concentration:0.3g}\t"
            )

    return {
        "args": args,
        "guide": guide,
        "losses": losses,
    }


def main(args):
    print(args)
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    dataset = load_data(args)
    logger.info("Loaded dataset with (T, P, S) = ({}, {}, {})".format(dataset['T'], dataset['P'], dataset['S']))

    guide = fit_mle(args, dataset)['guide']

    median = guide.median()
    log_rate_coef = median['log_rate_coef']
    log_rate_coef_abs = log_rate_coef.abs()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="1-step-ahead forecasting"
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
    parser.add_argument("--learning-rate", default=0.03, type=float)
    parser.add_argument("--lrd", default=0.1, type=float)
    parser.add_argument("--num-steps", default=31, type=int)
    parser.add_argument("--num-particles", default=3, type=int)
    parser.add_argument("--mutation-cutoff", default=0.5, type=float)
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("-l", "--log-every", default=500, type=int)
    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    main(args)
