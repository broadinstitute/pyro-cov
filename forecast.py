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
from pyro.poutine import mask, trace, replay

from pyrocov import pangolin

from collections import defaultdict


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

    # 12 feature groups
    feature_groups = ['ORF7b', 'ORF7a', 'N', 'ORF8', 'ORF9b', 'ORF3a',
                      'ORF1b', 'M', 'ORF1a', 'E', 'S', 'ORF6']

    if args.feature_groups:  # tag features with 0, 1, 2, ..., 11
        feature_group_index = (-torch.ones(len(mutations))).long()

        for f, feature in enumerate(mutations):
            for g, group in enumerate(feature_groups):
                if feature.startswith(group):
                    feature_group_index[f] = g
            assert feature_group_index[f] != -1
    else:
        feature_group_index = None

    T, P, S = weekly_strains.shape

    return {"args": args, "weekly_strains": weekly_strains, "features": features, "T": T, "P": P, "S": S,
            "mutations": mutations, "feature_group_index": feature_group_index, "feature_groups": feature_groups}


def model(weekly_strains, features, feature_group_index=None,
          feature_scale=1.0, T_train=None, forecast=False):
    assert weekly_strains.shape[-1] == features.shape[0]
    if feature_group_index is not None:
        assert features.shape[-1:] == feature_group_index.shape

    T, P, S = weekly_strains.shape
    S, F = features.shape
    time_plate = pyro.plate("time", T, dim=-2)
    place_plate = pyro.plate("place", P, dim=-1)
    time = torch.arange(float(T)) * args.timestep / 365.25  # in years
    time -= time.max()

    if feature_group_index is not None:  # allow L1 regularization scale to depend on feature group
        feature_group_scale = feature_scale * pyro.sample("feature_group_scale", dist.Dirichlet(torch.ones(12)))
        feature_group_scale = feature_group_scale.index_select(-1, feature_group_index)
    else:
        feature_group_scale = feature_scale * features.new_ones(F)

    # Assume relative growth rate depends on mutation features but not time or place.
    log_rate_coef = pyro.sample(
        "log_rate_coef", dist.Laplace(0, feature_group_scale).to_event(1)
    )
    log_rate = pyro.deterministic("log_rate", log_rate_coef @ features.T)

    # Assume places differ only in their initial infection count.
    with place_plate:
        log_init = pyro.sample("log_init", dist.Normal(0, 10).expand([S]).to_event(1))

    # Finally observe overdispersed counts.
    strain_probs = (log_init + log_rate * time[:, None, None]).softmax(-1)
    concentration = pyro.sample("concentration", dist.LogNormal(2, 4))

    obs_dist = dist.DirichletMultinomial(
                total_count=weekly_strains.sum(-1).max(),
                concentration=concentration * strain_probs,
                is_sparse=True)  # uses a faster algorithm

    if T_train is not None:
        assert isinstance(T_train, int) and T_train < T
        time_mask = log_rate.new_ones(T)
        time_mask[T_train:] = 0.0
        time_mask = time_mask.bool().unsqueeze(-1)
    else:
        time_mask = True

    # if doing 1-step-ahead forecasting mask out held-out test data
    with time_plate, place_plate, mask(mask=time_mask):
        pyro.sample("obs", obs_dist, obs=weekly_strains)

    # compute LL for held-out test data
    if forecast:
        return obs_dist.log_prob(weekly_strains)[T_train:]  # T P


def init_loc_fn(site):
    if site["name"] in ("log_rate_coef", "log_rate", "log_init"):
        return torch.zeros(site["fn"].shape())
    if site["name"] == "concentration":
        return torch.full(site["fn"].shape(), 5.0)
    return init_to_median(site)


def _fit_map_filename(args, dataset, guide=None, without_feature=None):
    return f"results/rank_mutations.{guide is None}.{without_feature}.pt"


# @cached(_fit_map_filename)
def fit_map(args, dataset, T_train=None):
    logger.info("Fitting via MAP")
    pyro.clear_param_store()
    pyro.set_rng_seed(20210319)

    guide = AutoDelta(model, init_loc_fn=init_loc_fn)
    guide(dataset["weekly_strains"], dataset["features"], feature_group_index=dataset['feature_group_index'])

    num_params = sum(p.numel() for p in guide.parameters())
    logger.info(f"Training guide with {num_params} parameters:")

    optim = ClippedAdam({"lr": args.learning_rate, "betas": (0.8, 0.99),
                         "lrd": args.lrd ** (1 / args.num_steps)})
    svi = SVI(model, guide, optim, Trace_ELBO())
    num_obs = dataset["weekly_strains"].count_nonzero()
    losses = []

    for step in range(args.num_steps):
        loss = svi.step(dataset["weekly_strains"], dataset["features"], feature_scale=args.feature_scale,
                        feature_group_index=dataset["feature_group_index"], T_train=T_train) / num_obs
        assert not math.isnan(loss)
        losses.append(loss)
        if step % args.log_every == 0:
            concentration = guide.median()["concentration"].item()
            logger.info(
                f"step {step: >4d} loss = {loss:0.6g}\tconc. = {concentration:0.3g}\t"
            )

    # compute 1-step-ahead forecast LL
    if T_train is not None:
        guide_tr = trace(guide).get_trace(dataset["weekly_strains"], dataset["features"],
                        feature_group_index=dataset["feature_group_index"], T_train=T_train, forecast=True)
        model_tr = trace(replay(model, trace=guide_tr)).get_trace(dataset["weekly_strains"], dataset["features"],
                        feature_group_index=dataset["feature_group_index"], T_train=T_train, forecast=True)
        ll_forecast = model_tr.nodes['_RETURN']['value'].data[0]
        ll_forecast = ll_forecast[ll_forecast.nonzero(as_tuple=False)].mean().item()
    else:
        ll_forecast = None

    return {
        "args": args,
        "guide": guide,
        "losses": losses,
        "ll_forecast": ll_forecast,
    }


def main(args):
    print(args)
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    dataset = load_data(args)

    if args.forecast == 0:
        dataset['feature_group_index'] = None
        guide = fit_map(args, dataset)['guide']

        median = guide.median()
        log_rate_coef = median['log_rate_coef']
        log_rate_coef_abs = log_rate_coef.abs()

        print("sum(log_rate_coef_abs > 0.1)", sum(log_rate_coef_abs > 0.1).item())
        print("sum(log_rate_coef_abs > 0.01)", sum(log_rate_coef_abs > 0.01).item())
        print("sum(log_rate_coef_abs < 0.1)", sum(log_rate_coef_abs < 0.1).item())
        print("sum(log_rate_coef_abs < 1.0e-3)", sum(log_rate_coef_abs < 1.0e-3).item())
        print("sum(log_rate_coef_abs < 1.0e-5)", sum(log_rate_coef_abs < 1.0e-5).item())

        counter = defaultdict(lambda: 0)

        for k, v in enumerate(log_rate_coef_abs.data.cpu().numpy()):
            if v > 0.01:
                counter[dataset['feature_groups'][dataset['feature_group_index'][k].item()]] += 1

        print(counter)

        if args.feature_groups:
            print("median[feature_group_scale]", median['feature_group_scale'].data.cpu().numpy())

    else:  # do rolling forecasting
        ll_forecasts = []
        for t in range(1, args.forecast + 1)[::-1]:
             T_train = dataset['T'] - t
             print('Training with T_train = {}'.format(T_train))
             result = fit_map(args, dataset, T_train=T_train)
             print("ll forecast: {:.4f}".format(result['ll_forecast']))
             ll_forecasts.append(result['ll_forecast'])
        print("MEAN FORECAST LL OVER {} TIME PERIOD WINDOW: {:.4f}  (feature_scale = {:.3f})".format(args.forecast,
              np.mean(ll_forecasts), args.feature_scale))
        forecast = pd.DataFrame(np.mean(ll_forecasts), columns=['LL'], index=['{}-forecast'.format(args.forecast)])
        f = 'forecasts/forecast.f_{}.fs_{:.3f}.fg_{}.csv'.format(args.forecast, args.feature_scale, args.feature_groups)
        forecast.to_csv(f)


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
    parser.add_argument("--learning-rate", default=0.05, type=float)
    parser.add_argument("--lrd", default=0.1, type=float)
    parser.add_argument("--num-steps", default=4001, type=int)
    parser.add_argument("--forecast", default=0, type=int)
    parser.add_argument("--feature-scale", default=0.04, type=float)
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--feature-groups", action="store_true")
    parser.add_argument("-l", "--log-every", default=500, type=int)
    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    main(args)
