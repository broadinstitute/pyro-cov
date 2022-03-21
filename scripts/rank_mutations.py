# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import functools
import logging
import math
import os

import torch
from pyro import poutine

from pyrocov import mutrans

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def cached(filename):
    def decorator(fn):
        @functools.wraps(fn)
        def cached_fn(*args, **kwargs):
            f = filename(*args, **kwargs) if callable(filename) else filename
            if args[0].force or not os.path.exists(f):
                result = fn(*args, **kwargs)
                logger.info(f"saving {f}")
                torch.save(result, f)
            else:
                logger.info(f"loading cached {f}")
                result = torch.load(f, map_location=args[0].device)
            return result

        return cached_fn

    return decorator


@cached("results/mutrans.data.pt")
def load_data(args):
    return mutrans.load_gisaid_data(device=args.device)


@cached("results/rank_mutations.rank_mf_svi.pt")
def rank_mf_svi(args, dataset):
    result = mutrans.fit_mf_svi(
        dataset,
        mutrans.model,
        learning_rate=args.svi_learning_rate,
        num_steps=args.svi_num_steps,
        log_every=args.log_every,
        seed=args.seed,
    )
    result["args"] = (args,)
    sigma = result["mean"] / result["std"]
    result["ranks"] = sigma.sort(0, descending=True).indices
    result["cond_data"] = {
        "feature_scale": result["median"]["feature_scale"].item(),
        "concentration": result["median"]["concentration"].item(),
    }
    del result["guide"]
    return result


@cached("results/rank_mutations.rank_full_svi.pt")
def rank_full_svi(args, dataset):
    result = mutrans.fit_full_svi(
        dataset,
        mutrans.model,
        learning_rate=args.full_learning_rate,
        learning_rate_decay=args.full_learning_rate_decay,
        num_steps=args.full_num_steps,
        log_every=args.log_every,
        seed=args.seed,
    )
    result["args"] = (args,)
    result["mean"] = result["params"]["rate_coef_loc"]
    scale_tril = result["params"]["rate_coef_scale_tril"]
    result["cov"] = scale_tril @ scale_tril.T
    result["var"] = result["cov"].diag()
    result["std"] = result["var"].sqrt()
    sigma = result["mean"] / result["std"]
    result["ranks"] = sigma.sort(0, descending=True).indices
    result["cond_data"] = {
        "feature_scale": result["median"]["feature_scale"].item(),
        "concentration": result["median"]["concentration"].item(),
    }
    return result


@cached("results/rank_mutations.hessian.pt")
def compute_hessian(args, dataset, result):
    logger.info("Computing Hessian")
    features = dataset["features"]
    weekly_clades = dataset["weekly_clades"]
    rate_coef = result["median"]["rate_coef"].clone().requires_grad_()

    cond_data = result["median"].copy()
    cond_data.pop("rate")
    cond_data.pop("rate_coef")
    model = poutine.condition(mutrans.model, cond_data)

    def log_prob(rate_coef):
        with poutine.trace() as tr:
            with poutine.condition(data={"rate_coef": rate_coef}):
                model(weekly_clades, features)
        return tr.trace.log_prob_sum()

    hessian = torch.autograd.functional.hessian(
        log_prob,
        rate_coef,
        create_graph=False,
        strict=True,
    )

    result = {
        "args": args,
        "mutations": dataset["mutations"],
        "initial_ranks": result,
        "mean": result["mean"],
        "hessian": hessian,
    }

    logger.info("Computing covariance")
    result["cov"] = _sym_inverse(-hessian)
    result["var"] = result["cov"].diag()
    result["std"] = result["var"].sqrt()
    sigma = result["mean"] / result["std"]
    result["ranks"] = sigma.sort(0, descending=True).indices
    return result


def _sym_inverse(mat):
    eye = torch.eye(len(mat))
    e = None
    for exponent in [-math.inf] + list(range(-20, 1)):
        eps = 10 ** exponent
        try:
            u = torch.cholesky(eye * eps + mat)
        except RuntimeError as e:  # noqa F841
            continue
        logger.info(f"Added {eps:g} to Hessian diagonal")
        return torch.cholesky_inverse(u)
    raise e from None


def _fit_map_filename(args, dataset, cond_data, guide=None, without_feature=None):
    return f"results/rank_mutations.{guide is None}.{without_feature}.pt"


@cached(_fit_map_filename)
def fit_map(args, dataset, cond_data, guide=None, without_feature=None):
    if without_feature is not None:
        # Drop feature.
        dataset = dataset.copy()
        dataset["features"] = dataset["features"].clone()
        dataset["features"][:, without_feature] = 0

    # Condition model.
    cond_data = {k: torch.as_tensor(v) for k, v in cond_data.items()}
    model = poutine.condition(mutrans.model, cond_data)

    # Fit.
    result = mutrans.fit_map(
        dataset,
        model,
        guide,
        learning_rate=args.map_learning_rate,
        num_steps=args.map_num_steps,
        log_every=args.log_every,
        seed=args.seed,
    )

    result["args"] = args
    result["guide"] = guide
    if without_feature is None:
        result["mutation"] = None
    else:
        result["mutation"] = dataset["mutations"][without_feature]
    return result


def rank_map(args, dataset, initial_ranks):
    """
    Given an initial approximate ranking of features, compute MAP log
    likelihood ratios of the most significant features.
    """
    # Fit an initial model for warm-starting.
    cond_data = initial_ranks["cond_data"]
    if args.warm_start:
        guide = fit_map(args, dataset, cond_data)["guide"]
    else:
        guide = None

    # Evaluate on the null hypothesis + the most positive features.
    dropouts = {}
    for feature in [None] + initial_ranks["ranks"].tolist():
        dropouts[feature] = fit_map(args, dataset, cond_data, guide, feature)

    result = {
        "args": args,
        "mutations": dataset["mutations"],
        "initial_ranks": initial_ranks,
        "dropouts": dropouts,
    }
    logger.info("saving results/rank_mutations.pt")
    torch.save(result, "results/rank_mutations.pt")


def main(args):
    if args.double:
        torch.set_default_dtype(torch.double)
    if args.cuda:
        torch.set_default_tensor_type(
            torch.cuda.DoubleTensor if args.double else torch.cuda.FloatTensor
        )

    dataset = load_data(args)
    if args.full:
        initial_ranks = rank_full_svi(args, dataset)
    else:
        initial_ranks = rank_mf_svi(args, dataset)
    if args.hessian:
        compute_hessian(args, dataset, initial_ranks)
    if args.dropout:
        rank_map(args, dataset, initial_ranks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank mutations via SVI and leave-feature-out MAP"
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--full-num-steps", default=10001, type=int)
    parser.add_argument("--full-learning-rate", default=0.01, type=float)
    parser.add_argument("--full-learning-rate-decay", default=0.01, type=float)
    parser.add_argument("--svi-num-steps", default=1001, type=int)
    parser.add_argument("--svi-learning-rate", default=0.05, type=float)
    parser.add_argument("--map-num-steps", default=1001, type=int)
    parser.add_argument("--map-learning-rate", default=0.05, type=float)
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--hessian", action="store_true")
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--double", action="store_true", default=True)
    parser.add_argument("--single", action="store_false", dest="double")
    parser.add_argument(
        "--cuda", action="store_true", default=torch.cuda.is_available()
    )
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("--seed", default=20210319, type=int)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-l", "--log-every", default=100, type=int)
    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    main(args)
