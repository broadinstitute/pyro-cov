import argparse
import functools
import logging
import os

import pyro.distributions as dist
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


@cached("results/rank_mutations.data.pt")
def load_data(args):
    return mutrans.load_data(device=args.device)


@cached("results/rank_mutations.rank_svi.pt")
def rank_svi(args, dataset):
    result = mutrans.fit_svi(
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


@cached("results/rank_mutations.hessian.pt")
def compute_hessian(args, dataset, result):
    logger.info("Computing Hessian")
    features = dataset["features"]
    weekly_strains = dataset["weekly_strains"]
    log_rate_coef = result["median"]["log_rate_coef"].clone().requires_grad_()

    if True:
        cond_data = result["median"].copy()
        cond_data.pop("log_rate")
        cond_data.pop("log_rate_coef")
        model = poutine.condition(mutrans.model, cond_data)

        def log_prob(log_rate_coef):
            with poutine.trace() as tr:
                with poutine.condition(data={"log_rate_coef": log_rate_coef}):
                    model(weekly_strains, features)
            return tr.trace.log_prob_sum()

    else:
        # I haven't been able to get trace(condition(mutrans.model)) working;
        # instead we hand-reproduce the likelihood of mutrans.model.
        log_init = result["median"]["log_init"]
        concentration = result["median"]["concentration"]
        T, P, S = weekly_strains.shape
        time = torch.arange(float(T)) * mutrans.TIMESTEP / 365.25  # in years
        time -= time.max()

        def log_prob(log_rate_coef):
            log_rate = log_rate_coef @ features.T
            strain_probs = (log_init + log_rate * time[:, None, None]).softmax(-1)
            d = dist.DirichletMultinomial(
                total_count=weekly_strains.sum(-1).max(),
                concentration=concentration * strain_probs,
                is_sparse=True,  # uses a faster algorithm
            )
            return d.log_prob(weekly_strains).sum()

    hessian = torch.autograd.functional.hessian(
        log_prob,
        log_rate_coef,
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
    for exponent in range(-20, 1):
        try:
            u = torch.cholesky(eye * (10 ** exponent) + mat)
        except RuntimeError as e:  # noqa F841
            continue
        logger.info(f"Added 1e{exponent} to Hessian diagonal")
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
    for feature in [None] + initial_ranks["ranks"][: args.num_features].tolist():
        dropouts[feature] = fit_map(args, dataset, cond_data, guide, feature)

    result = {
        "args": args,
        "mutations": dataset["mutations"],
        "initial_ranks": initial_ranks,
        "dropouts": dropouts,
    }
    logger.info("saving results/rank_mutations.pt")
    torch.save(result, "results/rank_mutations.pt")


def rank_map_parallel(args, dataset, initial_ranks):
    # Condition model.
    cond_data = initial_ranks["cond_data"]
    cond_data = {k: torch.as_tensor(v) for k, v in cond_data.items()}
    model = poutine.condition(mutrans.dropout_model, cond_data)

    # Fit.
    result = mutrans.fit_map(
        dataset,
        model,
        learning_rate=args.map_learning_rate,
        num_steps=args.map_num_steps,
        log_every=args.log_every,
        seed=args.seed,
        vectorized=True,
    )

    result["args"] = args
    result["mutations"] = dataset["mutations"]
    return result


def main(args):
    if args.double:
        torch.set_default_dtype(torch.double)
    if args.cuda:
        torch.set_default_tensor_type(
            torch.cuda.DoubleTensor if args.double else torch.cuda.FloatTensor
        )

    dataset = load_data(args)
    initial_ranks = rank_svi(args, dataset)
    if args.hessian:
        compute_hessian(args, dataset, initial_ranks)
    if args.map_num_steps and args.num_features:
        rank_map(args, dataset, initial_ranks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank mutations via SVI and leave-feature-out MAP"
    )
    parser.add_argument("--svi-learning-rate", default=0.05, type=float)
    parser.add_argument("--svi-num-steps", default=1001, type=int)
    parser.add_argument("--map-learning-rate", default=0.05, type=float)
    parser.add_argument("--map-num-steps", default=1001, type=int)
    parser.add_argument("--num-features", type=int)
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
    parser.add_argument("-l", "--log-every", default=50, type=int)
    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    main(args)
