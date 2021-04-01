import argparse
import functools
import logging
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
                result = torch.load(f)
            return result

        return cached_fn

    return decorator


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
    median = result["guide"].median()
    result["cond_data"] = {
        "feature_scale": median["feature_scale"].item(),
        "concentration": median["concentration"].item(),
    }
    del result["guide"]
    return result


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

    dataset = mutrans.load_data(device=args.device)
    initial_ranks = rank_svi(args, dataset)
    if args.map_num_steps:
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
