import argparse
import functools
import logging
import os

import torch

from pyrocov import mutrans

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def cached(filename):
    def decorator(fn):
        @functools.wraps(fn)
        def cached_fn(*args, **kwargs):
            f = filename(*args, **kwargs) if callable(filename) else filename
            if not os.path.exists(f):
                result = fn(*args, **kwargs)
                logger.info(f"saving {f}")
                torch.save(result, f)
            else:
                logger.info(f"loading cached {f}")
                result = torch.load(f, map_location=torch.empty(()).device)
            return result

        return cached_fn

    return decorator


def _load_data_filename(args, **kwargs):
    return "results/mutrans.{}.pt".format(
        ".".join(["data"] + [f"{k}={v}" for k, v in sorted(kwargs.items())])
    )


@cached(_load_data_filename)
def load_data(args, **kwargs):
    return mutrans.load_data(device=args.device, **kwargs)


def _fit_filename(*args):
    strs = []
    for arg in args[2:]:
        if isinstance(arg, tuple):
            strs.append("-".join(f"{k}={v}" for k, v in arg))
        else:
            strs.append(str(arg))
    return "results/mutrans.fit.{}.pt".format(".".join(strs))


@cached(_fit_filename)
def fit(
    args,
    dataset,
    guide_type="mvn_dependent",
    n=1001,
    lr=0.01,
    lrd=0.1,
    holdout=(),
):
    result = mutrans.fit(
        dataset,
        guide_type=guide_type,
        num_steps=n,
        learning_rate=lr,
        learning_rate_decay=lrd,
        log_every=args.log_every,
        seed=args.seed,
    )

    result["args"] = args
    return result


def main(args):
    if args.double:
        torch.set_default_dtype(torch.double)
    if args.cuda:
        torch.set_default_tensor_type(
            torch.cuda.DoubleTensor if args.double else torch.cuda.FloatTensor
        )

    # Configure guides.
    # guide_type, n, lr, lrd
    guide_configs = [
        ("map", 1001, 0.05, 0.1),
        ("normal", 2001, 0.05, 0.1),
        ("mvn", 10001, 0.01, 0.1),
        ("mvn_dependent", 10001, 0.01, 0.1),
    ]
    best_config = (
        args.guide_type,
        args.num_steps,
        args.learning_rate,
        args.learning_rate_decay,
    )
    if args.best:
        dataset = load_data(args)
        fit(args, dataset, *best_config)
        return

    # Configure data holdouts.
    empty_holdout = ()
    holdouts = [
        (("virus_name_pattern", "^hCoV-19/USA/..-CDC-"),),
        (("virus_name_pattern", "^hCoV-19/USA/..-CDC-2-"),),
        (("location_pattern", "^North America / USA"),),
        (("location_pattern", "^Europe / United Kingdom"),),
    ]

    configs = [c + (empty_holdout,) for c in guide_configs]
    for holdout in holdouts:
        configs.append(best_config + holdout)

    # Sequentially fit models.
    result = {}
    for config in configs:
        holdout = dict(config[-1])
        dataset = load_data(args, **holdout)
        result[config] = fit(args, dataset, *config)
    logger.info("saving results/mutrans.pt")
    torch.save(result, "results/mutrans.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit mutation-transmissibility models")
    parser.add_argument("--best", action="store_true", help="fit only one config")
    parser.add_argument("-g", "--guide_type", default="mvn_dependent")
    parser.add_argument("-n", "--num-steps", default=10001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)

    parser.add_argument("--double", action="store_true", default=True)
    parser.add_argument("--single", action="store_false", dest="double")
    parser.add_argument(
        "--cuda", action="store_true", default=torch.cuda.is_available()
    )
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("--seed", default=20210319, type=int)
    parser.add_argument("-l", "--log-every", default=50, type=int)
    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    main(args)
