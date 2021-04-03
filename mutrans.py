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


@cached("results/mutrans.data.pt")
def load_data(args):
    return mutrans.load_data(device=args.device)


@cached(lambda *args: "results/mutrans.fit.{}.pt".format(".".join(map(str, args[2:]))))
def fit(
    args,
    dataset,
    guide_type="mvn_dependent",
    n=1001,
    lr=0.01,
    lrd=0.1,
    holdout=None,
):
    if holdout is not None:
        dataset = dataset.copy()
        raise NotImplementedError("TODO")

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

    dataset = load_data(args)

    # guide_type, n, lr, lrd
    configs = [
        ("map", 1001, 0.05, 0.1),
        ("normal", 2001, 0.05, 0.1),
        ("mvn", 10001, 0.01, 0.1),
        ("mvn_dependent", 10001, 0.01, 0.1),
    ]
    holdouts = [None]
    result = {}
    for config in configs:
        for holdout in holdouts:
            key = config + (holdout,)
            result[key] = fit(args, dataset, *key)
    logger.info("saving results/mutrans.pt")
    torch.save(result, "results/mutrans.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit mutation-transmissibility models")
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
