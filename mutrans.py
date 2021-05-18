#!/usr/bin/env python

import argparse
import functools
import logging
import math
import os
import re
from timeit import default_timer

import torch

from pyrocov import mutrans
from pyrocov.util import torch_map

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


def _safe_str(v):
    v = str(v)
    v = re.sub("[^A-Za-x0-9-]", "_", v)
    return v


def _load_data_filename(args, **kwargs):
    parts = ["data", "double" if args.double else "single"]
    for k, v in sorted(kwargs.get("include", {}).items()):
        parts.append(f"I{k}={_safe_str(v)}")
    for k, v in sorted(kwargs.get("exclude", {}).items()):
        parts.append(f"E{k}={_safe_str(v)}")
    return "results/mutrans.{}.pt".format(".".join(parts))


@cached(_load_data_filename)
def load_data(args, **kwargs):
    return mutrans.load_gisaid_data(device=args.device, max_obs=args.max_obs, **kwargs)


def _fit_filename(name, *args):
    strs = [name]
    for arg in args[2:]:
        if isinstance(arg, tuple):
            strs.append("-".join(f"{k}={_safe_str(v)}" for k, v in arg))
        else:
            strs.append(str(arg))
    return "results/mutrans.{}.pt".format(".".join(strs))


@cached(lambda *args: _fit_filename("svi", *args))
def fit_svi(
    args,
    dataset,
    guide_type="mvn_dependent",
    n=1001,
    p=1,
    lr=0.01,
    lrd=0.1,
    cn=10.0,
    holdout=(),
):
    start_time = default_timer()
    result = mutrans.fit_svi(
        dataset,
        guide_type=guide_type,
        num_steps=n,
        num_particles=p,
        learning_rate=lr,
        learning_rate_decay=lrd,
        clip_norm=cn,
        log_every=args.log_every,
        seed=args.seed,
    )
    result["walltime"] = default_timer() - start_time

    result["args"] = args
    return result


def main(args):
    torch.set_default_dtype(torch.double if args.double else torch.float)
    if args.cuda:
        torch.set_default_tensor_type(
            torch.cuda.DoubleTensor if args.double else torch.cuda.FloatTensor
        )
    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    # Configure guides.
    svi_config = (
        args.guide_type,
        args.num_steps,
        args.num_particles,
        args.learning_rate,
        args.learning_rate_decay,
        args.clip_norm,
    )
    if args.svi:
        dataset = load_data(args)
        fit_svi(args, dataset, *svi_config)
        return

    inference_configs = [
        svi_config,
        # ("map", 2001, 1, 0.05, 0.1, 10.0),
    ]

    # Add SVI configs.
    guide_types = [
        # "normal_delta",
        # "normal",
        # "mvn_delta",
        # "mvn_normal",
        # "normal_delta_dependent",
        # "mvn_delta_dependent",
        # "normal_dependent",
        # "mvn_normal_dependent",
    ]
    for guide_type in guide_types:
        inference_configs.append(
            (
                guide_type,
                args.num_steps,
                args.num_particles,
                args.learning_rate,
                args.learning_rate_decay,
                args.clip_norm,
            )
        )

    # Configure data holdouts.
    empty_holdout = ()
    holdouts = [
        # {"include": {"location": "^Europe"}},
        # {"exclude": {"location": "^Europe"}},
        # {"include": {"location": "^North America"}},
        # {"exclude": {"location": "^North America"}},
        # {"include": {"location": "^North America / USA"}},
        # {"exclude": {"location": "^North America / USA"}},
        # {"include": {"location": "^Europe / United Kingdom"}},
        # {"exclude": {"location": "^Europe / United Kingdom"}},
        # {"include": {"virus_name": "^hCoV-19/USA/..-CDC-"}},
        # {"include": {"virus_name": "^hCoV-19/USA/..-CDC-2-"}},
    ]
    configs = [c + (empty_holdout,) for c in inference_configs]
    for holdout in holdouts:
        holdout = tuple(
            (k, tuple(sorted(v.items()))) for k, v in sorted(holdout.items())
        )
        configs.append(svi_config + (holdout,))

    # Sequentially fit models.
    results = {}
    for config in configs:
        logger.info(f"Config: {config}")
        holdout = {k: dict(v) for k, v in config[-1]}
        dataset = load_data(args, **holdout)
        result = fit_svi(args, dataset, *config)
        result.pop("guide", None)  # to save space
        result["mutations"] = dataset["mutations"]
        result = torch_map(result, device="cpu", dtype=torch.float)  # to save space
        results[config] = result
    logger.info("saving results/mutrans.pt")
    torch.save(results, "results/mutrans.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit mutation-transmissibility models")
    parser.add_argument("--max-obs", default=math.inf, type=int)
    parser.add_argument("--svi", action="store_true", help="run only one SVI config")
    parser.add_argument("-g", "--guide-type", default="mvn_normal_dependent")
    parser.add_argument("-n", "--num-steps", default=10001, type=int)
    parser.add_argument("-p", "--num-particles", default=1, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("-cn", "--clip-norm", default=10.0, type=float)
    parser.add_argument("-fp64", "--double", action="store_true")
    parser.add_argument("-fp32", "--float", action="store_false", dest="double")
    parser.add_argument(
        "--cuda", action="store_true", default=torch.cuda.is_available()
    )
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("--seed", default=20210319, type=int)
    parser.add_argument("-l", "--log-every", default=50, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    main(args)
