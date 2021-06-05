#!/usr/bin/env python

import argparse
import functools
import gc
import logging
import os
import re

import pyro
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
                if not args[0].test:
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
    return mutrans.load_gisaid_data(device=args.device, **kwargs)


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
    model_type="",
    cond_data="",
    guide_type="mvn_dependent",
    n=1001,
    lr=0.01,
    lrd=0.1,
    cn=10.0,
    r=10,
    holdout=(),
):
    for kv in cond_data.split(","):
        if kv:
            k, v = kv.split("=")
            cond_data[k] = float(v)

    result = mutrans.fit_svi(
        dataset,
        model_type=model_type,
        cond_data={},
        guide_type=guide_type,
        num_steps=n,
        learning_rate=lr,
        learning_rate_decay=lrd,
        clip_norm=cn,
        rank=r,
        log_every=args.log_every,
        seed=args.seed,
    )
    result["args"] = args
    return result


@cached(lambda *args: _fit_filename("bootstrap", *args))
def fit_bootstrap(
    args,
    dataset,
    num_samples,
    model_type="",
    cond_data="",
    guide_type="mvn_dependent",
    n=1001,
    p=1,
    lr=0.01,
    lrd=0.1,
    cn=10.0,
    r=10,
    holdout=(),
):
    result = mutrans.fit_bootstrap(
        dataset,
        num_samples=num_samples,
        model_type=model_type,
        cond_data=cond_data,
        guide_type=guide_type,
        num_steps=n,
        learning_rate=lr,
        learning_rate_decay=lrd,
        clip_norm=cn,
        rank=r,
        log_every=args.log_every,
        seed=args.seed,
    )
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

    # Configure fits.
    configs = []
    empty_holdout = ()
    if args.vary_num_steps:
        grid = sorted(int(n) for n in args.vary_num_steps.split(","))
        for num_steps in grid:
            configs.append(
                (
                    args.model_type,
                    args.cond_data,
                    args.guide_type,
                    num_steps,
                    args.learning_rate,
                    args.learning_rate_decay,
                    args.clip_norm,
                    args.rank,
                    empty_holdout,
                )
            )
    elif args.vary_model_type:
        for model_type in args.vary_model_type.split(","):
            configs.append(
                (
                    model_type,
                    args.cond_data,
                    args.guide_type,
                    args.num_steps,
                    args.learning_rate,
                    args.learning_rate_decay,
                    args.clip_norm,
                    args.rank,
                    empty_holdout,
                )
            )
    elif args.vary_guide_type:
        for guide_type in args.vary_guide_type.split(","):
            configs.append(
                (
                    args.model_type,
                    args.cond_data,
                    guide_type,
                    args.num_steps,
                    args.learning_rate,
                    args.learning_rate_decay,
                    args.clip_norm,
                    args.rank,
                    empty_holdout,
                )
            )
    elif args.vary_holdout:
        grid = [
            {},
            {"include": {"location": "^Europe"}},
            {"exclude": {"location": "^Europe"}},
            # {"include": {"location": "^North America"}},
            # {"exclude": {"location": "^North America"}},
            # {"include": {"location": "^North America / USA"}},
            # {"exclude": {"location": "^North America / USA"}},
            # {"include": {"location": "^Europe / United Kingdom"}},
            # {"exclude": {"location": "^Europe / United Kingdom"}},
            # {"include": {"virus_name": "^hCoV-19/USA/..-CDC-"}},
            # {"include": {"virus_name": "^hCoV-19/USA/..-CDC-2-"}},
        ]
        for holdout in grid:
            holdout = tuple(
                (k, tuple(sorted(v.items()))) for k, v in sorted(holdout.items())
            )
            configs.append(
                (
                    args.model_type,
                    args.cond_data,
                    args.guide_type,
                    args.num_steps,
                    args.learning_rate,
                    args.learning_rate_decay,
                    args.clip_norm,
                    args.rank,
                    holdout,
                )
            )
    else:
        configs.append(
            (
                args.model_type,
                args.cond_data,
                args.guide_type,
                args.num_steps,
                args.learning_rate,
                args.learning_rate_decay,
                args.clip_norm,
                args.rank,
                empty_holdout,
            )
        )

    # Sequentially fit models.
    results = {}
    for config in configs:
        logger.info(f"Config: {config}")
        holdout = {k: dict(v) for k, v in config[-1]}
        dataset = load_data(args, **holdout)
        if args.bootstrap:
            result = fit_bootstrap(args, dataset, args.bootstrap, *config)
        else:
            result = fit_svi(args, dataset, *config)
        mutrans.log_stats(dataset, result)
        result["mutations"] = dataset["mutations"]
        result = torch_map(result, device="cpu", dtype=torch.float)  # to save space
        results[config] = result

        del dataset
        pyro.clear_param_store()
        gc.collect()

    if not args.test:
        logger.info("saving results/mutrans.pt")
        torch.save(results, "results/mutrans.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit mutation-transmissibility models")
    parser.add_argument("--cond-data", default="")
    parser.add_argument("--vary-model-type", help="comma delimited list of model types")
    parser.add_argument("--vary-guide-type", help="comma delimited list of guide types")
    parser.add_argument("--vary-num-steps", help="comma delimited list of num_steps")
    parser.add_argument("--vary-holdout", action="store_true")
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("-m", "--model-type", default="")
    parser.add_argument("-g", "--guide-type", default="custom")
    parser.add_argument("-n", "--num-steps", default=10001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("-cn", "--clip-norm", default=10.0, type=float)
    parser.add_argument("-r", "--rank", default=10, type=int)
    parser.add_argument("-fp64", "--double", action="store_true")
    parser.add_argument("-fp32", "--float", action="store_false", dest="double")
    parser.add_argument(
        "--cuda", action="store_true", default=torch.cuda.is_available()
    )
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("--seed", default=20210319, type=int)
    parser.add_argument("-l", "--log-every", default=50, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    main(args)
