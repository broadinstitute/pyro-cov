#!/usr/bin/env python

import argparse
import functools
import logging
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
    parts = ["data", str(args.max_feature_order)]
    for k, v in sorted(kwargs.get("include", {}).items()):
        parts.append(f"I{k}={_safe_str(v)}")
    for k, v in sorted(kwargs.get("exclude", {}).items()):
        parts.append(f"E{k}={_safe_str(v)}")
    return "results/mutrans.{}.pt".format(".".join(parts))


@cached(_load_data_filename)
def load_data(args, **kwargs):
    return mutrans.load_gisaid_data(
        max_feature_order=args.max_feature_order, device=args.device, **kwargs
    )


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
    lr=0.01,
    lrd=0.1,
    holdout=(),
):
    start_time = default_timer()
    result = mutrans.fit_svi(
        dataset,
        guide_type=guide_type,
        num_steps=n,
        learning_rate=lr,
        learning_rate_decay=lrd,
        log_every=args.log_every,
        seed=args.seed,
    )
    result["walltime"] = default_timer() - start_time

    result["args"] = args
    return result


@cached(lambda *args: _fit_filename("mcmc", *args))
def fit_mcmc(
    args,
    dataset,
    guide_type="naive",
    num_steps=10001,
    num_warmup=1000,
    num_samples=1000,
    num_chains=1,
    max_tree_depth=10,
    holdout=(),
):
    if guide_type == "naive":
        guide = None
    else:
        guide = fit_svi(
            args,
            dataset,
            guide_type,
            num_steps,
            0.01,
            0.1,
            holdout,
        )["guide"].double()

    start_time = default_timer()
    result = mutrans.fit_mcmc(
        dataset,
        guide,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        max_tree_depth=max_tree_depth,
        log_every=args.log_every,
        seed=args.seed,
    )
    result["walltime"] = default_timer() - start_time

    result["args"] = args
    return result


def main(args):
    torch.set_default_dtype(torch.double)
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    # Run MCMC.
    mcmc_config = (
        "mcmc",
        args.mcmc_type,
        args.num_steps,
        args.num_warmup,
        args.num_samples,
        args.num_chains,
        args.max_tree_depth,
    )
    if args.mcmc:
        dataset = load_data(args)
        fit_mcmc(args, dataset, *mcmc_config[1:])
        return

    # Configure guides.
    svi_config = (
        args.guide_type,
        args.num_steps,
        args.learning_rate,
        args.learning_rate_decay,
    )
    if args.svi:
        dataset = load_data(args)
        fit_svi(args, dataset, *svi_config)
        return
    guide_types = [
        # "normal_delta",
        # "normal",
        # "mvn_delta",
        # "mvn_normal",
        # "normal_delta_dependent",
        "mvn_delta_dependent",
        # "normal_dependent",
        # "mvn_normal_dependent",
    ]

    # Add SVI configs.
    inference_configs = [
        svi_config,
        ("map", 1001, 0.05, 1.0),
    ]
    for guide_type in guide_types:
        inference_configs.append(
            (
                guide_type,
                args.num_steps,
                args.learning_rate,
                args.learning_rate_decay,
            )
        )

    # Add mcmc configs.
    inference_configs.append(mcmc_config)
    if args.mcmc_experiments:
        inference_configs.append(
            (
                "mcmc",
                "naive",
                args.num_steps,
                args.num_warmup,
                args.num_samples,
                args.num_chains,
                args.max_tree_depth,
            )
        )
        for guide_type in guide_types:
            inference_configs.append(
                (
                    "mcmc",
                    guide_type,
                    args.num_steps,
                    args.num_warmup,
                    args.num_samples,
                    args.num_chains,
                    args.max_tree_depth,
                )
            )

    # Configure data holdouts.
    empty_holdout = ()
    holdouts = [
        {"include": {"location": "^North America / USA"}},
        {"exclude": {"location": "^North America / USA"}},
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
        configs.append(mcmc_config + (holdout,))

    # Sequentially fit models.
    results = {}
    for config in configs:
        logger.info(f"Config: {config}")
        holdout = {k: dict(v) for k, v in config[-1]}
        dataset = load_data(args, **holdout)
        if config[0] == "mcmc":
            result = fit_mcmc(args, dataset, *config[1:])
        else:
            result = fit_svi(args, dataset, *config)
            result.pop("guide", None)  # to save space
        result["mutations"] = dataset["mutations"]
        result = torch_map(result, device="cpu", dtype=torch.float)  # to save space
        results[config] = result
    logger.info("saving results/mutrans.pt")
    torch.save(results, "results/mutrans.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit mutation-transmissibility models")
    parser.add_argument("--max-feature-order", default=0, type=int)
    parser.add_argument("--svi", action="store_true", help="run only SVI inference")
    parser.add_argument("--mcmc", action="store_true", help="run only MCMC inference")
    parser.add_argument("--mcmc-experiments", action="store_true")
    parser.add_argument("-g", "--guide-type", default="mvn_delta_dependent")
    parser.add_argument("-m", "--mcmc-type", default="mvn_delta_dependent")
    parser.add_argument("-n", "--num-steps", default=10001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("-w", "--num-warmup", default=500, type=int)
    parser.add_argument("-s", "--num-samples", default=500, type=int)
    parser.add_argument("-c", "--num-chains", default=2, type=int)
    parser.add_argument("-t", "--max-tree-depth", default=10, type=int)
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
