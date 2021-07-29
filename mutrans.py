#!/usr/bin/env python

import argparse
import functools
import gc
import logging
import os
import re
from typing import Callable, Union

import pyro
import torch

from pyrocov import mutrans, pangolin
from pyrocov.sarscov2 import GENE_TO_POSITION
from pyrocov.util import torch_map

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def cached(filename: Union[str, Callable]):
    """
    Simple utiltity to cache results based on filename.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def cached_fn(*args, **kwargs):
            base_args = args[0]
            if base_args.no_cache:
                return fn(*args, **kwargs)
            f = filename(*args, **kwargs) if callable(filename) else filename
            if os.path.exists(f):
                logger.info(f"loading cached {f}")
                return torch.load(f, map_location=torch.empty(()).device)
            if base_args.no_new:
                raise ValueError(f"Missing {f}")
            result = fn(*args, **kwargs)
            if not args[0].test:
                logger.info(f"saving {f}")
                torch.save(result, f)
            return result

        return cached_fn

    return decorator


def _safe_str(v):
    v = str(v)
    v = re.sub("[^A-Za-x0-9-]", "_", v)
    return v


def holdout_to_hashable(holdout):
    return tuple((k, tuple(sorted(v.items()))) for k, v in sorted(holdout.items()))


def hashable_to_holdout(holdout):
    return {k: dict(v) for k, v in holdout}


def _load_data_filename(args, **kwargs):
    parts = ["data", "double" if args.double else "single"]
    for k, v in sorted(kwargs.get("include", {}).items()):
        parts.append(f"I{k}={_safe_str(v)}")
    for k, v in sorted(kwargs.get("exclude", {}).items()):
        parts.append(f"E{k}={_safe_str(v)}")
    parts.append(str(kwargs.get("end_day")))
    return "results/mutrans.{}.pt".format(".".join(parts))


@cached(_load_data_filename)
def load_data(args, **kwargs):
    """
    Cached wrapper to load GISAID data.
    """
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
    cond_data="",
    guide_type="mvn_dependent",
    n=1001,
    lr=0.01,
    lrd=0.1,
    cn=10.0,
    r=200,
    f=6,
    end_day=None,
    holdout=(),
):
    """
    Cached wrapper to fit a model via SVI.
    """
    cond_data = [kv.split("=") for kv in cond_data.split(",") if kv]
    cond_data = {k: float(v) for k, v in cond_data}
    holdout = hashable_to_holdout(holdout)

    result = mutrans.fit_svi(
        dataset,
        cond_data=cond_data,
        guide_type=guide_type,
        num_steps=n,
        learning_rate=lr,
        learning_rate_decay=lrd,
        clip_norm=cn,
        rank=r,
        forecast_steps=f,
        log_every=args.log_every,
        seed=args.seed,
        jit=args.jit,
        num_samples=args.num_samples,
    )

    if "lineage" in holdout.get("exclude", {}):
        # Save only what's needed to evaluate loo predictions.
        result = {
            "median": {
                "coef": result["median"]["coef"].float(),  # [F]
                "rate_loc": result["median"]["rate_loc"].float(),  # [S]
            },
        }

    result["args"] = args
    return result


def vary_leaves(args, default_config):
    """
    Run a leave-one-out experiment over a set of leaf lineages, saving results
    to ``results/mutrans.vary_leaves.pt``.
    """
    # Load a single common dataset.
    dataset = load_data(args)
    lineage_id = {name: i for i, name in enumerate(dataset["lineage_id_inv"])}
    descendents = pangolin.find_descendents(dataset["lineage_id_inv"])
    if args.only_gene:
        for m in dataset["mutations"]:
            assert m.startswith(args.only_gene + ":"), m

    # Run default config to get a ranking of leaves.
    def make_config(**holdout):
        if args.only_gene:
            include = holdout.setdefault("include", {})
            include["gene"] = args.only_gene
        config = list(default_config)
        config[-1] = holdout_to_hashable(holdout)
        config = tuple(config)
        return config

    result = fit_svi(args, dataset, *make_config())

    # Rank lineages by divergence from parent.
    lineages = mutrans.rank_loo_lineages(dataset, result)
    lineages = lineages[: args.vary_leaves]
    logger.info(
        "\n".join(
            [f"Leave-one-out predicting growth rate of {len(lineages)} lineages:"]
            + lineages
        )
    )

    # Run inference for each lineage. This is very expensive.
    results = {}
    for lineage in lineages:
        config = make_config(exclude={"lineage": "^" + lineage + "$"})
        logger.info(f"Config: {config}")

        # Construct a leave-one-out dataset by zeroing out a subclade.
        clade = [lineage_id[lineage]]
        for descendent in descendents[lineage]:
            clade.append(lineage_id[descendent])
        loo_dataset = dataset.copy()
        loo_dataset["weekly_strains"] = dataset["weekly_strains"].clone()
        loo_dataset["weekly_strains"][:, :, clade] = 0

        # Run SVI
        result = fit_svi(args, loo_dataset, *config)
        result["mutations"] = dataset["mutations"]
        result["location_id"] = dataset["location_id"]
        result["lineage_id_inv"] = dataset["lineage_id_inv"]
        results[config] = result

        # Cleanup
        del result
        pyro.clear_param_store()
        gc.collect()

    if not args.test:
        logger.info("saving results/mutrans.vary_leaves.pt")
        torch.save(results, "results/mutrans.vary_leaves.pt")


def vary_gene(args, default_config):
    """
    Train on the whole genome and on various single genes, saving results to
    ``results/mutrans.vary_gene.pt``.
    """
    # Collect a set of genes.
    if args.vary_gene == "all":
        genes = list(GENE_TO_POSITION)
    else:
        genes = args.vary_gene.split(",")
    logger.info("Fitting to each of genes: {}".format(", ".join(genes)))

    # Construct a grid of holdouts.
    grid = [{}]
    for gene in genes:
        grid.append({"include": {"gene": f"^{gene}:"}})
        grid.append({"exclude": {"gene": f"^{gene}:"}})

    def make_config(**holdout):
        config = list(default_config)
        config[-1] = holdout_to_hashable(holdout)
        config = tuple(config)
        return config

    results = {}
    for holdout in grid:
        # Fit a single model.
        logger.info(f"Holdout: {holdout}")
        dataset = load_data(args, **holdout)
        result = fit_svi(args, dataset, *make_config(**holdout))

        # Save metrics.
        key = holdout_to_hashable(holdout)
        results[key] = mutrans.log_stats(dataset, result)

        # Clean up to save memory.
        del dataset, result
        pyro.clear_param_store()
        gc.collect()

    if not args.test:
        logger.info("saving results/mutrans.vary_gene.pt")
        torch.save(results, "results/mutrans.vary_gene.pt")


def main(args):
    """Main Entry Point"""

    # Torch configuration
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
    empty_end_day = None
    default_config = (
        args.cond_data,
        args.guide_type,
        args.num_steps,
        args.learning_rate,
        args.learning_rate_decay,
        args.clip_norm,
        args.rank,
        args.forecast_steps,
        empty_end_day,
        empty_holdout,
    )

    if args.vary_leaves:
        return vary_leaves(args, default_config)
    if args.vary_gene:
        return vary_gene(args, default_config)

    if args.vary_num_steps:
        grid = sorted(int(n) for n in args.vary_num_steps.split(","))
        for num_steps in grid:
            configs.append()
    elif args.vary_guide_type:
        for guide_type in args.vary_guide_type.split(","):
            configs.append(
                (
                    args.cond_data,
                    guide_type,
                    args.num_steps,
                    args.learning_rate,
                    args.learning_rate_decay,
                    args.clip_norm,
                    args.rank,
                    args.forecast_steps,
                    empty_end_day,
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
            if args.only_gene:
                include = holdout.setdefault("include", {})
                include["gene"] = f"^{args.only_gene}:"
            configs.append(
                (
                    args.cond_data,
                    args.guide_type,
                    args.num_steps,
                    args.learning_rate,
                    args.learning_rate_decay,
                    args.clip_norm,
                    args.rank,
                    args.forecast_steps,
                    empty_end_day,
                    holdout_to_hashable(holdout),
                )
            )
    elif args.backtesting_max_day:
        for max_day in args.backtesting_max_day.split(","):
            max_day = int(max_day)
            configs.append(
                (
                    args.cond_data,
                    args.guide_type,
                    args.num_steps,
                    args.learning_rate,
                    args.learning_rate_decay,
                    args.clip_norm,
                    args.rank,
                    args.forecast_steps,
                    max_day,
                    empty_holdout,
                )
            )
    else:
        configs.append(default_config)

    # Sequentially fit models.
    results = {}
    for config in configs:
        logger.info(f"Config: {config}")

        # Holdout is the last in the config
        holdout = hashable_to_holdout(config[-1])
        # end_day is second from last
        end_day = config[-2]

        # load dataset
        dataset = load_data(args, end_day=end_day, **holdout)

        # Run SVI
        result = fit_svi(args, dataset, *config)
        mutrans.log_stats(dataset, result)

        # Save the results for this config

        # Augment gisaid dataset with JHU timeseries counts
        dataset.update(mutrans.load_jhu_data(dataset))

        # Generate results
        result["mutations"] = dataset["mutations"]
        result["weekly_strains"] = dataset["weekly_strains"]
        result["weekly_cases"] = dataset["weekly_cases"]
        result["weekly_strains_shape"] = tuple(dataset["weekly_strains"].shape)
        result["location_id"] = dataset["location_id"]
        result["lineage_id_inv"] = dataset["lineage_id_inv"]

        result = torch_map(result, device="cpu", dtype=torch.float)  # to save space
        results[config] = result

        # Ensure number of regions match
        assert dataset["weekly_strains"].shape[1] == result["mean"]["probs"].shape[1]
        assert dataset["weekly_cases"].shape[1] == result["mean"]["probs"].shape[1]

        # Cleanup
        del dataset
        pyro.clear_param_store()
        gc.collect()

    if args.vary_holdout:
        mutrans.log_holdout_stats({k[-1]: v for k, v in results.items()})

    if not args.test:
        logger.info("saving results/mutrans.pt")
        torch.save(results, "results/mutrans.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit mutation-transmissibility models")
    parser.add_argument("--vary-guide-type", help="comma delimited list of guide types")
    parser.add_argument("--vary-num-steps", help="comma delimited list of num_steps")
    parser.add_argument("--vary-holdout", action="store_true")
    parser.add_argument(
        "--vary-leaves", type=int, help="min number of samples per held out lineage"
    )
    parser.add_argument(
        "--vary-gene",
        help="a comma-separated list of genes or multiple genes, e.g. S,N,S|N",
    )
    parser.add_argument("--only-gene")
    parser.add_argument("-cd", "--cond-data", default="coef_scale=0.5")
    parser.add_argument("-g", "--guide-type", default="custom")
    parser.add_argument("-n", "--num-steps", default=10001, type=int)
    parser.add_argument("-s", "--num-samples", default=1000, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("-cn", "--clip-norm", default=10.0, type=float)
    parser.add_argument("-r", "--rank", default=200, type=int)
    parser.add_argument("-f", "--forecast-steps", default=6, type=int)
    parser.add_argument("-fp64", "--double", action="store_true")
    parser.add_argument("-fp32", "--float", action="store_false", dest="double")
    parser.add_argument(
        "--cuda", action="store_true", default=torch.cuda.is_available()
    )
    parser.add_argument("-b", "--backtesting-max-day", default=None)
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("--jit", action="store_true", default=False)
    parser.add_argument("--no-jit", dest="jit", action="store_false")
    parser.add_argument("--seed", default=20210319, type=int)
    parser.add_argument("-l", "--log-every", default=50, type=int)
    parser.add_argument("--no-new", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    main(args)
