"""
This experiment aims to assess posterior accuracy of a variational approach to
phylogenetic inference. We examine by default the dataset DS4 [2] from Whidden
and Matsen [1]. This dataset has 41 taxa and 1137 characters, many of which are
unobserved for many taxa.

[1] Chris Whidden, Frederick A. Matsen, IV (2015)
    "Quantifying MCMC Exploration of Phylogenetic Tree Space"
    https://academic.oup.com/sysbio/article/64/3/472/1632660
[2] TreeBase dataset M487
    https://treebase.org/treebase-web/search/study/matrices.html?id=965
"""

import argparse
import io
import logging
import math
import sys
from collections import Counter

import pyro
import pyro.poutine as poutine
import setuptools  # noqa F401
import torch
import torch.multiprocessing as mp
from Bio import AlignIO
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoNormal
from pyro.optim import ClippedAdam

from pyrophylo.bethe import BetheModel
from pyrophylo.phylo import Phylogeny

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.DEBUG)


def load_data(args):
    logger.info(f"loading data from {args.nexus_infile}")

    # Truncate file to work around bug in Bio.Nexus reader.
    lines = []
    with open(args.nexus_infile) as f:
        for line in f:
            if line.startswith("BEGIN CODONS"):
                break
            lines.append(line)
    f = io.StringIO("".join(lines))
    alignment = AlignIO.read(f, "nexus")

    num_taxa = min(len(alignment), args.max_taxa)
    num_characters = min(len(alignment[0]), args.max_characters)
    data = torch.zeros((num_taxa, num_characters), dtype=torch.long)
    mask = torch.zeros((num_taxa, num_characters), dtype=torch.bool)
    mapping = {
        "?": (False, 0),  # unobserved
        "A": (True, 0),
        "C": (True, 1),
        "G": (True, 2),
        "T": (True, 3),
        "-": (True, 4),  # insertion/deletion
    }
    for i in range(num_taxa):
        seq = alignment[i].seq
        for j in range(num_characters):
            mask[i, j], data[i, j] = mapping[seq[j]]

    times = torch.zeros(num_taxa)
    return times, data, mask


def train_guide(args, model):
    logger.info("Training via SVI")

    # Configure a guide.
    # Note AutoDelta guides fail due to EM-style mode collapse.
    guide_config = {"init_loc_fn": model.init_loc_fn, "init_scale": 0.01}
    if args.guide_rank == 0:
        guide = AutoNormal(model, **guide_config)
    else:
        guide_config["rank"] = args.guide_rank
        guide = AutoLowRankMultivariateNormal(model, **guide_config)

    # Train the guide via SVI.
    optim = ClippedAdam({"lr": args.learning_rate,
                         "lrd": args.learning_rate_decay ** (1 / args.num_steps)})
    svi = SVI(model, guide, optim, Trace_ELBO())
    t0 = args.init_temperature
    t1 = args.final_temperature
    num_observations = model.leaf_mask.sum()
    losses = []
    for step in range(args.num_steps):
        model.temperature = t0 * (t1 / t0) ** (step / (args.num_steps - 1))
        loss = svi.step() / num_observations
        if step == 0:
            logger.info("guide has {} parameters".format(
                sum(p.numel() for p in guide.parameters())))
        if step % args.log_every == 0:
            logger.info(f"step {step: >4} loss = {loss:0.4g}")
        assert math.isfinite(loss)
        losses.append(loss)

    # Log diagnostics.
    median = guide.median()
    message = ["median latent variables:"]
    for name, value in sorted(median.items()):
        if value.numel() == 1:
            message.append(f"{name} = {value:0.3g}")
        else:
            message.append(f"{name}.shape:{tuple(value.shape)}, "
                           f"lb mean ub: {value.min().item():0.3g} "
                           f"{value.mean().item():0.3g} "
                           f"{value.max().item():0.3g}")
    logger.info("\n".join(message))

    return guide, losses


@torch.no_grad()
def _predict_task(args):
    args, model, guide, i = args
    torch.set_default_dtype(torch.double)
    pyro.set_rng_seed(args.seed + i)
    pyro.enable_validation(__debug__)

    guide_trace = poutine.trace(guide).get_trace()
    with poutine.replay(trace=guide_trace):
        tree, codes = model(sample_tree=True)
    if i % args.log_every == 0:
        sys.stderr.write(".")
        sys.stderr.flush()
    return tree, codes


def predict(args, model, guide):
    logger.info(f"Drawing {args.num_samples} posterior samples")
    map_ = mp.Pool().map if args.parallel else map
    samples = list(map_(_predict_task, [
        (args, model, guide, i)
        for i in range(args.num_samples)
    ]))
    trees = Phylogeny.stack([tree for tree, _ in samples])
    codes = torch.stack([code for _, code in samples])
    return trees, codes


def pretty_tree(t):
    if isinstance(t, frozenset):
        x, y = sorted(map(pretty_tree, t))
        return f"({x} {y})"
    return str(t)


@torch.no_grad()
def evaluate(args, trees):
    # Compute histogram over top k trees.
    # This aims to reproduce Figure 3 Section 6.5 of [1].
    # [1] Vu Dinh, Arman Bilge, Cheng Zhang, Frederick A. Matsen IV
    #     "Probabilistic Path Hamiltonian Monte Carlo"
    #     http://proceedings.mlr.press/v70/dinh17a.html
    counts = Counter(t.hash_topology() for t in trees)
    top_k = counts.most_common(args.top_k)
    logger.info("Estimated posterior distribution for the top {} trees:\n{}"
                .format(args.top_k,
                        ", ".join("{:0.3g}".format(count / args.num_samples)
                                  for key, count in top_k)))
    for i, (tree, count) in enumerate(top_k):
        logger.info(f"Tree {i}:\n{pretty_tree(tree)}")


def main(args):
    mp.set_start_method("spawn")
    torch.set_default_dtype(torch.double if args.double else torch.float)
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)

    # Run the pipeline.
    leaf_times, leaf_data, leaf_mask = load_data(args)
    model = BetheModel(leaf_times, leaf_data, leaf_mask,
                       embedding_dim=args.embedding_dim,
                       bp_iters=args.bp_iters)
    guide, losses = train_guide(args, model)
    trees, codes = predict(args, model, guide)
    evaluate(args, trees)

    # Save results for bethe_vi.ipynb.
    if args.outfile:
        logger.info(f"saving to {args.outfile}")
        results = {
            "data": {
                "leaf_times": leaf_times,
                "leaf_data": leaf_data,
                "leaf_mask": leaf_mask,
            },
            "model": model,
            "guide": guide,
            "losses": losses,
            "samples": {
                "trees": trees,
                "codes": codes,
            },
        }
        torch.save(results, args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tree learning experiment")
    parser.add_argument("--nexus-infile", default="data/treebase/M487.nex")
    parser.add_argument("--outfile", default="results/bethe_vi.pt")
    parser.add_argument("--max-taxa", default=int(1e6), type=int)
    parser.add_argument("--max-characters", default=int(1e6), type=int)
    parser.add_argument("-e", "--embedding-dim", default=10, type=int)
    parser.add_argument("-r", "--guide-rank", default=0, type=int)
    parser.add_argument("-t0", "--init-temperature", default=1.0, type=float)
    parser.add_argument("-t1", "--final-temperature", default=0.01, type=float)
    parser.add_argument("-bp", "--bp-iters", default=30, type=int)
    parser.add_argument("-n", "--num-steps", default=501, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.2, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("-s", "--num-samples", default=1000, type=int)
    parser.add_argument("--double", default=True, action="store_true")
    parser.add_argument("--single", action="store_false", dest="double")
    parser.add_argument("--parallel", default=True, action="store_true")
    parser.add_argument("--sequential", action="store_false", dest="parallel")
    parser.add_argument("--top-k", default=10, type=int)
    parser.add_argument("--seed", default=20201103, type=int)
    parser.add_argument("-l", "--log-every", default=1, type=int)
    args = parser.parse_args()

    # Disable multiprocessing when running under pdb.
    main_module = sys.modules["__main__"]
    if not hasattr(main_module, "__spec__"):
        args.parallel = False

    main(args)
