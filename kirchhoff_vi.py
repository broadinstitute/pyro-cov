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

import pyro
import pyro.poutine as poutine
import setuptools  # noqa F401
import torch
from Bio import AlignIO
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoNormal
from pyro.optim import Adam

from pyrophylo.kirchhoff import KirchhoffModel

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

    guide_config = {"init_loc_fn": model.init_loc_fn, "init_scale": 0.01}
    if args.guide_rank == 0:
        guide = AutoNormal(model, **guide_config)
    else:
        guide_config["rank"] = args.guide_rank
        guide = AutoLowRankMultivariateNormal(model, **guide_config)

    optim = Adam({"lr": args.learning_rate})
    svi = SVI(model, guide, optim, Trace_ELBO())
    num_observations = model.leaf_mask.sum()
    losses = []
    for step in range(args.num_steps):
        loss = svi.step() / num_observations
        if step % args.log_every == 0:
            logger.info(f"step {step: >4} loss = {loss:0.4g}")
        assert math.isfinite(loss)
        losses.append(loss)


@torch.no_grad()
def predict(args, model, guide):
    logger.info(f"Drawing {args.num_samples} posterior samples")
    num_edges = model.num_nodes - 1
    trees = torch.empty((args.num_samples, num_edges, 2), dtype=torch.long)
    for i in range(args.num_samples):
        guide_trace = poutine.trace(guide).get_trace()
        with poutine.replay(trace=guide_trace):
            tree = model(sample_tree=True)
            trees[i] = tree
    return trees


@torch.no_grad()
def evaluate(args, trees):
    pass  # TODO


def main(args):
    torch.set_default_dtype(torch.double)
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)

    leaf_times, leaf_data, leaf_mask = load_data(args)
    model = KirchhoffModel(leaf_times, leaf_data, leaf_mask,
                           temperature=args.temperature)
    guide = train_guide(args, model)
    trees = predict(args, model, guide)
    evaluate(args, trees)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tree learning experiment")
    parser.add_argument("--nexus-infile", default="data/treebase/M487.nex")
    parser.add_argument("--max-taxa", default=int(1e6), type=int)
    parser.add_argument("--max-characters", default=int(1e6), type=int)
    parser.add_argument("-t", "--temperature", default=1.0, type=float)
    parser.add_argument("--guide-rank", default=0, type=int)
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("-s", "--num-samples", default=1000, type=int)
    parser.add_argument("--seed", default=20201103, type=int)
    parser.add_argument("--log-every", default=100, type=int)
    args = parser.parse_args()
    main(args)
