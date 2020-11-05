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

import pyro
import torch
from Bio import AlignIO
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.optim import Adam

from pyrophylo.kirchhoff import KirchhoffModel

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.DEBUG)


def load_data(args):
    # Truncate file to work around bug in Bio.Nexus reader.
    lines = []
    with open(args.nexus_infile) as f:
        for line in f:
            if line.startswith("BEGIN CODONS"):
                break
            lines.append(line)
    f = io.StringIO("".join(lines))
    alignment = AlignIO.read(f, "nexus")

    num_taxa = len(alignment)
    num_characters = len(alignment[0])
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
    for i, seq in enumerate(alignment):
        for j, value in enumerate(seq.seq):
            mask[i, j], data[i, j] = mapping[value]

    times = torch.zeros(num_taxa)
    return times, data, mask


def main(args):
    torch.set_default_dtype(torch.double)
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)

    leaf_times, leaf_data, leaf_mask = load_data(args)
    num_leaves = len(leaf_times)

    model = KirchhoffModel(leaf_times, leaf_data, leaf_mask)
    guide = AutoLowRankMultivariateNormal(model)
    optim = Adam({"lr": args.learning_rate})
    svi = SVI(model, guide, optim, Trace_ELBO())
    losses = []
    for step in range(args.num_steps):
        loss = svi.step() / num_leaves
        if step % 100 == 0:
            logging.info(f"step {step: >4} loss = {loss:0.4g}")
        losses.append(loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tree learning experiment")
    parser.add_argument("--nexus-infile", default="data/treebase/M487.nex")
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("--seed", default=20201103, type=int)
    args = parser.parse_args()
    main(args)
