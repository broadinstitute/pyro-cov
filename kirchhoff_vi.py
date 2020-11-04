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

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.DEBUG)


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
    raise NotImplementedError("TODO")

    return data, mask


def main(args):
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.seed)

    leaf_times, leaf_states = load_data(args)

    model = KirchhoffModel(leaf_times, leaf_states)
    logging.info("Fitting embedding")
    model.fit_embedding()

    logging.info("Fitting posterior")
    guide = AutoLowRankMultivariateNormal(model)
    optim = Adam({"lr": args.learning_rate})
    svi = SVI(model, guide, optim, Trace_ELBO())
    for step in range(args.num_steps):
        loss = svi.step()
        if step % 100 == 0:
            logging.info(f"step {step: >4} loss = {loss:0.4g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tree learning experiment")
    parser.add_argument("--nexus-infile", default="data/treebase/M487.nex")
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("--seed", default=20201103, type=int)
    args = parser.parse_args()
    main(args)
