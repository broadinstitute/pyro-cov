import argparse
import os

import torch

from pyrophylo.io import stack_nexus_trees

if not os.path.exists("results"):
    os.makedirs("results")


def main(args):
    phylo = stack_nexus_trees(
        args.infile, max_num_trees=args.max_num_trees, processes=args.processes
    )
    torch.save(phylo, args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max_num_trees", default=1000000, type=int)
    parser.add_argument("-i", "--infile", default="data/GTR4G_posterior.trees")
    parser.add_argument("-o", "--outfile", default="results/GTR4G_posterior.pt")
    parser.add_argument("-p", "--processes", type=int)
    args = parser.parse_args()
    main(args)
