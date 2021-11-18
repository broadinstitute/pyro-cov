# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import pickle
from collections import defaultdict

from pyrocov.usher import prune_mutation_tree

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def main(args):
    logger.info(f"Loading {args.columns_file_in}")
    with open(args.columns_file_in, "rb") as f:
        columns = pickle.load(f)

    # Aggregate ambiguous clade counts.
    weights = defaultdict(float)
    for clades in columns["clades"]:
        clades = clades.split(",")
        weight = 1 / len(clades)
        for clade in clades:
            weights[clade] += weight

    # Compress the tree, minimizing distortion.
    logger.info(f"Loading {args.tree_file_in}")
    old_to_new = prune_mutation_tree(
        args.tree_file_in, args.tree_file_out, args.max_num_nodes, weights
    )
    logger.info(f"Saved {args.tree_file_out}")

    # Update columns.
    columns["clade"] = [old_to_new.get(c, c) for c in columns["clades"]]
    columns["clades"] = [
        ",".join(old_to_new.get(c, c) for c in cs.split(","))
        for cs in columns["clades"]
    ]

    with open(args.columns_file_out, "wb") as f:
        pickle.dump(columns, f)
    logger.info(f"Saved {args.columns_file_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress mutation tree")
    parser.add_argument("--tree-file-in", default="results/aligndb/lineageTree.fine.pb")
    parser.add_argument("--columns-file-in", default="results/usher.columns.pkl")
    parser.add_argument("--max-num-nodes", default=10000, type=int)
    args = parser.parse_args()
    args.tree_file_out = f"results/lineageTree.{args.max_num_nodes}.pb"
    args.columns_file_out = f"results/columns.{args.max_num_nodes}.pkl"
    main(args)
