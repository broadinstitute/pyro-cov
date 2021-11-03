# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

from pyrocov.fasta import PANGOLEARN_DATA
from pyrocov.usher import load_mutation_tree


def main(args):
    mutations_by_lineage = load_mutation_tree(args.tree_file_in)
    assert mutations_by_lineage
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess pangolin mutations")
    parser.add_argument(
        "--tree-file-in", default=os.path.join(PANGOLEARN_DATA, "lineageTree.pb")
    )
    args = parser.parse_args()
    main(args)
