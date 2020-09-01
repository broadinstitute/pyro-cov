import os

import pytest
from Bio import Phylo

from pyrophylo.io import parse_nexus_trees

ROOT = os.path.dirname(os.path.dirname(__file__))
FILENAME = os.path.join(ROOT, "data", "GTR4G_posterior.trees")


@pytest.mark.xfail(reason="Python <3.8 cannot .read() large files")
def test_bio_phylo_parse(args):
    trees = Phylo.parse(FILENAME, format="nexus")
    for tree in trees:
        print(tree.count_terminals())


def test_parse_nexus_trees():
    for i, tree in enumerate(parse_nexus_trees(FILENAME)):
        assert tree.count_terminals() == 772
        if i == 5:
            break
