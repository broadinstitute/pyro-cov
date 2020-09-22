import os

import pytest
from Bio import Phylo

from pyrophylo.io import read_nexus_trees, stack_nexus_trees

ROOT = os.path.dirname(os.path.dirname(__file__))
FILENAME = os.path.join(ROOT, "data", "GTR4G_posterior.trees")


@pytest.mark.xfail(reason="Python <3.8 cannot .read() large files")
def test_bio_phylo_parse():
    trees = Phylo.parse(FILENAME, format="nexus")
    for tree in trees:
        print(tree.count_terminals())


@pytest.mark.parametrize("processes", [0, 2])
def test_read_nexus_trees(processes):
    trees = read_nexus_trees(FILENAME, max_num_trees=5, processes=processes)
    trees = list(trees)
    assert len(trees) == 5
    for tree in trees:
        assert tree.count_terminals() == 772


@pytest.mark.parametrize("processes", [0, 2])
def test_stack_nexus_trees(processes):
    phylo = stack_nexus_trees(FILENAME, max_num_trees=5, processes=processes)
    assert phylo.batch_shape == (5,)
