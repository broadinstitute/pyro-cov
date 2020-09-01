import os

import pytest
from Bio import Phylo

from pyrophylo.io import iter_nexus_trees, stack_nexus_trees

ROOT = os.path.dirname(os.path.dirname(__file__))
FILENAME = os.path.join(ROOT, "data", "GTR4G_posterior.trees")


@pytest.mark.xfail(reason="Python <3.8 cannot .read() large files")
def test_bio_phylo_parse(args):
    trees = Phylo.parse(FILENAME, format="nexus")
    for tree in trees:
        print(tree.count_terminals())


def test_iter_nexus_trees():
    for i, tree in enumerate(iter_nexus_trees(FILENAME)):
        assert tree.count_terminals() == 772
        if i == 5:
            break


def test_stack_nexus_trees():
    phylo = stack_nexus_trees(FILENAME, max_num_trees=5)
    assert phylo.batch_shape == (5,)


@pytest.mark.xfail(reason="unknown")
def test_iter_nexus_trees_async():
    trees = []
    for i, tree in enumerate(iter_nexus_trees(FILENAME, processes=2)):
        trees.append(tree)
        if i == 5:
            break
    for async_result in trees:
        tree = async_result.get(timeout=0.5)
        assert tree.count_terminals() == 772


@pytest.mark.xfail(reason="unknown")
def test_stack_nexus_trees_async():
    phylo = stack_nexus_trees(FILENAME, max_num_trees=5, processes=2, timeout=None)
    assert phylo.batch_shape == (5,)
