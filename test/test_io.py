# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import glob
import os

import pytest
import torch
from Bio import Phylo

from pyrocov.io import read_alignment, read_nexus_trees, stack_nexus_trees

ROOT = os.path.dirname(os.path.dirname(__file__))
FILENAME = os.path.join(ROOT, "data", "GTR4G_posterior.trees")


@pytest.mark.skipif(not os.path.exists(FILENAME), reason="file unavailable")
@pytest.mark.xfail(reason="Python <3.8 cannot .read() large files", run=False)
def test_bio_phylo_parse():
    trees = Phylo.parse(FILENAME, format="nexus")
    for tree in trees:
        print(tree.count_terminals())


@pytest.mark.skipif(not os.path.exists(FILENAME), reason="file unavailable")
@pytest.mark.parametrize("processes", [0, 2])
def test_read_nexus_trees(processes):
    trees = read_nexus_trees(FILENAME, max_num_trees=5, processes=processes)
    trees = list(trees)
    assert len(trees) == 5
    for tree in trees:
        assert tree.count_terminals() == 772


@pytest.mark.skipif(not os.path.exists(FILENAME), reason="file unavailable")
@pytest.mark.parametrize("processes", [0, 2])
def test_stack_nexus_trees(processes):
    phylo = stack_nexus_trees(FILENAME, max_num_trees=5, processes=processes)
    assert phylo.batch_shape == (5,)


@pytest.mark.parametrize("filename", glob.glob("data/treebase/DS*.nex"))
def test_read_alignment(filename):
    probs = read_alignment(filename)
    assert probs.dim() == 3
    assert torch.isfinite(probs).all()
