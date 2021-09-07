# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyrocov.phylo import Phylogeny


@pytest.mark.parametrize("num_leaves", range(1, 50))
def test_smoke(num_leaves):
    phylo = Phylogeny.generate(num_leaves)
    phylo.num_lineages()
    phylo.hash_topology()
    phylo.time_mrca()


@pytest.mark.parametrize("num_leaves", range(1, 10))
@pytest.mark.parametrize("num_samples", range(1, 5))
def test_smoke_batch(num_leaves, num_samples):
    phylo = Phylogeny.generate(num_leaves, num_samples=num_samples)
    phylo.num_lineages()
    phylo.hash_topology()
    phylo.time_mrca()


def test_time_mrca():
    #     0       0
    #    1 \      1
    #   /|  2     2
    #  3 4  |\    3
    #       5 \   4
    #      / \ 6  5
    #     7   8   6
    times = torch.tensor([0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0])
    parents = torch.tensor([-1, 0, 0, 1, 1, 2, 2, 5, 5])
    leaves = torch.tensor([3, 4, 6, 7, 8])
    phylo = Phylogeny(times, parents, leaves)

    actual = phylo.time_mrca()
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
            [0.0, 1.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 2.0, 4.0, 4.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 5.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 2.0, 6.0, 4.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 2.0, 4.0, 6.0],
        ]
    )
    assert (actual == expected).all()

    actual = phylo.leaf_time_mrca()
    expected = torch.tensor(
        [
            [3.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 6.0, 4.0],
            [0.0, 0.0, 2.0, 4.0, 6.0],
        ]
    )
    assert (actual == expected).all()
