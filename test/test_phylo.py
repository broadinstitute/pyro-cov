import pytest
import torch

from pyrophylo.phylo import Phylogeny


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
    times = torch.tensor([0., 1., 2., 3., 3., 4., 5., 6., 6.])
    parents = torch.tensor([-1, 0, 0, 1, 1, 2, 2, 5, 5])
    leaves = torch.tensor([3, 4, 6, 7, 8])
    phylo = Phylogeny(times, parents, leaves)
    actual = phylo.time_mrca()
    expected = torch.tensor([
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 1., 1., 0., 0., 0., 0.],
        [0., 0., 2., 0., 0., 2., 2., 2., 2.],
        [0., 1., 0., 3., 1., 0., 0., 0., 0.],
        [0., 1., 0., 1., 3., 0., 0., 0., 0.],
        [0., 0., 2., 0., 0., 4., 2., 4., 4.],
        [0., 0., 2., 0., 0., 2., 5., 2., 2.],
        [0., 0., 2., 0., 0., 4., 2., 6., 4.],
        [0., 0., 2., 0., 0., 4., 2., 4., 6.],
    ])
    assert (actual == expected).all()
