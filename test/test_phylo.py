import pytest

from pyrophylo.phylo import Phylogeny


@pytest.mark.parametrize("num_leaves", range(1, 50))
def test_generate(num_leaves):
    Phylogeny.generate(num_leaves)


@pytest.mark.parametrize("num_leaves", range(1, 10))
@pytest.mark.parametrize("num_samples", range(1, 5))
def test_generate_batch(num_leaves, num_samples):
    Phylogeny.generate(num_leaves, num_samples=num_samples)
