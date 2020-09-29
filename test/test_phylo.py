import pyro.distributions as dist
import pytest
import torch

from pyrophylo.phylo import MarkovTree, Phylogeny, _interpolate_lmve, _mpm


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("size", [5])
def test_mpm(shape, size):
    matrix = torch.randn(shape + (size, size)).exp()
    matrix /= matrix.logsumexp(dim=-2, keepdim=True)  # Make stochastic.
    matrix = (matrix + 4 * torch.eye(size)) / 5  # Make diagonally dominant.
    vector = torch.randn(shape + (size, 1))

    for t in range(0, 10):
        expected = matrix.matrix_power(t) @ vector
        actual = _mpm(matrix, torch.tensor(float(t)), vector)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("size", [5])
@pytest.mark.parametrize("duration", [1, 2, 3, 4, 5])
def test_interpolate_lmve_smoke(shape, size, duration):
    matrix = torch.randn(shape + (duration, size, size)).exp()
    log_vector = torch.randn(shape + (size,))
    t0 = -0.6
    while t0 < duration + 0.9:
        t1 = t0 + 0.2
        while t1 < duration + 0.9:
            actual = _interpolate_lmve(torch.tensor(t0), torch.tensor(t1),
                                       matrix, log_vector)
            assert actual.shape == log_vector.shape
            t1 += 1
        t0 += 1


@pytest.mark.parametrize("num_leaves", range(1, 50))
def test_generate(num_leaves):
    Phylogeny.generate(num_leaves)


@pytest.mark.parametrize("num_leaves", range(1, 10))
@pytest.mark.parametrize("num_samples", range(1, 5))
def test_generate_batch(num_leaves, num_samples):
    Phylogeny.generate(num_leaves, num_samples=num_samples)


@pytest.mark.xfail(reason="disagreement")
@pytest.mark.parametrize("num_leaves", [2, 12, 22])
@pytest.mark.parametrize("num_states", [3, 7])
@pytest.mark.parametrize("duration", [1, 5])
def test_markov_tree_log_prob(duration, num_leaves, num_states):
    phylo = Phylogeny.generate(num_leaves, num_samples=4)
    phylo.times.mul_(duration * 0.25).add_(0.75 * duration)
    phylo.times = phylo.times.round()  # Required for agreement.

    leaf_state = dist.Categorical(torch.ones(num_states)).sample([num_leaves])

    state_trans = torch.randn(duration, num_states, num_states).mul(0.1).exp()
    state_trans += torch.eye(num_states)
    state_trans /= state_trans.sum(dim=-2, keepdim=True)
    state_trans.requires_grad_()

    dist1 = MarkovTree(phylo, state_trans, method="naive")
    dist2 = MarkovTree(phylo, state_trans, method="likelihood")

    logp1 = dist1.log_prob(leaf_state)
    logp2 = dist2.log_prob(leaf_state)
    assert torch.allclose(logp1, logp2)

    # FIXME method="naive" does not support gradients.
    # grad1 = torch.autograd.grad(logp1.sum(), [state_trans])[0]
    torch.autograd.grad(logp2.sum(), [state_trans])[0]
    # assert torch.allclose(grad1, grad2)
