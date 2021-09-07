# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import pyro.distributions as dist
import pytest
import torch

from pyrocov.markov_tree import MarkovTree, _interpolate_lmve, _mpm
from pyrocov.phylo import Phylogeny


def grad(output, inputs, **kwargs):
    if not output.requires_grad:
        return list(map(torch.zeros_like, inputs))
    return torch.autograd.grad(output, inputs, **kwargs)


@pytest.mark.parametrize("size", range(2, 10))
def test_mpm(size):
    matrix = torch.randn(size, size).exp()
    matrix /= matrix.sum(dim=-1, keepdim=True)  # Make stochastic.
    matrix = (matrix + 4 * torch.eye(size)) / 5  # Make diagonally dominant.
    vector = torch.randn(size)

    for t in range(0, 10):
        expected = vector @ matrix.matrix_power(t)
        actual = _mpm(matrix, torch.tensor(float(t)), vector)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("size", [2, 3, 4, 5])
@pytest.mark.parametrize("duration", [1, 2, 3, 4, 5])
def test_interpolate_lmve_smoke(size, duration):
    matrix = torch.randn(duration, size, size).exp()
    log_vector = torch.randn(size)
    t0 = -0.6
    while t0 < duration + 0.9:
        t1 = t0 + 0.2
        while t1 < duration + 0.9:
            actual = _interpolate_lmve(
                torch.tensor(t0), torch.tensor(t1), matrix, log_vector
            )
            assert actual.shape == log_vector.shape
            t1 += 1
        t0 += 1


@pytest.mark.parametrize("num_states", [3, 7])
@pytest.mark.parametrize("num_leaves", [4, 16, 17])
@pytest.mark.parametrize("duration", [1, 5])
@pytest.mark.parametrize("num_samples", [1, 2, 3])
def test_markov_tree_log_prob(num_samples, duration, num_leaves, num_states):
    phylo = Phylogeny.generate(num_leaves, num_samples=num_samples)
    phylo.times.mul_(duration * 0.25).add_(0.75 * duration)
    phylo.times.round_()  # Required for naive-vs-likelihood agreement.

    leaf_state = dist.Categorical(torch.ones(num_states)).sample([num_leaves])

    state_trans = torch.randn(duration, num_states, num_states).mul(0.1).exp()
    state_trans /= state_trans.sum(dim=-1, keepdim=True)
    state_trans += 4 * torch.eye(num_states)
    state_trans /= state_trans.sum(dim=-1, keepdim=True)
    state_trans.requires_grad_()

    dist1 = MarkovTree(phylo, state_trans, method="naive")
    dist2 = MarkovTree(phylo, state_trans, method="likelihood")

    logp1 = dist1.log_prob(leaf_state)
    logp2 = dist2.log_prob(leaf_state)
    assert torch.allclose(logp1, logp2)

    grad1 = grad(logp1.logsumexp(0), [state_trans], allow_unused=True)[0]
    grad2 = grad(logp2.logsumexp(0), [state_trans], allow_unused=True)[0]
    grad1 = grad1 - grad1.mean(dim=-1, keepdim=True)
    grad2 = grad2 - grad2.mean(dim=-1, keepdim=True)
    assert torch.allclose(grad1, grad2, rtol=1e-4, atol=1e-4)
