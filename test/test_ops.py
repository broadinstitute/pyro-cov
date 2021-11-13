# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import pyro.distributions as dist
import pytest
import torch
from torch.autograd import grad

from pyrocov.ops import (
    logistic_logsumexp,
    sparse_multinomial_likelihood,
    sparse_poisson_likelihood,
)


@pytest.mark.parametrize("T,P,S", [(5, 6, 7)])
@pytest.mark.parametrize("backend", ["sequential"])
def test_logistic_logsumexp(T, P, S, backend):
    alpha = torch.randn(P, S, requires_grad=True)
    beta = torch.randn(P, S, requires_grad=True)
    delta = torch.randn(P, S, requires_grad=True)
    tau = torch.randn(T, P)

    expected = logistic_logsumexp(alpha, beta, delta, tau, backend="naive")
    actual = logistic_logsumexp(alpha, beta, delta, tau, backend=backend)
    assert torch.allclose(actual, expected)

    probe = torch.randn(expected.shape)
    expected_grads = grad((probe * expected).sum(), [alpha, beta, delta])
    actual_grads = grad((probe * actual).sum(), [alpha, beta, delta])
    for e, a, name in zip(expected_grads, actual_grads, ["alpha", "beta", "delta"]):
        assert torch.allclose(a, e), name


@pytest.mark.parametrize("T,P,S", [(2, 3, 4), (5, 6, 7), (8, 9, 10)])
def test_sparse_poisson_likelihood(T, P, S):
    log_rate = torch.randn(T, P, S)
    d = dist.Poisson(log_rate.exp())
    value = d.sample()
    assert 0.1 < (value == 0).float().mean() < 0.9, "weak test"
    expected = d.log_prob(value).sum()

    full_log_rate = log_rate.logsumexp(-1)
    nnz = value.nonzero(as_tuple=True)
    nonzero_value = value[nnz]
    nonzero_log_rate = log_rate[nnz]
    actual = sparse_poisson_likelihood(full_log_rate, nonzero_log_rate, nonzero_value)
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("T,P,S", [(2, 3, 4), (5, 6, 7), (8, 9, 10)])
def test_sparse_multinomial_likelihood(T, P, S):
    logits = torch.randn(T, P, S)
    value = dist.Poisson(logits.exp()).sample()

    d = dist.Multinomial(logits=logits, validate_args=False)
    assert 0.1 < (value == 0).float().mean() < 0.9, "weak test"
    expected = d.log_prob(value).sum()

    logits = logits.log_softmax(-1)
    total_count = value.sum(-1)
    nnz = value.nonzero(as_tuple=True)
    nonzero_value = value[nnz]
    nonzero_logits = logits[nnz]
    actual = sparse_multinomial_likelihood(total_count, nonzero_logits, nonzero_value)
    assert torch.allclose(actual, expected)
