# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.autograd import grad

from pyrocov.ops import logistic_logsumexp


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
