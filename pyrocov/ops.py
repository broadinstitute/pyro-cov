# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import torch


def logistic_logsumexp(alpha, beta, delta, tau, *, backend="sequential"):
    """
    Computes::

        (alpha + beta * (delta + tau[:, None])).logsumexp(-1)

    where::

        alpha.shape == [P, S]
        beta.shape == [P, S]
        delta.shape == [P, S]
        tau.shape == [T, P]

    :param str backend: One of "naive", "sequential".
    """
    assert alpha.dim() == 2
    assert alpha.shape == beta.shape == delta.shape
    assert tau.dim() == 2
    assert tau.size(1) == alpha.size(0)
    assert not tau.requires_grad

    if backend == "naive":
        return (alpha + beta * (delta + tau[:, :, None])).logsumexp(-1)
    if backend == "sequential":
        return LogisticLogsumexp.apply(alpha, beta, delta, tau)
    raise ValueError(f"Unknown backend: {repr(backend)}")


class LogisticLogsumexp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, beta, delta, tau):
        P, S = alpha.shape
        T, P = tau.shape
        output = alpha.new_zeros(T, P)
        for t in range(len(tau)):
            logits = (delta + tau[t, :, None]).mul_(beta).add_(alpha)  # [P, S]
            output[t] = logits.logsumexp(-1)  # [P]

        ctx.save_for_backward(alpha, beta, delta, tau, output)
        return output  # [T, P]

    @staticmethod
    def backward(ctx, grad_output):
        alpha, beta, delta, tau, output = ctx.saved_tensors

        grad_alpha = torch.zeros_like(alpha)  # [P, S]
        grad_beta = torch.zeros_like(beta)  # [P, S]
        for t in range(len(tau)):
            delta_tau = delta + tau[t, :, None]  # [P, S]
            logits = (delta_tau * beta).add_(alpha)  # [ P, S]
            softmax_logits = logits.sub_(output[t, :, None]).exp_()  # [P, S]
            grad_logits = softmax_logits * grad_output[t, :, None]  # [P, S]
            grad_alpha += grad_logits  # [P, S]
            grad_beta += delta_tau * grad_logits  # [P, S]

        grad_delta = beta * grad_alpha  # [P, S]
        return grad_alpha, grad_beta, grad_delta, None


def sparse_poisson_likelihood(full_log_rate, nonzero_log_rate, nonzero_value):
    """
    The following are equivalent::

        # Version 1. dense
        log_prob = Poisson(log_rate.exp()).log_prob(value).sum()

        # Version 2. sparse
        nnz = value.nonzero(as_tuple=True)
        log_prob = sparse_poisson_likelihood(
            log_rate.logsumexp(-1),
            log_rate[nnz],
            value[nnz],
        )
    """
    # Let p = Poisson(log_rate.exp()). Then
    # p.log_prob(value)
    #   = log_rate * value - log_rate.exp() - (value + 1).lgamma()
    # p.log_prob(0) = -log_rate.exp()
    # p.log_prob(value) - p.log_prob(0)
    #   = log_rate * value - log_rate.exp() - (value + 1).lgamma() + log_rate.exp()
    #   = log_rate * value - (value + 1).lgamma()
    return (
        torch.dot(nonzero_log_rate, nonzero_value)
        - (nonzero_value + 1).lgamma().sum()
        - full_log_rate.exp().sum()
    )
