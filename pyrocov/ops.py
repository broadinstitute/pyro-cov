# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import weakref
from typing import Dict

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


_log_factorial_cache: Dict[int, torch.Tensor] = {}


def log_factorial_sum(x: torch.Tensor) -> torch.Tensor:
    if x.requires_grad:
        return (x + 1).lgamma().sum()
    key = id(x)
    if key not in _log_factorial_cache:
        weakref.finalize(x, _log_factorial_cache.pop, key, None)
        _log_factorial_cache[key] = (x + 1).lgamma().sum()
    return _log_factorial_cache[key]


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
        - log_factorial_sum(nonzero_value)
        - full_log_rate.exp().sum()
    )


def sparse_multinomial_likelihood(total_count, nonzero_logits, nonzero_value):
    """
    The following are equivalent::

        # Version 1. dense
        log_prob = Multinomial(logits=logits).log_prob(value).sum()

        # Version 2. sparse
        nnz = value.nonzero(as_tuple=True)
        log_prob = sparse_multinomial_likelihood(
            value.sum(-1),
            (logits - logits.logsumexp(-1))[nnz],
            value[nnz],
        )
    """
    return (
        log_factorial_sum(total_count)
        - log_factorial_sum(nonzero_value)
        + torch.dot(nonzero_logits, nonzero_value)
    )


def sparse_categorical_kl(log_q, p_support, log_p):
    """
    Computes the restricted Kl divergence::

        sum_i restrict(q)(i) (log q(i) - log p(i))

    where ``p`` is a uniform prior, ``q`` is the posterior, and
    ``restrict(q))`` is the posterior restricted to the support of ``p`` and
    renormalized. Note for degenerate ``p=delta(i)`` this reduces to the log
    likelihood ``log q(i)``.
    """
    assert log_q.dim() == 1
    assert log_p.dim() == 1
    assert p_support.shape == log_p.shape + log_q.shape
    q = log_q.exp()
    sum_q = torch.mv(p_support, q)
    sum_q_log_q = torch.mv(p_support, q * log_q)
    sum_r_log_q = sum_q_log_q / sum_q  # restrict and normalize
    kl = sum_r_log_q - log_p  # note sum_r_log_p = log_p because p is uniform
    return kl.sum()
