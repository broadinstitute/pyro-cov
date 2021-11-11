# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import torch


def logsumexp_logistic(alpha, beta, delta, tau, *, backend="naive"):
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
