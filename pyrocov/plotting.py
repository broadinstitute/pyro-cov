# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import torch


@torch.no_grad()
def force_apart(*X, radius=[0.05, 0.005], iters=10, stepsize=2, xshift=0.01):
    X = torch.stack([torch.as_tensor(x) for x in X], dim=-1)
    assert len(X.shape) == 2
    radius = torch.as_tensor(radius)
    scale = X.max(0).values - X.min(0).values
    X /= scale
    for _ in range(iters):
        XX = X - X[:, None]
        r = (XX / radius).square().sum(-1, True)
        kernel = r.neg().exp()
        F = (XX * radius.square().sum() / radius ** 2 * kernel).sum(0)
        F_norm = F.square().sum(-1, True).sqrt().clamp(min=1e-20)
        F *= F_norm.neg().expm1().neg() / F_norm
        F *= stepsize
        if xshift is not None:
            F[:, 0].clamp_(min=0)
        X += F / iters
    if xshift is not None:
        X[:, 0] += xshift
    X *= scale
    return X.unbind(-1)
