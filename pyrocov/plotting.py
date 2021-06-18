import torch


@torch.no_grad()
def force_apart(*X, radius=0.01, iters=10, xshift=0.01):
    X = torch.stack([torch.as_tensor(x) for x in X], dim=-1)
    assert len(X.shape) == 2
    scale = X.max(0).values - X.min(0).values
    X /= scale
    for _ in range(iters):
        XX = X - X[:, None]
        r = (XX / radius).square().sum(-1, True).sqrt()
        F = (XX * r.neg().exp()).sum(0)
        f = F.square().sum(-1, True).sqrt().clamp(min=1e-20)
        F /= f
        F *= 1 - f.neg().exp()
        if xshift is not None:
            F[:, 0].clamp_(min=0)
        X += F
    if xshift is not None:
        X[:, 0] += xshift
    X *= scale
    return X.unbind(-1)
