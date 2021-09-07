# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from scipy.special import log_ndtr


def hpd_interval(p: float, samples: torch.Tensor):
    assert 0.5 < p < 1
    assert samples.shape
    pad = int(round((1 - p) * len(samples)))
    assert pad > 0, "too few samples"
    width = samples[-pad:] - samples[:pad]
    lb = width.max(0).indices
    ub = len(samples) - lb - 1
    i = torch.stack([lb, ub])
    return samples.gather(0, i)


def confidence_interval(p: float, samples: torch.Tensor):
    assert 0.5 < p < 1
    assert samples.shape
    pad = (1 - p) / 2
    lk = int(round(pad * (len(samples) - 1)))
    uk = int(round((1 - pad) * (len(samples) - 1)))
    assert pad > 0, "too few samples"
    lb = samples.kthvalue(lk, 0).values
    ub = samples.kthvalue(uk, 0).values
    return torch.stack([lb, ub])


def normal_log10bf(mean, std=1.0):
    """
    Returns ``log10(P[x>0] / P[x<0])`` for ``x ~ N(mean, std)``.
    """
    z = mean / std
    return (log_ndtr(z) - log_ndtr(-z)) / np.log(10)
