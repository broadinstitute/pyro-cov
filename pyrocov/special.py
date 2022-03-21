# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

# Adapted from @viswackftw
# https://github.com/pytorch/pytorch/issues/52973#issuecomment-787587188

import math

import torch


def ndtr(value: torch.Tensor):
    """
    Based on the SciPy implementation of ndtr
    """
    sqrt_half = torch.sqrt(torch.tensor(0.5, dtype=value.dtype))
    x = value * sqrt_half
    z = abs(x)
    y = 0.5 * torch.erfc(z)
    output = torch.where(
        z < sqrt_half, 0.5 + 0.5 * torch.erf(x), torch.where(x > 0, 1 - y, y)
    )
    return output


def log_ndtr(value: torch.Tensor):
    """
    Function to compute the log of the normal CDF at value.
    This is based on the TFP implementation.
    """
    dtype = value.dtype
    if dtype == torch.float64:
        lower, upper = -20, 8
    elif dtype == torch.float32:
        lower, upper = -10, 5
    else:
        raise TypeError("value needs to be either float32 or float64")

    # When x < lower, then we perform a fixed series expansion (asymptotic)
    # = log(cdf(x)) = log(1 - cdf(-x)) = log(1 / 2 * erfc(-x / sqrt(2)))
    # = log(-1 / sqrt(2 * pi) * exp(-x ** 2 / 2) / x * (1 + sum))
    # When x >= lower and x <= upper, then we simply perform log(cdf(x))
    # When x > upper, then we use the approximation log(cdf(x)) = log(1 - cdf(-x)) \approx -cdf(-x)
    return torch.where(
        value > upper,
        torch.log1p(-ndtr(-value)),
        torch.where(value >= lower, torch.log(ndtr(value)), log_ndtr_series(value)),
    )


def log_ndtr_series(value: torch.Tensor, num_terms=3):
    """
    Function to compute the asymptotic series expansion of the log of normal CDF
    at value.
    This is based on the TFP implementation.
    """
    # sum = sum_{n=1}^{num_terms} (-1)^{n} (2n - 1)!! / x^{2n}))
    value_sq = value ** 2
    t1 = -0.5 * (math.log(2 * math.pi) + value_sq) - torch.log(-value)
    t2 = torch.zeros_like(value)
    value_even_power = value_sq.clone()
    double_fac = 1
    multiplier = -1
    for n in range(1, num_terms + 1):
        t2.add_(multiplier * double_fac / value_even_power)
        value_even_power.mul_(value_sq)
        double_fac *= 2 * n - 1
        multiplier *= -1
    return t1 + torch.log1p(t2)
