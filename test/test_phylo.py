import pytest
import torch

from pyrophylo.phylo import _interpolate_lmve, _mpm


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("size", [5])
def test_mpm(shape, size):
    matrix = torch.randn(shape + (size, size)).exp()
    matrix /= matrix.logsumexp(dim=-2, keepdim=True)  # Make stochastic.
    matrix = (matrix + 4 * torch.eye(size)) / 5  # Make diagonally dominant.
    vector = torch.randn(shape + (size, 1))

    for t in range(0, 10):
        expected = matrix.matrix_power(t) @ vector
        actual = _mpm(matrix, torch.tensor(float(t)), vector)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("size", [5])
@pytest.mark.parametrize("duration", [1, 2, 3, 4, 5])
def test_interpolate_lmve_smoke(shape, size, duration):
    matrix = torch.randn(shape + (duration, size, size)).exp()
    log_vector = torch.randn(shape + (size,))
    t0 = -0.6
    while t0 < duration + 0.9:
        t1 = t0 + 0.2
        while t1 < duration + 0.9:
            actual = _interpolate_lmve(torch.tensor(t0), torch.tensor(t1),
                                       matrix, log_vector)
            assert actual.shape == log_vector.shape
            t1 += 1
        t0 += 1
