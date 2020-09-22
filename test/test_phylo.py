import pytest
import torch

from pyrophylo.phylo import _interpolate_lmve, _lmve


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)])
@pytest.mark.parametrize("size", [5])
def test_lmve(shape, size):
    matrix = torch.randn(shape + (size, size)).exp()
    log_tensor = torch.randn(shape + (size,))

    expected = (matrix @ log_tensor.exp().unsqueeze(-1)).squeeze(-1).log()
    actual = _lmve(matrix, log_tensor)

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)])
@pytest.mark.parametrize("size", [5])
@pytest.mark.parametrize("duration", [
    pytest.param(1, marks=[pytest.mark.xfail(reason="matrix_power supports only int powers")]),
    pytest.param(2, marks=[pytest.mark.xfail(reason="not implemented")]),
    pytest.param(3, marks=[pytest.mark.xfail(reason="not implemented")]),
    pytest.param(4, marks=[pytest.mark.xfail(reason="not implemented")]),
    pytest.param(5, marks=[pytest.mark.xfail(reason="not implemented")]),
])
def test_lmve_smoke(shape, size, duration):
    matrix = torch.randn(shape + (duration, size, size)).exp()
    log_tensor = torch.randn(shape + (size,))
    t0 = -0.6
    while t0 < duration + 0.9:
        t1 = t0 + 0.2
        while t1 < duration + 0.9:
            actual = _interpolate_lmve(t0, t1, matrix, log_tensor)
            assert actual.shape == log_tensor.shape
            t1 += 1
        t0 += 1
