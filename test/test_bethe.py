import pyro.distributions as dist
import pytest
import torch

from pyrophylo.bethe import Encoder


@pytest.mark.parametrize("E", [3, 10])
@pytest.mark.parametrize("C", [5])
@pytest.mark.parametrize("D", [2, 4])
@pytest.mark.parametrize("N", [1, 2, 3])
def test_expected_lob_prob(E, C, D, N):
    codes = torch.randn(N, E)
    probs = torch.randn(N, C, D).softmax(-1)
    encoder = Encoder(E, (C, D))
    actual = encoder.expected_log_prob(codes, probs)

    num_samples = 100000
    states = dist.OneHotCategorical(probs).sample([num_samples])
    expected = encoder.log_prob(codes, states) / num_samples
    assert torch.allclose(actual, expected, atol=0.01)
