import pytest
import torch
from pyro.distributions.testing.gof import auto_goodness_of_fit

from pyrocov.distributions import SmoothLaplace

TEST_FAILURE_RATE = 1e-2


@pytest.mark.parametrize(
    "Dist, params",
    [
        (SmoothLaplace, {"loc": 0.0, "scale": 1.0}),
        (SmoothLaplace, {"loc": 1.0, "scale": 1.0}),
        (SmoothLaplace, {"loc": 0.0, "scale": 10.0}),
    ],
)
def test_gof(Dist, params):
    num_samples = 50000
    d = Dist(**params)
    samples = d.sample(torch.Size([num_samples]))
    probs = d.log_prob(samples).exp()

    # Test each batch independently.
    probs = probs.reshape(num_samples, -1)
    samples = samples.reshape(probs.shape + d.event_shape)
    for b in range(probs.size(-1)):
        gof = auto_goodness_of_fit(samples[:, b], probs[:, b])
        assert gof > TEST_FAILURE_RATE
