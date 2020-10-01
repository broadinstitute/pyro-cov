import os

import pytest
import torch

from pyrophylo.models import CountyModel


@pytest.fixture
def model_inputs():
    if not os.path.exists("results/model_inputs.pt"):
        pytest.skip("missing results/model_inputs.pt try running model_1.ipynb")
    return torch.load("results/model_inputs.pt")


@pytest.mark.xfail(reason="not implemented", run=False)
@pytest.mark.parametrize("num_particles", [1, 10])
def test_county_model(model_inputs, num_particles):
    model_inputs["trees"] = model_inputs["trees"][:2]  # Subsample for speed.
    model = CountyModel(**model_inputs)
    model.fit_svi(num_particles=num_particles, log_every=10)
