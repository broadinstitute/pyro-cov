import os

import pytest
import torch

from pyrophylo.models import CountyModel


@pytest.fixture
def model_inputs():
    if not os.path.exists("results/model_inputs.pt"):
        pytest.skip("missing results/model_inputs.pt try running model_1.ipynb")
    return torch.load("results/model_inputs.pt")


@pytest.mark.parametrize("num_trees", [2])
def test_county_model(model_inputs, num_trees):
    if num_trees is None:
        model_inputs["trees"] = model_inputs["trees"][0]
    else:
        model_inputs["trees"] = model_inputs["trees"][:num_trees]
    model = CountyModel(**model_inputs)
    model.fit_svi(num_particles=1, log_every=1, num_steps=2)
    model.predict()
