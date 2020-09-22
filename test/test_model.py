import os

import pytest
import torch

from pyrophylo.models import CountyModel


@pytest.fixture
def model_inputs():
    if not os.path.exists("results/model_inputs.pt"):
        pytest.skip("missing results/model_inputs.pt try running model_1.ipynb")
    return torch.load("results/model_inputs.pt")


@pytest.mark.xfail(reason="not implemented")
def test_county_model(model_inputs):
    model = CountyModel(**model_inputs)
    model.fit_svi(guide_rank=0, num_samples=10, log_every=10)
