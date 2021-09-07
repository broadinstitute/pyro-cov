# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import pyro.poutine as poutine
import pytest
import torch
from pyro.infer.autoguide import AutoDelta

from pyrocov.substitution import GeneralizedTimeReversible, JukesCantor69


@pytest.mark.parametrize("Model", [JukesCantor69, GeneralizedTimeReversible])
def test_matrix_exp(Model):
    model = Model()
    guide = AutoDelta(model)
    guide()
    trace = poutine.trace(guide).get_trace()
    t = torch.randn(10).exp()
    with poutine.replay(trace=trace):
        m = model()
        assert torch.allclose(model(), m)

        exp_mt = (m * t[:, None, None]).matrix_exp()
        actual = model.matrix_exp(t)
        assert torch.allclose(actual, exp_mt, atol=1e-6)

        actual = model.log_matrix_exp(t)
        log_exp_mt = exp_mt.log()
        assert torch.allclose(actual, log_exp_mt, atol=1e-6)
