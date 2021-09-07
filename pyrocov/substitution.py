# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import pyro.distributions as dist
import torch
from pyro.nn import PyroModule, PyroSample, pyro_method


class SubstitutionModel(PyroModule):
    """
    Probabilistic substitution model among a finite number of states
    (typically 4 for nucleotides or 20 for amino acids).

    This returns a continuous time transition matrix.
    """

    @pyro_method
    def matrix_exp(self, dt):
        m = self().to(dt.dtype)
        return (m * dt[:, None, None]).matrix_exp()

    @pyro_method
    def log_matrix_exp(self, dt):
        m = self.matrix_exp(dt)
        m.data.clamp_(torch.finfo(m.dtype).eps)
        return m.log()


class JukesCantor69(SubstitutionModel):
    """
    A simple uniform substition model with a single latent rate parameter.

    This provides a weak Exponential(1) prior over the rate parameter.

    [1] T.H. Jukes, C.R. Cantor (1969) "Evolution of protein molecules"
    [2] https://en.wikipedia.org/wiki/Models_of_DNA_evolution#JC69_model_(Jukes_and_Cantor_1969)
    """

    def __init__(self, *, dim=4):
        assert isinstance(dim, int) and dim > 0
        super().__init__()
        self.dim = dim
        self.rate = PyroSample(dist.Exponential(1.0))

    def forward(self):
        D = self.dim
        return self.rate * (1.0 / D - torch.eye(D))

    @pyro_method
    def matrix_exp(self, dt):
        D = self.dim
        rate = torch.as_tensor(self.rate, dtype=dt.dtype)
        p = dt.mul(-rate).exp()[:, None, None]
        q = (1 - p) / D
        return torch.where(torch.eye(D, dtype=torch.bool), p + q, q)

    @pyro_method
    def log_matrix_exp(self, dt):
        D = self.dim
        rate = torch.as_tensor(self.rate, dtype=dt.dtype)
        p = dt.mul(-rate).exp()[:, None, None]
        q = (1 - p) / D
        q.data.clamp_(min=torch.finfo(q.dtype).eps)
        on_diag = (p + q).log()
        off_diag = q.log()
        return torch.where(torch.eye(D, dtype=torch.bool), on_diag, off_diag)


class GeneralizedTimeReversible(SubstitutionModel):
    """
    Generalized time-reversible substitution model among ``dim``-many states.

    This provides a weak Dirichlet(2) prior over the steady state distribution
    and a weak Exponential(1) prior over mutation rates.
    """

    def __init__(self, *, dim=4):
        assert isinstance(dim, int) and dim > 0
        super().__init__()
        self.dim = dim
        self.stationary = PyroSample(dist.Dirichlet(torch.full((dim,), 2.0)))
        self.rates = PyroSample(
            dist.Exponential(torch.ones(dim * (dim - 1) // 2)).to_event(1)
        )
        i = torch.arange(dim)
        self._index = (i > i[:, None]).nonzero(as_tuple=False).T

    def forward(self):
        p = self.stationary
        i, j = self._index
        m = torch.zeros(self.dim, self.dim)
        m[i, j] = self.rates
        m = m + m.T * (p / p[:, None])
        m = m - m.sum(dim=-1).diag_embed()
        return m
