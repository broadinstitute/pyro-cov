import pyro.distributions as dist
import torch
from pyro.nn import PyroModule, PyroSample


class SubstitutionModel(PyroModule):
    """
    Probabilistic substitution model among a finite number of states
    (typically 4 for nucleotides or 20 for amino acids).

    This returns a continuous time transition matrix.
    """


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
        self.rate = PyroSample(dist.Exponential(1.))

    def forward(self):
        D = self.dim
        return self.rate * (1. / D - torch.eye(D))


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
        self.stationary = PyroSample(dist.Dirichlet(torch.full((dim,), 2.)))
        self.rates = PyroSample(
            dist.Exponential(torch.ones(dim * (dim - 1) // 2)).to_event(1))
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
