import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from pyro.nn import PyroModule, PyroSample
from sklearn.cluster import AgglomerativeClustering
from torch.distributions import constraints


def _vv(x, y):
    return (x[..., None, :] @ y[..., None]).squeeze(-1).squeeze(-1)


class GTRSubstitutionModel(PyroModule):
    def __init__(self, dim=4):
        self.dim = dim
        self.stationary = PyroSample(dist.Dirichlet(torch.full(2., (dim,))))
        self.rates = PyroSample(
            dist.Exponential(1.).expand([dim * (dim - 1) // 2]).to_event(1))
        i = torch.arange(dim)
        self._index = (i > i[:, None]).nonzero(as_tuple=False).T

    @property
    def transition(self):
        p = self.stationary
        i, j = self._index
        m = torch.zeros(self.dim, self.dim)
        m[i, j] = self.rates
        m = m + m.T * (p / p[:, None])
        m = m - m.sum(dim=-1).diag_embed()
        return m

    def forward(self, times, states):
        m = self.transition
        return states @ (m * times[..., None, None]).matrix_exp().T


def log_count_spanning_trees(w):
    """
    Uses Kirchhoff's matrix tree theorem to count weighted spanning trees.

    :param Tensor w: A symmetric matrix of edge weights.
    :returns: The log sum-over-trees product-over-edges.
    """
    assert w.dim() == 2
    L = w.sum(dim=-1).diag_embed() - w
    truncated = L[:-1, :-1]
    try:
        import gpytorch
        return gpytorch.lazy.NonLazyTensor(truncated).logdet()
    except ImportError:
        return torch.cholesky(truncated).diag().log().sum() * 2


class KirchhoffModel(PyroModule):
    def __init__(self, leaf_times, leaf_data, leaf_mask, *,
                 temperature=1.):
        assert leaf_times.dim() == 1
        assert (leaf_times[:-1] <= leaf_times[1:]).all()
        N = self.leaf_times.size(0)

        self.leaf_times = leaf_times
        self.leaf_data = leaf_data
        self.leaf_mask = leaf_mask
        self.subs_model = GTRSubstitutionModel()
        self.temperature = temperature

        is_leaf = torch.full(False, (2 * N - 1,), dtype=torch.bool)
        is_leaf[:N] = True
        self._leaf_leaf = is_leaf[:, None] & is_leaf
        self._leaf_internal = is_leaf[:, None] & ~is_leaf

    def forward(self):
        N = self.leaf_times.size(0)
        D = self.subs_model.dim

        # Impute missing leaf states.
        with poutine.mask(mask=~self.leaf_mask):
            leaf_probs = pyro.sample(
                "leaf_probs",
                dist.Uniform(0, 1).expand([N, D]).to_event(2))
            leaf_states = pyro.sample(
                "leaf_states",
                dist.RelaxedOneHotCategoricalStraightThrough(
                    self.temperature, leaf_probs).to_event(2))
        leaf_states = torch.where(self.leaf_mask[..., None],
                                  self.leaf_data, leaf_states)

        # Sample internal nodes.
        internal_times = pyro.sample(
            "internal_times",
            dist.ImproperUniform(constraints.greater_than(self.leaf_times[1:]),
                                 (), (N - 1,)))
        internal_states = pyro.sample(
            "internal_states",
            dist.ImproperUniform(constraints.real, (), (N - 1, D)))

        times = torch.cat([self.leaf_times, internal_times])
        states = torch.cat([leaf_states, internal_states], dim=0)
        w = self.kernel(states, times)
        pyro.factor("topology", log_count_spanning_trees(w))

    def kernel(self, states, times):
        N, C, D = states.shape
        assert times.shape == (N,)

        # Start with a naive mutation kernel.
        dt = times[:, None] - times
        x = states
        y = self.subs_model(dt, x[:, None])
        w = _vv(x, y).add(1e-4).log().logsumexp(dim=-1)

        # Exclude leaf-leaf edges and out-of-order leaf-internal edges.
        ooo = self._leaf_internal & (dt <= 0)
        w = w.masked_fill(self._leaf_leaf | ooo | ooo.transpose(-1, -2), 0.)

        return w

    def init_loc_fn(self, site):
        """
        Heuristic to initialize ``internal_times`` and ``internal_states``.
        """
        if not hasattr(self, "clustering"):
            self.clustering = AgglomerativeClustering(
                distance_threshold=0, n_clusters=None).fit(self.leaf_states)
        clustering = self.clustering
        assert clustering
        raise NotImplementedError("TODO")
