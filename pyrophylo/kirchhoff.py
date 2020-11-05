import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from pyro.nn import PyroModule, PyroSample
from sklearn.cluster import AgglomerativeClustering


def _vv(x, y):
    return (x[..., None, :] @ y[..., None]).squeeze(-1).squeeze(-1)


class GTRSubstitutionModel(PyroModule):
    """
    Generalized time-reversible substitution model among ``dim``-many states.
    """
    def __init__(self, dim=4):
        super().__init__()
        self.dim = dim
        self.stationary = PyroSample(dist.Dirichlet(torch.full((dim,), 2.)))
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
        return states @ (m * times[..., None]).matrix_exp().transpose(-1, -2)


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
        super().__init__()
        assert leaf_times.dim() == 1
        assert (leaf_times[:-1] <= leaf_times[1:]).all()
        assert leaf_data.dim() == 2
        assert leaf_mask.shape == leaf_data.shape
        assert leaf_data.shape[:1] == leaf_times.shape
        L, C = leaf_data.shape
        N = 2 * L - 1
        D = 1 + leaf_data.max().item() - leaf_data.min().item()

        self.leaf_times = leaf_times
        self.leaf_mask = leaf_mask
        self.mask = torch.cat([~leaf_mask, leaf_mask.new_ones(L - 1, C)], dim=0)
        self.leaf_states = torch.ones(L, C, D).scatter_(-1, leaf_data[..., None], 1)
        self.subs_model = GTRSubstitutionModel(dim=D)
        self.temperature = temperature

        is_leaf = torch.full((N,), False, dtype=torch.bool)
        is_leaf[:L] = True
        self._leaf_leaf = is_leaf[:, None] & is_leaf
        self._leaf_internal = is_leaf[:, None] & ~is_leaf

    def forward(self):
        L, C, D = self.leaf_states.shape
        N = 2 * L - 1

        # Impute missing states.
        with pyro.plate("nodes", N, dim=-2), \
             pyro.plate("characters", C, dim=-1), \
             poutine.mask(mask=self.mask):
            # TODO reparametrize this with a SoftmaxReparam
            states = pyro.sample(
                "states",
                dist.RelaxedOneHotCategorical(self.temperature, torch.ones(D)))
        # Interleave with observed states.
        leaf_states = torch.where(self.leaf_mask[..., None],
                                  self.leaf_states, states[:L])
        states = torch.cat([leaf_states, states[L:]], dim=0)
        assert states.shape == (N, C, D)

        # Sample times of internal nodes.
        if not (self.leaf_times == 0).all():
            raise NotImplementedError("TODO")
        internal_times = pyro.sample(
            "internal_times",
            # TODO replace with CoalescentTimes(self.leaf_times, ordered=False)
            dist.Exponential(torch.ones(L - 1)).to_event(1).mask(False))

        times = torch.cat([self.leaf_times, internal_times])
        w = self.kernel(states, times)
        pyro.factor("topology", log_count_spanning_trees(w))

    def kernel(self, states, times):
        N, C, D = states.shape
        assert times.shape == (N,)

        # Start with a naive mutation kernel.
        dt = times[:, None] - times
        x = states
        y = self.subs_model(dt[..., None], x[:, None])
        assert y.shape == (N, N, C, D)
        w = _vv(x, y).add(1e-6).log().logsumexp(dim=-1)
        assert w.shape == (N, N)

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
