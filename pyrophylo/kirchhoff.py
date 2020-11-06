import logging
import math

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from pyro.distributions import constraints
from pyro.distributions.spanning_tree import make_complete_graph
from pyro.nn import PyroModule, PyroSample
from pyro.util import warn_if_nan
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


def _vv(x, y):
    return (x[..., None, :] @ y[..., None]).squeeze(-1).squeeze(-1)


class FakeCoalescentTimes(dist.TransformedDistribution):
    support = constraints.less_than(0.)

    def __init__(self, leaf_times):
        if not (leaf_times == 0).all():
            raise NotImplementedError
        L = len(leaf_times)
        super().__init__(
            dist.Exponential(torch.ones(L - 1)).to_event(1),
            dist.transforms.AffineTransform(0., -1.))


class GTRSubstitutionModel(PyroModule):
    """
    Generalized time-reversible substitution model among ``dim``-many states.
    """
    def __init__(self, dim=4):
        super().__init__()
        self.dim = dim
        self.stationary = PyroSample(dist.Dirichlet(torch.full((dim,), 2.)))
        self.rates = PyroSample(
            dist.Exponential(torch.ones(dim * (dim - 1) // 2)).to_event(1))
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
        warn_if_nan(m, "transition")
        times = times.abs()
        return states @ (m * times[..., None]).matrix_exp().transpose(-1, -2)


def log_count_spanning_trees(w):
    """
    Uses Kirchhoff's matrix tree theorem to count weighted spanning trees.

    :param Tensor w: A symmetric matrix of edge weights.
    :returns: The log sum-over-trees product-over-edges.
    """
    assert w.dim() == 2
    L = w.sum(dim=-1).diag_embed() - w  # Construct laplacian.
    m = L[:-1, :-1]  # Truncate.

    # Numerically stabilize.
    scale = m.diag()
    shift = scale.log().sum()
    scale = scale.sqrt()
    m = m / (scale * scale[:, None])

    # Defer to logdet.
    try:
        import gpytorch
        result = gpytorch.lazy.NonLazyTensor(m).logdet()
    except ImportError:
        result = torch.cholesky(m).diag().log().sum() * 2

    return result + shift


class KirchhoffModel(PyroModule):
    def __init__(self, leaf_times, leaf_data, leaf_mask, *,
                 temperature=1.):
        super().__init__()
        assert leaf_times.dim() == 1
        assert (leaf_times[:-1] <= leaf_times[1:]).all()
        assert leaf_data.dim() == 2
        assert leaf_mask.shape == leaf_data.shape
        assert leaf_data.shape[:1] == leaf_times.shape
        assert temperature > 0
        L, C = leaf_data.shape
        N = 2 * L - 1
        D = 1 + leaf_data.max().item() - leaf_data.min().item()

        self.leaf_times = leaf_times
        self.leaf_states = torch.zeros(L, C, D).scatter_(-1, leaf_data[..., None], 1)
        self.subs_model = GTRSubstitutionModel(dim=D)
        self.temperature = torch.tensor(float(temperature))

        # Precompute masks.
        self.leaf_mask = leaf_mask
        self.is_latent = torch.cat([~leaf_mask, leaf_mask.new_ones(L - 1, C)], dim=0)
        is_leaf = torch.full((N,), False, dtype=torch.bool)
        is_leaf[:L] = True
        self.leaf_internal = is_leaf[:, None] & ~is_leaf
        leaf_leaf = is_leaf[:, None] & is_leaf
        self.infeasible = leaf_leaf | torch.eye(N, dtype=torch.bool)
        self.is_leaf = is_leaf

        self._initialize()

    @property
    def num_nodes(self):
        return len(self.is_latent)

    def forward(self, sample_tree=False):
        L, C, D = self.leaf_states.shape
        N = 2 * L - 1

        # Impute missing states.
        with pyro.plate("nodes", N, dim=-2), \
             pyro.plate("characters", C, dim=-1), \
             poutine.mask(mask=self.is_latent):
            # TODO reparametrize this with a SoftmaxReparam
            states = pyro.sample(
                "states",
                dist.RelaxedOneHotCategorical(self.temperature, torch.ones(D)))
        # Interleave with observed states.
        states = torch.cat([torch.where(self.leaf_mask[..., None],
                                        self.leaf_states, states[:L]),
                            states[L:]], dim=0)
        assert states.shape == (N, C, D)
        warn_if_nan(states, "states")

        # Sample times of internal nodes.
        # TODO replace with CoalescentTimes(self.leaf_times, ordered=False)
        internal_times = pyro.sample("internal_times",
                                     FakeCoalescentTimes(self.leaf_times))
        warn_if_nan(internal_times, "internal_times")
        times = torch.cat([self.leaf_times, internal_times])

        # Account for random tree structure.
        w = self.kernel(states, times)
        if not sample_tree:
            # During training, analytically marginalize over trees.
            pyro.factor("state_likelihood", log_count_spanning_trees(w))
        else:
            # During prediction, simply sample a tree.
            i, j = make_complete_graph(N)
            edge_logits = w.detach()[i, j].log()
            return pyro.sample(
                "tree",
                dist.SpanningTree(edge_logits, {"backend": "cpp"}))

    def kernel1(self, states, times):
        N, C, D = states.shape
        assert times.shape == (N,)

        # Start with a naive mutation kernel.
        dt = times[:, None] - times
        x = states
        y = self.subs_model(dt[..., None], x[:, None])
        assert y.shape == (N, N, C, D)
        w = _vv(x, y).add(1e-6).log().logsumexp(dim=-1)
        assert w.shape == (N, N)

        # Exclude self edges, leaf-leaf edges, and out-of-order leaf-internal edges.
        ooo = self.leaf_internal & (dt <= 0)
        w = w.masked_fill(self.infeasible | ooo | ooo.transpose(-1, -2), 0.)
        assert w.isfinite().all()

        return w

    def kernel2(self, states, times):
        """
        This kernel is cheaper but is less forgiving of unlikely values,
        and ends up leading to near zero mutation likelihood.
        """
        N, C, D = states.shape
        assert times.shape == (N,)
        L = (N + 1) // 2

        # Select feasible pairs.
        with torch.no_grad():
            feasible = times[:, None] < times
            feasible[:L] = False  # Leaves are terminal.
            ancestor, descendent = feasible.nonzero(as_tuple=False).unbind(-1)
            F = len(ancestor)

        # Convert dense -> sparse.
        x0 = states[ancestor]
        x1 = states[descendent]
        dt = 1e-6 + times[descendent] - times[ancestor]

        # Accumulate sufficient statistics over characters.
        stats = torch.einsum("fcd,fce->fde", x0, x1)
        assert stats.shape == (F, D, D)

        # Interpolate mutation likelihood to relaxed values.
        m = self.subs_model.transition
        exp_mt = (dt[:, None, None] * m).matrix_exp()
        w_sparse = torch.einsum("fde,fde->f", exp_mt.add(1e-6).log(), stats).exp()
        assert w_sparse.isfinite().all()

        # Convert sparse -> dense.
        w = torch.zeros(N, N)
        w[feasible] = w_sparse
        w.T[feasible] = w_sparse
        return w

    kernel = kernel1

    def _initialize(self):
        logger.info("Initializing via agglomerative clustering")

        # Deterministically impute, only used by initialization.
        missing = ~self.leaf_mask
        self.leaf_states[missing] = 1. / self.subs_model.dim

        # Heuristically initialize hierarchy.
        L, C, D = self.leaf_states.shape
        N = 2 * L - 1
        data = self.leaf_states.reshape(L, C * D)
        clustering = AgglomerativeClustering(
            distance_threshold=0, n_clusters=None).fit(data)
        children = clustering.children_
        assert children.shape == (L - 1, 2)

        times = torch.full((N,), math.nan)
        states = torch.full((N, C, D), math.nan)
        times[:L] = self.leaf_times
        states[:L] = self.leaf_states * 0.99 + 0.01 / D
        for p, (c1, c2) in enumerate(children):
            times[L + p] = min(times[c1], times[c2]) - 1
            states[L + p] = (states[c1] + states[c2]) / 2
        assert times.isfinite().all()
        assert states.isfinite().all()
        self.init_internal_times = times[L:].clone()
        self.init_states = states

    def init_loc_fn(self, site):
        """
        Heuristic initialization for guides.
        """
        if site["name"] == "states":
            return self.init_states
        if site["name"] == "internal_times":
            return self.init_internal_times
        if site["name"].endswith("subs_model.rates"):
            # Initialize to low mutation rate.
            return torch.full(site["fn"].shape(), 0.01)
        if site["name"].endswith("subs_model.stationary"):
            D, = site["fn"].event_shape
            return torch.ones(D) / D
        raise ValueError("unknown site {}".format(site["name"]))
