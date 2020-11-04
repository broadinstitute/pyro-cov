import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints

from sklearn.cluster import AgglomerativeClustering


class MaskedGaussianKernel:
    def __init__(self, num_leaves, embedding_dim):
        N = self.num_leaves = num_leaves
        self.embeding_dim = embedding_dim

        is_leaf = torch.full(False, (2 * N - 1,), dtype=torch.bool)
        is_leaf[:N] = True
        self.leaf_leaf = is_leaf[:, None] & is_leaf
        self.leaf_internal = is_leaf[:, None] & ~is_leaf

    def __call__(self, codes, times):
        N = self.num_leaves
        D = self.embedding_dim
        assert times.shape == (2 * N - 1,)
        assert codes.shape == (2 * N - 1, D)

        # Start with a Brownian bridge kernel.
        r2 = (codes[:, None] - codes).pow(2).sum(-1)
        dt = times[:, None] - times
        abs_dt = dt.abs() + 1e-2  # avoid nans
        w = r2.div(abs_dt).mul(-0.5).exp() / abs_dt.pow(D / 2)

        # Exclude leaf-leaf edges and out-of-order leaf-internal edges.
        ooo = self.leaf_internal & (dt <= 0)
        w = w.masked_fill(self.leaf_leaf | ooo | ooo.transpose(-1, -2), 0.)

        return w


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


class KirchhoffModel:
    def __init__(self, leaf_times, leaf_states, *,
                 embedding_dim=20,
                 kernel=None):
        assert leaf_times.dim() == 1
        assert (leaf_times[:-1] <= leaf_times[1:]).all()
        if kernel is None:
            kernel = MaskedGaussianKernel(len(leaf_times), embedding_dim)

        self.leaf_times = leaf_times
        self.leaf_states = leaf_states
        self.embedding_dim = embedding_dim
        self.kernel = kernel

    def fit_embedding(self, *, num_steps=1000):
        input_dim = self.leaf_states.size(-2) * self.leaf_states.size(-1)
        encode = torch.nn.Linear(input_dim, self.embedding_dim)
        w_true = "TODO"
        for step in range(num_steps):
            codes = encode(self.leaf_states)
            w_approx = self.kernel(codes)  # TODO integrate over time or sth
            loss = (w_true - w_approx).abs().sum()
            loss.backward()
            raise NotImplementedError("TODO")

        # Save the learned embedding.
        with torch.no_grad():
            self.leaf_codes = encode(self.leaf_states)

    def __call__(self):
        N = self.leaf_times.size(0)
        D = self.embedding_dim

        internal_times = pyro.sample(
            "internal_times",
            dist.ImproperUniform(constraints.greater_than(self.leaf_times[1:]),
                                 (), (N - 1,)))
        internal_codes = pyro.sample(
            "internal_codes",
            dist.ImproperUniform(constraints.real, (), (N - 1, D)))

        times = torch.cat([self.leaf_times, internal_times])
        codes = torch.cat([self.leaf_codes, internal_codes], dim=0)
        w = self.kernel(codes, times)
        pyro.factor("topology", log_count_spanning_trees(w))

    def init_loc_fn(self, site):
        """
        Heuristic to initialize ``internal_times`` and ``internal_codes``.
        """
        if not hasattr(self, "clustering"):
            self.clustering = AgglomerativeClustering(
                distance_threshold=0, n_clusters=None).fit(self.leaf_codes)
        clustering = self.clustering
        assert clustering
        raise NotImplementedError("TODO")
