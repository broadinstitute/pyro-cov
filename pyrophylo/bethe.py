import logging
import math

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from pyro.distributions import constraints
from pyro.nn import PyroModule
from sklearn.cluster import AgglomerativeClustering

from .substitution import JukesCantor69

logger = logging.getLogger(__name__)


# TODO replace with CoalescentTimes(self.leaf_times, ordered=False)
class CoalescentTimes(dist.TransformedDistribution):
    support = constraints.less_than(0.)

    def __init__(self, leaf_times):
        if not (leaf_times == 0).all():
            raise NotImplementedError
        L = len(leaf_times)
        super().__init__(
            dist.Exponential(torch.ones(L - 1)).to_event(1),
            dist.transforms.AffineTransform(0., -1.))


class Decoder(nn.Module):
    def __init__(self, input_dim, output_shape):
        super().__init__()
        self.shape = torch.Size(output_shape)
        self.linear = torch.nn.Linear(input_dim, self.shape.numel())

    def forward(self, code, temperature):
        logits = self.linear(code)
        logits = logits.reshape(code.shape[:-1] + self.shape)
        return logits.div(temperature).softmax(dim=-1)


class BetheModel(PyroModule):
    """
    A phylogenetic tree model that marginalizes over tree structure and relaxes
    over the states of internal nodes.
    """
    def __init__(self, leaf_times, leaf_data, leaf_mask, *,
                 embedding_dim=20, temperature=1.):
        super().__init__()
        assert leaf_times.dim() == 1
        assert (leaf_times[:-1] <= leaf_times[1:]).all()
        assert leaf_data.dim() == 2
        assert leaf_mask.shape == leaf_data.shape
        assert leaf_data.shape[:1] == leaf_times.shape
        assert isinstance(embedding_dim, int) and embedding_dim > 0
        assert temperature > 0
        L, C = leaf_data.shape
        D = 1 + leaf_data.max().item() - leaf_data.min().item()

        self.leaf_times = leaf_times
        self.leaf_data = leaf_data
        self.leaf_mask = leaf_mask
        self.leaf_states = torch.zeros(L, C, D).scatter_(-1, leaf_data[..., None], 1)
        self.subs_model = JukesCantor69(dim=D)
        self.temperature = torch.tensor(float(temperature))
        self.num_nodes = 2 * L - 1
        self.decoder = Decoder(embedding_dim, (C, D))

        self._initialize()

    def forward(self, sample_tree=False):
        L, C, D = self.leaf_states.shape
        N = 2 * L - 1

        # Sample genetic sequences of all nodes, leaves + internal.
        with pyro.plate("nodes", N, dim=-2), \
             pyro.plate("code_plate", 20, dim=-1):
            codes = pyro.sample("codes", dist.Normal(0, 1))
        states = self.decoder(codes, self.temperature)

        # Interleave samples with observations.
        with pyro.plate("leaves", L, dim=-2), \
             pyro.plate("characters", C, dim=-1), \
             poutine.mask(mask=self.leaf_mask):
            pyro.sample("leaf_likelihood", dist.Categorical(states[:L]),
                        obs=self.leaf_data)
        imputed_leaf_states = torch.where(self.leaf_mask[..., None],
                                          self.leaf_states, states[:L])
        states = torch.cat([imputed_leaf_states, states[L:]], dim=0)

        # Sample times of internal nodes.
        internal_times = pyro.sample("internal_times",
                                     CoalescentTimes(self.leaf_times))
        times = pyro.deterministic("times",
                                   torch.cat([self.leaf_times, internal_times]))

        # Account for random tree structure.
        logits = self.kernel(states.float(), times.float())
        tree_dist = dist.OneTwoMatching(logits, bp_iters=10)
        if not sample_tree:
            # During training, analytically marginalize over trees.
            pyro.factor("tree_likelihood",
                        tree_dist.log_partition_function.to(times.dtype))
        else:
            # During prediction, simply sample a tree.
            pyro.sample("tree", tree_dist)

    def kernel(self, states, times):
        """
        Given states and times, compute pairwise transition log probability
        between every undirected pair of states. This will be -inf for
        infeasible pairs, namely leaf-leaf and internal-after-leaf.
        """
        N, C, D = states.shape
        assert times.shape == (N,)
        L = (N + 1) // 2

        # Select feasible time-ordered pairs.
        with torch.no_grad():
            feasible = times[:, None] < times
            feasible[:L] = False  # Leaves are terminal.
            v0, v1 = feasible.nonzero(as_tuple=True)

        # Convert dense square -> sparse.
        x0 = states[v0]
        x1 = states[v1]
        dt = times[v1] - times[v0] + 1e-6
        m = self.subs_model().to(states.dtype)

        # There are multiple ways to extend the mutation likelihood function to
        # the interior of the relaxed space.
        exp_mt = (dt[:, None, None] * m).matrix_exp()
        kernel_version = 1
        if kernel_version == 0:
            sparse_logits = torch.einsum("fcd,fce,fde->fc",
                                         x0, x1, exp_mt).log().sum(-1)
        elif kernel_version == 1:
            # Accumulate sufficient statistics over characters.
            stats = torch.einsum("fcd,fce->fde", x0, x1)
            sparse_logits = torch.einsum("fde,fde->f",
                                         (exp_mt + 1e-6).log(), stats)
        assert sparse_logits.isfinite().all()

        # Convert sparse -> dense matching.
        num_sources = N - 1  # Everything but the root.
        num_destins = N - L  # All internal nodes.
        assert num_sources == 2 * num_destins
        root = times.min(0).indices.item()
        source_id = torch.arange(N)
        source_id[root] = N
        source_id[root + 1:] -= 1
        destin_id = torch.cat([torch.full((L,), N), torch.arange(N - L)])
        logits = sparse_logits.new_full((num_sources, num_destins), -math.inf)
        logits[source_id[v1], destin_id[v0]] = sparse_logits
        return logits

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

        # Heuristically initialize times and states.
        times = torch.full((N,), math.nan)
        states = torch.full((N, C, D), math.nan)
        times[:L] = self.leaf_times
        states[:L] = self.leaf_states * 0.99 + 0.01 / D
        for p, (c1, c2) in enumerate(children):
            times[L + p] = min(times[c1], times[c2]) - 1 - torch.rand(()) * 0.1
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
        if site["name"] == "states_uniform":
            # This is the GumbelSoftmaxReparam latent variable.
            return 0.1 + 0.8 * self.init_states
        if site["name"] == "internal_times":
            return self.init_internal_times
        if site["name"].endswith("subs_model.rate") or \
                site["name"].endswith("subs_model.rates"):
            # Initialize to low mutation rate.
            return torch.full(site["fn"].shape(), 0.1)
        if site["name"].endswith("subs_model.stationary"):
            D, = site["fn"].event_shape
            return torch.ones(D) / D
        if site["name"] == "codes":
            return torch.randn(site["fn"].shape())
        raise ValueError("unknown site {}".format(site["name"]))
