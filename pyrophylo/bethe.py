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

from .phylo import Phylogeny
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

    def forward(self, code):
        logits = self.linear(code)
        logits = logits.reshape(code.shape[:-1] + self.shape)
        return logits.softmax(dim=-1)


class BetheModel(PyroModule):
    """
    A phylogenetic tree model that marginalizes over tree structure and relaxes
    over the states of internal nodes.
    """
    def __init__(self, leaf_times, leaf_data, leaf_mask, *,
                 embedding_dim=20, bp_iters=30):
        super().__init__()
        assert leaf_times.dim() == 1
        assert (leaf_times[:-1] <= leaf_times[1:]).all()
        assert leaf_data.dim() == 2
        assert leaf_mask.shape == leaf_data.shape
        assert leaf_data.shape[:1] == leaf_times.shape
        assert isinstance(embedding_dim, int) and embedding_dim > 0
        L, C = leaf_data.shape
        D = 1 + leaf_data.max().item() - leaf_data.min().item()

        self.leaf_times = leaf_times
        self.leaf_data = leaf_data
        self.leaf_mask = leaf_mask
        self.leaf_states = torch.zeros(L, C, D).scatter_(-1, leaf_data[..., None], 1)
        self.subs_model = JukesCantor69(dim=D)
        self.bp_iters = bp_iters
        self.num_nodes = 2 * L - 1
        self.embedding_dim = embedding_dim
        self.decoder = Decoder(embedding_dim, (C, D))

        self._initialize()

    def forward(self, pretrain=False, sample_tree=False):
        L, C, D = self.leaf_states.shape
        N = 2 * L - 1

        # Sample genetic sequences of all nodes, leaves + internal.
        with pyro.plate("nodes", N, dim=-2), \
             pyro.plate("code_plate", self.embedding_dim, dim=-1):
            codes = pyro.sample("codes", dist.Normal(0, 1))
        states = self.decoder(codes)

        # Interleave samples with observations.
        with pyro.plate("leaves", L, dim=-2), \
             pyro.plate("characters", C, dim=-1), \
             poutine.mask(mask=self.leaf_mask):
            # We could account for sequencing errors here by multiplying by a
            # confusion matrix, possibly depending on site or batch.
            pyro.sample("leaf_likelihood", dist.Categorical(states[:L]),
                        obs=self.leaf_data)
        if pretrain:  # If we're training only self.decoder,
            return    # then we can ignore the rest of the model.
        imputed_leaf_states = torch.where(self.leaf_mask[..., None],
                                          self.leaf_states, states[:L])
        states = torch.cat([imputed_leaf_states, states[L:]], dim=0)

        # Sample times of internal nodes.
        internal_times = pyro.sample("internal_times",
                                     CoalescentTimes(self.leaf_times))
        times = pyro.deterministic("times",
                                   torch.cat([self.leaf_times, internal_times]))

        # Account for random tree structure.
        logits, sources, destins = self.kernel(states.float(), times.float())
        tree_dist = dist.OneTwoMatching(logits, bp_iters=self.bp_iters)
        if not sample_tree:
            # During training, analytically marginalize over trees.
            pyro.factor("tree_likelihood",
                        tree_dist.log_partition_function.to(times.dtype))
        else:
            # During prediction, simply sample a tree.
            # TODO implement OneTwoMatching.sample(); until then use .mode().
            tree_dist = dist.Delta(tree_dist.mode(), event_dim=1)
            tree = pyro.sample("tree", tree_dist)

            # Convert sparse matching to phylogeny.
            parents = tree.new_full((N,), -1)
            parents[sources] = destins[tree]
            leaves = torch.arange(L)
            phylo, old2new, new2old = Phylogeny.sort(times, parents, leaves)
            codes = codes[new2old]
            return phylo, codes

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
        dt = times[v1] - times[v0]
        m = self.subs_model().to(states.dtype)
        exp_mt = (dt[:, None, None] * m).matrix_exp()
        # Accumulate sufficient statistics over characters.
        stats = torch.einsum("icd,jce->ijde", states, states)[v0, v1]
        sparse_logits = torch.einsum("fde,fde->f", exp_mt.log(), stats)
        assert sparse_logits.isfinite().all()

        # Convert sparse -> dense matching.
        num_sources = N - 1  # Everything but the root.
        num_destins = N - L  # All internal nodes.
        assert num_sources == 2 * num_destins
        root = times.min(0).indices.item()
        decode_source = torch.cat([torch.arange(root), torch.arange(root + 1, N)])
        decode_destin = torch.arange(L, N)
        encode = torch.full((2, N), N)
        encode[0, decode_source] = torch.arange(N - 1)
        encode[1, decode_destin] = torch.arange(N - L)
        logits = sparse_logits.new_full((num_sources, num_destins), -math.inf)
        logits[encode[0, v1], encode[1, v0]] = sparse_logits
        return logits, decode_source, decode_destin

    def _initialize(self):
        logger.info("Initializing via PCA + agglomerative clustering")

        # Deterministically impute, used only by initialization.
        missing = ~self.leaf_mask
        self.leaf_states[missing] = 1. / self.subs_model.dim

        # Heuristically initialize leaf codes via PCA.
        L = self.leaf_states.size(0)
        N = 2 * L - 1
        E = self.embedding_dim
        leaf_codes = torch.pca_lowrank(self.leaf_states.reshape(L, -1), E)[0]

        # Heuristically initialize hierarchy.
        clustering = AgglomerativeClustering(
            distance_threshold=0, n_clusters=None).fit(leaf_codes)
        children = clustering.children_
        assert children.shape == (L - 1, 2)

        # Heuristically initialize internal times and codes.
        times = torch.full((N,), math.nan)
        codes = torch.full((N, E), math.nan)
        times[:L] = self.leaf_times
        codes[:L] = leaf_codes + torch.randn(L, E) * 0.1
        for p, (c1, c2) in enumerate(children):
            times[L + p] = min(times[c1], times[c2]) - 0.1 - torch.rand(())
            codes[L + p] = (codes[c1] + codes[c2]) / 2 + torch.randn(E) * 0.1
        assert times.isfinite().all()
        assert codes.isfinite().all()
        self.init_internal_times = times[L:].clone()
        self.init_codes = codes

    def init_loc_fn(self, site):
        """
        Heuristic initialization for guides.
        """
        if site["name"] == "internal_times":
            return self.init_internal_times
        if site["name"] == "codes":
            return self.init_codes
        if site["name"].endswith("subs_model.rate") or \
                site["name"].endswith("subs_model.rates"):
            # Initialize to low mutation rate.
            return torch.full(site["fn"].shape(), 0.1)
        if site["name"].endswith("subs_model.stationary"):
            D, = site["fn"].event_shape
            return torch.ones(D) / D
        raise ValueError("unknown site {}".format(site["name"]))
