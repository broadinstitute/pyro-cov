import logging
import math

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.distributions import constraints
from pyro.nn import PyroModule
from sklearn.cluster import AgglomerativeClustering

from .substitution import JukesCantor69

logger = logging.getLogger(__name__)


# TODO replace with CoalescentTimes(self.leaf_times, ordered=False)
class FakeCoalescentTimes(dist.TransformedDistribution):
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
    max_plate_nesting = 2

    def __init__(self, leaf_times, leaf_logits, *,
                 embedding_dim=20, bp_iters=30):
        super().__init__()
        assert leaf_times.dim() == 1
        assert (leaf_times[:-1] <= leaf_times[1:]).all()
        assert leaf_logits.dim() == 3
        assert leaf_logits.shape[:1] == leaf_times.shape
        assert isinstance(embedding_dim, int) and embedding_dim > 0
        L, C, D = leaf_logits.shape

        self.num_nodes = 2 * L - 1
        self.num_leaves = L
        self.num_observations = L * C
        self.leaf_times = leaf_times
        self.leaf_logits = leaf_logits
        self.subs_model = JukesCantor69(dim=D)
        self.bp_iters = bp_iters
        self.embedding_dim = embedding_dim
        self.decoder = Decoder(embedding_dim, (C, D))

        self._initialize()

    def forward(self, mode="train"):
        L, C, D = self.leaf_logits.shape
        N = 2 * L - 1
        node_plate = pyro.plate("node_plate", N, dim=-2)
        code_plate = pyro.plate("code_plate", self.embedding_dim, dim=-1)

        # Sample genetic sequences of all nodes, leaves + internal.
        with node_plate, code_plate:
            codes = pyro.sample("codes", dist.Normal(0, 1).mask(False))
            probs = self.decoder(codes)
        assert probs.shape == (N, C, D)

        # Condition on observations.
        if mode in ("train", "pretrain"):
            pyro.factor("leaf_likelihood",
                        torch.einsum("lcd,lcd->", probs[:L], self.leaf_logits))
        if mode == "pretrain":  # If we're training only self.decoder,
            return              # then we can ignore the rest of the model.

        # Sample times of internal nodes.
        internal_times = pyro.sample("internal_times",
                                     FakeCoalescentTimes(self.leaf_times).mask(False))
        times = pyro.deterministic("times",
                                   torch.cat([self.leaf_times, internal_times]))

        # Account for random tree structure.
        logits, sources, destins = self.kernel(probs.float(), times.float())
        tree_dist = dist.OneTwoMatching(logits, bp_iters=self.bp_iters)
        if mode == "train":
            # During training, analytically marginalize over trees.
            pyro.factor("tree_likelihood",
                        tree_dist.log_partition_function.to(times.dtype))
        elif mode == "predict":
            # During prediction, simply sample a tree.
            # TODO implement OneTwoMatching.sample(); until then use .mode().
            tree_dist = dist.Delta(tree_dist.mode(), event_dim=1)
            tree = pyro.sample("tree", tree_dist)

            # Convert sparse matching to phylogeny inputs.
            parents = tree.new_full((N,), -1)
            parents[sources] = destins[tree]
            pyro.deterministic("parents", parents)
            return codes, times, parents

    def kernel(self, probs, times):
        """
        Given probs and times, compute pairwise transition log probability
        between every undirected pair of states. This will be -inf for
        infeasible pairs, namely leaf-leaf and internal-after-leaf.
        """
        finfo = torch.finfo(probs.dtype)
        N, C, D = probs.shape
        assert times.shape == (N,)
        L = (N + 1) // 2

        # Select feasible time-ordered pairs.
        times.data.clamp_(min=-1 / finfo.eps)
        feasible = times.data[:, None] < times.data
        feasible[:L] = False  # Leaves are terminal.
        v0, v1 = feasible.nonzero(as_tuple=True)

        # Convert dense square -> sparse.
        dt = times[v1] - times[v0]
        transition = self.subs_model.log_matrix_exp(dt)
        # Accumulate sufficient statistics over characters.
        stats = torch.einsum("icd,jce->ijde", probs, probs)[v0, v1]
        sparse_logits = torch.einsum("fde,fde->f", stats, transition)
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

    @torch.no_grad()
    def _initialize(self):
        logger.info("Initializing via PCA + agglomerative clustering")

        # Heuristically initialize leaf codes via PCA.
        L = self.leaf_logits.size(0)
        N = 2 * L - 1
        E = self.embedding_dim
        leaf_codes = torch.pca_lowrank(self.leaf_logits.exp().reshape(L, -1), E)[0]
        leaf_codes *= L ** 0.5  # Force unit variance.

        # Heuristically initialize hierarchy.
        clustering = AgglomerativeClustering(
            distance_threshold=0, n_clusters=None).fit(leaf_codes)
        children = clustering.children_
        assert children.shape == (L - 1, 2)

        # Heuristically initialize internal times and codes.
        times = torch.full((N,), math.nan)
        codes = torch.full((N, E), math.nan)
        times[:L] = self.leaf_times
        codes[:L] = leaf_codes + torch.randn(L, E) * 0.01
        timescale = 0.1
        for p, (c1, c2) in enumerate(children):
            times[L + p] = min(times[c1], times[c2]) - timescale * (0.5 + torch.rand(()))
            codes[L + p] = (codes[c1] + codes[c2]) / 2 + torch.randn(E) * 0.01
        assert times.isfinite().all()
        assert codes.isfinite().all()
        self.init_internal_times = times[L:].clone()
        self.init_codes = codes

    def init_loc_fn(self, site):
        """
        Heuristic initialization for guides.
        """
        name = site["name"]
        if name == "internal_times":
            return self.init_internal_times
        if name == "codes":
            return self.init_codes
        if name.endswith("subs_model.rate") or name.endswith("subs_model.rates"):
            # Initialize to unit mutation rate.
            return torch.full(site["fn"].shape(), 1.0)
        if name.endswith("subs_model.stationary"):
            D, = site["fn"].event_shape
            return torch.ones(D) / D
        raise ValueError(f"unknown site: {name}")
