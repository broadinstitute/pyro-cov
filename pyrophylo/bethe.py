import logging
import math

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from opt_einsum import contract as einsum
from pyro.distributions import constraints
from pyro.nn import PyroModule
from sklearn.cluster import AgglomerativeClustering

from .substitution import JukesCantor69

logger = logging.getLogger(__name__)


# TODO move upstream to Pyro
class UnorderedCoalescentTimes(dist.CoalescentTimes):
    support = constraints.less_than(0.)

    def __init__(self, leaf_times, rate):
        if not (leaf_times == 0).all():
            raise NotImplementedError("TODO")
        super().__init__(leaf_times, rate)

    def log_prob(self, value):
        value = value.sort(dim=-1).values
        return super().log_prob(value) - math.lgamma(1 + value.size(-1))


class Decoder(nn.Module):
    def __init__(self, embedding_dim, shape):
        super().__init__()
        self.shape = torch.Size(shape)
        self.linear = torch.nn.Linear(embedding_dim, self.shape.numel())

    def forward(self, code):
        logits = self.linear(code)
        logits = logits.reshape(code.shape[:-1] + self.shape)
        logits = logits.softmax(dim=-1)
        logits.data.clamp_(min=torch.finfo(logits.dtype).eps)
        return logits


class Encoder(nn.Module):
    def __init__(self, embedding_dim, shape):
        super().__init__()
        self.shape = torch.Size(shape)
        self.linear = torch.nn.Linear(self.shape.numel(), embedding_dim)
        self.log_scale = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, probs):
        shape = probs.shape[:-len(self.shape)] + (self.shape.numel(),)
        loc = self.linear(probs.reshape(shape))
        scale = self.log_scale.exp()
        return loc, scale

    def log_prob(self, codes, state):
        loc, scale = self(state)
        return dist.Normal(loc, scale).log_prob(codes).sum()

    def expected_log_prob(self, codes, probs):
        assert len(codes) == len(probs)
        N, E = codes.shape
        N, C, D = probs.shape
        loc, scale = self(probs)
        mean_part = dist.Normal(loc, scale).log_prob(codes).sum()
        cov_sum = probs.sum(0).diag_embed() - einsum("nci,ncj->cij", probs, probs)
        linear = self.linear.weight.reshape(E, C, D)
        cov_part = einsum("e,eci,ecj,cij->", scale.pow(-2), linear, linear, cov_sum)
        return mean_part - 0.5 * cov_part


class BetheModel(PyroModule):
    """
    A phylogenetic tree model that marginalizes over tree structure and relaxes
    over the states of internal nodes.
    """
    max_plate_nesting = 2

    def __init__(self, leaf_times, leaf_logits, *,
                 embedding_dim=20, bp_iters=30, min_dt=0.01):
        super().__init__()
        assert leaf_times.dim() == 1
        assert (leaf_times[:-1] <= leaf_times[1:]).all()
        assert leaf_logits.dim() == 3
        assert leaf_logits.shape[:1] == leaf_times.shape
        assert isinstance(embedding_dim, int) and embedding_dim > 0
        assert isinstance(min_dt, float) and min_dt >= 0
        L, C, D = leaf_logits.shape

        self.num_nodes = 2 * L - 1
        self.num_leaves = L
        self.num_observations = L * C
        self.leaf_times = leaf_times
        self.leaf_logits = leaf_logits
        self.subs_model = JukesCantor69(dim=D)
        self.bp_iters = bp_iters
        self.embedding_dim = embedding_dim
        self.min_dt = min_dt
        self.decoder = Decoder(embedding_dim, (C, D))
        self.encoder = Encoder(embedding_dim, (C, D))

        self._initialize()

    def forward(self, mode="train"):
        L, C, D = self.leaf_logits.shape
        N = 2 * L - 1
        E = self.embedding_dim

        # Sample relaxed genetic sequences of all nodes, leaves + internal.
        with pyro.plate("nodes", N, dim=-2), pyro.plate("embedding", E, dim=-1):
            codes = pyro.sample("codes", dist.Normal(0, 1).mask(False))
            probs = self.decoder(codes)

        if mode in ("train", "pretrain"):
            # Condition on observations.
            pyro.factor("leaf_likelihood",
                        einsum("lcd,lcd->", probs[:L], self.leaf_logits))

            # Add hierarchical elbo terms.
            pyro.factor("entropy", -(probs * probs.log()).sum())
            pyro.factor("hierarchy", self.encoder.expected_log_prob(codes, probs))
        if mode == "pretrain":  # If we're training only self.decoder,
            return              # then we can ignore the rest of the model.

        # Sample constant coalescent rate from the Jeffreys prior.
        coalescent_rate = pyro.sample(
            "coalescent_rate",
            dist.TransformedDistribution(
                dist.ImproperUniform(constraints.real, (), ()),
                dist.transforms.ExpTransform()))

        # Sample times of internal nodes.
        internal_times = pyro.sample(
            "internal_times",
            UnorderedCoalescentTimes(self.leaf_times, coalescent_rate))
        times = pyro.deterministic("times",
                                   torch.cat([self.leaf_times, internal_times]))

        # Account for random tree structure.
        # TODO try running this in double precision.
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
        if self.min_dt > 0:
            # Ensure gradients wrt time are smooth.
            dt = (dt ** 2 + self.min_dt ** 2).sqrt()
        transition = self.subs_model.log_matrix_exp(dt)
        # Accumulate sufficient statistics over characters.
        stats = einsum("icd,jce->ijde", probs, probs)[v0, v1]
        sparse_logits = einsum("fde,fde->f", stats, transition)
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
        if name.endswith("coalescent_rate"):
            return torch.ones(())
        raise ValueError(f"unknown site: {name}")
