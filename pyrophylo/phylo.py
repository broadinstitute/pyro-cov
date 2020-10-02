import logging
import math

import numpy as np
import pyro.distributions as dist
import torch
from pyro.distributions import constraints, is_validation_enabled
from pyro.distributions.util import broadcast_shape
from pyro.ops.special import safe_log
from pyro.util import warn_if_nan

from .util import deduplicate_tensor, weak_memoize_by_id

logger = logging.getLogger(__name__)


class Phylogeny:
    """
    Tensor data structure to represent a (batched) phylogenetic tree.

    The tree is timed and is assumed to have only binary nodes; polysemy is
    represented as multiple binary nodes but with zero branch length.

    :param Tensor times: float tensor of times of each node. Must be ordered.
    :param Tensor parents: int tensor of parent id of each node. The root node
        must be first and have null id ``-1``.
    :param Tensor leaves: int tensor of ids of all leaf nodes.
    """
    _fields = ("times", "parents", "leaves")

    def __init__(self, times, parents, leaves):
        num_nodes = times.size(-1)
        assert num_nodes % 2 == 1, "expected odd number of nodes"
        num_leaves = (num_nodes + 1) // 2
        assert parents.shape == times.shape
        assert leaves.shape == times.shape[:-1] + (num_leaves,)
        assert (times[..., :-1] <= times[..., 1:]).all(), "expected nodes ordered by time"
        assert (parents[..., 0] == -1).all(), "expected root node first"
        assert (parents[..., 1:] >= 0).all(), "multiple root nodes"
        if __debug__:
            _parents = parents[..., 1:]
            is_leaf_1 = torch.ones_like(parents, dtype=torch.bool)
            is_leaf_1.scatter_(-1, _parents, False)
            is_leaf_2 = torch.zeros_like(is_leaf_1)
            is_leaf_2.scatter_(-1, leaves, True)
            assert (is_leaf_1.sum(-1) == num_leaves).all()
            assert (is_leaf_2.sum(-1) == num_leaves).all()
            assert (is_leaf_2 == is_leaf_1).all()
        super().__init__()
        self.times = times
        self.parents = parents
        self.leaves = leaves

    @property
    def num_nodes(self):
        return self.times.size(-1)

    @property
    def num_leaves(self):
        return self.leaves.size(-1)

    @property
    def batch_shape(self):
        return self.times.shape[:-1]

    def __len__(self):
        return self.batch_shape[0]

    def __getitem__(self, index):
        kwargs = {name: getattr(self, name)[index] for name in self._fields}
        return Phylogeny(**kwargs)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def contiguous(self):
        kwargs = {name: getattr(self, name).contiguous() for name in self._fields}
        return Phylogeny(**kwargs)

    def num_lineages(self):
        _parents = self.parents[..., 1:]
        sign = torch.ones_like(self.parents).scatter_(-1, _parents, -1.)
        num_lineages = sign.flip(-1).cumsum(-1).flip(-1)
        return num_lineages

    @staticmethod
    def stack(phylogenies):
        """
        :param iterable phylogenies: An iterable of :class:`Phylogeny` objects
            of identical shape.
        :returns: A batched phylogeny.
        :rtype: Phylogeny
        """
        phylogenies = list(phylogenies)
        kwargs = {name: torch.stack([getattr(x, name) for x in phylogenies])
                  for name in Phylogeny._fields}
        return Phylogeny(**kwargs)

    @staticmethod
    def from_bio_phylo(tree):
        """
        Builds a :class:`Phylogeny` object from a biopython tree structure.

        :param Bio.Phylo.BaseTree.Clade tree: A phylogenetic tree.
        :returns: A single phylogeny.
        :rtype: Phylogeny
        """
        # Compute time as cumulative branch length.
        def get_branch_length(clade):
            branch_length = clade.branch_length
            return 1.0 if branch_length is None else branch_length

        # Collect times and parents.
        clades = list(tree.find_clades())
        clade_to_time = {tree.root: get_branch_length(tree.root)}
        clade_to_parent = {}
        for clade in clades:
            time = clade_to_time[clade]
            for child in clade:
                clade_to_time[child] = time + get_branch_length(child)
                clade_to_parent[child] = clade
        clades.sort(key=lambda c: (clade_to_time[c], c.name))
        assert clades[0] not in clade_to_parent, "invalid root"
        # TODO binarize the tree
        clade_to_id = {clade: i for i, clade in enumerate(clades)}
        times = torch.tensor([float(clade_to_time[clade]) for clade in clades])
        parents = torch.tensor([-1] + [clade_to_id[clade_to_parent[clade]]
                                       for clade in clades[1:]])

        # Construct leaf index ordered by clade.name.
        leaves = [clade for clade in clades if len(clade) == 0]
        leaves.sort(key=lambda clade: clade.name)
        leaves = torch.tensor([clade_to_id[clade] for clade in leaves])

        return Phylogeny(times, parents, leaves)

    @staticmethod
    def generate(num_leaves, *, num_samples=None):
        """
        Generate a random (arbitrarily distributed) phylogeny for testing.
        """
        if num_samples is not None:
            return Phylogeny.stack(Phylogeny.generate(num_leaves)
                                   for _ in range(num_samples))
        num_nodes = 2 * num_leaves - 1
        times = torch.randn(num_nodes)
        nodes = list(range(num_leaves))
        parents = torch.zeros(num_nodes, dtype=torch.long)
        for w in range(num_leaves, num_nodes):
            i, j = np.random.choice(len(nodes), 2, replace=False)
            u = nodes[i]
            v = nodes[j]
            nodes[i] = w
            del nodes[j]
            parents[u] = w
            parents[v] = w
            times[w] = torch.min(times[u], times[v]) - torch.rand(()) / num_leaves
        assert len(nodes) == 1
        leaves = torch.arange(num_leaves)

        # Sort results.
        times, new2old = times.sort()
        old2new = torch.empty(num_nodes, dtype=torch.long)
        old2new[new2old] = torch.arange(num_nodes)
        assert new2old[0] == nodes[0]
        leaves = old2new[leaves]
        parents = old2new[parents[new2old]]
        parents[0] = -1
        return Phylogeny(times, parents, leaves)


class MarkovTree(dist.TorchDistribution):
    """
    Joint distribution over states of leaves of a phylogenetic tree whose
    branches define a discrete-state Markov process.

    :param Phylogeny phylo: A phylogeny or batch of phylogenies.
    :param Tensor state_trans: Either a homogeneous reverse-time state
        transition matrix, or a heterogeneous grid of ``T`` transition matrices
        applying to time intervals ``(-inf,1]``, ``(1,2]``, ..., ``(T-1,inf)``.
        These are oriented such that ``earlier_probs = later_probs @
        state_trans``.
    :param str method: Either "likeliood" for vectorized computation or "naive"
        for sequential computation. The "likelihood" method rounds times to the
        nearest integer; the "naive" method allows non-integer times but
        linearly approximates matrix exponentials within short time intervals.
    """
    arg_constraints = {
        "transition": constraints.IndependentConstraint(constraints.simplex, 2),
    }

    def __init__(self, phylogeny, transition, *, method="likelihood"):
        assert isinstance(transition, torch.Tensor)
        assert isinstance(phylogeny, Phylogeny)
        assert transition.dim() >= 2
        self.num_states = transition.size(-1)
        assert transition.size(-2) == self.num_states
        assert isinstance(method, str)
        self.method = method
        self.phylogeny = phylogeny
        self.transition = transition
        batch_shape = broadcast_shape(phylogeny.batch_shape, transition.shape[:-3])
        event_shape = torch.Size([phylogeny.num_leaves])
        super().__init__(batch_shape, event_shape)

    # TODO def expand(self, shape): ...

    @constraints.dependent_property
    def support(self):
        return constraints.IndependentConstraint(
            constraints.integer_interval(0, self.num_states), 1)

    def log_prob(self, leaf_state):
        """
        :param Tensor leaf_state: int tensor of states of all leaf nodes.
        """
        if self.method == "naive":
            return markov_log_prob(self.phylogeny, leaf_state, self.transition)
        elif self.method == "likelihood":
            # This likelihood object is memoized across model executions.
            leaf_state = deduplicate_tensor(leaf_state)  # FIXME leaks memory.
            likelihood = markov_tree_likelihood(self.phylogeny, leaf_state)
            return likelihood(self.transition)
        else:
            raise NotImplementedError(f"Unknown implementation: {self.method}")


def markov_log_prob(phylo, leaf_state, state_trans):
    """
    Compute the marginal log probability of a Markov tree with given edge
    transitions and leaf observations. This can be used for either mutation or
    phylogeographic mugration models, but does not allow state-dependent
    reproduction rate as in the structured coalescent [1].

    **References**

    [1] T. Vaughan, D. Kuhnert, A. Popinga, D. Welch, A. Drummond (2014)
        `Efficient Bayesian inference under the structured coalescent`
        https://academic.oup.com/bioinformatics/article/30/16/2272/2748160

    :param Phylogeny phylo: A phylogeny or batch of phylogenies.
    :param Tensor leaf_state: int tensor of states of all leaf nodes.
    :param Tensor state_trans: Either a homogeneous reverse-time state
        transition matrix, or a heterogeneous grid of ``T`` transition matrices
        applying to time intervals ``(-inf,1]``, ``(1,2]``, ..., ``(T-1,inf)``.
        These are oriented such that ``earlier_probs = later_probs @
        state_trans``.
    :returns: Marginal log probability of data ``leaf_state``.
    :rtype: Tensor
    """
    batch_shape = phylo.batch_shape
    if batch_shape:
        # TODO vectorize.
        return torch.stack([markov_log_prob(p, leaf_state, state_trans)
                            for p in phylo])
    num_nodes = phylo.num_nodes
    num_leaves = phylo.num_leaves
    num_states = state_trans.size(-1)
    assert leaf_state.shape == (num_leaves,)
    assert state_trans.dim() in (2, 3)  # homogeneous, heterogeneous
    assert state_trans.shape[-2:] == (num_states, num_states)
    if is_validation_enabled():
        constraints.simplex.check(state_trans)
    times = phylo.times
    parents = phylo.parents
    leaves = phylo.leaves
    finfo = torch.finfo(state_trans.dtype)

    # Convert (leaves,leaf_state) to initial state log density.
    logp = state_trans.new_zeros(num_nodes, num_states)
    logp[leaves] = -math.inf
    logp[leaves, leaf_state] = 0
    logp = list(logp)  # Work around non-differentiability of in-place update.

    # Dynamic programming along the tree.
    for i in range(-1, -num_nodes, -1):
        j = parents[i]
        logp[j] = logp[j] + _interpolate_lmve(times[j], times[i], state_trans, logp[i])
    logp = logp[0].clamp(min=finfo.min).logsumexp(dim=-1)
    warn_if_nan(logp, "logp")
    return logp


def _mpm(m, t, v):
    """
    Like ``v @ m.matrix_power(t)`` but
    approximates fractional powers via linear interpolation.
    """
    assert t >= 0
    if t == 0:
        return v
    t_int = t.floor()
    t_frac = t - t_int
    if t_int:
        v = v @ m.matrix_power(int(t_int))
    if t_frac:
        v = v + t_frac * (v @ m - v)  # linear approximation
    return v


def _interpolate_mm(t0, t1, m, v):
    """
    Like ``v @ m.matrix_power(t1 - t2)`` but allows time-varying ``m``.

    ``m`` specifies a piecewise constant unit-time transition matrix over time
    intervals ``(-inf,1]``, ``(1,2]``, ..., ``(T-1,inf)``, where ``T =
    m.size(-3)``. Note if ``h`` is the differential transition matrix, then ``m
    = exp(h)``.
    """
    if is_validation_enabled():
        assert (t0 <= t1).all()

    # Homogeneous.
    if m.dim() == 2:
        return _mpm(m, t1 - t0, v)
    T = m.size(-3)
    if T == 1:
        return _mpm(m[..., 0, :, :], t1 - t0, v)

    # Heterogeneous.
    assert t0.numel() == 1 and t1.numel() == 1
    # Handle trivial cases.
    if t0 >= T - 1:  # After grid.
        return _mpm(m[..., T - 1, :, :], t1 - t0, v)
    if t1 <= 1:  # Before grid.
        return _mpm(m[..., 0, :, :], t1 - t0, v)
    if t1.ceil() - t0.floor() <= 1:  # Single grid cell.
        t = int(t0.floor())
        return _mpm(m[..., t, :, :], t1 - t0, v)

    # Handle case when (t0,t1) spans at least two grid cells.
    t1_floor = min(T - 1, int(t1.floor()))
    t0_ceil = max(1, int(t0.ceil()))

    # Note this assumes m encodes a reverse-time transition.
    t = t1_floor
    v = _mpm(m[..., t, :, :], t1 - t1_floor, v)

    for t in range(t1_floor - 1, t0_ceil - 1, -1):
        v = v @ m[..., t, :, :]

    t = t0_ceil - 1
    v = _mpm(m[..., t, :, :], t0_ceil - t0, v)

    return v


def _interpolate_lmve(t0, t1, matrix, log_vector):
    """
    Like ``log(exp(v) @ m.matrix_power(t1-t2))`` but allows time-varying ``m``.
    """
    if t0 == t1:
        return log_vector
    finfo = torch.finfo(log_vector.dtype)
    shift = log_vector.detach().max(-1, True).values.clamp_(min=finfo.min)
    v = (log_vector - shift).exp()
    v = _interpolate_mm(t0, t1, matrix, v)
    return safe_log(v) + shift


class MarkovTreeLikelihood:
    """
    Like the :class:`MarkovTree` distribution but allowing a different form of
    partial evaluation.  This is differentiable wrt ``state_trans`` but not wrt
    ``phylo`` or ``leaf_state``.

    The following are equivalent::

        dist = MarkovTree(phylo, state_trans)
        return dist.log_prob(leaf_state)

        like = MarkovTreeLikelihood(phylo, leaf_state, state_trans.size(-1))
        return like(state_trans)

    This is invariant under adding new single-child nodes along branches;
    thus polysemy can be faithfully encoded by converting a >2-way branching
    tree to a binary tree by adding fake nodes.

    :param Phylogeny phylo: A phylogeny or batch of phylogenies.
    :param Tensor leaf_state: int tensor of states of all leaf nodes.
        Nonnegative values are observed. The value ``-1`` denods missing data.
    """
    def __init__(
        self,
        phylo,  # batched Phylogeny
        leaf_state,  # leaf_id -> category
    ):
        times = phylo.times.round().long()  # batch * node_id -> time
        batch_size, num_nodes = times.shape
        parents = phylo.parents  # batch * node_id -> {-1} + node_id
        leaves = phylo.leaves  # leaf_id -> node_id
        T0 = times.min().item()
        T1 = times.max().item()

        # Drop leaves that have not been observed.
        observed = (leaf_state != -1).nonzero(as_tuple=True)[-1]
        if observed.numel():
            leaves = leaves[..., observed]
            leaf_state = leaf_state[..., observed]

        # Flatten the batch of trees to a forest.
        # Let bnode_id = batch * num_nodes + node_id.
        #     bleaf_id = batch * num_leaves + leaf_id.
        times = times.reshape(-1)
        batch_expand = (torch.arange(batch_size) * num_nodes).unsqueeze(-1)
        parents = parents + batch_expand
        parents[:, 0] = -1  # Retain nonce.
        parents = parents.reshape(-1)
        leaves = (leaves + batch_expand).reshape(-1)
        leaf_state = leaf_state.expand(batch_size, -1).reshape(-1)
        assert (parents == -1).sum() == batch_size
        # Now times : bnode_id -> time
        #     parents : bnode_id -> bnode_id
        #     leaves : Set[bnode_id]
        #     leaf_state : bleaf_id -> category

        # Collapse parent-grandparent relationships of zero duration,
        # as required by the propagation algorithm in .__call__().
        has_parent = (parents != -1)
        while True:
            has_gparent = has_parent & has_parent[parents]
            ptimes = times[parents]
            gptimes = ptimes[parents]
            children = (has_gparent & (gptimes == ptimes)).nonzero()
            if not children.numel():
                break
            logger.debug(f"collapsing parent-grandparent of {len(children)} children")
            parents[children] = parents[parents[children]]

        # Sort and stratify leaves by time. Strata are represented as
        # leaf_strata : time -> Set[bleaf_id].
        order = times[leaves].sort().indices
        leaves = leaves[order]
        leaf_state = leaf_state[order]
        ones = torch.ones(()).expand_as(leaves)
        leaf_strata = torch.zeros(T1 - T0 + 2) \
                           .scatter_add_(0, times[leaves] - T0 + 1, ones) \
                           .cumsum(0).long()
        assert leaf_strata[0] == 0
        assert leaf_strata[-1] == len(leaves)

        # Save.
        self.batch_size = batch_size
        self.sizes = T0, T1
        self.times = times
        self.parents = parents
        self.leaves = leaves
        self.leaf_state = leaf_state
        self.leaf_strata = leaf_strata

    def __call__(
        self,
        state_trans,  # time * category * category -> prob
    ):
        """
        :param Tensor state_trans: Either a homogeneous reverse-time state
            transition matrix, or a heterogeneous grid of ``T`` transition
            matrices applying to time intervals ``(-inf,1]``, ``(1,2]``, ...,
            ``(T-1,inf)``. These are oriented such that ``earlier_probs =
            later_probs @ state_trans``.
        :returns: Marginal log probability of data ``leaf_state``.
        :rtype: Tensor
        """
        T, N, N = state_trans.shape
        T0, T1 = self.sizes
        times = self.times
        parents = self.parents
        leaves = self.leaves
        leaf_state = self.leaf_state
        leaf_strata = self.leaf_strata
        finfo = torch.finfo(state_trans.dtype)

        # Sequentially propagate backward in time.
        nodes = parents.new_empty(0)
        logps = state_trans.new_empty(0, N)
        for t in range(T1, T0 - 1, -1):
            # Update each existing lineage's state distribution independently.
            if nodes.numel():
                shift = logps.detach().max(-1, True).values.clamp_(min=finfo.min)
                v = (logps - shift).exp()
                v = v @ state_trans[max(0, min(T - 1, t))]
                logps = safe_log(v) + shift  # TODO use safe_log?
                # TODO Lazily convert between (exp,shift) <--> log representations.

            # Add new leaf lineages ("birth").
            beg, end = leaf_strata[t - T0:t - T0 + 2]
            new_nodes = leaves[beg:end]
            if new_nodes.numel():
                new_logps = _log_one_hot(new_nodes.shape + (N,), leaf_state[beg:end])
                nodes = torch.cat([nodes, new_nodes], 0)
                logps = torch.cat([logps, new_logps], 0)

            # Merge lineages, replacing children by parents ("death").
            pi = parents[nodes]
            to_merge = times[pi] == t
            if to_merge.any():
                nodes, order = torch.where(to_merge, pi, nodes).unique(return_inverse=True)
                order = order.unsqueeze(-1).expand_as(logps)
                logps = logps.new_zeros(nodes.shape + (N,)).scatter_add(0, order, logps)
        assert nodes.shape == (self.batch_size,)

        # Marginalize over root states.
        logps = logps.clamp(min=finfo.min).logsumexp(dim=-1)
        logps = logps[nodes.sort().indices].contiguous()
        print(f"DEBUG logps.shape = {logps.shape}")
        warn_if_nan(logps, "logps")
        return logps


markov_tree_likelihood = weak_memoize_by_id(MarkovTreeLikelihood)


def _log_one_hot(shape, index):
    return torch.full(shape, -math.inf).scatter_(-1, index.unsqueeze(-1), 0.)
