# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict, namedtuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SubClade = namedtuple("_SubClade", ("clade", "name"))


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
        assert (
            times[..., :-1] <= times[..., 1:]
        ).all(), "expected nodes ordered by time"
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
        sign = torch.ones_like(self.parents).scatter_(-1, _parents, -1.0)
        num_lineages = sign.flip(-1).cumsum(-1).flip(-1)
        return num_lineages

    def hash_topology(self):
        """
        Returns a hashable binary tree represented as nested frozensets.
        """
        if self.batch_shape:
            return tuple(p.hash_topology() for p in self)
        trees = defaultdict(list)
        for leaf, v in enumerate(self.leaves.tolist()):
            trees[v] = leaf
        for v, parent in enumerate(self.parents[1:].tolist()):
            trees[parent].append(trees[v + 1])

        def freeze(x):
            if isinstance(x, int):
                return x
            assert len(x) == 2
            return frozenset(map(freeze, x))

        return freeze(trees[0])

    def time_mrca(self):
        """
        Computes all-pairs times to most recent common ancestor.
        """
        if self.batch_shape:
            return torch.stack([p.time_mrca() for p in self])
        descendents = torch.eye(self.num_nodes, dtype=torch.bool)
        children = defaultdict(list)
        result = self.times.new_zeros(descendents.shape)
        result.diagonal()[:] = self.times
        for c in range(self.num_nodes - 1, 0, -1):
            p = self.parents[c].item()
            descendents[p] |= descendents[c]
            children[p].append(c)
            if len(children[p]) == 2:
                c1, c2 = children[p]
                d1 = descendents[c1]
                d2 = descendents[c2]
                mask = d1 & d2[:, None]
                mask[p] |= d1 | d2
                mask |= mask.T
                result[mask] = self.times[p]
        return result

    def leaf_time_mrca(self):
        """
        Computes times to most recent common ancestors for all leaves.
        For phylogenies whose leaves are all at time 0, the result is an
        ultrametric.
        """
        if self.batch_shape:
            return torch.stack([p.leaf_time_mrca() for p in self])
        result = self.time_mrca()
        return result[self.leaves[..., None], self.leaves]

    @staticmethod
    def stack(phylogenies):
        """
        :param iterable phylogenies: An iterable of :class:`Phylogeny` objects
            of identical shape.
        :returns: A batched phylogeny.
        :rtype: Phylogeny
        """
        phylogenies = list(phylogenies)
        kwargs = {
            name: torch.stack([getattr(x, name) for x in phylogenies])
            for name in Phylogeny._fields
        }
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
        clade_to_children = defaultdict(list)
        for clade in clades:
            time = clade_to_time[clade]
            for child in clade:
                clade_to_time[child] = time + get_branch_length(child)
                clade_to_parent[child] = clade
                clade_to_children[clade].append(child)

        # Binarize the tree.
        for parent, children in clade_to_children.items():
            while len(children) > 2:
                c1 = children.pop()
                c2 = children.pop()
                c12 = _SubClade(parent, f"{parent.name}.{len(children):0>4d}")
                clades.append(c12)
                children.append(c12)
                clade_to_time[c12] = clade_to_time[parent]
                clade_to_parent[c1] = c12
                clade_to_parent[c2] = c12
                clade_to_parent[c12] = parent
        del clade_to_children

        # Serialize clades.
        clades.sort(key=lambda c: (clade_to_time[c], str(c.name)))
        assert clades[0] not in clade_to_parent, "invalid root"
        clade_to_id = {clade: i for i, clade in enumerate(clades)}
        times = torch.tensor([float(clade_to_time[clade]) for clade in clades])
        parents = torch.tensor(
            [-1] + [clade_to_id[clade_to_parent[clade]] for clade in clades[1:]]
        )

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
            return Phylogeny.stack(
                Phylogeny.generate(num_leaves) for _ in range(num_samples)
            )
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
        return Phylogeny.from_unsorted(times, parents, leaves)

    @staticmethod
    def sort(times, parents, leaves):
        if times.dim() > 1:
            raise NotImplementedError("Phylogeny.sort() does not support batching")
        num_nodes = times.size(-1)
        times, new2old = times.sort(-1)
        old2new = torch.empty(num_nodes, dtype=torch.long)
        old2new[new2old] = torch.arange(num_nodes)
        leaves = old2new[leaves]
        parents = old2new[parents[new2old]]
        parents[0] = -1
        return Phylogeny(times, parents, leaves), old2new, new2old

    @staticmethod
    def from_unsorted(times, parents, leaves):
        return Phylogeny.sort(times, parents, leaves)[0]
