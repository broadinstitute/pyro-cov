# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

import pyro.distributions as dist
import torch

from pyrocov.phylo import Phylogeny


class SoftmaxTree(dist.Distribution):
    """
    Samples a :class:`~pyrocov.phylo.Phylogeny` given parameters of a tree
    embedding.

    :param torch.Tensor bit_times: Tensor of times of each bit in the
        embedding.
    :param torch.Tensor logits: ``(num_leaves, num_bits)``-shaped tensor
        parametrizing the independent Bernoulli distributions over bits
        in each leaf's embedding.
    """

    has_rsample = True  # only wrt times, not parents

    def __init__(self, leaf_times, bit_times, logits):
        assert leaf_times.dim() == 1
        assert bit_times.dim() == 1
        assert logits.dim() == 2
        assert logits.shape == leaf_times.shape + bit_times.shape
        self.leaf_times = leaf_times
        self.bit_times = bit_times
        self._bernoulli = dist.Bernoulli(logits=logits)
        super().__init__()

    @property
    def probs(self):
        return self._bernoulli.probs

    @property
    def logits(self):
        return self._bernoulli.logits

    @property
    def num_leaves(self):
        return self.logits.size(0)

    @property
    def num_bits(self):
        return self.logits.size(1)

    def entropy(self):
        return self._bernoulli.entropy().sum([-1, -2])

    def sample(self, sample_shape=torch.Size()):
        if sample_shape:
            raise NotImplementedError
        raise NotImplementedError("TODO")

    def rsample(self, sample_shape=torch.Size()):
        if sample_shape:
            raise NotImplementedError
        bits = self._bernoulli.sample()
        num_leaves, num_bits = bits.shape
        phylogeny = _decode(self.leaf_times, self.bit_times, bits, self.probs)
        return phylogeny

    def log_prob(self, phylogeny):
        """
        :param ~pyrocov.phylo.Phylogeny phylogeny:
        """
        return self.entropy()  # Is this right?


# TODO Implement a C++ version.
# This costs O(num_bits * num_leaves) sequential time.
def _decode(leaf_times, bit_times, bits, probs):
    # Sort bits by time.
    bit_times, index = bit_times.sort()
    bits = bits[..., index]
    probs = probs[..., index]

    # Construct internal nodes.
    num_leaves, num_bits = bits.shape
    assert num_leaves >= 2
    times = torch.cat([leaf_times, leaf_times.new_empty(num_leaves - 1)])
    parents = torch.empty(2 * num_leaves - 1, dtype=torch.long)
    leaves = torch.arange(num_leaves)

    next_id = num_leaves

    def get_id():
        nonlocal next_id
        next_id += 1
        return next_id - 1

    root = get_id()
    parents[root] = -1
    partitions = [{frozenset(range(num_leaves)): root}]
    for t, b in zip(*bit_times.sort()):
        partitions.append({})
        for partition, p in partitions[-2].items():
            children = defaultdict(set)
            for n in partition:
                bit = bits[n, b].item()
                # TODO Clamp bit if t is later than node n.
                children[bit].add(n)
            if len(children) == 1:
                partitions[-1][partition] = p
                continue
            assert len(children) == 2
            for child in children.values():
                if len(child) == 1:
                    # Terminate at a leaf.
                    c = child.pop()
                else:
                    # Create a new internal node.
                    c = get_id()
                    partitions[-1][frozenset(child)] = c
                parents[c] = p
            times[p] = t
    # Create binarized fans for remaining leaves.
    for partition, p in partitions[-1].items():
        t = times[torch.tensor(list(partition))].min()
        times[p] = t
        partition = set(partition)
        while len(partition) > 2:
            c = get_id()
            times[c] = t
            parents[c] = p
            parents[partition.pop()] = p
            p = c
        parents[partition.pop()] = p
        parents[partition.pop()] = p
        assert not partition

    return Phylogeny.from_unsorted(times, parents, leaves)
