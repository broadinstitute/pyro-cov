import re

import torch


def _murmur64(h):
    """
    A cheap simple hash function : int64 -> int64.
    """
    h ^= h >> 33
    h *= -49064778989728563  # == 0xff51afd7ed558ccd
    h ^= h >> 33
    h *= -4265267296055464877  # == 0xc4ceb9fe1a85ec53
    h ^= h >> 33
    return h


def count_bits(bits):
    """
    Returns a lookup table for the ``count_bits()`` function on binary numbers
    ``bits``.
    """
    assert isinstance(bits, int) and bits > 0
    result = torch.zeros((2,) * bits, dtype=torch.float)
    for b in range(bits):
        index = [slice(None)] * bits
        index[b] = 1
        result[tuple(index)] += 1
    return result.reshape(-1)


class KmerSketcher:
    """
    Clustering via LSH of k-mers.
    """
    def __init__(self, *, min_k=2, max_k=6, bits=12):
        assert 1 <= min_k <= max_k <= 32
        assert bits < 23, "too many bits for float storage"
        self.min_k = min_k
        self.max_k = max_k
        self.bits = bits
        self._bit_mask = 0xFFFFFFFF >> (32 - self.bits)
        self._bits = 2 ** torch.arange(self.bits)
        self._kmer = 2 ** torch.arange(self.max_k * 2, dtype=torch.float)
        self._codebook = torch.full((256, 2), -1, dtype=torch.float)
        for i, base in enumerate("ACGT"):
            self._codebook[ord(base), 0] = i & 1
            self._codebook[ord(base), 1] = i & 2

    def string_to_soft_hash(self, string, out):
        assert out.shape == (self.bits,)
        assert out.dtype == torch.float
        out.fill_(0)
        for substring in re.findall("[ACGT]+", string):
            seq = self._codebook[list(map(ord, substring))].reshape(-1)
            for k in range(self.min_k, min(self.max_k + 1, len(substring))):
                n = len(substring) - k + 1
                kmers = seq.as_strided((n, k * 2), (2, 1))
                ints = (kmers @ self._kmer[:k * 2]).long()
                ints = _murmur64(ints)
                signs = (ints[:, None] & self._bits).bool().float().sum(0) * 2 - n
                out += signs

    def soft_to_hard_hashes(self, soft_hashes):
        assert soft_hashes.dim() == 2
        assert soft_hashes.size(-1) == self.bits
        signs = (soft_hashes > soft_hashes.median(0).values).float()
        powers_of_two = 2 ** torch.arange(self.bits, dtype=torch.float)
        hard_hashes = (signs @ powers_of_two).long()
        return hard_hashes

    def find_clusters(self, hard_hashes, *, radius=2):
        assert hard_hashes.dim() == 1
        assert radius >= 1

        # Construct a kernel for non-maximum supression.
        kernel = count_bits(self.bits) <= radius
        k = torch.arange(2 ** self.bits)

        # Aggregate hash counts.
        counts = torch.zeros(2 ** self.bits, dtype=torch.float)
        ones = torch.ones(()).expand_as(hard_hashes)
        counts.scatter_add_(-1, hard_hashes, ones)

        # Greedily extract clusters.
        clusters = []
        while counts.max() > 0:
            c = counts.max(0).indices.item()
            counts[kernel[k ^ c]] = 0
            clusters.append(c)
        return torch.tensor(clusters)

    def cdist(self, hard_hashes, clusters):
        assert hard_hashes.dim() == 1
        assert clusters.dim() == 1
        return count_bits(self.bits)[hard_hashes[:, None] ^ clusters]
