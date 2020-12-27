import logging
import re
from collections import namedtuple

import torch

logger = logging.getLogger(__name__)

MeanStd = namedtuple("MeanStd", ("mean", "std"))


def murmur64(h):
    """
    A cheap simple hash function : int64 -> int64.
    """
    if isinstance(h, torch.Tensor):
        h ^= h >> 33
        h *= -49064778989728563  # == 0xff51afd7ed558ccd
        h ^= h >> 33
        h *= -4265267296055464877  # == 0xc4ceb9fe1a85ec53
        h ^= h >> 33
    else:
        h ^= h >> 33
        h *= 0xff51afd7ed558ccd
        h &= 0xffffffffffffffff
        h ^= h >> 33
        h *= 0xc4ceb9fe1a85ec53
        h &= 0xffffffffffffffff
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


class AMSSketcher:
    """
    Clustering via AMS sketching of k-mer counts followed by LSH.
    """
    def __init__(self, *, min_k=2, max_k=12, bits=16, backend="cpp"):
        assert 1 <= min_k <= max_k
        assert max_k * 2 <= 64, "max_k is too large"
        assert bits <= 24, "too many bits for float storage"
        self.min_k = min_k
        self.max_k = max_k
        self.bits = bits
        self.backend = backend

    def string_to_soft_hash(self, strings, out):
        assert out.dim() == 1
        assert out.dtype == torch.float
        if isinstance(strings, str):
            strings = re.findall("[ACGT]+", strings)
        if self.backend == "python":
            impl = string_to_soft_hash
        elif self.backend == "cpp":
            impl = _get_cpp_module().string_to_soft_hash
        else:
            raise NotImplementedError(f"Unsupported backend: {self.backend}")
        out.fill_(0)
        for string in strings:
            impl(self.min_k, self.max_k, string, out)

    def soft_to_hard_hashes(self, soft_hashes):
        assert soft_hashes.dim() == 2
        soft_hashes = soft_hashes[:, :self.bits]
        signs = (soft_hashes > soft_hashes.median(0).values).float()
        powers_of_two = 2 ** torch.arange(self.bits, dtype=torch.float)
        hard_hashes = (signs @ powers_of_two).long()
        return hard_hashes

    def find_clusters(self, hard_hashes, *, radius=4):
        assert hard_hashes.dim() == 1
        assert radius >= 1

        # Aggregate hash counts.
        counts = torch.zeros(2 ** self.bits, dtype=torch.float)
        ones = torch.ones(()).expand_as(hard_hashes)
        counts.scatter_add_(-1, hard_hashes, ones)

        # Add counts from single and double bit flips via inclusion-exclusion.
        B = self.bits
        B2 = B * (B - 1) / 2
        counts = counts.reshape((2,) * B)
        conv = counts.clone()
        for b in range(B):
            conv -= counts.sum(b, True) / B
            for b2 in range(b):
                conv += counts.sum([b, b2], True) / B2
        counts = conv.reshape(-1)

        # Greedily detect clusters, suppressing nearby maxima.
        mask = count_bits(self.bits) <= radius
        k = torch.arange(2 ** self.bits)
        clusters = []
        while counts.max() > 0:
            c = counts.max(0).indices.item()
            counts[mask[k ^ c]] = 0
            clusters.append(c)
        return torch.tensor(clusters)

    def cdist(self, hard_hashes, clusters):
        assert hard_hashes.dim() == 1
        assert clusters.dim() == 1
        return count_bits(self.bits)[hard_hashes[:, None] ^ clusters]


class ClockSketch:
    def __init__(self, clocks, count):
        assert clocks.shape[:-1] == count.shape
        assert clocks.dtype == torch.int8
        assert count.dtype == torch.int16
        self.clocks = clocks
        self.count = count

    def __getitem__(self, i):
        return ClockSketch(self.clocks[i], self.count[i])

    def __setitem__(self, i, value):
        self.clocks[i] = value.clocks
        self.count[i] = value.count

    def __len__(self):
        return len(self.count)

    @property
    def shape(self):
        return self.count.shape

    def clone(self):
        return ClockSketch(self.clocks.clone(), self.count.clone())


class ClockSketcher:
    """
    Sketches each bag of k-mers as bank of 8-bit clocks plus a single total
    k-mer counter.
    """
    def __init__(self, k, *, num_clocks=256, backend="cpp"):
        assert num_clocks > 0 and num_clocks % 64 == 0
        self.k = k
        self.num_clocks = num_clocks
        self.backend = backend

    def init_sketch(self, *shape):
        clocks = torch.zeros(shape + (self.num_clocks,), dtype=torch.int8)
        count = torch.zeros(shape, dtype=torch.int16)
        return ClockSketch(clocks, count)

    def string_to_hash(self, strings, sketch):
        assert sketch.shape == ()
        if isinstance(strings, str):
            strings = re.findall("[ACGT]+", strings)
        if self.backend == "python":
            impl = string_to_clock_hash
        elif self.backend == "cpp":
            impl = _get_cpp_module().string_to_clock_hash
        else:
            raise NotImplementedError(f"Unsupported backend: {self.backend}")
        for string in strings:
            impl(self.k, string, sketch.clocks, sketch.count)
        sketch.clocks.mul_(2).sub_(sketch.count)

    def cdiff(self, x, y):
        clocks = x.clocks.unsqueeze(-2) - y.clocks.unsqueeze(-3)
        count = x.count.unsqueeze(-1) - y.count.unsqueeze(-2)
        return ClockSketch(clocks, count)

    def estimate_set_difference(self, x, y):
        r"""
        Estimates the multiset difference ``|x\y|``.
        Returns the mean and standard deviation of the estimate.
        """
        clocks = x.clocks.unsqueeze(-2) - y.clocks.unsqueeze(-3)
        count = x.count.unsqueeze(-1) - y.count.unsqueeze(-2)

        # Use a moment-matching estimator with Gaussian approximation.
        V_hx_minus_hy = clocks.float().square_().mean(-1)
        E_x_minus_y = count.float().add_(V_hx_minus_hy).clamp_(min=0).mul_(0.5)
        std_x_minus_y = V_hx_minus_hy.mul_((0.5 / self.num_clocks) ** 0.5)

        # Give up in case of numerical overflow.
        overflow = (count.abs_() > 85).any(-1)
        std_x_minus_y[overflow] = float(x.count.max())
        return MeanStd(E_x_minus_y, std_x_minus_y)

    def set_difference(self, x, y):
        r"""
        Estimates the multiset difference ``|x\y|``.
        """
        # An optimized version of estimate_set_difference().
        clocks = x.clocks.unsqueeze(-2) - y.clocks.unsqueeze(-3)
        count = x.count.unsqueeze(-1) - y.count.unsqueeze(-2)
        V_hx_minus_hy = clocks.float().square_().mean(-1)
        E_x_minus_y = V_hx_minus_hy.add_(count).clamp_(min=0).mul_(0.5)
        return E_x_minus_y

    def find_clusters(self, data, num_clusters, *,
                      radius=100, shuffle=True,
                      epochs=10, batch_size=100, log_every=10000):
        r"""
        Greedy clustering algorithm by reservoir sampling.
        """
        assert len(data.clocks) == len(data.count)
        N, = data.shape
        K = min(N, num_clusters)
        B = batch_size
        log_every = (log_every + B - 1) // B * B
        if shuffle:
            data = data[torch.randperm(N)]

        clusters = data[N - K:].clone()
        weights = torch.ones(K, dtype=torch.float)
        probs = torch.ones(B, K + 1, dtype=torch.float)
        num_batches = (N + B - 1) // B
        num_steps = num_batches * epochs
        for epoch in range(epochs):
            for i in range(0, N, B):
                x = data[i: i + B]
                b = len(x)
                weights *= 1 - 1 / N
                x_minus_c = self.set_difference(x, clusters)
                probs[:b, :K].copy_(x_minus_c).mul_(-1 / radius).exp_().mul_(weights)
                c = probs[:b].multinomial(1).squeeze()

                existing = c < K
                replaced = ~existing
                num_replaced = replaced.sum().item()
                if existing.any():
                    # Add weight to random existing clusters.
                    weights.scatter_add_(0, c % K, existing.float())
                if num_replaced:
                    # Replace random existing cluster with the datum.
                    c = weights.reciprocal().multinomial(num_replaced)
                    clusters[c] = x[replaced]
                    weights[c] = 1

                if log_every and i % log_every == 0:
                    step = i // B + num_batches * epoch
                    p = weights / weights.sum()
                    perplexity = p.log().neg().mul(p).sum().exp()
                    logger.info(f"step {step: >5d}/{num_steps} "
                                f"weight = {weights.sum():0.1f}\t"
                                f"perplexity = {perplexity:0.2f}")

        weights, i = weights.sort(descending=True)
        clusters = clusters[i]
        return clusters, weights

    def classify(self, data, clusters, weights, *, radius=100, batch_size=10000):
        N, = data.shape
        K, = clusters.shape
        probs = weights.new_empty(N, K)
        for i in range(0, N, batch_size):
            x = data[i: i + batch_size]
            probs[i:i + len(x)] = self.set_difference(x, clusters)
        probs.mul_(-1 / radius).add_(weights.log())
        probs.sub_(probs.logsumexp(-1, True)).exp_()
        return probs


_cpp_module = None


def _get_cpp_module():
    """
    JIT compiles the cpp module.
    """
    global _cpp_module
    if _cpp_module is None:
        from torch.utils.cpp_extension import load
        assert __file__.endswith(".py")
        path = __file__[:-3] + ".cpp"
        _cpp_module = load(name="cpp_cluster",
                           sources=[path],
                           extra_cflags=['-O2'],
                           verbose=False)
    return _cpp_module


def string_to_soft_hash(min_k, max_k, seq, out):
    to_bits = {"A": 0, "C": 1, "G": 2, "T": 3}

    bits = out.size(-1)
    for k in range(min_k, max_k + 1):
        salt = murmur64(1 + k)
        for pos in range(len(seq) - k + 1):
            hash_ = salt
            for i in range(k):
                hash_ ^= to_bits[seq[pos + i]] << (i + i)
            hash_ = murmur64(hash_)
            for b in range(bits):
                out[b] += 1 if (hash_ & (1 << b)) else -1


def string_to_clock_hash(k, seq, clocks, count):
    to_bits = {"A": 0, "C": 1, "G": 2, "T": 3}

    num_kmers = len(seq) - k + 1
    count.add_(num_kmers)
    for pos in range(num_kmers):
        hash_ = 0
        for i in range(k):
            hash_ ^= to_bits[seq[pos + i]] << (i + i)
        for w in range(len(clocks) // 64):
            hash_w = murmur64(murmur64(1 + w) ^ hash_)
            for b in range(64):
                wb = w * 64 + b
                b_ = (b // 8) + 8 * (b % 8)  # for vectorized math in C++
                clocks[wb] += (hash_w >> b_) & 1
                clocks[wb] &= 0x7F
