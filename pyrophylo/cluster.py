import re

import torch


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

    def string_to_soft_hash(self, string, out):
        assert out.dim() == 1
        assert out.dtype == torch.float
        if self.backend == "python":
            impl = string_to_soft_hash
        elif self.backend == "cpp":
            impl = _get_cpp_module().string_to_soft_hash
        else:
            raise NotImplementedError(f"Unsupported backend: {self.backend}")
        out.fill_(0)
        for substring in re.findall("[ACGT]+", string):
            impl(self.min_k, self.max_k, substring, out)

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


class ClockSketcher:
    """
    Sketches bags of k-mers as 64 8-bit clocks plus a k-mer counter.
    """
    def __init__(self, k, *, backend="cpp"):
        self.k = k
        self.backend = backend

    def init_hash(self, *batch_shape):
        clocks = torch.zeros(batch_shape + (64,), dtype=torch.int8)
        count = torch.zeros(batch_shape, dtype=torch.int16)
        return clocks, count

    def string_to_hash(self, string, clocks, count):
        assert clocks.dtype == torch.int8
        assert clocks.shape == (64,)
        assert count.shape == ()
        if self.backend == "python":
            impl = string_to_clock_hash
        elif self.backend == "cpp":
            impl = _get_cpp_module().string_to_clock_hash
        else:
            raise NotImplementedError(f"Unsupported backend: {self.backend}")
        for substring in re.findall("[ACGT]+", string):
            impl(self.k, substring, clocks, count)

    def cdiff(self, x_clocks, x_count, y_clocks, y_count):
        clock = x_clocks.unsqueeze(-2) - y_clocks.unsqueeze(-3)
        count = x_count.unsqueeze(-1) - y_count.unsqueeze(-2)
        diff = clock.to(dtype=torch.int16)
        diff.mul_(2).sub_(count.unsqueeze(-1))
        diff.add_(256).bitwise_and_(511).sub_(256)
        return diff


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
                           verbose=True)
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
        hash_ = murmur64(1 + k)
        for i in range(k):
            hash_ ^= to_bits[seq[pos + i]] << (i + i)
        hash_ = murmur64(hash_)
        for b in range(64):
            clocks[b] += (hash_ >> b) & 1
