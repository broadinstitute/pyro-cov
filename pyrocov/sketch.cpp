// Copyright Contributors to the Pyro-Cov project.
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/stl.h>
#include <torch/extension.h>

inline uint64_t murmur64(uint64_t h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccd;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53;
  h ^= h >> 33;
  return h;
}

at::Tensor get_32mers(const std::string& seq) {
  static std::vector<int64_t> to_bits(256, false);
  to_bits['A'] = 0;
  to_bits['C'] = 1;
  to_bits['G'] = 2;
  to_bits['T'] = 3;

  int size = seq.size() - 32 + 1;
  if (size <= 0) {
    return at::empty(0, at::kLong);
  }
  at::Tensor out = at::empty(size, at::kLong);
  int64_t * const out_data = static_cast<int64_t*>(out.data_ptr());

  int64_t kmer = 0;
  for (int pos = 0; pos < 31; ++pos) {
    kmer <<= 2;
    kmer ^= to_bits[seq[pos]];
  }
  for (int pos = 0; pos < size; ++pos) {
    kmer <<= 2;
    kmer ^= to_bits[seq[pos + 31]];
    out_data[pos] = kmer;
  }
  return out;
}

struct KmerCounter {
  KmerCounter() : to_bits(256, false), counts() {
    to_bits['A'] = 0;
    to_bits['C'] = 1;
    to_bits['G'] = 2;
    to_bits['T'] = 3;
  }

  void update(const std::string& seq) {
    int size = seq.size() - 32 + 1;
    if (size <= 0) {
      return;
    }

    int64_t kmer = 0;
    for (int pos = 0; pos < 31; ++pos) {
      kmer <<= 2;
      kmer ^= to_bits[seq[pos]];
    }
    for (int pos = 0; pos < size; ++pos) {
      kmer <<= 2;
      kmer ^= to_bits[seq[pos + 31]];
      counts[kmer] += 1;
    }
  }

  void truncate_below(int64_t threshold) {
    for (auto i = counts.begin(); i != counts.end();) {
      if (i->second < threshold) {
         counts.erase(i++);
      } else {
         ++i;
      }
    }
  }

  std::unordered_map<int64_t, int64_t> to_dict() const { return counts; }

  std::vector<int64_t> to_bits;
  std::unordered_map<int64_t, int64_t> counts;
};

void string_to_soft_hash(int min_k, int max_k, const std::string& seq, at::Tensor out) {
  static std::vector<uint64_t> to_bits(256, false);
  to_bits['A'] = 0;
  to_bits['C'] = 1;
  to_bits['G'] = 2;
  to_bits['T'] = 3;

  static std::vector<uint64_t> salts(33);
  assert(max_k < salts.size());
  for (int k = min_k; k <= max_k; ++k) {
    salts[k] = murmur64(1 + k);
  }

  float * const data_begin = static_cast<float*>(out.data_ptr());
  float * const data_end = data_begin + out.size(-1);
  for (int pos = 0, end = seq.size(); pos != end; ++pos) {
    if (max_k > end - pos) {
      max_k = end - pos;
    }
    uint64_t hash = 0;
    for (int k = 1; k <= max_k; ++k) {
      int i = k - 1;
      hash ^= to_bits[seq[pos + i]] << (i + i);
      if (k < min_k) continue;
      uint64_t hash_k = murmur64(salts[k] ^ hash);
      for (float *p = data_begin; p != data_end; ++p) {
        *p += static_cast<int64_t>(hash_k & 1UL) * 2L - 1L;
        hash_k >>= 1UL;
      }
    }
  }
}

void string_to_clock_hash_v0(int k, const std::string& seq, at::Tensor clocks, at::Tensor count) {
  static std::vector<uint64_t> to_bits(256, false);
  to_bits['A'] = 0;
  to_bits['C'] = 1;
  to_bits['G'] = 2;
  to_bits['T'] = 3;

  int8_t * const clocks_data = static_cast<int8_t*>(clocks.data_ptr());
  const int num_kmers = seq.size() - k + 1;
  count.add_(num_kmers);
  for (int pos = 0; pos < num_kmers; ++pos) {
    uint64_t hash = murmur64(1 + k);
    for (int i = 0; i < k; ++i) {
      hash ^= to_bits[seq[pos + i]] << (i + i);
    }
    hash = murmur64(hash);
    for (int i = 0; i < 64; ++i) {
      clocks_data[i] += (hash >> i) & 1;
    }
  }
}

void string_to_clock_hash(int k, const std::string& seq, at::Tensor clocks, at::Tensor count) {
  static std::vector<uint64_t> to_bits(256, false);
  to_bits['A'] = 0;
  to_bits['C'] = 1;
  to_bits['G'] = 2;
  to_bits['T'] = 3;

  const int num_words = clocks.size(-1) / 64;
  uint64_t * const clocks_data = static_cast<uint64_t*>(clocks.data_ptr());
  const int num_kmers = seq.size() - k + 1;
  count.add_(num_kmers);
  for (int pos = 0; pos < num_kmers; ++pos) {
    uint64_t hash = 0;
    for (int i = 0; i != k; ++i) {
      hash ^= to_bits[seq[pos + i]] << (i + i);
    }
    for (int w = 0; w != num_words; ++w) {
      const uint64_t hash_w = murmur64(murmur64(1 + w) ^ hash);
      for (int b = 0; b != 8; ++b) {
        const int wb = w * 8 + b;
        clocks_data[wb] = (clocks_data[wb] + ((hash_w >> b) & 0x0101010101010101UL))
                        & 0x7F7F7F7F7F7F7F7FUL;
      }
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_32mers", &get_32mers, "Extract list of 32-mers from a string");
  m.def("string_to_soft_hash", &string_to_soft_hash, "Convert a string to a soft hash");
  m.def("string_to_clock_hash", &string_to_clock_hash, "Convert a string to a clock hash");
  py::class_<KmerCounter>(m, "KmerCounter")
    .def(py::init<>())
    .def("update", &KmerCounter::update)
    .def("truncate_below", &KmerCounter::truncate_below)
    .def("to_dict", &KmerCounter::to_dict);
}
