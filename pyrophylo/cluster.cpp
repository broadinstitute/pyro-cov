#include <vector>
#include <string>

#include <torch/extension.h>

inline uint64_t murmur64(uint64_t h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccd;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53;
  h ^= h >> 33;
  return h;
}

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

void string_to_clock_hash(int k, const std::string& seq, at::Tensor clocks, at::Tensor count) {
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("string_to_soft_hash", &string_to_soft_hash, "Convert a string to a soft hash");
  m.def("string_to_clock_hash", &string_to_clock_hash, "Convert a string to a clock hash");
}
