# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import heapq


class RandomSubDict:
    def __init__(self, max_size):
        assert isinstance(max_size, int) and max_size > 0
        self.key_to_value = {}
        self.hash_to_key = {}
        self.heap = []
        self.max_size = max_size

    def __len__(self):
        return len(self.heap)

    def __setitem__(self, key, value):
        assert key not in self.key_to_value

        # Add (key,value) pair.
        self.key_to_value[key] = value
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        self.hash_to_key[h] = key
        heapq.heappush(self.heap, h)

        # Truncate via k-min-hash.
        if len(self.heap) > self.max_size:
            h = heapq.heappop(self.heap)
            key = self.hash_to_key.pop(h)
            self.key_to_value.pop(key)

    def keys(self):
        assert len(self.key_to_value) <= self.max_size
        return self.key_to_value.keys()

    def values(self):
        assert len(self.key_to_value) <= self.max_size
        return self.key_to_value.values()

    def items(self):
        assert len(self.key_to_value) <= self.max_size
        return self.key_to_value.items()
