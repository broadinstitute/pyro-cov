from collections import Counter

import torch

# See https://cov-lineages.org/lineages.html or
# https://github.com/cov-lineages/pangoLEARN/blob/master/pangoLEARN/supporting_information/lineage_notes.txt
PANGOLIN_ALIASES = {
    "C": "B.1.1.1",
    "D": "B.1.1.25",
    "E": "B.1.416",
    "F": "B.1.36.17",
    "G": "B.1.258.2",
    "H": "B.1.1.67",
    "I": "B.1.1.217",
    "J": "B.1.1.250",
    "K.1": "B.1.1.277",
    "L": "B.1.1.10",
    "M": "B.1.1.294",
    "N": "B.1.1.33",
    "P": "B.1.1.28",
}

DECOMPRESS = PANGOLIN_ALIASES.copy()
COMPRESS = {v: k for k, v in DECOMPRESS.items()}


def decompress(name):
    """
    Decompress an alias like C.10 to a full lineage like B.1.1.1.10.
    """
    try:
        return DECOMPRESS[name]
    except KeyError:
        pass
    if name[0] in "AB":
        DECOMPRESS[name] = name
        return name
    for key, value in PANGOLIN_ALIASES.items():
        if name.startswith(key):
            result = value + name[len(key) :]
            DECOMPRESS[name] = result
            COMPRESS[result] = name
            return result
    raise ValueError(f"Unknown alias: {name}")


def compress(name):
    """
    Compress a full lineage like B.1.1.1.10 to an alias like C.10.
    """
    return COMPRESS.get(name, name)


def _get_parent(longname):
    return longname.rsplit(".", 1)[0]


def find_edges(names):
    """
    Given a set of short lineages, return a list of pairs of parent-child
    relationships among lineages.
    """
    short_to_long = {compress(name): decompress(name) for name in names}
    long_to_short = {v: k for k, v in short_to_long.items()}
    assert len(short_to_long) == len(long_to_short)
    edges = [("A", "B")]
    for x, longx in short_to_long.items():
        longy = _get_parent(longx)
        while longy not in names:
            longy = _get_parent(longy)
        if longy == longx:
            continue
        if longy != longx:
            y = long_to_short[longy]
            assert x != y
            edges.append((x, y) if x < y else (y, x))
    assert len(set(edges)) == len(edges)
    assert len(edges) == len(names) - 1
    return edges


def merge_lineages(counts, min_count):
    """
    Given a dict of lineage counts and a min_count, returns a mapping from all
    lineages to merged lineages.
    """
    assert isinstance(counts, dict)
    assert isinstance(min_count, int)
    assert min_count > 0

    # Merge rare children into their parents.
    counts = Counter({decompress(k): v for k, v in counts.items()})
    mapping = {}
    for child in sorted(counts, key=lambda k: (-len(k), k)):
        if counts[child] < min_count:
            parent = _get_parent(child)
            if parent == child:
                continue  # at a root
            counts[parent] += counts.pop(child)
            mapping[child] = parent

    # Transitively close.
    for old, new in list(mapping.items()):
        while new in mapping:
            new = mapping[new]
        mapping[old] = new

    # Recompress.
    mapping = {compress(k): compress(v) for k, v in mapping.items()}
    return mapping


def classify(lineage):
    """
    Given a list of compressed lineages, return a torch long tensor of class
    ids and a list of pairs of integers representing parent-child lineage
    relationships between classes.
    """
    # Construct a class tensor.
    names = sorted(set(lineage))
    position = {name: i for i, name in enumerate(names)}
    classes = torch.zeros(len(lineage), dtype=torch.long)
    for i, name in enumerate(lineage):
        classes[i] = position[name]

    # Construct a tree.
    short_to_long = {name: decompress(name) for name in names}
    long_to_short = {v: k for k, v in short_to_long.items()}
    assert len(short_to_long) == len(long_to_short)
    edges = [(position["A"], position["B"])]
    for x, longx in short_to_long.items():
        i = position[x]
        longy = longx.rsplit(".", 1)[0]
        if longy != longx:
            j = position[long_to_short[longy]]
            assert i != j
            edges.append((i, j) if i < j else (j, i))
    assert len(set(edges)) == len(edges)
    assert len(edges) == len(names) - 1

    return classes, edges
