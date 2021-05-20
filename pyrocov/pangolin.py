import glob
import os
import re
import warnings
from collections import Counter

import torch

PANGOLIN_REPO = os.environ.get(
    "PANGOLIN_REPO", "~/github/cov-lineages/lineages-website"
)

# TODO replace this with
# https://github.com/cov-lineages/pango-designation/blob/master/alias_key.json
# See https://cov-lineages.org/lineages.html or
# https://github.com/cov-lineages/lineages-website/lineages/*.md
# This list can be updated via update_aliases() below.
PANGOLIN_ALIASES = {
    "AA": "B.1.177.15",
    "AB": "B.1.160.16",
    "AC": "B.1.1.405",
    "AD": "B.1.1.315",
    "AE": "B.1.1.306",
    "AF": "B.1.1.305",
    "AG": "B.1.1.297",
    "AH": "B.1.1.241",
    "AJ": "B.1.1.240",
    "AK": "B.1.1.232",
    "AL": "B.1.1.231",
    "AM": "B.1.1.216",
    "AN": "B.1.1.200",
    "AP": "B.1.1.70",
    "AQ": "B.1.1.39",
    "AS": "B.1.1.317",
    "AT": "B.1.1.370",
    "AU": "B.1.466.2",
    "AV": "B.1.1.482",
    "C": "B.1.1.1",
    "D": "B.1.1.25",
    "E": "B.1.416",
    "F": "B.1.36.17",
    "G": "B.1.258.2",
    "H": "B.1.1.67",
    "I": "B.1.1.217",
    "J": "B.1.1.250",
    "K": "B.1.1.277",
    "L": "B.1.1.10",
    "M": "B.1.1.294",
    "N": "B.1.1.33",
    "P": "B.1.1.28",
    "Q": "B.1.1.7",
    "R": "B.1.1.316",
    "S": "B.1.1.217",
    "U": "B.1.177.60",
    "V": "B.1.177.54",
    "W": "B.1.177.53",
    "Y": "B.1.177.52",
    "Z": "B.1.177.50",
}

DECOMPRESS = PANGOLIN_ALIASES.copy()
COMPRESS = {}


def update_aliases():
    repo = os.path.expanduser(PANGOLIN_REPO)
    for filename in glob.glob(f"{repo}/lineages/lineage_*.md"):
        with open(filename) as f:
            lineage = None
            parent = None
            for line in f:
                line = line.strip()
                if line.startswith("lineage: "):
                    lineage = line.split()[-1]
                elif line.startswith("parent: "):
                    parent = line.split()[-1]
            if lineage and "." in lineage and parent:
                alias = lineage.rsplit(".", 1)[0]
                if parent != alias:
                    PANGOLIN_ALIASES[alias] = parent
    return PANGOLIN_ALIASES


try:
    update_aliases()
except Exception:
    warnings.warn(
        f"Failed to find {PANGOLIN_REPO}, pangolin aliases may be stale", RuntimeWarning
    )


def decompress(name):
    """
    Decompress an alias like C.10 to a full lineage like B.1.1.1.10.
    """
    try:
        return DECOMPRESS[name]
    except KeyError:
        pass
    if name.split(".")[0] in ("A", "B"):
        DECOMPRESS[name] = name
        return name
    for key, value in PANGOLIN_ALIASES.items():
        if name == key or name.startswith(key + "."):
            result = value + name[len(key) :]
            DECOMPRESS[name] = result
            return result
    raise ValueError(f"Unknown alias: {repr(name)}")


def compress(name):
    """
    Compress a full lineage like B.1.1.1.10 to an alias like C.10.
    """
    result = COMPRESS.get(name)
    if result is None:
        result = name
        if name.count(".") > 3:
            for key, value in PANGOLIN_ALIASES.items():
                if key == "I":
                    continue  # obsolete
                if name == value or name.startswith(value + "."):
                    result = key + name[len(value) :]
                    break
                COMPRESS[name] = result
    assert re.match(r"^[A-Z]+(\.[0-9]+)*$", result), result
    return result


def get_parent(name):
    assert decompress(name) == name, "expected a decompressed name"
    if name == "A":
        return None
    if name == "B":
        return "A"
    assert "." in name, name
    return name.rsplit(".", 1)[0]


def find_edges(names):
    """
    Given a set of short lineages, return a list of pairs of parent-child
    relationships among lineages.
    """
    longnames = [decompress(name) for name in names]
    edges = []
    for x in longnames:
        if x == "A":
            continue  # A is root
        y = get_parent(x)
        while y not in longnames:
            y = get_parent(y)
        if y != x:
            edges.append((x, y) if x < y else (y, x))
    edges = [(compress(x), compress(y)) for x, y in edges]
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
            parent = get_parent(child)
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
