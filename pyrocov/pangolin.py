# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import warnings
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import torch

PANGOLIN_REPO = os.path.expanduser(
    os.environ.get("PANGOLIN_REPO", "~/github/cov-lineages/pango-designation")
)

# See https://cov-lineages.org/lineage_list.html or
# https://github.com/cov-lineages/pango-designation/blob/master/pango_designation/alias_key.json
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
    "AW": "B.1.1.464",
    "AY": "B.1.617.2",
    "AZ": "B.1.1.318",
    "BA": "B.1.1.529",
    "BB": "B.1.621.1",
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
    "XA": "B.1.1.7",
    "XB": "B.1.634",
    "XC": "B.1.617.2.29",  # i.e. AY.29
}

# From https://www.who.int/en/activities/tracking-SARS-CoV-2-variants/
WHO_ALIASES = {
    # Variants of concern.
    "Alpha": ["B.1.1.7", "Q"],
    "Beta": ["B.1.351", "B.1.351.2", "B.1.351.3"],
    "Gamma": ["P.1", "P.1.1", "P.1.2"],
    "Delta": ["B.1.617.2", "AY"],
    "Omicron": ["B.1.1.529", "BA"],
    # Variants of interest.
    "Lambda": ["C.37"],
    "Mu": ["B.1.621"],
    # Former variants of interest.
    # Epsilon (B.1.427/B.1.429), Zeta (P.2), Theta (P.3)
    "Eta": ["B.1.525"],
    "Iota": ["B.1.526"],
    "Kappa": ["B.1.617.1"],
}
WHO_VOC = ["Alpha", "Beta", "Gamma", "Delta", "Omicron"]
WHO_VOI = ["Lambda", "Mu"]


def update_aliases():
    repo = os.path.expanduser(PANGOLIN_REPO)
    with open(f"{repo}/pango_designation/alias_key.json") as f:
        for k, v in json.load(f).items():
            if isinstance(v, str) and v:
                PANGOLIN_ALIASES[k] = v
    return PANGOLIN_ALIASES


try:
    update_aliases()
except Exception as e:
    warnings.warn(
        f"Failed to find {PANGOLIN_REPO}, pangolin aliases may be stale.\n{e}",
        RuntimeWarning,
    )

DECOMPRESS = PANGOLIN_ALIASES.copy()
COMPRESS: Dict[str, str] = {}
RE_PANGOLIN = re.compile(r"^[A-Z]+(\.[0-9]+)*$")


def is_pango_lineage(name: str) -> bool:
    """
    Returns whether the name looks like a PANGO lineage e.g. "AY.4.2".
    """
    return RE_PANGOLIN.match(name) is not None


def decompress(name: str) -> str:
    """
    Decompress an alias like C.10 to a full lineage like "B.1.1.1.10".
    """
    if name.startswith("fine"):
        return name
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
            assert result
            DECOMPRESS[name] = result
            return result
    raise ValueError(f"Unknown alias: {repr(name)}")


def compress(name: str) -> str:
    """
    Compress a full lineage like "B.1.1.1.10" to an alias like "C.10".
    """
    if name.startswith("fine"):
        return name
    try:
        return COMPRESS[name]
    except KeyError:
        pass
    if name.count(".") <= 3:
        result = name
    else:
        for key, value in PANGOLIN_ALIASES.items():
            if key == "I":
                continue  # obsolete
            if name == value or name.startswith(value + "."):
                result = key + name[len(value) :]
                break
    assert is_pango_lineage(result), result
    COMPRESS[name] = result
    return result


assert compress("B.1.1.7") == "B.1.1.7"


def get_parent(name: str) -> Optional[str]:
    """
    Given a decompressed lineage name ``name``, find the decompressed name of
    its parent lineage.
    """
    assert decompress(name) == name, "expected a decompressed name"
    if name in ("A", "fine"):
        return None
    if name == "B":
        return "A"
    assert "." in name, name
    return name.rsplit(".", 1)[0]


def get_most_recent_ancestor(name: str, ancestors: Set[str]) -> Optional[str]:
    """
    Given a decompressed lineage name ``name``, find the decompressed name of
    its the lineage's most recent ancestor from ``ancestors``. This is like
    :func:`get_parent` but may skip one or more generations in case a parent is
    not in ``ancestors``.
    """
    ancestor = get_parent(name)
    while ancestor is not None and ancestor not in ancestors:
        ancestor = get_parent(ancestor)
    return ancestor


def find_edges(names: List[str]) -> List[Tuple[str, str]]:
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
        while y is not None and y not in longnames:
            y = get_parent(y)
        if y is None:
            continue
        if y != x:
            edges.append((x, y) if x < y else (y, x))
    edges = [(compress(x), compress(y)) for x, y in edges]
    assert len(set(edges)) == len(edges)
    assert len(edges) == len(names) - 1
    return edges


def find_descendents(names: List[str]) -> Dict[str, List[str]]:
    """
    Given a set of short lineages, returns a dict mapping short lineage to its
    list of descendents.
    """
    longnames = [decompress(name) for name in names]
    descendents: Dict[str, List[str]] = {}
    for long1, short1 in zip(longnames, names):
        prefix = long1 + "."
        descendents1 = descendents[short1] = []
        for long2, short2 in zip(longnames, names):
            if long2.startswith(prefix):
                descendents1.append(short2)
    return descendents


def merge_lineages(counts: Dict[str, int], min_count: int) -> Dict[str, str]:
    """
    Given a dict of lineage counts and a min_count, returns a mapping from all
    lineages to merged lineages.
    """
    assert isinstance(counts, dict)
    assert isinstance(min_count, int)
    assert min_count > 0

    # Merge rare children into their parents.
    counts: Dict[str, int] = Counter({decompress(k): v for k, v in counts.items()})
    mapping = {}
    for child in sorted(counts, key=lambda k: (-len(k), k)):
        if counts[child] < min_count:
            parent = get_parent(child)
            if parent is None:
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


def classify(lineage: List[str]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
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
