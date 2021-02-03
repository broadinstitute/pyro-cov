import torch

# See https://cov-lineages.org/lineages.html
PANGOLIN_ALIASES = {
    "C": "B.1.1.1",
    "D": "B.1.1.25",
    "E": "B.1.416",
    "F": "B.1.36.17",
    "G": "B.1.258.2",
    "H": "B.1.1.67",
    "I": "B.1.1.217",
    "J": "B.1.1.250",
    "K": "B.1.1.277",  # FIXME
    "L": "B.1.1.10",
    "M": "B.1.1.294",
    "N": "B.1.1.33",
    "P": "B.1.1.28",
}


def canonize(name):
    prefix = name[0]
    if prefix in "AB":
        return name
    return PANGOLIN_ALIASES[prefix] + name[1:]


def find_edges(names):
    short_to_long = {name: canonize(name) for name in names}
    long_to_short = {v: k for k, v in short_to_long.items()}
    assert len(short_to_long) == len(long_to_short)
    edges = [("A", "B")]
    for x, longx in short_to_long.items():
        longy = longx.rsplit(".", 1)[0]
        if longy != longx:
            y = long_to_short[longy]
            assert x != y
            edges.append((x, y) if x < y else (y, x))
    assert len(set(edges)) == len(edges)
    assert len(edges) == len(names) - 1
    return edges


def classify(lineage):
    # Construct a class tensor.
    names = sorted(set(lineage))
    position = {name: i for i, name in enumerate(names)}
    classes = torch.zeros(len(lineage), dtype=torch.long)
    for i, name in enumerate(lineage):
        classes[i] = position[name]

    # Construct a tree.
    short_to_long = {name: canonize(name) for name in names}
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
