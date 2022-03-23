# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import gzip
import heapq
import logging
import shutil
import warnings
from collections import defaultdict, namedtuple
from typing import Dict, FrozenSet, Optional, Set, Tuple

import tqdm
from Bio.Phylo.NewickIO import Parser, Writer

from . import pangolin
from .external.usher import parsimony_pb2

logger = logging.getLogger(__name__)

Mutation = namedtuple("Mutation", ["position", "ref", "mut"])

NUCLEOTIDE = "ACGT"


def load_proto(filename):
    open_ = gzip.open if filename.endswith(".gz") else open
    with open_(filename, "rb") as f:
        proto = parsimony_pb2.data.FromString(f.read())  # type: ignore
    newick = proto.newick.replace(";", "")  # work around unescaped node names
    tree = next(Parser.from_string(newick).parse())
    return proto, tree


def load_usher_clades(filename: str) -> Dict[str, Tuple[str, str]]:
    """
    Loads usher's output clades.txt and extracts the best lineage and a list of
    possible lineages, for each sequence.
    """
    clades: Dict[str, Tuple[str, str]] = {}
    with open(filename) as f:
        for line in f:
            name, lineages = line.strip().split("\t")
            # Split histograms like B.1.1.161*|B.1.1(2/3),B.1.1.161(1/3) into points
            # like B.1.1.161 and lists like B.1.1,B.1.1.161.
            if "*|" in lineages:
                lineage, lineages = lineages.split("*|")
                lineages = ",".join(part.split("(")[0] for part in lineages.split(","))
            else:
                assert "*" not in lineages
                assert "|" not in lineages
                lineage = lineages
            clades[name] = lineage, lineages
    return clades


def load_mutation_tree(
    filename: str,
) -> Tuple[Dict[str, FrozenSet[Mutation]], object, object]:
    """
    Loads an usher lineageTree.pb or lineageTree.pb.gz annotated with mutations
    and pango lineages, and creates a mapping from lineages to their set of
    mutations.
    """
    logger.info(f"Loading tree from {filename}")
    proto, tree = load_proto(filename)

    # Extract phylogenetic tree.
    clades = list(tree.find_clades())
    assert len(proto.metadata) == len(clades)
    assert len(proto.node_mutations) == len(clades)

    # Map lineages to clades.
    lineage_to_clade = {
        str(meta.clade): clade
        for clade, meta in zip(clades, proto.metadata)
        if meta and meta.clade
    }

    # Accumulate mutations in each clade, which are overwritten at each position.
    logger.info(f"Accumulating mutations on {len(clades)} nodes")
    clade_to_muts: Dict[object, Dict[int, Mutation]] = defaultdict(dict)
    for clade, muts in zip(tqdm.tqdm(clades), proto.node_mutations):
        for mut in muts.mutation:
            clade_to_muts[clade][mut.position] = Mutation(
                mut.position,
                NUCLEOTIDE[mut.ref_nuc],
                "".join(NUCLEOTIDE[n] for n in mut.mut_nuc),
            )
        for c in clade.clades:
            clade_to_muts[c].update(clade_to_muts[clade])

    mutations_by_lineage = {
        k: frozenset(clade_to_muts[v].values()) for k, v in lineage_to_clade.items()
    }
    return mutations_by_lineage, proto, tree


def refine_mutation_tree(filename_in: str, filename_out: str) -> Dict[str, str]:
    """
    Refines a mutation tree clade metadata from pango lineages like B.1.1 to
    full node addresses like fine.0.12.4.1. Among clones, only the basal clade
    with have a .clade attribute, all descendents will have metadata.clade ==
    "". The tree structure remains unchanged.
    """
    proto, tree = load_proto(filename_in)

    # Extract phylogenetic tree.
    clades = list(tree.find_clades())
    logger.info(f"Refining a tree with {len(clades)} nodes")
    assert len(proto.metadata) == len(clades)
    assert len(proto.node_mutations) == len(clades)
    metadata = dict(zip(clades, proto.metadata))
    mutations = dict(zip(clades, proto.node_mutations))

    # Add refined clades, collapsing clones.
    num_children: Dict[str, int] = defaultdict(int)
    clade_to_fine = {clades[0]: "fine"}
    fine_to_clade = {"fine": clades[0]}
    for parent in clades:
        parent_fine = clade_to_fine[parent]
        for child in parent.clades:
            if mutations[child].mutation:
                # Create a new fine id.
                n = num_children[parent_fine]
                fine = f"{parent_fine}.{n - 1}" if n else parent_fine + "."
                num_children[parent_fine] += 1
                clade_to_fine[child] = fine
                fine_to_clade[fine] = child
            else:
                # Collapse clone into parent.
                clade_to_fine[child] = parent_fine

    # Save basal fine clades and the fine -> coarse mapping.
    fine_to_coarse = {}
    for clade, meta in metadata.items():
        fine = clade_to_fine[clade]
        if meta.clade and pangolin.is_pango_lineage(meta.clade):
            fine_to_coarse[fine] = pangolin.compress(meta.clade)
        meta.clade = fine if clade is fine_to_clade[fine] else ""
    # Propagate basal clade metadata downward.
    for parent in clades:
        parent_coarse = fine_to_coarse[clade_to_fine[parent]]
        for child in parent.clades:
            fine_to_coarse.setdefault(clade_to_fine[child], parent_coarse)

    with open(filename_out, "wb") as f:
        f.write(proto.SerializeToString())

    logger.info(f"Found {len(clades) - len(fine_to_coarse)} clones")
    logger.info(f"Refined {len(set(fine_to_coarse.values()))} -> {len(fine_to_coarse)}")
    return fine_to_coarse


def prune_mutation_tree(
    filename_in: str,
    filename_out: str,
    max_num_nodes: int,
    weights: Optional[Dict[str, int]] = None,
) -> Set[str]:
    """
    Condenses a mutation tree by greedily pruning nodes with least value
    under the error-minimizing objective function::

        value(node) = num_mutations(node) * weights(node)

    Returns a restricted set of clade names.
    """
    proto, tree = load_proto(filename_in)
    num_pruned = len(proto.node_mutations) - max_num_nodes
    if num_pruned < 0:
        shutil.copyfile(filename_in, filename_out)
        return {m.clade for m in proto.node_metadata if m.clade}

    # Extract phylogenetic tree.
    clades = list(tree.find_clades())
    logger.info(f"Pruning {num_pruned}/{len(clades)} nodes")
    assert len(clades) == len(set(clades))
    clade_to_id = {c: i for i, c in enumerate(clades)}
    assert len(proto.metadata) == len(clades)
    assert len(proto.node_mutations) == len(clades)
    metadata = dict(zip(clades, proto.metadata))
    mutations = dict(zip(clades, proto.node_mutations))
    name_set = {m.clade for m in proto.metadata if m.clade}

    # Initialize weights and topology.
    if weights is None:
        weights = {c: 1 for c in clades}
    else:
        assert set(weights).issubset(name_set)
        old_weights = weights.copy()
        weights = {}
        for c, m in metadata.items():
            weights[c] = old_weights.pop(m.clade, 0) if m.clade else 0
        assert not old_weights, list(old_weights)
    parents = {c: parent for parent in clades for c in parent.clades}
    assert tree.root not in parents

    def get_loss(clade):
        return weights[clade] * len(mutations[clade].mutation)

    # Greedily prune nodes.
    heap = [(get_loss(c), clade_to_id[c]) for c in clades[1:]]  # don't prune the root
    heapq.heapify(heap)
    for step in tqdm.tqdm(range(num_pruned)):
        # Find the clade with lowest loss.
        stale_loss, i = heapq.heappop(heap)
        clade = clades[i]
        loss = get_loss(clade)
        while loss != stale_loss:
            # Reinsert clades whose loss was stale.
            stale_loss, i = heapq.heappushpop(heap, (loss, i))
            clade = clades[i]
            loss = get_loss(clade)

        # Prune this clade.
        parent = parents.pop(clade)
        weights[parent] += weights.pop(clade, 0)  # makes the parent loss stale
        parent.clades.remove(clade)
        parent.clades.extend(clade.clades)
        mutation = list(mutations.pop(clade).mutation)
        for child in clade.clades:
            parents[child] = parent
            m = mutations[child].mutation
            cat = mutation + list(m)  # order so as to be compatible with reversions
            del m[:]
            m.extend(cat)
    clades = list(tree.find_clades())
    assert len(clades) == max_num_nodes

    # Create the pruned proto.
    proto.newick = next(iter(Writer([tree]).to_strings()))
    del proto.metadata[:]
    del proto.node_mutations[:]
    proto.metadata.extend(metadata[clade] for clade in clades)
    proto.node_mutations.extend(mutations[clade] for clade in clades)
    with open(filename_out, "wb") as f:
        f.write(proto.SerializeToString())

    return {metadata[clade].clade for clade in clades if metadata[clade].clade}


def apply_mutations(ref: str, mutations: FrozenSet[Mutation]) -> str:
    """
    Applies a set of mutations to a reference sequence.
    """
    seq = list(ref)
    for m in mutations:
        if m.mut == m.ref:
            continue
        if m.ref != seq[m.position - 1]:
            warnings.warn(f"invalid reference: {m.ref} vs {seq[m.position - 1]}")
        seq[m.position - 1] = m.mut
    return "".join(seq)


class FineToMeso:
    """
    Mapping from fine clade names like ``fine.1...3.`` to ancestors in
    ``meso_set`` like ``fine.1..`` .
    """

    def __init__(self, meso_set):
        self.meso_set = frozenset(meso_set)
        self._cache = {}

    def __call__(self, fine):
        meso = self._cache.get(fine, None)
        if meso is None:
            meso = fine if fine in self.meso_set else self(fine.rsplit(".", 1)[0])
            self._cache[fine] = meso
        return meso
