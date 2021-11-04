# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections import defaultdict, namedtuple
from typing import Dict, FrozenSet

from Bio.Phylo.NewickIO import Parser

from .external.usher import parsimony_pb2

Mutation = namedtuple("Mutation", ["position", "ref", "mut"])

NUCLEOTIDE = "ACGT"


def load_mutation_tree(filename: str) -> Dict[str, FrozenSet[Mutation]]:
    """
    Loads an usher lineageTree.pb annotated with mutations and pango lineages,
    and creates a mapping from lineages to their set of mutations.
    """
    with open(filename, "rb") as f:
        proto = parsimony_pb2.data.FromString(f.read())  # type: ignore

    tree = next(Parser.from_string(proto.newick).parse())
    clades = list(tree.find_clades())
    assert len(proto.metadata) == len(clades)
    assert len(proto.node_mutations) == len(clades)

    clade_to_muts: Dict[object, Dict[int, Mutation]] = defaultdict(dict)
    lineage_to_clade = {}
    for clade, meta, muts in zip(clades, proto.metadata, proto.node_mutations):
        if meta and meta.clade:
            lineage_to_clade[str(meta.clade)] = clade
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
    return mutations_by_lineage


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
