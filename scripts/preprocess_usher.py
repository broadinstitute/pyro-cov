# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import math
import pickle
from collections import defaultdict

import torch

from pyrocov.sarscov2 import nuc_mutations_to_aa_mutations
from pyrocov.usher import (
    FineToMeso,
    load_mutation_tree,
    prune_mutation_tree,
    refine_mutation_tree,
)

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def prune_tree(args, coarse_to_fine, columns):
    # To ensure pango lineages remain distinct, set their weights to infinity.
    weights = defaultdict(float, {fine: math.inf for fine in coarse_to_fine.values()})

    # Add weights of ambiguous clades.
    for clades in columns["clades"]:
        clades = clades.split(",")
        weight = 1 / len(clades)
        for clade in clades:
            weights[clade] += weight
    for clade in columns["clade"]:
        assert clade in weights

    # Prune the tree, minimizing the number of incorrect mutations.
    args.max_num_clades = max(args.max_num_clades, len(coarse_to_fine))
    tree_filename = f"results/lineageTree.{args.max_num_clades}.pb"
    meso_set = prune_mutation_tree(
        args.tree_file_in, tree_filename, args.max_num_clades, weights
    )
    assert len(meso_set) == args.max_num_clades
    return FineToMeso(meso_set), tree_filename


def main(args):
    # Extract mappings between coarse lineages and fine clades.
    coarse_proto = args.tree_file_in
    fine_proto = args.tree_file_out
    fine_to_coarse = refine_mutation_tree(coarse_proto, fine_proto)
    coarse_to_fines = defaultdict(list)
    for fine, coarse in fine_to_coarse.items():
        coarse_to_fines[coarse].append(fine)
    # Choose the basal representative.
    # FIXME is this actually the most recent common ancestor?
    coarse_to_fine = {c: min(fs) for c, fs in coarse_to_fines.items()}

    # Prune tree, updating data structures to use meso-scale clades.
    logger.info(f"Loading {args.columns_file_in}")
    with open(args.columns_file_in, "rb") as f:
        columns = pickle.load(f)
    fine_to_meso, tree_filename = prune_tree(args, coarse_to_fine, columns)

    fine_to_coarse = {fine_to_meso(f): c for f, c in fine_to_coarse.items()}
    coarse_to_fine = {c: fine_to_meso(f) for c, f in coarse_to_fine.items()}
    columns["clade"] = [fine_to_meso(c) for c in columns["clade"]]
    clade_set = set(columns["clade"])
    assert len(clade_set) <= args.max_num_clades
    columns["clades"] = [
        ",".join(fine_to_meso(c) for c in cs.split(",")) for cs in columns["clades"]
    ]
    clade_set.update(*(c.split(",") for c in columns["clades"]))
    assert len(clade_set) <= args.max_num_clades
    with open(args.columns_file_out, "wb") as f:
        pickle.dump(columns, f)
    logger.info(f"Saved {args.columns_file_out}")

    # Convert from nucleotide mutations to amino acid mutations.
    nuc_mutations_by_clade = load_mutation_tree(tree_filename)
    assert nuc_mutations_by_clade
    aa_mutations_by_clade = {
        clade: nuc_mutations_to_aa_mutations(mutations)  # FIXME type error
        for clade, mutations in nuc_mutations_by_clade.items()
    }

    # Create dense aa features.
    clades = sorted(nuc_mutations_by_clade)
    clade_ids = {k: i for i, k in enumerate(clades)}
    aa_mutations = sorted(set().union(*aa_mutations_by_clade.values()))
    logger.info(f"Found {len(aa_mutations)} amino acid mutations")
    mutation_ids = {k: i for i, k in enumerate(aa_mutations)}
    aa_features = torch.zeros(len(clade_ids), len(mutation_ids), dtype=torch.bool)
    for clade, ms in aa_mutations_by_clade.items():
        i = clade_ids[clade]
        for m in ms:
            j = mutation_ids.get(m)
            aa_features[i, j] = True

    # Save features.
    features = {
        "clades": clades,
        "clade_to_lineage": fine_to_coarse,
        "lineage_to_clade": coarse_to_fine,
        "aa_mutations": aa_mutations,
        "aa_features": aa_features,
    }
    logger.info(
        f"saving {tuple(aa_features.shape)} aa features to {args.features_file_out}"
    )
    torch.save(features, args.features_file_out)
    logger.info(f"Saved {args.features_file_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess pangolin mutations")
    parser.add_argument("--columns-file-in", default="results/usher.columns.pkl")
    parser.add_argument("--tree-file-in", default="results/usher/all.masked.pb")
    parser.add_argument("--tree-file-out", default="results/lineageTree.fine.pb")
    parser.add_argument("--features-file-out", default="")
    parser.add_argument("--columns-file-out", default="")
    parser.add_argument("-c", "--max-num-clades", type=int, default=5000)
    args = parser.parse_args()
    if not args.features_file_out:
        args.features_file_out = f"results/features.{args.max_num_clades}.pt"
    if not args.columns_file_out:
        args.columns_file_out = f"results/columns.{args.max_num_clades}.pkl"
    main(args)
