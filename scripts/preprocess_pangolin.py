# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import math
import os
import pickle
from collections import Counter, defaultdict

import torch

from pyrocov import pangolin
from pyrocov.align import NEXTSTRAIN_DATA, AlignDB
from pyrocov.usher import apply_mutations, load_mutation_tree, prune_mutation_tree

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


def main(args):
    # Extract mappings between coarse lineages and fine clades.
    db = AlignDB()
    fine_to_coarse = db.fine_to_coarse.copy()
    coarse_to_fines = defaultdict(list)
    for fine, coarse in db.fine_to_coarse.items():
        coarse_to_fines[coarse].append(fine)
    # Choose the basal representative.
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
    if not args.columns_file_out:
        args.columns_file_out = f"results/columns.{args.max_num_clades}.pkl"
    with open(args.columns_file_out, "wb") as f:
        pickle.dump(columns, f)
    logger.info(f"Saved {args.columns_file_out}")

    # Extract mutations from an annotated tree.
    nuc_mutations_by_clade = load_mutation_tree(tree_filename)
    assert nuc_mutations_by_clade
    nuc_mutations = frozenset(m for ms in nuc_mutations_by_clade.values() for m in ms)
    logger.info(
        f"Found {len(nuc_mutations)} mutations in {len(nuc_mutations_by_clade)} clades"
    )

    # Load reference sequence.
    with open(os.path.join(NEXTSTRAIN_DATA, "reference.fasta")) as f:
        ref = "".join(line.strip() for line in f if not line.startswith(">"))
    assert len(ref) == 29903, len(ref)

    # Convert from nucleotide mutations to amino acid mutations.
    aa_mutations_by_clade = {}

    def collect_mutations(clade, row):
        ms = row["aaSubstitutions"]
        aa_mutations_by_clade[clade] = ms.split(",") if ms else []

    logger.info(f"Aligning {len(nuc_mutations_by_clade)} sequences with nextclade")
    for clade, mutations in sorted(nuc_mutations_by_clade.items()):
        seq = apply_mutations(ref, mutations)
        db.schedule(seq, collect_mutations, clade)
    db.wait()

    # Create dense aa features.
    clades = sorted(nuc_mutations_by_clade)
    clade_ids = {k: i for i, k in enumerate(clades)}
    aa_mutations = Counter()
    for ms in aa_mutations_by_clade.values():
        aa_mutations.update(ms)
    aa_mutations = [
        m for m, c in aa_mutations.most_common() if c >= args.min_num_mutations
    ]
    logger.info(f"Found {len(aa_mutations)} amino acid mutations")
    mutation_ids = {k: i for i, k in enumerate(aa_mutations)}
    aa_features = torch.zeros(len(clade_ids), len(mutation_ids), dtype=torch.bool)
    for clade, ms in aa_mutations_by_clade.items():
        i = clade_ids[clade]
        for m in ms:
            j = mutation_ids.get(m)
            if j is not None:
                aa_features[i, j] = True

    # Create a dense ancestry matrix.
    ancestry = torch.eye(len(clades))
    for child, parent in pangolin.find_edges(clades):
        ancestry[clade_ids[parent], clade_ids[child]] = 1
    while True:  # Transitively close.
        square = (ancestry @ ancestry).clamp_(max=1)
        if torch.allclose(ancestry, square):
            break
        ancestry = square

    # Create dense nucleotide features.
    nuc_mutations_by_clade = {
        clade: [f"{m.ref}{m.position}{m.mut}" for m in ms]
        for clade, ms in nuc_mutations_by_clade.items()
    }
    nuc_mutations = sorted(set(m for ms in nuc_mutations_by_clade.values() for m in ms))
    mutation_ids = {k: i for i, k in enumerate(nuc_mutations)}
    nuc_features = torch.zeros(len(clade_ids), len(mutation_ids), dtype=torch.bool)
    for clade, ms in nuc_mutations_by_clade.items():
        i = clade_ids[clade]
        for m in ms:
            j = mutation_ids[m]
            nuc_features[i, j] = True

    # Save features.
    if not args.features_file_out:
        args.features_file_out = (
            f"results/features.{args.max_num_clades}.{args.min_num_mutations}.pt"
        )
    features = {
        "clades": clades,
        "ancestry": ancestry,
        "clade_to_lineage": fine_to_coarse,
        "lineage_to_clade": coarse_to_fine,
        "aa_mutations": aa_mutations,
        "aa_features": aa_features,
        "nuc_mutations": nuc_mutations,
        "nuc_features": nuc_features,
    }
    logger.info(
        f"saving {tuple(aa_features.shape)} aa features and "
        f"{tuple(nuc_features.shape)} nucleotide features "
        f"to {args.features_file_out}"
    )
    torch.save(features, args.features_file_out)
    logger.info(f"Saved {args.features_file_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess pangolin mutations")
    parser.add_argument("--columns-file-in", default="results/usher.columns.pkl")
    parser.add_argument("--tree-file-in", default="results/aligndb/lineageTree.fine.pb")
    parser.add_argument("--features-file-out", default="")
    parser.add_argument("--columns-file-out", default="")
    parser.add_argument("-c", "--max-num-clades", type=int, default=5000)
    parser.add_argument("-m", "--min-num-mutations", type=int, default=1)
    args = parser.parse_args()
    main(args)
