# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os

import torch

from pyrocov.fasta import NEXTSTRAIN_DATA, PANGOLEARN_DATA, NextcladeDB
from pyrocov.usher import apply_mutations, load_mutation_tree

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def main(args):
    # Extract mutations from an annotated tree.
    nuc_mutations_by_lineage = load_mutation_tree(args.tree_file_in)
    assert nuc_mutations_by_lineage
    nuc_mutations = frozenset(m for ms in nuc_mutations_by_lineage.values() for m in ms)
    logger.info(
        f"Found {len(nuc_mutations)} mutations in "
        f"{len(nuc_mutations_by_lineage)} lineages"
    )

    # Load reference sequence.
    with open(os.path.join(NEXTSTRAIN_DATA, "reference.fasta")) as f:
        lines = []
        for line in f:
            if not line.startswith(">"):
                lines.append(line.strip())
    ref = "".join(lines)
    assert len(ref) == 29903, len(ref)

    # Convert from nucleotide mutations to amino acid mutations.
    aa_mutations_by_lineage = {}

    def collect_mutations(lineage, row):
        ms = row["aaSubstitutions"]
        aa_mutations_by_lineage[lineage] = ms.split(",") if ms else []

    logger.info(f"Aligning {len(nuc_mutations_by_lineage)} sequences with nextclade")
    db = NextcladeDB(max_fasta_count=args.max_fasta_count)
    for lineage, mutations in sorted(nuc_mutations_by_lineage.items()):
        seq = apply_mutations(ref, mutations)
        db.schedule(seq, collect_mutations, lineage)
    db.wait()

    # Create dense aa features.
    lineages = sorted(nuc_mutations_by_lineage)
    lineage_ids = {k: i for i, k in enumerate(lineages)}
    aa_mutations = sorted(set(m for ms in aa_mutations_by_lineage.values() for m in ms))
    logger.info(f"Found {len(aa_mutations)} amino acid mutations")
    mutation_ids = {k: i for i, k in enumerate(aa_mutations)}
    aa_features = torch.zeros(len(lineage_ids), len(mutation_ids), dtype=torch.bool)
    for lineage, ms in aa_mutations_by_lineage.items():
        i = lineage_ids[lineage]
        for m in ms:
            j = mutation_ids[m]
            aa_features[i, j] = True

    # Create dense nucleotide features.
    nuc_mutations_by_lineage = {
        lineage: [f"{m.ref}{m.position}{m.mut}" for m in ms]
        for lineage, ms in nuc_mutations_by_lineage.items()
    }
    nuc_mutations = sorted(
        set(m for ms in nuc_mutations_by_lineage.values() for m in ms)
    )
    mutation_ids = {k: i for i, k in enumerate(nuc_mutations)}
    nuc_features = torch.zeros(len(lineage_ids), len(mutation_ids), dtype=torch.bool)
    for lineage, ms in nuc_mutations_by_lineage.items():
        i = lineage_ids[lineage]
        for m in ms:
            j = mutation_ids[m]
            nuc_features[i, j] = True

    # TODO create pairwise features.

    result = {
        "lineages": lineages,
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
    torch.save(result, args.features_file_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess pangolin mutations")
    parser.add_argument(
        "--tree-file-in", default=os.path.join(PANGOLEARN_DATA, "lineageTree.pb")
    )
    parser.add_argument("--features-file-out", default="results/usher.features.pt")
    parser.add_argument("--max-fasta-count", default=4000, type=int)
    args = parser.parse_args()
    main(args)
