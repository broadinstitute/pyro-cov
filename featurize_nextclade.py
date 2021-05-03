#!/usr/bin/env python

import argparse
import logging
import re
from collections import Counter, defaultdict

import torch

from pyrocov.fasta import NextcladeDB

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def count_mutations(counts, row):
    # Check whether row is valid
    status = row["qc.overallStatus"]
    if status != "good":
        counts["error"] += 1  # hack to count errors
        return
    counts[None] += 1  # hack to count number of lineages
    for col in ["aaSubstitutions", "aaDeletions"]:
        ms = row[col]
        if not ms:
            continue
        ms = ms.split(",")
        counts.update(ms)
        # Add within-gene pairs of mutations.
        by_gene = defaultdict(list)
        for m in ms:
            g, m = m.split(":")
            by_gene[g].append(m)
        for g, ms in by_gene.items():
            # Sort by position, then alphabetical.
            ms.sort(key=lambda m: (int(re.search(r"\d+", m).group(0)), m))
            for i, m1 in enumerate(ms):
                for m2 in ms[i + 1 :]:
                    counts[f"{g}:{m1},{m2}"] += 1


def main(args):
    # Load the subsampled tsv file.
    subsamples = defaultdict(dict)
    with open(args.subset_file_in, "rt") as f:
        for line in f:
            line = line.strip()
            lineage, accession_id, seq = line.split("\t")
            subsamples[lineage][accession_id] = seq.replace("_", "\n")

    # Count mutations via nextclade.
    lineage_mutation_counts = defaultdict(Counter)
    db = NextcladeDB()
    for lineage, sequences in subsamples.items():
        mutations = lineage_mutation_counts[lineage]
        for seq in sequences.values():
            db.schedule(seq, count_mutations, mutations)
    db.wait()
    num_errors = sum(v.pop("error", 0) for v in lineage_mutation_counts.values())
    logger.info(f"Found {num_errors} sequencing errors")
    lineage_counts = {
        k: v.pop(None) for k, v in lineage_mutation_counts.items() if None in v
    }

    # Filter to features that occur in the majority of at least one lineage.
    total = len(set().union(*lineage_mutation_counts.values()))
    mutations = set()
    for lineage, mutation_counts in list(lineage_mutation_counts.items()):
        if not mutation_counts:
            lineage_mutation_counts.pop(lineage)
            continue
        denominator = lineage_counts[lineage]
        for m, count in mutation_counts.items():
            if count / denominator >= args.thresh:
                mutations.add(m)
    by_num = Counter(m.count(",") for m in mutations)
    logger.info(
        "Keeping only ({} single + {} double) = {} of {} mutations".format(
            by_num[0], by_num[1], len(mutations), total
        )
    )

    # Convert to dense features.
    lineages = sorted(lineage_counts)
    mutations = sorted(mutations, key=lambda m: (m.count(","), m))
    lineage_ids = {k: i for i, k in enumerate(lineages)}
    mutation_ids = {k: i for i, k in enumerate(mutations)}
    features = torch.zeros(len(lineage_ids), len(mutation_ids))
    for lineage, counts in lineage_mutation_counts.items():
        i = lineage_ids[lineage]
        denominator = lineage_counts[lineage]
        for mutation, count in counts.items():
            j = mutation_ids.get(mutation, None)
            if j is not None:
                features[i, j] = count / denominator

    result = {
        "lineages": lineages,
        "mutations": mutations,
        "features": features,
    }
    logger.info(f"saving {tuple(features.shape)}-features to {args.features_file_out}")
    torch.save(result, args.features_file_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize nextclade mutations")
    parser.add_argument("--thresh", default=0.5, type=float)
    parser.add_argument("--subset-file-in", default="results/gisaid.subset.tsv")
    parser.add_argument("--features-file-out", default="results/nextclade.features.pt")
    args = parser.parse_args()
    main(args)
