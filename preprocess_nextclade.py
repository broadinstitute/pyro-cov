# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import pickle
import re
from collections import Counter, defaultdict

import torch

from pyrocov.fasta import NextcladeDB
from pyrocov.util import open_tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def process_row(
    id_to_lineage,
    mutation_counts,
    status_counts,
    accession_id,
    row,
):
    # Check whether row is valid
    lineage = row["lineage"]
    status = row["qc.overallStatus"]
    status_counts[lineage][status] += 1
    if status != "good":
        id_to_lineage[accession_id] = None
        return

    # Collect stats on a single lineage.
    id_to_lineage[accession_id] = lineage
    mutation_counts = mutation_counts[lineage]
    mutation_counts[None] += 1  # hack to count number of lineages

    for col in ["aaSubstitutions", "aaDeletions"]:
        ms = row[col]
        if not ms:
            continue
        ms = ms.split(",")
        mutation_counts.update(ms)
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
                    mutation_counts[f"{g}:{m1},{m2}"] += 1


def main(args):
    db = NextcladeDB(max_fasta_count=args.max_fasta_count)
    if args.repair:
        return db.repair()

    # Load the filtered accession ids.
    logger.info(f"Loading {args.columns_file_in}")
    with open(args.columns_file_in, "rb") as f:
        columns = pickle.load(f)
        id_to_lineage = dict(zip(columns["accession_id"], columns["lineage"]))
        del columns

    # Count mutations via nextclade.
    # This is batched and cached under the hood.
    logger.info(f"Loading {args.gisaid_file_in}")
    mutation_counts = defaultdict(Counter)
    status_counts = defaultdict(Counter)
    for line in open_tqdm(args.gisaid_file_in, "rt"):
        datum = json.loads(line)

        # Filter to desired sequences.
        accession_id = datum["covv_accession_id"]
        if accession_id not in id_to_lineage:
            continue

        # Schedule sequence for alignment.
        seq = datum["sequence"].replace("\n", "")
        db.schedule(
            seq,
            process_row,
            id_to_lineage,
            mutation_counts,
            status_counts,
            accession_id,
        )
    db.wait()

    message = ["Total quality:"]
    counts = Counter()
    for c in status_counts.values():
        counts.update(c)
    for s, c in counts.most_common():
        message.append(f"{s}: {c}")
    logger.info("\n\t".join(message))

    message = ["Lineages with fewest good samples:"]
    for c, l in sorted((c["good"], l) for l, c in status_counts.items())[:20]:
        message.append(f"{l}: {c}")
    logger.info("\n\t".join(message))

    # Update columns with usher-computed lineages.
    with open(args.columns_file_in, "rb") as f:
        old_columns = pickle.load(f)
    del old_columns["lineage"]
    columns = defaultdict(list)
    for row in zip(*old_columns.values()):
        row = dict(zip(old_columns, row))
        lineage = id_to_lineage.get(row["accession_id"])
        if lineage is None:
            continue  # drop the row
        columns["lineage"].append(lineage)
        for k, v in row.items():
            columns[k].append(v)
    del old_columns
    columns = dict(columns)
    with open(args.columns_file_out, "wb") as f:
        pickle.dump(columns, f)

    # Collect a set of all single mutations observed in this subsample.
    agg_counts = Counter()
    for ms in mutation_counts.values():
        for m, count in ms.items():
            if m is not None and "," not in m:
                agg_counts[m] += count
    all_mutations = sorted(agg_counts)
    logger.info(f"saving {args.counts_file_out}")
    with open(args.counts_file_out, "wb") as f:
        pickle.dump(dict(agg_counts), f)

    # Filter to lineages with at least a few good samples.
    for lineage, counts in list(status_counts.items()):
        if counts["good"] < args.min_good_samples:
            logger.info(f"Dropping {lineage} with {counts}")
            mutation_counts.pop(lineage, None)
            status_counts.pop(lineage, None)

    # Filter to features that occur in the majority of at least one lineage.
    lineage_counts = {k: v.pop(None) for k, v in mutation_counts.items() if None in v}
    mutations = set()
    for lineage, counts in list(mutation_counts.items()):
        if not counts:
            mutation_counts.pop(lineage)
            continue
        denominator = lineage_counts[lineage]
        for m, count in counts.items():
            if count / denominator >= 0.5:
                mutations.add(m)
    by_num = Counter(m.count(",") for m in mutations)
    logger.info(
        "Keeping only ({} single + {} double) = {} of {} mutations".format(
            by_num[0], by_num[1], len(mutations), len(all_mutations)
        )
    )

    # Convert to dense features.
    lineages = sorted(lineage_counts)
    mutations = sorted(mutations, key=lambda m: (m.count(","), m))
    lineage_ids = {k: i for i, k in enumerate(lineages)}
    mutation_ids = {k: i for i, k in enumerate(mutations)}
    features = torch.zeros(len(lineage_ids), len(mutation_ids))
    for lineage, counts in mutation_counts.items():
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
        "all_mutations": all_mutations,
    }
    logger.info(f"saving {tuple(features.shape)}-features to {args.features_file_out}")
    torch.save(result, args.features_file_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize nextclade mutations")
    parser.add_argument("--gisaid-file-in", default="results/gisaid.json")
    parser.add_argument("--columns-file-in", default="results/gisaid.columns.pkl")
    parser.add_argument("--columns-file-out", default="results/usher.columns.pkl")
    parser.add_argument("--features-file-out", default="results/nextclade.features.pt")
    parser.add_argument("--counts-file-out", default="results/nextclade.counts.pkl")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--min-nchars", default=29000, type=int)
    parser.add_argument("--max-nchars", default=31000, type=int)
    parser.add_argument("--min-good-samples", default=5, type=float)
    parser.add_argument("--max-fasta-count", default=4000, type=int)
    args = parser.parse_args()
    main(args)
