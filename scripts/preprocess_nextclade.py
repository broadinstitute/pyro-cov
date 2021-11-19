# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import pickle
import re
from collections import Counter, defaultdict

from pyrocov.align import AlignDB
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
    id_to_lineage[accession_id] = lineage, row["lineages"], row["clade"], row["clades"]
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
    # Load the filtered accession ids.
    logger.info(f"Loading {args.columns_file_in}")
    with open(args.columns_file_in, "rb") as f:
        columns = pickle.load(f)
        id_to_lineage = {aid: None for aid in columns["accession_id"]}
        del columns

    # Count mutations via nextclade.
    # This is batched and cached under the hood.
    logger.info(f"Loading {args.gisaid_file_in}")
    mutation_counts = defaultdict(Counter)
    status_counts = defaultdict(Counter)
    db = AlignDB()
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
        llcc = id_to_lineage.get(row["accession_id"])
        if llcc is None:
            continue  # drop the row
        lineage, lineages, clade, clades = llcc
        columns["clade"].append(clade)
        columns["clades"].append(clades)
        columns["lineage"].append(lineage)
        columns["lineages"].append(lineages)
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
    logger.info(f"saving {args.counts_file_out}")
    with open(args.counts_file_out, "wb") as f:
        pickle.dump(dict(agg_counts), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize nextclade mutations")
    parser.add_argument("--gisaid-file-in", default="results/gisaid.json")
    parser.add_argument("--columns-file-in", default="results/gisaid.columns.pkl")
    parser.add_argument("--columns-file-out", default="results/usher.columns.pkl")
    parser.add_argument("--counts-file-out", default="results/nextclade.counts.pkl")
    args = parser.parse_args()
    main(args)
