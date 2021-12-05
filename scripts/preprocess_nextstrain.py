# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import logging
import pickle
from collections import Counter, defaultdict

import torch

from pyrocov.mutrans import START_DATE
from pyrocov.util import gzip_open_tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def parse_date(string):
    return datetime.datetime.strptime(string, "%Y-%m-%d")


def main(args):
    columns = defaultdict(list)
    stats = defaultdict(Counter)
    skipped = Counter()

    # Process rows one at a time.
    logger.info(f"Reading {args.metadata_file_in}")
    header = None
    for line in gzip_open_tqdm(args.metadata_file_in, "rt"):
        line = line.strip().split("\t")
        if header is None:
            header = line
            continue
        row = dict(zip(header, line))

        # Parse date.
        try:
            date = parse_date(row["date"])
        except ValueError:
            skipped["date"] += 1
            continue
        day = (date - args.start_date).days

        # Parse location.
        location = row["location"]
        if location in ("", "?"):
            skipped["location"] += 1
            continue

        # Parse lineage.
        lineage = row["pango_lineage"]
        if lineage in ("", "?", "unclassifiable"):
            skipped["lineage"] += 1
            continue
        assert lineage[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ", lineage

        # Append row.
        columns["day"].append(day)
        columns["location"].append(location)
        columns["lineage"].append(lineage)

        # Record stats.
        stats["day"][day] += 1
        stats["location"][location] += 1
        stats["lineage"][lineage] += 1
        for aa in row["aaSubstitutions"].split(","):
            stats["aa"][aa] += 1
            stats["lineage_aa"][lineage, aa] += 1
    columns = dict(columns)
    stats = dict(stats)
    logger.info(f"Skipped {sum(skipped.values())} due to:\n{dict(skipped)}")

    logger.info(f"Saving {args.stats_file_out}")
    with open(args.stats_file_out, "wb") as f:
        pickle.dump(stats, f)

    logger.info(f"Saving {args.columns_file_out}")
    with open(args.columns_file_out, "wb") as f:
        pickle.dump(columns, f)

    # Create contiguous coordinates.
    lineages = sorted(stats["lineage"])
    aa_counts = Counter()
    for (lineage, aa), count in stats["lineage_aa"].most_common():
        if count * 2 >= stats["lineage"][lineage]:
            aa_counts[aa] += count
    logger.info(f"kept {len(aa_counts)}/{len(stats['aa'])} aa substitutions")
    aa_mutations = [aa for aa, _ in aa_counts.most_common()]

    # Create a dense feature matrix.
    aa_features = torch.zeros(len(lineages), len(aa_mutations), dtype=torch.float)
    logger.info(
        f"Saving {tuple(aa_features.shape)} features to {args.features_file_out}"
    )
    for s, lineage in enumerate(lineages):
        for f, aa in enumerate(aa_mutations):
            count = stats["lineage_aa"].get((lineage, aa))
            if count is None:
                continue
            aa_features[s, f] = count / stats["lineage"][lineage]
    features = {
        "lineages": lineages,
        "aa_mutations": aa_mutations,
        "aa_features": aa_features,
    }
    with open(args.features_file_out, "wb") as f:
        torch.save(features, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Nextstrain open data")
    parser.add_argument(
        "--metadata-file-in", default="results/nextstrain/metadata.tsv.gz"
    )
    parser.add_argument("--columns-file-out", default="results/nextstrain.columns.pkl")
    parser.add_argument("--stats-file-out", default="results/nextstrain.stats.pkl")
    parser.add_argument("--features-file-out", default="results/nextstrain.features.pt")
    parser.add_argument("--start-date", default=START_DATE)
    args = parser.parse_args()
    args.start_date = parse_date(args.start_date)
    main(args)
