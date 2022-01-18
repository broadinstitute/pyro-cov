# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

"""
Preprocess Nextstrain open data.

This script aggregates the metadata.tsv.gz file available from:
https://docs.nextstrain.org/projects/ncov/en/latest/reference/remote_inputs.html
This file is mirrored on S3 and GCP:
https://data.nextstrain.org/files/ncov/open/metadata.tsv.gz
s3://nextstrain-data/files/ncov/open/metadata.tsv.gz
gs://nextstrain-data/files/ncov/open/metadata.tsv.gz
"""

import argparse
import datetime
import logging
import pickle
from collections import Counter, defaultdict

import torch

from pyrocov.growth import START_DATE, dense_to_sparse
from pyrocov.util import gzip_open_tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def parse_date(string):
    return datetime.datetime.strptime(string, "%Y-%m-%d")


def coarsen_locations(args, counts):
    """
    Select regions that have at least ``args.min_region_size`` samples.
    Remaining regions will be coarsely aggregated up to country level.
    """
    locations = set()
    coarsen_location = {}
    for location, count in counts.items():
        if " / " in location and counts[location] < args.min_region_size:
            old = location
            location = location.split(" / ")[0]
            coarsen_location[old] = location
        locations.add(location)
    locations = sorted(locations)
    logger.info(f"kept {len(locations)}/{len(counts)} locations")
    return locations, coarsen_location


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
        location = row["country"]
        if location in ("", "?"):
            skipped["location"] += 1
            continue
        division = row["division"]
        if division not in ("", "?"):
            location += " / " + division

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
    logger.info(f"kept {len(columns['location'])} rows")
    logger.info(f"skipped {sum(skipped.values())} due to:\n{dict(skipped)}")
    for k, v in stats.items():
        logger.info(f"found {len(v)} {k}s")

    logger.info(f"saving {args.stats_file_out}")
    with open(args.stats_file_out, "wb") as f:
        pickle.dump(stats, f)

    logger.info(f"saving {args.columns_file_out}")
    with open(args.columns_file_out, "wb") as f:
        pickle.dump(columns, f)

    # Create contiguous coordinates.
    locations = sorted(stats["location"])
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
        f"saving {tuple(aa_features.shape)} features to {args.features_file_out}"
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

    # Create a dense dataset.
    locations, coarsen_location = coarsen_locations(args, stats["location"])
    location_id = {location: i for i, location in enumerate(locations)}
    lineage_id = {lineage: i for i, lineage in enumerate(lineages)}
    T = max(stats["day"]) // args.time_step_days + 1
    P = len(locations)
    S = len(lineages)
    counts = torch.zeros(T, P, S)
    for day, location, lineage in zip(
        columns["day"], columns["location"], columns["lineage"]
    ):
        location = coarsen_location.get(location, location)
        t = day // args.time_step_days
        p = location_id[location]
        s = lineage_id[lineage]
        counts[t, p, s] += 1
    logger.info(f"counts data is {counts.ne(0).float().mean().item()*100:0.3g}% dense")
    sparse_counts = dense_to_sparse(counts)
    place_lineage_index = counts.ne(0).any(0).reshape(-1).nonzero(as_tuple=True)[0]
    logger.info(f"saving {tuple(counts.shape)} counts to {args.dataset_file_out}")
    dataset = {
        "start_date": args.start_date,
        "time_step_days": args.time_step_days,
        "locations": locations,
        "lineages": lineages,
        "mutations": aa_mutations,
        "features": aa_features,
        "weekly_counts": counts,
        "sparse_counts": sparse_counts,
        "place_lineage_index": place_lineage_index,
    }
    with open(args.dataset_file_out, "wb") as f:
        torch.save(dataset, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-file-in", default="results/nextstrain/metadata.tsv.gz"
    )
    parser.add_argument("--columns-file-out", default="results/nextstrain.columns.pkl")
    parser.add_argument("--stats-file-out", default="results/nextstrain.stats.pkl")
    parser.add_argument("--features-file-out", default="results/nextstrain.features.pt")
    parser.add_argument("--dataset-file-out", default="results/nextstrain.data.pt")
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--time-step-days", default=14, type=int)
    parser.add_argument("--min-region-size", default=50, type=int)
    args = parser.parse_args()
    args.start_date = parse_date(args.start_date)
    main(args)
