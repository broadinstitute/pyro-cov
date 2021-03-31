#!/usr/bin/env python

import argparse
import datetime
import glob
import json
import logging
import os
import pickle
import re
from collections import Counter, defaultdict
from subprocess import check_call

from pyrocov import pangolin
from pyrocov.fasta import ShardedFastaWriter
from pyrocov.hashsubset import RandomSubDict

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

DATE_FORMATS = {4: "%Y", 7: "%Y-%m", 10: "%Y-%m-%d"}


def parse_date(string):
    return datetime.datetime.strptime(string, DATE_FORMATS[len(string)])


FIELDS = ["virus_name", "accession_id", "collection_date", "location", "add_location"]


def main(args):
    logger.info(f"Filtering {args.gisaid_file_in}")
    if not os.path.exists(args.gisaid_file_in):
        raise OSError(
            "Each user must independently request a data feed from gisaid.org"
        )
    if not os.path.exists("results"):
        os.makedirs("results")

    columns = defaultdict(list)
    stats = defaultdict(Counter)
    covv_fields = ["covv_" + key for key in FIELDS]
    subsamples = defaultdict(lambda: RandomSubDict(args.samples_per_lineage))

    with open(args.gisaid_file_in) as f:
        for i, line in enumerate(f):
            # Filter out bad data.
            datum = json.loads(line)
            if len(datum["covv_collection_date"]) < 7:
                continue  # Drop rows with no month information.
            date = parse_date(datum["covv_collection_date"])
            if date < args.start_date:
                continue  # Drop rows before start date.
            lineage = datum["covv_lineage"]
            if lineage in (None, "None"):
                continue  # Drop rows with unknown lineage.
            lineage = pangolin.decompress(lineage)

            # Collate.
            columns["lineage"].append(lineage)
            for covv_key, key in zip(covv_fields, FIELDS):
                columns[key].append(datum[covv_key])
            columns["day"].append((date - args.start_date).days)

            # Aggregate statistics.
            stats["date"][datum["covv_collection_date"]] += 1
            stats["location"][datum["covv_location"]] += 1
            stats["lineage"][lineage] += 1

            # Collect samples.
            if args.fasta_file_out:
                seq = datum["sequence"].replace("\n", "")
                parts = re.findall("[ACGT]+", seq)
                if args.min_nchars <= sum(map(len, parts)) <= args.max_nchars:
                    subsamples[lineage][datum["covv_accession_id"]] = datum["sequence"]

            if i % args.log_every == 0:
                print(".", end="", flush=True)
            if i >= args.truncate:
                break

    num_dropped = i + 1 - len(columns["day"])
    logger.info(f"dropped {num_dropped}/{i+1} = {num_dropped/(i+1):0.2g}% rows")

    logger.info(f"saving {args.columns_file_out}")
    with open(args.columns_file_out, "wb") as f:
        pickle.dump(dict(columns), f)

    logger.info(f"saving {args.stats_file_out}")
    with open(args.stats_file_out, "wb") as f:
        pickle.dump(dict(stats), f)

    num_sequences = sum(len(v) for v in subsamples.values())
    logger.info(f"saving {num_sequences} sequences to {args.fasta_file_out}")
    with ShardedFastaWriter(args.fasta_file_out) as f:
        for lineage, samples in subsamples.items():
            for accession_id, sequence in samples.items():
                f.write(f"{lineage} {accession_id}", sequence)

    for fasta_filename in glob.glob(args.fasta_file_out):
        tsv_filename = fasta_filename.replace(".fasta", ".tsv")
        cmd = [
            "nextclade",
            "--input-fasta",
            fasta_filename,
            "--output-tsv",
            tsv_filename,
        ]
        logger.info(" ".join(cmd))
        check_call(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GISAID data")
    parser.add_argument(
        "--gisaid-file-in", default=os.path.expanduser("~/data/gisaid/provision.json")
    )
    parser.add_argument("--columns-file-out", default="results/gisaid.columns.pkl")
    parser.add_argument("--fasta-file-out", default="results/gisaid.subset.*.fasta")
    parser.add_argument("--stats-file-out", default="results/gisaid.stats.pkl")
    parser.add_argument("--start-date", default="2019-12-01")
    parser.add_argument("--min-nchars", default=29000, type=int)
    parser.add_argument("--max-nchars", default=31000, type=int)
    parser.add_argument("-s", "--samples-per-lineage", default=30, type=int)
    parser.add_argument("-l", "--log-every", default=1000, type=int)
    parser.add_argument("--truncate", default=int(1e10), type=int)
    args = parser.parse_args()
    args.start_date = parse_date(args.start_date)
    main(args)
