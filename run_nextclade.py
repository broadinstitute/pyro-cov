#!/usr/bin/env python

import argparse
import json
import logging
import os
import pickle
from collections import Counter

from pyrocov.fasta import NextcladeDB

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def count_mutations(mutation_counts, row):
    # Check whether row is valid
    if row["qc.overallStatus"] != "good":
        return
    for col in ["aaSubstitutions", "aaDeletions"]:
        ms = row[col]
        if ms:
            mutation_counts.update(ms.split(","))


def main(args):
    logger.info(f"Filtering {args.gisaid_file_in}")
    if not os.path.exists(args.gisaid_file_in):
        raise OSError(
            "Each user must independently request a data feed from gisaid.org"
        )
    os.makedirs("results", exist_ok=True)

    db = NextcladeDB()
    schedule = db.maybe_schedule if args.no_new else db.schedule
    mutation_counts = Counter()
    with open(args.gisaid_file_in) as f:
        for i, line in enumerate(f):
            seq = json.loads(line)["sequence"]

            # Filter by length.
            nchars = sum(seq.count(b) for b in "ACGT")
            if args.min_nchars <= nchars <= args.max_nchars:
                schedule(seq, count_mutations, mutation_counts)

            if i % args.log_every == 0:
                print(".", end="", flush=True)
    db.wait(log_every=args.log_every)

    logger.info(f"saving {args.counts_file_out}")
    with open(args.counts_file_out, "wb") as f:
        pickle.dump(dict(mutation_counts), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NextClade on all sequences")
    parser.add_argument(
        "--gisaid-file-in", default=os.path.expanduser("~/data/gisaid/provision.json")
    )
    parser.add_argument("--counts-file-out", default="results/nextclade.counts.pkl")
    parser.add_argument("--min-nchars", default=29000, type=int)
    parser.add_argument("--max-nchars", default=31000, type=int)
    parser.add_argument("--no-new", action="store_true")
    parser.add_argument("-l", "--log-every", default=1000, type=int)
    args = parser.parse_args()
    main(args)
