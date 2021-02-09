#!/usr/bin/env python

import argparse
import datetime
import json
import logging
import os
import pickle
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

DATE_FORMATS = {4: "%Y", 7: "%Y-%m", 10: "%Y-%m-%d"}


def parse_date(string):
    return datetime.datetime.strptime(string, DATE_FORMATS[len(string)])


FIELDS = ["accession_id", "collection_date", "location", "add_location", "lineage"]


def main(args):
    logger.info(f"Filtering {args.gisaid_file_in}")
    if not os.path.exists(args.gisaid_file_in):
        raise OSError("Each user must independently request a data feed from gisaid.org")
    if not os.path.exists("results"):
        os.makedirs("results")

    columns = defaultdict(list)
    stats = defaultdict(Counter)
    covv_fields = ["covv_" + key for key in FIELDS]

    with open(args.gisaid_file_in) as f:
        for i, line in enumerate(f):
            # Filter out bad data.
            datum = json.loads(line)
            if len(datum["covv_collection_date"]) < 7:
                continue  # Drop rows with no month information.
            date = parse_date(datum["covv_collection_date"])
            if date < args.start_date:
                continue  # Drop rows before start date.
            if datum["covv_lineage"] in (None, "None"):
                continue  # Drop rows with unknown lineage.

            # Collate.
            for covv_key, key in zip(covv_fields, FIELDS):
                columns[key].append(datum[covv_key])
            columns["day"].append((date - args.start_date).days)

            # Aggregate statistics.
            stats["date"][datum["covv_collection_date"]] += 1
            stats["location"][datum["covv_location"]] += 1
            stats["lineage"][datum["covv_lineage"]] += 1

            if i % args.log_every == 0:
                print(".", end="", flush=True)

    num_dropped = i + 1 - len(columns["day"])
    logger.info(f"dropped {num_dropped}/{i+1} = {num_dropped/(i+1):0.2g}% rows")

    logger.info(f"saving {args.columns_file_out}")
    with open(args.columns_file_out, "wb") as f:
        pickle.dump(dict(columns), f)

    logger.info(f"saving {args.stats_file_out}")
    with open(args.stats_file_out, "wb") as f:
        pickle.dump(dict(stats), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GISAID data")
    parser.add_argument("--gisaid-file-in",
                        default=os.path.expanduser("~/data/gisaid/provision.json"))
    parser.add_argument("--columns-file-out", default="results/gisaid.columns.pkl")
    parser.add_argument("--stats-file-out", default="results/gisaid.stats.pkl")
    parser.add_argument("--start-date", default="2019-12-01")
    parser.add_argument("-l", "--log-every", default=1000, type=int)
    args = parser.parse_args()
    args.start_date = parse_date(args.start_date)
    main(args)
