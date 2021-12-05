# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import json
import logging
import os
import pickle
import warnings
from collections import Counter, defaultdict

from pyrocov import pangolin
from pyrocov.geo import gisaid_normalize
from pyrocov.mutrans import START_DATE
from pyrocov.util import open_tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

DATE_FORMATS = {4: "%Y", 7: "%Y-%m", 10: "%Y-%m-%d"}


def parse_date(string):
    fmt = DATE_FORMATS.get(len(string))
    if fmt is None:
        # Attempt to fix poorly formated dates like 2020-09-1.
        parts = string.split("-")
        parts = parts[:1] + [f"{int(p):>02d}" for p in parts[1:]]
        string = "-".join(parts)
        fmt = DATE_FORMATS[len(string)]
    return datetime.datetime.strptime(string, fmt)


FIELDS = ["virus_name", "accession_id", "collection_date", "location", "add_location"]


def main(args):
    logger.info(f"Filtering {args.gisaid_file_in}")
    if not os.path.exists(args.gisaid_file_in):
        raise OSError(f"Missing {args.gisaid_file_in}; you may need to request a feed")
    os.makedirs("results", exist_ok=True)

    columns = defaultdict(list)
    stats = defaultdict(Counter)
    covv_fields = ["covv_" + key for key in FIELDS]

    for i, line in enumerate(open_tqdm(args.gisaid_file_in)):
        # Optimize for faster reading.
        line, _ = line.split(', "sequence": ', 1)
        line += "}"

        # Filter out bad data.
        datum = json.loads(line)
        if len(datum["covv_collection_date"]) < 7:
            continue  # Drop rows with no month information.
        date = parse_date(datum["covv_collection_date"])
        if date < args.start_date:
            date = args.start_date  # Clip rows before start date.
        lineage = datum["covv_lineage"]
        if lineage in (None, "None", ""):
            continue  # Drop rows with unknown lineage.
        try:
            lineage = pangolin.compress(lineage)
            lineage = pangolin.decompress(lineage)
            assert lineage
        except (ValueError, AssertionError) as e:
            warnings.warn(str(e))
            continue

        # Fix duplicate locations.
        datum["covv_location"] = gisaid_normalize(datum["covv_location"])

        # Collate.
        columns["lineage"].append(lineage)
        for covv_key, key in zip(covv_fields, FIELDS):
            columns[key].append(datum[covv_key])
        columns["day"].append((date - args.start_date).days)

        # Aggregate statistics.
        stats["date"][datum["covv_collection_date"]] += 1
        stats["location"][datum["covv_location"]] += 1
        stats["lineage"][lineage] += 1

        if i >= args.truncate:
            break

    num_dropped = i + 1 - len(columns["day"])
    logger.info(f"dropped {num_dropped}/{i+1} = {num_dropped*100/(i+1):0.2g}% rows")

    logger.info(f"saving {args.columns_file_out}")
    with open(args.columns_file_out, "wb") as f:
        pickle.dump(dict(columns), f)

    logger.info(f"saving {args.stats_file_out}")
    with open(args.stats_file_out, "wb") as f:
        pickle.dump(dict(stats), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GISAID data")
    parser.add_argument("--gisaid-file-in", default="results/gisaid.json")
    parser.add_argument("--columns-file-out", default="results/gisaid.columns.pkl")
    parser.add_argument("--stats-file-out", default="results/gisaid.stats.pkl")
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--truncate", default=int(1e10), type=int)
    args = parser.parse_args()
    args.start_date = parse_date(args.start_date)
    main(args)
