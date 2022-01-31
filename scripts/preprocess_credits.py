# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import lzma
import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def main(args):
    logger.info(f"Loading {args.columns_file_in}")
    with open(args.columns_file_in, "rb") as f:
        columns = pickle.load(f)

    logger.info(f"Saving {args.credits_file_out}")
    with lzma.open(args.credits_file_out, "wt") as f:
        f.write("\n".join(sorted(columns["index"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save gisaid accession numbers")
    parser.add_argument("---columns-file-in", default="results/columns.3000.pkl")
    parser.add_argument("--credits-file-out", default="paper/accession_ids.txt.xz")
    args = parser.parse_args()
    main(args)
