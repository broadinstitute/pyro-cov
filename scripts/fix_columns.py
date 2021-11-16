# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import glob
import logging
import os
import pickle

import tqdm

from pyrocov.geo import gisaid_normalize

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def main():
    """
    Fixes columns["location"] via gisaid_normalize().
    """
    tempfile = "results/temp.columns.pkl"
    for infile in glob.glob("results/*.columns.pkl"):
        if "temp" in infile:
            continue
        logger.info(f"Processing {infile}")
        with open(infile, "rb") as f:
            columns = pickle.load(f)
        columns["location"] = [
            gisaid_normalize(x) for x in tqdm.tqdm(columns["location"])
        ]
        with open(tempfile, "wb") as f:
            pickle.dump(columns, f)
        os.rename(tempfile, infile)  # atomic


if __name__ == "__main__":
    main()
