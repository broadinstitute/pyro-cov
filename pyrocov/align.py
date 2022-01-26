# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import os

# Source: https://samtools.github.io/hts-specs/SAMv1.pdf
CIGAR_CODES = "MIDNSHP=X"  # Note minimap2 uses only "MIDNSH"

ROOT = os.path.dirname(os.path.dirname(__file__))
NEXTCLADE_DATA = os.path.expanduser("~/github/nextstrain/nextclade/data/sars-cov-2")
PANGOLEARN_DATA = os.path.expanduser("~/github/cov-lineages/pangoLEARN/pangoLEARN/data")
