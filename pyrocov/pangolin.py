# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import warnings
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple
import logging

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

PANGOLIN_REPO = os.path.expanduser(
    os.environ.get("PANGOLIN_REPO", "~/github/cov-lineages/pango-designation")
)

# See https://cov-lineages.org/lineage_list.html or
# https://github.com/cov-lineages/pango-designation/blob/master/pango_designation/alias_key.json
# This list can be updated via update_aliases() below.
PANGOLIN_ALIASES = {
    # "AA": "B.1.177.15",
    # "AB": "B.1.160.16",
    # "AC": "B.1.1.405",
    # "AD": "B.1.1.315",
    # "AE": "B.1.1.306",
    # "AF": "B.1.1.305",
    # "AG": "B.1.1.297",
    # "AH": "B.1.1.241",
    # "AJ": "B.1.1.240",
    # "AK": "B.1.1.232",
    # "AL": "B.1.1.231",
    # "AM": "B.1.1.216",
    # "AN": "B.1.1.200",
    # "AP": "B.1.1.70",
    # "AQ": "B.1.1.39",
    # "AS": "B.1.1.317",
    # "AT": "B.1.1.370",
    # "AU": "B.1.466.2",
    # "AV": "B.1.1.482",
    # "AW": "B.1.1.464",
    # "AY": "B.1.617.2",
    # "AZ": "B.1.1.318",
    # "BA": "B.1.1.529",
    # "BB": "B.1.621.1",
    # "C": "B.1.1.1",
    # "D": "B.1.1.25",
    # "E": "B.1.416",
    # "F": "B.1.36.17",
    # "G": "B.1.258.2",
    # "H": "B.1.1.67",
    # "I": "B.1.1.217",
    # "J": "B.1.1.250",
    # "K": "B.1.1.277",
    # "L": "B.1.1.10",
    # "M": "B.1.1.294",
    # "N": "B.1.1.33",
    # "P": "B.1.1.28",
    # "Q": "B.1.1.7",
    # "R": "B.1.1.316",
    # "S": "B.1.1.217",
    # "U": "B.1.177.60",
    # "V": "B.1.177.54",
    # "W": "B.1.177.53",
    # "Y": "B.1.177.52",
    # "Z": "B.1.177.50",
    # "XA": "B.1.1.7",
    # "XB": "B.1.634",
    # "XC": "B.1.617.2.29",  # i.e. AY.29
    "C": "B.1.1.1",
    "D": "B.1.1.25",
    "G": "B.1.258.2",
    "K": "B.1.1.277",
    "L": "B.1.1.10",
    "M": "B.1.1.294",
    "N": "B.1.1.33",
    "P": "B.1.1.28",
    "Q": "B.1.1.7",
    "R": "B.1.1.316",
    "S": "B.1.1.217",
    "U": "B.1.177.60",
    "V": "B.1.177.54",
    "W": "B.1.177.53",
    "Y": "B.1.177.52",
    "Z": "B.1.177.50",
    "AA": "B.1.177.15",
    "AB": "B.1.160.16",
    "AC": "B.1.1.405",
    "AD": "B.1.1.315",
    "AE": "B.1.1.306",
    "AF": "B.1.1.305",
    "AG": "B.1.1.297",
    "AH": "B.1.1.241",
    "AJ": "B.1.1.240",
    "AK": "B.1.1.232",
    "AL": "B.1.1.231",
    "AM": "B.1.1.216",
    "AN": "B.1.1.200",
    "AP": "B.1.1.70",
    "AQ": "B.1.1.39",
    "AS": "B.1.1.317",
    "AT": "B.1.1.370",
    "AU": "B.1.466.2",
    "AV": "B.1.1.482",
    "AW": "B.1.1.464",
    "AY": "B.1.617.2",
    "AZ": "B.1.1.318",
    "BA": "B.1.1.529",
    "BB": "B.1.621.1",
    "BC": "B.1.1.529.1.1.1",
    "BD": "B.1.1.529.1.17.2",
    "BE": "B.1.1.529.5.3.1",
    "BF": "B.1.1.529.5.2.1",
    "BG": "B.1.1.529.2.12.1",
    "BH": "B.1.1.529.2.38.3",
    "BJ": "B.1.1.529.2.10.1",
    "BK": "B.1.1.529.5.1.10",
    "BL": "B.1.1.529.2.75.1",
    "BM": "B.1.1.529.2.75.3",
    "BN": "B.1.1.529.2.75.5",
    "BP": "B.1.1.529.2.3.16",
    "BQ": "B.1.1.529.5.3.1.1.1.1",
    "BR": "B.1.1.529.2.75.4",
    "BS": "B.1.1.529.2.3.2",
    "BT": "B.1.1.529.5.1.21",
    "BU": "B.1.1.529.5.2.16",
    "BV": "B.1.1.529.5.2.20",
    "BW": "B.1.1.529.5.6.2",
    "BY": "B.1.1.529.2.75.6",
    "BZ": "B.1.1.529.5.2.3",
    "CA": "B.1.1.529.2.75.2",
    "CB": "B.1.1.529.2.75.9",
    "CC": "B.1.1.529.5.3.1.1.1.2",
    "CD": "B.1.1.529.5.2.31",
    "CE": "B.1.1.529.5.2.33",
    "CF": "B.1.1.529.5.2.27",
    "CG": "B.1.1.529.5.2.26",
    "CH": "B.1.1.529.2.75.3.4.1.1",
    "CJ": "B.1.1.529.2.75.3.1.1.1",
    "CK": "B.1.1.529.5.2.24",
    "CL": "B.1.1.529.5.1.29",
    "CM": "B.1.1.529.2.3.20",
    "CN": "B.1.1.529.5.2.21",
    "CP": "B.1.1.529.5.2.6",
    "CQ": "B.1.1.529.5.3.1.4.1.1",
    "CR": "B.1.1.529.5.2.18",
    "CS": "B.1.1.529.4.1.10",
    "CT": "B.1.1.529.5.2.36",
    "CU": "B.1.1.529.5.1.26",
    "CV": "B.1.1.529.2.75.3.1.1.3",
    "CW": "B.1.1.529.5.3.1.1.1.1.1.1.14",
    "CY": "B.1.1.529.5.2.7",
    "CZ": "B.1.1.529.5.3.1.1.1.1.1.1.1",
    "DA": "B.1.1.529.5.2.38",
    "DB": "B.1.1.529.5.2.25",
    "DC": "B.1.1.529.4.6.5",
    "DD": "B.1.1.529.2.3.21",
    "DE": "B.1.1.529.5.1.23",
    "DF": "B.1.1.529.5.10.1",
    "DG": "B.1.1.529.5.2.24.2.1.1",
    "DH": "B.1.1.529.5.1.22",
    "DJ": "B.1.1.529.5.1.25",
    "DK": "B.1.1.529.5.3.1.1.1.1.1.1.7",
    "DL": "B.1.1.529.5.1.15",
    "DM": "B.1.1.529.5.3.1.1.1.1.1.1.15",
    "DN": "B.1.1.529.5.3.1.1.1.1.1.1.5",
    "DP": "B.1.1.529.5.3.1.1.1.1.1.1.8",
    "DQ": "B.1.1.529.5.2.47",
    "DR": "B.1.1.529.5.3.1.1.1.1.1.1.3",
    "DS": "B.1.1.529.2.75.5.1.3.1",
    "DT": "B.1.1.529.5.3.1.1.1.1.1.1.32",
    "DU": "B.1.1.529.5.3.1.1.1.1.1.1.2",
    "DV": "B.1.1.529.2.75.3.4.1.1.1.1.1",
    "DW": "B.1.1.529.5.3.1.1.2.1",
    "DY": "B.1.1.529.5.2.48",
    "DZ": "B.1.1.529.5.2.49",
    "EA": "B.1.1.529.5.3.1.1.1.1.1.1.52",
    "EB": "B.1.1.529.5.1.35",
    "EC": "B.1.1.529.5.3.1.1.1.1.1.10.1",
    "ED": "B.1.1.529.5.3.1.1.1.1.1.1.18",
    "EE": "B.1.1.529.5.3.1.1.1.1.1.1.4",
    "EF": "B.1.1.529.5.3.1.1.1.1.1.1.13",
    "EG": "XBB.1.9.2",
    "EH": "B.1.1.529.5.3.1.1.1.1.1.1.28",
    "EJ": "B.1.1.529.2.75.5.1.3.8",
    "EK": "XBB.1.5.13",
    "EL": "XBB.1.5.14",
    "EM": "XBB.1.5.7",
    "EN": "B.1.1.529.5.3.1.1.1.1.1.1.46",
    "EP": "B.1.1.529.2.75.3.1.1.4",
    "EQ": "B.1.1.529.5.1.33",
    "ER": "B.1.1.529.5.3.1.1.1.1.1.1.22",
    "ES": "B.1.1.529.5.3.1.1.1.1.1.1.65",
    "ET": "B.1.1.529.5.3.1.1.1.1.1.1.35",
    "EU": "XBB.1.5.26",
    "EV": "B.1.1.529.5.3.1.1.1.1.1.1.71",
    "EW": "B.1.1.529.5.3.1.1.1.1.1.1.38",
    "EY": "B.1.1.529.5.3.1.1.1.1.1.1.13.1.1.1",
    "EZ": "B.1.1.529.5.3.1.1.1.1.1.1.43",
    "FA": "B.1.1.529.5.3.1.1.1.1.1.1.10",
    "FB": "B.1.1.529.5.3.1.1.1.1.1.2.1",
    "FC": "B.1.1.529.5.3.1.1.1.1.1.1.72",
    "FD": "XBB.1.5.15",
    "FE": "XBB.1.18.1",
    "FF": "B.1.1.529.5.3.1.1.1.1.1.8.2",
    "FG": "XBB.1.5.16",
    "FH": "XBB.1.5.17",
    "FJ": "B.1.1.529.2.75.3.4.1.1.1.1.19",
    "FK": "B.1.1.529.2.75.3.4.1.1.1.1.17",
    "FL": "XBB.1.9.1",
    "FM": "B.1.1.529.5.3.1.1.1.1.1.1.53",
    "FN": "B.1.1.529.5.3.1.1.1.1.1.1.74",
    "FP": "XBB.1.11.1",
    "FQ": "B.1.1.529.5.3.1.1.1.1.1.1.39",
    "FR": "B.1.1.529.2.75.5.1.2.3",
    "FS": "B.1.1.529.2.75.3.4.1.1.1.1.12",
    "FT": "XBB.1.5.39",
    "FU": "XBB.1.16.1",
    "FV": "B.1.1.529.2.3.20.8.1.1",
    "FW": "XBB.1.28.1",
    "FY": "XBB.1.22.1",
    "FZ": "XBB.1.5.47",
    "GA": "XBB.1.17.1",
    "GB": "XBB.1.5.46",
    "GC": "XBB.1.5.21",
    "GD": "XBB.1.9.3",
    "GE": "XBB.2.3.10",
    "GF": "XBB.1.5.24",
    "GG": "XBB.1.5.38",
    "GH": "XBB.2.6.1",
    "GJ": "XBB.2.3.3",
    "GK": "XBB.1.5.70",
    "GL": "XAY.1.1.1",
    "GM": "XBB.2.3.6",
    "GN": "XBB.1.5.73",
    "GP": "B.1.1.529.2.75.3.4.1.1.1.1.11",
    "GQ": "B.1.1.529.2.75.3.4.1.1.1.1.3",
    "GR": "XBB.1.5.42",
    "GS": "XBB.2.3.11",
    "GT": "XBC.1.6.1",
    "GU": "XBB.1.5.41",
    "GV": "XBB.1.5.48",
    "GW": "XBB.1.19.1",
    "GY": "XBB.1.16.2",
    "GZ": "XBB.2.3.4",
    "HA": "XBB.1.5.86",
    "HB": "XBB.1.34.2",
    "HC": "XBB.1.5.44",
    "HD": "XBB.1.5.93",
    "HE": "XBB.1.18.1.1.1.1",
    "HF": "XBB.1.16.13",
    "HG": "XBB.2.3.8",
    "HH": "XBB.2.3.2",
    "HJ": "XBB.1.5.1",
    "HK": "XBB.1.9.2.5.1.1",
    "HL": "XBB.1.42.2",
    "HM": "XBB.1.5.30",
    "HN": "XBB.1.9.1.1.5.1",
    "HP": "XBB.1.5.55",
    "HQ": "XBB.1.5.92",
    "HR": "XBB.1.5.77",
    "HS": "XBB.1.5.95",
    "HT": "XBB.1.5.49",
    "HU": "XBB.1.22.2",
    "HV": "XBB.1.9.2.5.1.6",
    "HW": "XBC.1.6.3",
    "HY": "XBB.1.5.100",
    "HZ": "XBB.1.5.68",
    "JA": "XBB.2.3.13",
    "JB": "XBB.1.5.53",
    "JC": "XBB.1.41.1",
    "JD": "XBB.1.5.102",
    "JE": "XBB.2.3.3.1.2.1",
    "JF": "XBB.1.16.6",
    "JG": "XBB.1.9.2.5.1.3",
    "JH": "B.1.1.529.5.3.1.1.1.1.1.2.2",
    "JJ": "XBB.1.9.2.5.1.4",
    "JK": "XBB.1.5.3",
    "JL": "B.1.1.529.2.75.3.4.1.1.1.1.17.1.3.2",
    "JM": "XBB.1.16.15",
    "JN": "B.1.1.529.2.86.1",
    "JP": "B.1.1.529.2.75.3.4.1.1.1.1.31",
    "JQ": "B.1.1.529.2.86.3",
    "JR": "XBB.1.9.2.5.1.11",
    "JS": "XBB.2.3.15",
    "JT": "XBC.1.6.6",
    "JU": "XBB.2.3.12",
    "JV": "B.1.1.529.2.75.3.4.1.1.1.1.1.7.1.2",
    "JW": "XBB.1.41.3",
    "JY": "XBB.2.3.19",
    "JZ": "XBB.1.5.107",
    "KA": "XBB.1.5.103",
    "KB": "XBB.1.9.2.5.1.8",
    "KC": "XBB.1.9.1.1.5.2",
    "KD": "XBC.1.3.1",
    "KE": "XBB.1.19.1.5.1.1",
    "KF": "XBB.1.9.1.15.1.1",
    "KG": "B.1.1.529.2.75.3.4.1.1.1.1.1.7.1.5",
    "KH": "XBB.2.3.3.1.2.1.1.1.1",
    "KJ": "XBB.1.16.32",
    "KK": "XBB.1.5.102.1.1.8",
    "KL": "XBB.1.9.2.5.1.6.1.6.1"
}

PANGOLIN_RECOMBINANTS = {
    # "XA": ["B.1.1.7", "B.1.177"],
    # "XB": ["B.1.634", "B.1.631"],
    # "XC": ["AY.29", "B.1.1.7"],
    # "XD": ["B.1.617.2*", "BA.1*"],
    # "XE": ["BA.1*", "BA.2*"],
    # "XF": ["B.1.617.2*", "BA.1*"],
    # "XG": ["BA.1*", "BA.2*"],
    # "XH": ["BA.1*", "BA.2*"],
    # "XJ": ["BA.1*", "BA.2*"],
    # "XK": ["BA.1*", "BA.2*"],
    # "XL": ["BA.1*", "BA.2*"],
    # "XM": ["BA.1.1*", "BA.2*"],
    # "XN": ["BA.1*", "BA.2*"],
    # "XP": ["BA.1.1*", "BA.2*"],
    # "XQ": ["BA.1.1*", "BA.2*"],
    # "XR": ["BA.1.1*", "BA.2*"],
    # "XS": ["B.1.617.2*", "BA.1.1*"],
    # "XT": ["BA.2*", "BA.1*"],
    # "XU": ["BA.1*", "BA.2*"],
    # "XV": ["BA.1*", "BA.2*"],
    # "XW": ["BA.1*", "BA.2*"],
    # "XY": ["BA.1*", "BA.2*"],
    # "XZ": ["BA.2*", "BA.1*"],
    # "XAA": ["BA.1*", "BA.2*"],
    # "XAB": ["BA.1*", "BA.2*"],
    # "XAC": ["BA.2*", "BA.1*", "BA.2*"],
    # "XAD": ["BA.2*", "BA.1*"],
    # "XAE": ["BA.2*", "BA.1*"],
    # "XAF": ["BA.1*", "BA.2*"],
    # "XAG": ["BA.1*", "BA.2*"],
    # "XAH": ["BA.2*", "BA.1*"],
    # "XAJ": ["BA.2.12.1*", "BA.4*"],
    # "XAK": ["BA.2*", "BA.1*", "BA.2*"],
    # "XAL": ["BA.1*", "BA.2*"],
    # "XAM": ["BA.1.1", "BA.2.9"],
    # "XAN": ["BA.2*", "BA.5.1"],
    # "XAP": ["BA.2*", "BA.1*"],
    # "XAQ": ["BA.1*", "BA.2*"],
    # "XAR": ["BA.1*", "BA.2*"],
    # "XAS": ["BA.5*", "BA.2*"],
    # "XAT": ["BA.2.3.13", "BA.1*"],
    # "XAU": ["BA.1.1*", "BA.2.9*"],
    # "XAV": ["BA.2*", "BA.5*"],
    # "XAZ": ["BA.2.5", "BA.5", "BA.2.5"],
    # "XBA": ["AY.45", "BA.2"],
    # "XBB": ["BJ.1", "BM.1.1.1"],
    # "XBC": ["BA.2*", "B.1.617.2*", "BA.2*", "B.1.617.2*"]
    "XA": ["B.1.1.7","B.1.177"],
    "XB": ["B.1.634","B.1.631"],
    "XC": ["AY.29","B.1.1.7"],
    "XD": ["B.1.617.2*","BA.1*"],
    "XE": ["BA.1*","BA.2*"],
    "XF": ["B.1.617.2*","BA.1*"],
    "XG": ["BA.1*","BA.2*"],
    "XH": ["BA.1*","BA.2*"],
    "XJ": ["BA.1*","BA.2*"],
    "XK": ["BA.1*","BA.2*"],
    "XL": ["BA.1*","BA.2*"],
    "XM": ["BA.1.1*","BA.2*"],
    "XN": ["BA.1*","BA.2*"],
    "XP": ["BA.1.1*","BA.2*"],
    "XQ": ["BA.1.1*","BA.2*"],
    "XR": ["BA.1.1*","BA.2*"],
    "XS": ["B.1.617.2*","BA.1.1*"],
    "XT": ["BA.2*","BA.1*"],
    "XU": ["BA.1*","BA.2*"],
    "XV": ["BA.1*","BA.2*"],
    "XW": ["BA.1*","BA.2*"],
    "XY": ["BA.1*","BA.2*"],
    "XZ": ["BA.2*","BA.1*"],
    "XAA": ["BA.1*","BA.2*"],
    "XAB": ["BA.1*","BA.2*"],
    "XAC": ["BA.2*","BA.1*","BA.2*"],
    "XAD": ["BA.2*","BA.1*"],
    "XAE": ["BA.2*","BA.1*"],
    "XAF": ["BA.1*","BA.2*"],
    "XAG": ["BA.1*","BA.2*"],
    "XAH": ["BA.2*","BA.1*"],
    "XAJ": ["BA.2.12.1*","BA.4*"],
    "XAK": ["BA.2*","BA.1*","BA.2*"],
    "XAL": ["BA.1*","BA.2*"],
    "XAM": ["BA.1.1","BA.2.9"],
    "XAN": ["BA.2*","BA.5.1"],
    "XAP": ["BA.2*","BA.1*"],
    "XAQ": ["BA.1*","BA.2*"],
    "XAR": ["BA.1*","BA.2*"],
    "XAS": ["BA.5*","BA.2*"],
    "XAT": ["BA.2.3.13","BA.1*"],
    "XAU": ["BA.1.1*","BA.2.9*"],
    "XAV": ["BA.2*","BA.5*"],
    "XAW": ["BA.2*","AY.122"],
    "XAY": ["BA.2*","AY.45","BA.2*","AY.45","BA.2*"],
    "XAZ": ["BA.2.5","BA.5","BA.2.5"],
    "XBA": ["BA.2*","AY.45","BA.2*","AY.45","BA.2*"],
    "XBB": ["BJ.1","BM.1.1.1"],
    "XBC": ["BA.2*","B.1.617.2*","BA.2*","B.1.617.2*"],
    "XBD": ["BA.2.75.2","BF.5"],
    "XBE": ["BA.5.2","BE.4.1"],
    "XBF": ["BA.5.2.3","CJ.1"],
    "XBG": ["BA.2.76","BA.5.2"],
    "XBH": ["BA.2.3.17","BA.2.75.2"],
    "XBJ": ["BA.2.3.20","BA.5.2"],
    "XBK": ["BA.5.2","CJ.1"],
    "XBL": ["XBB.1.5.57","BA.2.75*","XBB.1.5.57"],
    "XBM": ["BA.2.76","BF.3"],
    "XBN": ["BA.2.75","XBB.3"],
    "XBP": ["BA.2.75*","BQ.1*"],
    "XBQ": ["BA.5.2","CJ.1"],
    "XBR": ["BA.2.75","BQ.1"],
    "XBS": ["BA.2.75","BQ.1"],
    "XBT": ["BA.5.2.34","BA.2.75","BA.5.2.34"],
    "XBU": ["BA.2.75.3","BQ.1","BA.2.75.3"],
    "XBV": ["CR.1","XBB.1"],
    "XBW": ["XBB.1.5","BQ.1.14"],
    "XBY": ["BR.2.1","XBF"],
    "XBZ": ["BA.5.2*","EF.1.3"],
    "XCA": ["BA.2.75*","BQ.1*"],
    "XCB": ["BF.31.1","BQ.1.10*"],
    "XCC": ["CH.1.1.1","XBB.1.9.1"],
    "XCD": ["XBB.1*","BQ.1.1.25*"],
    "XCE": ["BQ.1*","FY.1"],
    "XCF": ["XBB*","FE.1"],
    "XCG": ["BA.5.2*","XBB.1"],
    "XCH": ["GK.1.3","XBB.1.9*","GK.1.3"],
    "XCJ": ["GM.2","FP.1","GM.2"],
    "XCK": ["FL.1.5.1","BQ.1.2.2"],
    "XCL": ["XBB*","GK.2.1.1"],
    "XCM": ["XBB.2.3","DV.7.1"],
    "XCN": ["FR.1.1","EG.5.1.1"],
    "XCP": ["EG.5.1.10","JD.1.1"],
    "XCQ": ["XBB.2.3","XBB.1.5"],
    "XCR": ["GK.1.1.1","FU.1.1.1"],
    "XCS": ["GK.1.9","XBB.1.16.25"],
    "XCT": ["JG.4","DV.7.1","JG.4"],
    "XCU": ["XBC.1.7.1","FL.23.2.1"],
    "XCV": ["XBB.1.16.19","EG.5.1.3","XBB.1.16.19"],
    "XCW": ["XBB.2.3.20","XBB.1.16.15"],
    "XCY": ["EG.5.1.3","GK.4"],
    "XCZ": ["EG.5.1.1","GK.1.1"],
    "XDA": ["XBB.1.16","HN.5"],
    "XDB": ["XBB.1.16.19","XBB"],
    "XDC": ["HK.3","XBB.1.16"],
    "XDD": ["EG.5.1.1","JN.1","EG.5.1.1"],
    "XDE": ["GW.5.1","FL.13.4"],
    "XDF": ["XBB*","EG.5.1.3"],
    "XDG": ["FL.37","EG.5.2.4"],
    "XDH": ["BN.1.2.8","XBB.1.9.1"],
    "XDJ": ["XBB.1.16.6","HK.3.1"],
    "XDK": ["XBB.1.16.11","JN.1.1.1"],
    "XDL": ["EG.5.1.1","XBB*"],
    "XDM": ["XDA","GW.5","XDA"],
    "XDN": ["JN.1.1","JD.1*"],
    "XDP": ["JN.1.4","FL.15"]
}


# From https://www.who.int/en/activities/tracking-SARS-CoV-2-variants/
WHO_ALIASES = {
    # Variants of concern.
    "Alpha": ["B.1.1.7", "Q"],
    "Beta": ["B.1.351", "B.1.351.2", "B.1.351.3"],
    "Gamma": ["P.1", "P.1.1", "P.1.2"],
    "Delta": ["B.1.617.2", "AY"],
    "Omicron": ["B.1.1.529", "BA"],
    # Variants of interest.
    "Lambda": ["C.37"],
    "Mu": ["B.1.621"],
    # Former variants of interest.
    # Epsilon (B.1.427/B.1.429), Zeta (P.2), Theta (P.3)
    "Eta": ["B.1.525"],
    "Iota": ["B.1.526"],
    "Kappa": ["B.1.617.1"],
}
WHO_VOC = ["Alpha", "Beta", "Gamma", "Delta", "Omicron"]
WHO_VOI = ["Lambda", "Mu"]


def update_aliases():
    repo = os.path.expanduser(PANGOLIN_REPO)
    with open(f"{repo}/pango_designation/alias_key.json") as f:
        for k, v in json.load(f).items():
            if isinstance(v, list) and len(v) == 1:
                v = v[0]
            if isinstance(v, str) and v:
                PANGOLIN_ALIASES[k] = v
    return PANGOLIN_ALIASES


def update_recombinants():
    repo = os.path.expanduser(PANGOLIN_REPO)
    with open(f"{repo}/pango_designation/alias_key.json") as f:
        for k, v in json.load(f).items():
            if isinstance(v, list) and v:
                PANGOLIN_RECOMBINANTS[k] = v
    return PANGOLIN_RECOMBINANTS


try:
    update_aliases()
    update_recombinants()
except Exception as e:
    warnings.warn(
        f"Failed to find {PANGOLIN_REPO}, pangolin aliases may be stale.\n{e}",
        RuntimeWarning,
    )

DECOMPRESS = PANGOLIN_ALIASES.copy()
COMPRESS: Dict[str, str] = {}
RE_PANGOLIN = re.compile(r"^[A-Z]+(\.[0-9]+)*$")


def is_pango_lineage(name: str) -> bool:
    """
    Returns whether the name looks like a PANGO lineage e.g. "AY.4.2".
    """
    return RE_PANGOLIN.match(name) is not None


def decompress(name: str) -> str:
    """
    Decompress an alias like C.10 to a full lineage like "B.1.1.1.10".
    """
    if name.startswith("fine"):
        return name
    try:
        return DECOMPRESS[name]
    except KeyError:
        pass
    if name.split(".")[0] in ("A", "B"):
        DECOMPRESS[name] = name
        return name
    for key, value in PANGOLIN_ALIASES.items():
        if name == key or name.startswith(key + "."):
            result = value + name[len(key) :]
            assert result
            DECOMPRESS[name] = result
            return result
    if not any([name.startswith(r) for r in PANGOLIN_RECOMBINANTS.keys()]):
        logger.info("Not an alias or recombinant: {}".format(name))
    return name
#    raise ValueError(f"Unknown alias: {repr(name)}")


def compress(name: str) -> str:
    """
    Compress a full lineage like "B.1.1.1.10" to an alias like "C.10".
    """
    if name.startswith("fine"):
        return name
    try:
        return COMPRESS[name]
    except KeyError:
        pass
    if name.count(".") <= 3:
        result = name
    else:
        for key, value in PANGOLIN_ALIASES.items():
            if key == "I":
                continue  # obsolete
            if name == value or name.startswith(value + "."):
                result = key + name[len(value) :]
                break
    assert is_pango_lineage(result), result
    COMPRESS[name] = result
    return result


assert compress("B.1.1.7") == "B.1.1.7"


def get_parent(name: str) -> Optional[str]:
    """
    Given a decompressed lineage name ``name``, find the decompressed name of
    its parent lineage.
    """
#    assert decompress(name) == name, "expected a decompressed name"
    if name in ("A", "fine"):
        return None
    if name == "B":
        return "A"
#    assert "." in name, name
    if decompress(name) == name and "." in name:
        return name.rsplit(".", 1)[0]
    if name in PANGOLIN_RECOMBINANTS.keys():
        assert isinstance(PANGOLIN_RECOMBINANTS[name], list), name
        parents = PANGOLIN_RECOMBINANTS[name]
        if any(['*' in p for p in parents]):
            parents = [p.rstrip('*') for p in parents]
        return parents[0]


def get_most_recent_ancestor(name: str, ancestors: Set[str]) -> Optional[str]:
    """
    Given a decompressed lineage name ``name``, find the decompressed name of
    its the lineage's most recent ancestor from ``ancestors``. This is like
    :func:`get_parent` but may skip one or more generations in case a parent is
    not in ``ancestors``.
    """
    ancestor = get_parent(name)
    while ancestor is not None and ancestor not in ancestors:
        ancestor = get_parent(ancestor)
    return ancestor


def find_edges(names: List[str]) -> List[Tuple[str, str]]:
    """
    Given a set of short lineages, return a list of pairs of parent-child
    relationships among lineages.
    """
    longnames = [decompress(name) for name in names]
    edges = []
    for x in longnames:
        if x == "A":
            continue  # A is root
        y = get_parent(x)
        while y is not None and y not in longnames:
            y = get_parent(y)
        if y is None:
            continue
        if y != x:
            edges.append((x, y) if x < y else (y, x))
    edges = [(compress(x), compress(y)) for x, y in edges]
    assert len(set(edges)) == len(edges)
    assert len(edges) == len(names) - 1
    return edges


def find_descendents(names: List[str]) -> Dict[str, List[str]]:
    """
    Given a set of short lineages, returns a dict mapping short lineage to its
    list of descendents.
    """
    longnames = [decompress(name) for name in names]
    descendents: Dict[str, List[str]] = {}
    for long1, short1 in zip(longnames, names):
        prefix = long1 + "."
        descendents1 = descendents[short1] = []
        for long2, short2 in zip(longnames, names):
            if long2.startswith(prefix):
                descendents1.append(short2)
    return descendents


def merge_lineages(counts: Dict[str, int], min_count: int) -> Dict[str, str]:
    """
    Given a dict of lineage counts and a min_count, returns a mapping from all
    lineages to merged lineages.
    """
    assert isinstance(counts, dict)
    assert isinstance(min_count, int)
    assert min_count > 0

    # Merge rare children into their parents.
    counts: Dict[str, int] = Counter({decompress(k): v for k, v in counts.items()})
    mapping = {}
    for child in sorted(counts, key=lambda k: (-len(k), k)):
        if counts[child] < min_count:
            parent = get_parent(child)
            if parent is None:
                continue  # at a root
            counts[parent] += counts.pop(child)
            mapping[child] = parent

    # Transitively close.
    for old, new in list(mapping.items()):
        while new in mapping:
            new = mapping[new]
        mapping[old] = new

    # Recompress.
    mapping = {compress(k): compress(v) for k, v in mapping.items()}
    return mapping


def classify(lineage: List[str]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Given a list of compressed lineages, return a torch long tensor of class
    ids and a list of pairs of integers representing parent-child lineage
    relationships between classes.
    """
    # Construct a class tensor.
    names = sorted(set(lineage))
    position = {name: i for i, name in enumerate(names)}
    classes = torch.zeros(len(lineage), dtype=torch.long)
    for i, name in enumerate(lineage):
        classes[i] = position[name]

    # Construct a tree.
    short_to_long = {name: decompress(name) for name in names}
    long_to_short = {v: k for k, v in short_to_long.items()}
    assert len(short_to_long) == len(long_to_short)
    edges = [(position["A"], position["B"])]
    for x, longx in short_to_long.items():
        i = position[x]
        longy = longx.rsplit(".", 1)[0]
        if longy != longx:
            j = position[long_to_short[longy]]
            assert i != j
            edges.append((i, j) if i < j else (j, i))
    assert len(set(edges)) == len(edges)
    assert len(edges) == len(names) - 1

    return classes, edges
