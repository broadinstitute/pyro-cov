# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
import os
import re
import typing
from collections import OrderedDict, defaultdict
from typing import Dict, List

import pandas as pd
import torch

logger = logging.getLogger(__name__)

JHU_DIRNAME = os.path.expanduser(
    "~/github/CSSEGISandData/COVID-19/csse_covid_19_data/csse_covid_19_time_series"
)

# To update see explore_gisaid.ipynb
GISAID_NORMALIZE = {
    "Africa / Botswana / Mochud": "Africa / Botswana / Mochudi",
    "Africa / Botswana / Selebe Phikwe": "Africa / Botswana / Selebi Phikwe",
    "Africa / Canary Islands / Las Palmas de Gran Canaria": "Africa / Canary Islands / Las Palmas",
    "Africa / Djibouti / Djbouti": "Africa / Djibouti / Djibouti",
    "Africa / Ethiopia / Addisababa": "Africa / Ethiopia / Addis Ababa",
    "Africa / Ghana / Accra Metro": "frica / Ghana / Accra",
    "Africa / Ghana / Central": "Africa / Ghana / Central Region",
    "Africa / Ghana / Greater Accra, Ghana": "Africa / Ghana / Greater Accra",
    "Africa / Kenya / Meru County": "Africa / Kenya / Meru",
    "Africa / Mali / Mopti Region": "Africa / Mali / Mopti",
    "Africa / Mauritius / Plaine Wilhems": "Africa / Mauritius / Plaines Wilhems",
    "Africa / Mozambique / Inhambabe": "Africa / Mozambique / Inhambane",
    "Africa / Mozambique / Maputo Cidade": "Africa / Mozambique / Maputo",
    "Africa / Togo / Lome Golfe": "Africa / Togo / Lome",
    "Africa / Zimbabwe / Mash. East": "Africa / Zimbabwe / Mashonaland East",
    "Asia / Bangladesh / Cox,s Bazar": "Asia / Bangladesh / Cox's Bazar",
    "Asia / Bangladesh / Mymemensingh": "Asia / Bangladesh / Mymensingh",
    "Asia / India / Maharasthra": "Asia / India / Maharashtra",
    "Asia / India / Pondicherry": "Asia / India / Puducherry",
    "Asia / Indonesia / North Sumatera": "Asia / Indonesia / North Sumatra",
    "Asia / Turkey": "Europe / Turkey",
    "Asia / Vietnam / Langson": "Asia / Vietnam / Lang Son",
    "Asia / Vietnam / Namdinh": "Asia / Vietnam / Nam Dinh",
    "Asia / Vietnam / Nghean": "Asia / Vietnam / Nghe An",
    "Asia / Vietnam / Phutho": "Asia / Vietnam / Phu Tho",
    "Asia / Vietnam / Thanhhoa": "Asia / Vietnam / Thanh Hoa",
    "Asia / Vietnam / Vinhphuc": "Asia / Vietnam / Vinh Phuc",
    "Europe / Croatia / Osijek Baranjacounty": "Europe / Croatia / Osijek Baranja",
    "Europe / England": "Europe / United Kingdom / England",
    "Europe / England / Suffolk": "Europe / United Kingdom / England / Suffolk",
    "Europe / Slovak Republic": "Europe / Slovakia",
    "Europe / Sweden / Vastra Gotalandsregionen": "Europe / Sweden / Vastra Gotalands",
    "North America / U.s Virgin Islands / St Croix": "North America / USA / U.s Virgin Islands / St Croix",
    "North America / U.s Virgin Islands / St Thomas": "North America / USA / U.s Virgin Islands / St Thomas",
    "North America / U.s Virgin Islands": "North America / USA / U.s Virgin Islands",
    "North America / USA / Us Virgin Islands": "North America / USA / U.s Virgin Islands",
    "North America / USA / Virgin Islands Of The U.s": "North America / USA / U.s Virgin Islands",
    "Frica / Ghana / Accra": "Africa / Ghana / Accra",
}

# To update see explore-jhu-time-series.ipynb
GISAID_TO_JHU = {
    "a": ("us",),  # typo?
    "anguilla": ("united kingdom", "anguilla"),
    "antigua": ("antigua and barbuda",),
    "aruba": ("netherlands", "aruba"),
    "belgique": ("belgium",),
    "bermuda": ("united kingdom", "bermuda"),
    "british virgin islands": ("united kingdom", "british virgin islands"),
    "bonaire": ("netherlands", "bonaire, sint eustatius and saba"),
    "bosni and herzegovina": ("bosni and herzegovina",),
    "burkinafaso": ("burkina faso",),
    "caribbean": ("dominican republic",),  # most populous island
    "cayman islands": ("united kingdom", "cayman islands"),
    "canary islands": ("spain",),
    "cote d ivoire": ("cote d'ivoire",),
    "cote divoire": ("cote d'ivoire",),
    "crimea": ("ukraine",),  # or "russia"?
    "curacao": ("netherlands", "curacao"),
    "czech repubic": ("czechia",),
    "congo": ("congo (kinshasa)",),
    "cotedivoire": ("cote d'ivoire",),
    "czech republic": ("czechia",),
    "côte d'ivoire": ("cote d'ivoire",),
    "democratic republic of the congo": ("congo (kinshasa)",),
    "england": ("united kingdom",),
    "faroe islands": ("denmark", "faroe islands"),
    "french guiana": ("france", "french guiana"),
    "french polynesia": ("france", "french polynesia"),
    "gaborone": ("botswana",),
    "gibraltar": ("united kingdom", "gibraltar"),
    "guadeloupe": ("france", "guadeloupe"),
    "guinea bissau": ("guinea-bissau",),
    "guam": ("us", "guam"),
    "guyane": ("france", "french guiana"),
    "guyane francaise": ("france", "french guiana"),
    "hong kong": ("china", "hong kong"),
    "kavadarci": ("north macedonia",),
    "kazkahstan": ("kazakhstan",),
    "kochani": ("north macedonia",),
    "la reunion": ("france", "reunion"),
    "martinique": ("france", "martinique"),
    "mayotte": ("france", "mayotte"),
    "méxico": ("mexico",),
    "montserrat": ("united kingdom", "montserrat"),
    "myanmar": ("burma",),
    "netherlans": ("netherlands",),
    "niogeria": ("nigeria",),
    "northern mariana islands": ("us", "northern mariana islands"),
    "palestine": ("israel",),  # ?
    "polynesia": ("france", "french polynesia"),
    "puerto rico": ("us", "puerto rico"),
    "republic of congo": ("congo (brazzaville)",),
    "republic of the congo": ("congo (brazzaville)",),
    "reunion": ("france", "reunion"),
    "réunion": ("france", "reunion"),
    "romaina": ("romania",),
    "saint barthelemy": ("france", "saint barthelemy"),
    "saint barthélemy": ("france", "saint barthelemy"),
    "saint martin": ("france", "st martin"),
    "sint eustatius": ("netherlands", "bonaire, sint eustatius and saba"),
    "sint maarten": ("france", "st martin"),
    "slovaia": ("slovakia",),
    "south korea": ("korea, south",),
    "st eustatius": ("netherlands", "bonaire, sint eustatius and saba"),
    "st. lucia": ("saint lucia",),
    "swizterland": ("switzerland",),
    "taiwan": ("china",),  # ?
    "trinidad": ("trinidad and tobago",),
    "turks and caicos islands": ("united kingdom", "turks and caicos islands"),
    "united states": ("us",),
    "usa": ("us",),
    "usa? ohio": ("us", "ohio"),
    "union of the comoros": ("comoros",),
    "viet nam": ("vietnam",),
    "wallis and futuna": ("france", "wallis and futuna islands"),
    "slovak republic": ("slovakia",),
    "the bahamas": ("bahamas",),
    "timor leste": ("timor-leste",),
    "rio de janeiro": ("brazil",),
    "parana": ("brazil",),
    "u.s virgin islands": ("us", "us virgin islands"),
    "u.s. virgin islands": ("us", "us virgin islands"),
    "wallis and futuna islands": ("france", "wallis and futuna islands"),
}

# To update see explore-jhu-time-series.ipynb
JHU_TO_UN = {
    "bolivia": "bolivia (plurinational state of)",
    "brunei": "brunei darussalam",
    "burma": "myanmar",
    "congo (brazzaville)": "congo",
    "congo (kinshasa)": "democratic republic of the congo",
    "cote d'ivoire": "côte d'ivoire",
    "diamond princess": None,  # cruise ship
    "iran": "iran (islamic republic of)",
    "korea, south": "republic of korea",
    "kosovo": "serbia",
    "laos": "lao people's democratic republic",
    "moldova": "republic of moldova",
    "ms zaandam": None,  # cruise ship
    "russia": "russian federation",
    "summer olympics 2020": None,  # event
    "syria": "syrian arab republic",
    "taiwan*": "china, taiwan province of china",
    "tanzania": "united republic of tanzania",
    "us": "united states of america",
    "venezuela": "venezuela (bolivarian republic of)",
    "vietnam": "viet nam",
    "west bank and gaza": "israel",
    "slovak republic": "slovakia",
    "the bahamas": "bahamas",
    "parana": "brazil",
}


def read_csv(basename):
    return pd.read_csv(os.path.join(JHU_DIRNAME, basename), header=0)


def pd_to_torch(df, *, columns):
    if isinstance(columns, slice):
        columns = df.columns[columns]
    df = df[columns]
    return torch.from_numpy(df.to_numpy()).type_as(torch.tensor(()))


def parse_date(string):
    month, day, year_since_2000 = map(int, string.split("/"))
    return datetime.datetime(day=day, month=month, year=2000 + year_since_2000)


def gisaid_normalize(gisaid_location):
    if gisaid_location in GISAID_NORMALIZE:
        return GISAID_NORMALIZE[gisaid_location]
    x = gisaid_location

    # Clean up slashes and truncate.
    x = " / ".join(p for p in re.split(r"\s*/\s*", x)[:3] if p)

    # Normalize unicode.
    x = " ".join(p.rstrip(".") for p in x.lower().split())
    x = x.replace("'", " ")
    x = x.replace("-", " ")
    x = x.replace("_", " ")
    x = x.replace("à", "a")
    x = x.replace("á", "a")
    x = x.replace("â", "a")
    x = x.replace("ã", "a")
    x = x.replace("ä", "a")
    x = x.replace("ç", "c")
    x = x.replace("é", "e")
    x = x.replace("ë", "e")
    x = x.replace("ì", "i")
    x = x.replace("í", "i")
    x = x.replace("î", "i")
    x = x.replace("ó", "o")
    x = x.replace("ô", "o")
    x = x.replace("ö", "oe")
    x = x.replace("ü", "ue")
    x = x.replace("ý", "y")
    x = x.replace("ą", "a")
    x = x.replace("ė", "e")
    x = x.replace("ł", "l")
    x = x.replace("ň", "n")
    x = x.replace("ś", "s")
    x = x.replace("š", "s")
    x = x.replace("ų", "u")
    x = x.replace("ž", "z")
    x = x.replace("ơ", "o")
    x = x.replace("ư", "u")
    x = x.replace("ˇ", "")
    x = x.replace("ầ", "a")
    x = x.replace("’", " ")
    x = x.replace("√º", "u")  # Zurich

    # Drop region suffixes.
    x = re.sub(
        " (city|district|metro|region|state|province|county|town|apskr|apskritis|r)$",
        "",
        x,
    )

    # Capitalize
    x = " ".join(p.capitalize() for p in x.split())
    x = re.sub(r"\bUsa\b", "USA", x)

    x = GISAID_NORMALIZE.setdefault(x, x)
    GISAID_NORMALIZE[gisaid_location] = x
    return x


GISAID_COUNTRY = {}


def gisaid_get_country(location: str) -> str:
    if location not in GISAID_COUNTRY:
        country = " / ".join(location.split(" / ")[:2])
        GISAID_COUNTRY[location] = country
    return GISAID_COUNTRY[location]


def gisaid_to_jhu_location(
    gisaid_locations: typing.List[str],
    jhu_us_df: pd.DataFrame,
    jhu_global_df: pd.DataFrame,
):
    """
    Fuzzily match GISAID locations with Johns Hopkins locations.

    :param list gisaid_locations: A list of (unique) GISAID location names.
    :param pandas.DataFrame jhu_us_df: Johns Hopkins daily cases dataframe,
        ``time_series_covid19_confirmed_US.csv``.
    :param pandas.DataFrame jhu_global_df: Johns Hopkins daily cases dataframe,
        ``time_series_covid19_confirmed_global.csv``.
    :returns: A nonnegative weight matrix of shape
        ``(len(gisaid_locations), len(jhu_us_df) + len(jhu_global_df))``
        assuming GISAID locations are non-overlapping.
    """
    assert isinstance(gisaid_locations, list)
    logger.info("Joining GISAID and JHU region codes")

    # Extract location tuples from JHU data.
    jhu_locations: List[tuple] = []
    for i, row in jhu_us_df[["Country_Region", "Province_State", "Admin2"]].iterrows():
        a, b, c = row
        if isinstance(c, str):
            jhu_locations.append((a.lower(), b.lower(), c.lower()))
        else:
            jhu_locations.append((a.lower(), b.lower()))
    for i, row in jhu_global_df[["Country/Region", "Province/State"]].iterrows():
        a, b = row
        if isinstance(b, str):
            jhu_locations.append((a.lower(), b.lower()))
        else:
            jhu_locations.append((a.lower(),))
    assert len(jhu_locations) == len(jhu_us_df) + len(jhu_global_df)
    logger.info(
        f"Matching {len(gisaid_locations)} GISAID regions "
        f"to {len(jhu_locations)} JHU fuzzy regions"
    )
    jhu_location_ids = {k: i for i, k in enumerate(jhu_locations)}

    # Extract location tuples from GISAID data.
    gisaid_to_jhu = OrderedDict(
        (key, tuple(p.strip() for p in key.lower().split("/")[1:]))
        for key in gisaid_locations
    )

    # Ensure each GISAID location maps at least approximately to some JHU tuple.
    jhu_prefixes = defaultdict(list)  # maps prefixes to full JHU locations
    for value in jhu_locations:
        for i in range(1 + len(value)):
            jhu_prefixes[value[:i]].append(value)
    for key, value in list(gisaid_to_jhu.items()):
        if value and value[0] in GISAID_TO_JHU:
            value = GISAID_TO_JHU[value[0]] + value[1:]
        while value not in jhu_prefixes:
            value = value[:-1]
            if not value:
                raise ValueError(f"Failed to find GISAID loctaion '{key}' in JHU data")
        gisaid_to_jhu[key] = value

    # Construct a matrix many-to-many matching GISAID locations to JHU locations.
    matrix = torch.zeros(len(gisaid_locations), len(jhu_locations))
    for i, (gisaid_tuple, jhu_prefix) in enumerate(gisaid_to_jhu.items()):
        for jhu_location in jhu_prefixes[jhu_prefix]:
            j = jhu_location_ids[jhu_location]
            matrix[i, j] = 1
    # Distribute JHU cases evenly among GISAID locations.
    matrix /= matrix.sum(-1, True)
    matrix[~(matrix > 0)] = 0  # remove NANs

    return matrix


def nextstrain_to_jhu_location(
    gisaid_locations: typing.List[str],
    jhu_us_df: pd.DataFrame,
    jhu_global_df: pd.DataFrame,
):
    raise NotImplementedError("TODO")


# From https://gist.github.com/rogerallen/1583593
us_state_to_abbrev: Dict[str, str] = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}

# invert the dictionary
abbrev_to_us_state: Dict[str, str] = {v: k for k, v in us_state_to_abbrev.items()}


def get_canonical_location_generator(recover_missing_USA_state=True, sep=" / "):
    """Generates a function that processes nextstrain metadata locations and converts them to
    canonical location strings, or None if they can't be resolved"""

    # Pre-compile regex
    re_type1 = re.compile(r"(?:USA|UnitedStates)\/([A-Z]{2})-[-_0-9A-Z]+/[0-9]{4}")

    def get_canonical_location_generator_inner(
        strain, region, country, division, location
    ):
        if country == "USA":
            if division == "USA":
                if recover_missing_USA_state:
                    # Division information incorrect, extract state from 'strain'
                    match_obj = re_type1.match(strain)
                    if match_obj:
                        state = match_obj.groups()[0]
                        if state in abbrev_to_us_state.keys():
                            return sep.join([region, country, state])
                        else:
                            return None
                else:
                    return None
            else:
                # Division information provided, convert to state abbr
                try:
                    state = us_state_to_abbrev[division]
                    return sep.join([region, country, state])
                except KeyError:
                    return None
        elif country == "United Kingdom":
            if division in ("England", "Scotland", "Northern Ireland", "Wales"):
                return sep.join([region, country, division])
            else:
                # These could also be discarded (n=622)
                return sep.join([region, country])
        elif country == "Germany":
            return sep.join([region, country, division])
        else:
            # For all other countries only return region and country
            return sep.join([region, country])

    return get_canonical_location_generator_inner
