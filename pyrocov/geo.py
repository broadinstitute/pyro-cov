import datetime
import logging
import os

import pandas as pd
import torch

logger = logging.getLogger(__name__)

JHU_DIRNAME = os.path.expanduser(
    "~/github/CSSEGISandData/COVID-19/csse_covid_19_data/csse_covid_19_time_series"
)

# To update see explore-jhu-time-series.ipynb
GISAID_TO_JHU = {
    "aruba": ("netherlands", "aruba"),
    "bermuda": ("united kingdom", "bermuda"),
    "british virgin islands": ("united kingdom", "british virgin islands"),
    "bonaire": ("netherlands", "bonaire, sint eustatius and saba"),
    "caribbean": ("dominican republic",),  # most populous island
    "cayman islands": ("united kingdom", "cayman islands"),
    "cote divoire": ("cote d'ivoire",),
    "crimea": ("ukraine",),  # or "russia"?
    "curacao": ("netherlands", "curacao"),
    "czech repubic": ("czechia",),
    "czech republic": ("czechia",),
    "côte d'ivoire": ("cote d'ivoire",),
    "democratic republic of the congo": ("congo (kinshasa)",),
    "faroe islands": ("denmark", "faroe islands"),
    "french guiana": ("france", "french guiana"),
    "gaborone": ("botswana",),
    "gibraltar": ("united kingdom", "gibraltar"),
    "guadeloupe": ("france", "guadeloupe"),
    "guam": ("us", "guam"),
    "guyane francaise": ("france", "french guiana"),
    "hong kong": ("china", "hong kong"),
    "kavadarci": ("north macedonia",),
    "kochani": ("north macedonia",),
    "mayotte": ("france", "mayotte"),
    "myanmar": ("burma",),
    "netherlans": ("netherlands",),
    "palestine": ("israel",),  # ?
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
    "south korea": ("korea, south",),
    "st eustatius": ("netherlands", "bonaire, sint eustatius and saba"),
    "st. lucia": ("saint lucia",),
    "taiwan": ("china",),  # ?
    "trinidad": ("trinidad and tobago",),
    "usa": ("us",),
    "viet nam": ("vietnam",),
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
    "syria": "syrian arab republic",
    "taiwan*": "china, taiwan province of china",
    "tanzania": "united republic of tanzania",
    "us": "united states of america",
    "venezuela": "venezuela (bolivarian republic of)",
    "vietnam": "viet nam",
    "west bank and gaza": "israel",
}


def read_csv(basename):
    return pd.read_csv(os.path.join(JHU_DIRNAME, basename), header=0)


def pd_to_torch(df, *, columns):
    if isinstance(columns, slice):
        columns = df.columns[columns]
    df = df[columns]
    return torch.from_numpy(df.to_numpy()).float()


def parse_date(string):
    month, day, year_since_2000 = map(int, string.split("/"))
    return datetime.datetime(day=day, month=month, year=2000 + year_since_2000)


def gisaid_to_jhu_location(gisaid_columns, jhu_us_df, jhu_global_df):
    """
    Fuzzily match GISAID locations with Johns Hopkins locations.
    """
    logger.info("Joining GISAID and JHU region codes")

    # Extract location tuples from JHU data.
    jhu_locations = []
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

    # Extract location tuples from GISAID data.
    gisaid_to_jhu = {
        key: tuple(p.strip() for p in key.lower().split("/")[1:])
        for key in set(gisaid_columns["location"])
    }

    # Ensure each GISAID location maps to a prefix of some JHU tuple.
    jhu_prefixes = set(jhu_locations) | {()}
    for value in jhu_locations:
        for i in range(1, len(value)):
            jhu_prefixes.add(value[:i])
    for key, value in list(gisaid_to_jhu.items()):
        if value and value[0] in GISAID_TO_JHU:
            value = GISAID_TO_JHU[value[0]] + value[1:]
        while value not in jhu_prefixes:
            value = value[:-1]
            if not value:
                raise ValueError(f"Failed to find GISAID loctaion '{key}' in JHU data")
        gisaid_to_jhu[key] = value

    # Construct a matrix projecting GISAID locations to JHU locations.
    gisaid_keys = {key: i for i, key in enumerate(sorted(gisaid_to_jhu))}
    gisaid_values = {gisaid_to_jhu[key]: i for key, i in gisaid_keys.items()}
    logger.info(
        f"Matching {len(gisaid_keys)} GISAID regions to {len(gisaid_values)} JHU fuzzy regions"
    )
    sample_matrix = torch.zeros(len(gisaid_to_jhu), len(jhu_locations))
    for j, value in enumerate(jhu_locations):
        for length in range(1 + len(value)):
            fuzzy_value = value[:length]
            i = gisaid_values.get(fuzzy_value, None)
            if i is not None:
                sample_matrix[i, j] = 1

    # Construct a sample_region of GISAID locations.
    sample_region = torch.empty(len(gisaid_columns["location"]), dtype=torch.long)
    for i, key in enumerate(gisaid_columns["location"]):
        sample_region[i] = gisaid_keys[key]

    # FIXME we should use inclusion-exlcusion in case some GISAID regions are
    # sub-regions of other GISAID regions.

    return sample_region, sample_matrix, jhu_locations
