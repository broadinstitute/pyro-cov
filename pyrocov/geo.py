import datetime
import logging
import os
import typing
from collections import OrderedDict, defaultdict

import pandas as pd
import torch

logger = logging.getLogger(__name__)

JHU_DIRNAME = os.path.expanduser(
    "~/github/CSSEGISandData/COVID-19/csse_covid_19_data/csse_covid_19_time_series"
)

# To update see explore-jhu-time-series.ipynb
GISAID_TO_JHU = {
    "a": ("us",),  # typo?
    "aruba": ("netherlands", "aruba"),
    "belgique": ("belgium",),
    "bermuda": ("united kingdom", "bermuda"),
    "british virgin islands": ("united kingdom", "british virgin islands"),
    "bonaire": ("netherlands", "bonaire, sint eustatius and saba"),
    "bosni and herzegovina": ("bosni and herzegovina",),
    "burkinafaso": ("burkina faso",),
    "caribbean": ("dominican republic",),  # most populous island
    "cayman islands": ("united kingdom", "cayman islands"),
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
    "myanmar": ("burma",),
    "netherlans": ("netherlands",),
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
    "united states": ("us",),
    "usa": ("us",),
    "union of the comoros": ("comoros",),
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
    return torch.from_numpy(df.to_numpy()).type_as(torch.tensor(()))


def parse_date(string):
    month, day, year_since_2000 = map(int, string.split("/"))
    return datetime.datetime(day=day, month=month, year=2000 + year_since_2000)


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
