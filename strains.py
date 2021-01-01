import argparse
import datetime
import logging
import os

import pandas as pd
import torch

from pyrophylo.strains import TimeSpaceStrainModel

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

DIRNAME = os.path.expanduser(
    "~/github/CSSEGISandData/COVID-19/"
    "csse_covid_19_data/csse_covid_19_time_series")

# See explore-jhu-time-series.ipynb
GISAID_TO_JHU = {
    "aruba": ("netherlands", "aruba"),
    "crimea": ("ukraine",),  # or "russia"?
    "curacao": ("netherlands", "curacao"),
    "czech repubic": ("czechia",),
    "czech republic": ("czechia",),
    "côte d'ivoire": ("cote d'ivoire",),
    "democratic republic of the congo": ("congo (kinshasa)",),
    "faroe islands": ("denmark", "faroe islands"),
    "gibraltar": ("united kingdom", "gibraltar"),
    "guadeloupe": ("france", "guadeloupe"),
    "guam": ("us", "guam"),
    "hong kong": ("china", "hong kong"),
    "myanmar": ("burma",),
    "palestine": ("israel",),  # ?
    "republic of congo": ("congo (brazzaville)",),
    "reunion": ("france", "reunion"),
    "saint barthélemy": ("france", "saint barthelemy"),
    "saint martin": ("france", "st martin"),
    "south korea": ("korea, south",),
    "st eustatius": ("netherlands", "bonaire, sint eustatius and saba"),
    "taiwan": ("china",),  # ?
    "trinidad": ("trinidad and tobago",),
    "usa": ("us",),
    "viet nam": ("vietnam",),
}


def gisaid_to_jhu_location(gisaid_columns, jhu_us_df, jhu_global_df):
    """
    Fuzzily match GISAID locations with Johns Hopkins locations.
    """
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
    gisaid_to_jhu = {key: tuple(key.lower().split(" / ")[1:])
                     for key in set(gisaid_columns["location"])}

    # Ensure each GISAID location maps to a prefix of some JHU tuple.
    jhu_prefixes = set(jhu_locations)
    for value in jhu_locations:
        for i in range(1, len(value)):
            jhu_prefixes.add(value[:i])
    for key, value in list(gisaid_to_jhu.items()):
        if value[0] in GISAID_TO_JHU:
            value = GISAID_TO_JHU[value[0]] + value[1:]
        while value not in jhu_prefixes:
            value = value[:-1]
            if not value:
                raise ValueError(f"Failed to find GISAID loctaion '{key}' in JHU data")
        gisaid_to_jhu[key] = value

    # Construct a matrix projecting GISAID locations to JHU locations.
    gisaid_index = {value: i for i, value in enumerate(sorted(gisaid_to_jhu.values()))}
    sample_matrix = torch.zeros(len(gisaid_index), len(jhu_locations))
    for j, value in enumerate(jhu_locations):
        for length in range(1, 1 + len(value)):
            fuzzy_value = value[:length]
            i = gisaid_index.get(fuzzy_value, None)
            if i is not None:
                sample_matrix[i, j] = 1

    # Construct a sample_region of GISAID locations.
    gisaid_index = {key: gisaid_index[value] for key, value in gisaid_to_jhu.items()}
    sample_region = torch.empty(len(gisaid_columns["loation"]))
    for i, key in enumerate(gisaid_columns["loation"]):
        sample_region[i] = gisaid_index[key]

    # FIXME we should use inclusion-exlcusion in case some GISAID regions are
    # sub-regions of other GISAID regions.

    return sample_region, sample_matrix


def read_csv(basename):
    return pd.read_csv(os.path.join(DIRNAME, basename), header=0)


def to_torch(df, *, columns):
    df = df[df.columns[columns]]
    return torch.from_numpy(df.to_numpy()).float()


def parse_date(string):
    month, day, year_since_2000 = map(int, string.split("/"))
    return datetime.datetime(day=day, month=month, year=2000 + year_since_2000)


def main(args):
    # Load time series data from JHU.
    us_cases_df = read_csv("time_series_covid19_confirmed_US.csv")
    us_deaths_df = read_csv("time_series_covid19_deaths_US.csv")
    global_cases_df = read_csv("time_series_covid19_confirmed_global.csv")
    global_deaths_df = read_csv("time_series_covid19_deaths_global.csv")
    case_data = torch.cat([to_torch(us_cases_df, columns=slice(11, None)),
                           to_torch(global_cases_df, columns=slice(4, None))])
    death_data = torch.cat([to_torch(us_deaths_df, columns=slice(12, None)),
                            to_torch(global_deaths_df, columns=slice(4, None))])
    assert case_data.shape == death_data.shape
    start_date = parse_date(us_cases_df.columns[11])

    # Convert from cumulative to density.
    case_data[:, 1:] -= case_data[:, :-1].clone()
    death_data[:, 1:] -= death_data[:, :-1].clone()

    # Load preprocessed GISAID data.
    columns = torch.load("results/gisaid.align.pt")["columns"]
    clustering = torch.load("results/gisaid.cluster.pt")
    strain_time = torch.tensor(columns["day"], dtype=torch.long)

    # Convert from daily to weekly observations.
    T, R = case_data.hape
    T /= 7
    case_data = case_data[:T * 7].reshape(T, 7, R).sum(-2)
    death_data = death_data[:T * 7].reshape(T, 7, R).sum(-2)
    strain_time += (args.start_date - start_date).days
    strain_time.clamp_(min=0).floor_divide_(7)

    # Match geographic regions across JUH and GISAID data.
    sample_region, sample_matrix = gisaid_to_jhu_location(
        columns, us_cases_df, global_cases_df)

    # Create a model.
    model = TimeSpaceStrainModel(
        case_data=case_data,
        death_data=death_data,
        transit_data="TODO",
        sample_time=strain_time,
        sample_region=sample_region,
        sample_strain=clustering["classes"],
        sample_matrix=sample_matrix,
        mutation_rate=clustering["transition_matrix"],
        death_rate=args.death_rate,
    )

    # Train.
    model.fit(
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        log_every=args.log_every,
    )
    torch.save(model, args.model_file_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a space-time-strain model")
    parser.add_argument("--model-file-out", "results/strains.pt")
    parser.add_argument("--start-date", default="2019-12-01")
    parser.add_argument("--learning-rate", default=0.02, type=float)
    parser.add_argument("--num-steps", default=1001, type=int)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()
    args.start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")

    main(args)
