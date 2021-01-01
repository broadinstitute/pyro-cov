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
    raise NotImplementedError("TODO")

    # Create a model.
    model = TimeSpaceStrainModel(
        case_data=case_data,
        death_data=death_data,
        transit_data="TODO",
        sample_time=strain_time,
        sample_region="TODO",
        sample_strain=clustering["classes"],
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
