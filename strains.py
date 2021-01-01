import argparse
import os

import pandas as pd
import torch

from pyrophylo.strains import TimeSpaceStrainModel


def main(args):
    # Load time series data from JHU.
    dirname = os.path.expanduser(
        "~/github/CSSEGISandData/COVID-19/"
        "csse_covid_19_data/csse_covid_19_time_series")

    def read_csv(basename):
        return pd.read_csv(os.path.join(dirname, basename), header=0)

    us_cases_df = read_csv("time_series_covid19_confirmed_US.csv")
    us_deaths_df = read_csv("time_series_covid19_deaths_US.csv")
    global_cases_df = read_csv("time_series_covid19_confirmed_global.csv")
    global_deaths_df = read_csv("time_series_covid19_deaths_global.csv")

    def to_torch(df, *, first_column):
        df = df[df.columns[first_column:]]
        return torch.from_numpy(df.to_numpy()).float()

    case_data = torch.cat([to_torch(us_cases_df, first_column=11),
                           to_torch(global_cases_df, first_column=4)])
    death_data = torch.cat([to_torch(us_deaths_df, first_column=12),
                            to_torch(global_deaths_df, first_column=4)])

    # Load preprocessed GISAID data.
    columns = torch.load("results/gisaid.align.pt")["columns"]
    clustering = torch.load("results/gisaid.cluster.pt")
    strain_time = columns["time"]  # FIXME align with case_data, death_data

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
    torch.save(model, args.model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a space-time-strain model")
    parser.add_argument("--model-file", "results/strains.pt")
    parser.add_argument("--learning-rate", default=0.02, type=float)
    parser.add_argument("--num-steps", default=1001, type=int)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    main(args)
