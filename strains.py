import argparse
import datetime
import logging
import math
import pickle

import pandas as pd
import torch

from pyrophylo import pangolin
from pyrophylo.geo import JHU_TO_UN, gisaid_to_jhu_location, parse_date, pd_to_torch, read_csv
from pyrophylo.strains import TimeSpaceStrainModel

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def extract_transit_features(args, us_df, global_df, region_tuples):
    R = len(us_df) + len(global_df)
    assert len(region_tuples) == R
    logger.info(f"Extracting transit features among {R} regions")

    # Extract geometric features.
    lat = torch.cat([torch.from_numpy(us_df["Lat"].to_numpy()),
                     torch.from_numpy(global_df["Lat"].to_numpy())]).float()
    lon = torch.cat([torch.from_numpy(us_df["Long_"].to_numpy()),
                     torch.from_numpy(global_df["Long"].to_numpy())]).float()
    lat.mul_(math.pi / 180)
    lon.mul_(math.pi / 180)
    earth_radius_km = 6.371e6
    xyz = torch.stack([lat.cos() * lon.cos(),
                       lat.cos() * lon.sin(),
                       lat.sin()], dim=-1).mul_(earth_radius_km)
    xyz[xyz.sum(-1).isnan()] = 0
    distance_km = torch.cdist(xyz, xyz)
    radii_km = torch.tensor([float(r) for r in args.radii_km.split(",")])
    geometric_features = (distance_km[..., None] / radii_km).square().mul(-0.5).exp()

    # Extract political features.
    country = {t: t[0] if len(t) >= 1 else None for t in region_tuples}
    state = {t: t[1] if len(t) >= 2 else None for t in region_tuples}
    city = {t: t[2] if len(t) >= 3 else None for t in region_tuples}
    country_index = {t: i for i, t in enumerate(set(country.values()))}
    state_index = {t: i for i, t in enumerate(set(state.values()))}
    city_index = {t: i for i, t in enumerate(set(city.values()))}
    place = torch.empty(R, 3)
    for i, t in enumerate(region_tuples):
        place[i, 0] = country_index[country[t]]
        place[i, 1] = state_index[state[t]]
        place[i, 2] = city_index[city[t]]
    same_country = place[:, 0] == place[:, None, 0]
    same_state = place[:, 1] == place[:, None, 1]
    same_city = place[:, 2] == place[:, None, 2]
    political_features = torch.stack([
        same_country,
        same_country & same_state,
        same_country & same_state & same_city,
    ], dim=-1)
    assert political_features.shape == (R, R, 3)

    def cross_features(x, y):
        cross = x[..., None] * y[..., None, :]
        return cross.reshape(*cross.shape[:-2], -1)

    features = torch.cat([
        geometric_features,
        political_features,
        cross_features(geometric_features, political_features),
    ], dim=-1)
    assert features.shape[:2] == (R, R)
    features.diagonal(dim1=0, dim2=1).fill_(0)

    logger.info(f"Created {features.size(-1)} transit features")
    return features


def main(args):
    # Load time series data from JHU.
    us_cases_df = read_csv("time_series_covid19_confirmed_US.csv")
    us_deaths_df = read_csv("time_series_covid19_deaths_US.csv")
    global_cases_df = read_csv("time_series_covid19_confirmed_global.csv")
    global_deaths_df = read_csv("time_series_covid19_deaths_global.csv")
    case_data = torch.cat([pd_to_torch(us_cases_df, columns=slice(11, None)),
                           pd_to_torch(global_cases_df, columns=slice(4, None))]).T
    death_data = torch.cat([pd_to_torch(us_deaths_df, columns=slice(12, None)),
                            pd_to_torch(global_deaths_df, columns=slice(4, None))]).T
    assert case_data.shape == death_data.shape
    start_date = parse_date(us_cases_df.columns[11])

    # Load population data from JHU and UN. These are used as upper bounds only.
    population = None
    if args.population_file_in:
        us_pop = pd_to_torch(us_deaths_df, columns="Population")
        df = pd.read_csv(args.population_file_in, header=0)
        df = df[df["Time"] == 2020]
        df = df[df["Variant"] == "High"]
        pop = {k.lower(): v * 1000
               for k, v in zip(df["Location"].to_list(), df["PopTotal"].to_list())}
        global_pop = []
        for name in global_cases_df["Country/Region"].tolist():
            name = name.lower()
            name = JHU_TO_UN.get(name, name)
            if name is None:  # cruise ship
                global_pop.append(10000.)
            else:
                global_pop.append(float(pop[name]))
        global_pop = torch.tensor(global_pop)
        population = torch.cat([us_pop, global_pop])
        population.clamp_(min=10000.)

    # Convert from cumulative to density.
    case_data[1:] -= case_data[:-1].clone()
    death_data[1:] -= death_data[:-1].clone()
    # Clamp due to retroactive data corrections.
    case_data.clamp_(min=0)
    death_data.clamp_(min=0)

    # Load preprocessed GISAID data.
    with open("results/gisaid.columns.pkl", "rb") as f:
        columns = pickle.load(f)
    strain_time = torch.tensor(columns["day"], dtype=torch.long)

    # Extract classes and mutation matrix.
    columns["lineage"] = list(map(pangolin.compress, columns["lineage"]))
    classes, edges = pangolin.classify(columns["lineage"])
    num_classes = classes.max().item() + 1
    mutation_matrix = torch.zeros(num_classes, num_classes)
    for i, j in edges:
        mutation_matrix[i, j] = 1
        mutation_matrix[j, i] = 1

    # Convert from daily to weekly observations.
    T, R = case_data.shape
    T //= 7
    case_data = case_data[:T * 7].reshape(T, 7, R).sum(-2)
    death_data = death_data[:T * 7].reshape(T, 7, R).sum(-2)
    strain_time += (args.start_date - start_date).days
    strain_time.clamp_(min=0).floor_divide_(7)

    # Match geographic regions across JUH and GISAID data.
    sample_region, sample_matrix, region_tuples = gisaid_to_jhu_location(
        columns, us_cases_df, global_cases_df)

    # Extract transit fetures (geographic + political).
    transit_data = extract_transit_features(
        args, us_cases_df, global_cases_df, region_tuples)

    # Create a model.
    model = TimeSpaceStrainModel(
        population=population,
        case_data=case_data,
        death_data=death_data,
        transit_data=transit_data,
        sample_time=strain_time,
        sample_region=sample_region,
        sample_strain=classes,
        sample_matrix=sample_matrix,
        mutation_matrix=mutation_matrix,
        death_rate=args.death_rate,
        normal_approx=args.normal_approx,
    )
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        model.cuda()
    else:
        print("WARNING it looks like you don't have a GPU or haven't set --cuda. "
              "Training on CPU may take a very long time...")

    # Train.
    losses = model.fit(
        haar=args.haar,
        guide_rank=args.guide_rank,
        learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        num_steps=args.num_steps,
        jit=args.jit,
        log_every=args.log_every,
    )
    torch.save({
        "args": args,
        "model": model,
        "losses": losses,
    }, args.model_file_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a space-time-strain model")
    parser.add_argument("--model-file-out", default="results/strains.pt")
    parser.add_argument("--start-date", default="2019-12-01")
    parser.add_argument("--radii-km", default="10,30,100,300,1000")
    parser.add_argument("--death-rate", default=0.03, type=float)
    parser.add_argument("--normal-approx", action="store_true")
    parser.add_argument("-r", "--guide-rank", default=0, type=int)
    parser.add_argument("--haar", default=True, action="store_true")
    parser.add_argument("--no-haar", action="store_false", dest="haar")
    parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("-n", "--num-steps", default=1501, type=int)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true",
                        default=torch.cuda.is_available())
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("-l", "--log-every", default=1, type=int)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--population-file-in")
    # Ignore population for now.
    # parser.add_argument("--population-file-in",
    #                     default="data/WPP2019_TotalPopulationBySex.csv")
    args = parser.parse_args()
    args.start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")

    main(args)
