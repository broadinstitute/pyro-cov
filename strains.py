import argparse
import datetime
import logging
import math
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
    "bermuda": ("united kingdom", "bermuda"),
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
    "guyane francaise": ("france", "french guiana"),
    "hong kong": ("china", "hong kong"),
    "mayotte": ("france", "mayotte"),
    "myanmar": ("burma",),
    "palestine": ("israel",),  # ?
    "republic of congo": ("congo (brazzaville)",),
    "reunion": ("france", "reunion"),
    "saint barthélemy": ("france", "saint barthelemy"),
    "saint martin": ("france", "st martin"),
    "south korea": ("korea, south",),
    "st eustatius": ("netherlands", "bonaire, sint eustatius and saba"),
    "st. lucia": ("saint lucia",),
    "taiwan": ("china",),  # ?
    "trinidad": ("trinidad and tobago",),
    "usa": ("us",),
    "viet nam": ("vietnam",),
}
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
    gisaid_to_jhu = {key: tuple(p.strip() for p in key.lower().split("/")[1:])
                     for key in set(gisaid_columns["location"])}

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
    logger.info(f"Matching {len(gisaid_keys)} GISAID regions to {len(gisaid_values)} JHU fuzzy regions")
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


def read_csv(basename):
    return pd.read_csv(os.path.join(DIRNAME, basename), header=0)


def to_torch(df, *, columns):
    if isinstance(columns, slice):
        columns = df.columns[columns]
    df = df[columns]
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
                           to_torch(global_cases_df, columns=slice(4, None))]).T
    death_data = torch.cat([to_torch(us_deaths_df, columns=slice(12, None)),
                            to_torch(global_deaths_df, columns=slice(4, None))]).T
    assert case_data.shape == death_data.shape
    start_date = parse_date(us_cases_df.columns[11])

    # Load population data from JHU and UN. These are used as upper bounds only.
    population = None
    if args.population_file_in:
        us_pop = to_torch(us_deaths_df, columns="Population")
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
    columns = torch.load("results/gisaid.align.pt")["columns"]
    clustering = torch.load("results/gisaid.cluster.pt")
    strain_time = torch.tensor(columns["day"], dtype=torch.long)

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
        sample_strain=clustering["classes"],
        sample_matrix=sample_matrix,
        mutation_matrix=clustering["transition_matrix"],
        death_rate=args.death_rate,
        normal_approx=args.normal_approx,
    )
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        model.cuda()

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
