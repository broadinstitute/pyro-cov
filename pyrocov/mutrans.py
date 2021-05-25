import datetime
import functools
import logging
import math
import pickle
import re
from collections import Counter, OrderedDict, defaultdict
from timeit import default_timer

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoStructured
from pyro.optim import ClippedAdam
from pyro.poutine.util import site_is_subsample

import pyrocov.geo
from pyrocov import pangolin

logger = logging.getLogger(__name__)

# Reasonable values might be week (7), fortnight (14), or month (28)
TIMESTEP = 7  # in days
GENERATION_TIME = 5.5  # in days
START_DATE = "2019-12-01"


def get_fine_countries(columns, min_samples=1000):
    """
    Select countries that have at least two subregions with at least
    ``min_samples`` samples. These will be finely partitioned into subregions.
    Remaining countries will be coarsely aggregated at country level.
    """
    # Count number of samples in each subregion.
    counts = Counter()
    for location in columns["location"]:
        parts = location.split("/")
        if len(parts) < 2:
            continue
        parts = tuple(p.strip() for p in parts[:3])
        counts[parts] += 1

    # Select fine countries.
    fine_countries = Counter()
    for parts, count in counts.items():
        if count >= min_samples:
            fine_countries[parts[1]] += 1
    fine_countries = frozenset(
        name for name, count in fine_countries.items() if count >= 2
    )
    logger.info(
        "Partitioning the following countries into subregions: {}".format(
            ", ".join(sorted(fine_countries))
        )
    )
    return fine_countries


def load_gisaid_data(
    *,
    device="cpu",
    include={},
    exclude={},
):
    logger.info("Loading data")
    include = {k: re.compile(v) for k, v in include.items()}
    exclude = {k: re.compile(v) for k, v in exclude.items()}
    with open("results/gisaid.columns.pkl", "rb") as f:
        columns = pickle.load(f)
    logger.info("Training on {} rows with columns:".format(len(columns["day"])))
    logger.info(", ".join(columns.keys()))
    fine_countries = get_fine_countries(columns)

    # Filter features into numbers of mutations.
    aa_features = torch.load("results/nextclade.features.pt")
    mutations = aa_features["mutations"]
    features = aa_features["features"].to(
        device=device, dtype=torch.get_default_dtype()
    )
    keep = [m.count(",") == 0 for m in mutations]
    mutations = [m for k, m in zip(keep, mutations) if k]
    features = features[:, keep]
    logger.info("Loaded {} feature matrix".format(" x ".join(map(str, features.shape))))

    # Aggregate regions.
    lineages = list(map(pangolin.compress, columns["lineage"]))
    lineage_id_inv = list(map(pangolin.compress, aa_features["lineages"]))
    lineage_id = {k: i for i, k in enumerate(lineage_id_inv)}
    sparse_data = Counter()
    location_id = OrderedDict()
    skipped = set()
    for virus_name, day, location, lineage in zip(
        columns["virus_name"], columns["day"], columns["location"], lineages
    ):
        row = {"virus_name": virus_name, "location": location}
        if lineage not in lineage_id:
            if lineage not in skipped:
                skipped.add(lineage)
                logger.warning(f"WARNING skipping unsampled lineage {lineage}")
            continue
        if not all(v.search(row[k]) for k, v in include.items()):
            continue
        if any(v.search(row[k]) for k, v in exclude.items()):
            continue
        parts = location.split("/")
        if len(parts) < 2:
            continue
        parts = [p.strip() for p in parts[:3]]
        if parts[1] not in fine_countries:
            parts = parts[:2]
        location = " / ".join(parts)
        p = location_id.setdefault(location, len(location_id))
        s = lineage_id[lineage]
        t = day // TIMESTEP
        sparse_data[t, p, s] += 1

    T = 1 + max(columns["day"]) // TIMESTEP
    P = len(location_id)
    S = len(lineage_id)
    weekly_strains = torch.zeros(T, P, S)
    for (t, p, s), n in sparse_data.items():
        weekly_strains[t, p, s] = n
    logger.info(
        f"Keeping {int(weekly_strains.sum())}/{len(lineages)} rows "
        f"(dropped {len(lineages) - int(weekly_strains.sum())})"
    )

    # Filter regions.
    num_times_observed = (weekly_strains > 0).max(2).values.sum(0)
    ok_regions = (num_times_observed >= 2).nonzero(as_tuple=True)[0]
    ok_region_set = set(ok_regions.tolist())
    logger.info(f"Keeping {len(ok_regions)}/{weekly_strains.size(1)} regions")
    weekly_strains = weekly_strains.index_select(1, ok_regions)
    locations = [k for k, v in location_id.items() if v in ok_region_set]
    location_id = OrderedDict(zip(locations, range(len(locations))))

    # Construct region-local time scales centered around observations.
    num_obs = weekly_strains.sum(-1)
    local_time = torch.arange(float(len(num_obs))) * TIMESTEP / GENERATION_TIME
    local_time = local_time[:, None]
    local_time = local_time - (local_time * num_obs).sum(0) / num_obs.sum(0)

    return {
        "location_id": location_id,
        "mutations": mutations,
        "weekly_strains": weekly_strains,
        "features": features,
        "lineage_id": lineage_id,
        "lineage_id_inv": lineage_id_inv,
        "local_time": local_time,
    }


def subset_gisaid_data(
    gisaid_dataset,
    location_queries=None,
    max_strains=math.inf,
    obs_scale=1.0,
    obs_max=math.inf,
    round_method=None,
):
    old = gisaid_dataset
    new = old.copy()

    # Select locations.
    if location_queries is not None:
        locations = sorted(
            {
                location
                for location in new["location_id"]
                if any(q in location for q in location_queries)
            }
        )
        ids = torch.tensor([old["location_id"][location] for location in locations])
        new["location_id"] = {name: i for i, name in enumerate(locations)}
        new["weekly_strains"] = new["weekly_strains"].index_select(1, ids)
        new["local_time"] = new["local_time"].index_select(1, ids)

    # Select strains.
    if new["weekly_strains"].size(-1) > max_strains:
        ids = (
            new["weekly_strains"]
            .sum([0, 1])
            .sort(0, descending=True)
            .indices[:max_strains]
        )
        new["weekly_strains"] = new["weekly_strains"].index_select(-1, ids)
        new["features"] = new["features"].index_select(0, ids)
        new["lineage_id_inv"] = [new["lineage_id_inv"][i] for i in ids.tolist()]
        new["lineage_id"] = {name: i for i, name in enumerate(new["lineage_id_inv"])}

    # Select mutations.
    gaps = new["features"].max(0).values - new["features"].min(0).values
    ids = (gaps >= 0.5).nonzero(as_tuple=True)[0]
    new["mutations"] = [new["mutations"][i] for i in ids.tolist()]
    new["features"] = new["features"].index_select(-1, ids)

    # Downsample highly-observed (time,place) bins for numerical stability. It
    # is unclear why this is needed, but e.g. England predictions are very bad
    # without downsampling, and @sfs believes this is a numerical issue.
    old_obs_sum = new["weekly_strains"].sum()
    if obs_scale != 1:
        new["weekly_strains"] = new["weekly_strains"] * obs_scale
    if obs_max not in (None, math.inf):
        new["weekly_strains"] = new["weekly_strains"] * (
            obs_max / new["weekly_strains"].sum(-1, True).clamp_(min=obs_max)
        )
    new["weekly_strains"] = round_counts(new["weekly_strains"], round_method)

    logger.info(
        "Selected {}/{} places, {}/{} strains, {}/{} mutations, {}/{} samples".format(
            len(new["location_id"]),
            len(old["location_id"]),
            len(new["lineage_id"]),
            len(old["lineage_id"]),
            len(new["mutations"]),
            len(old["mutations"]),
            int(new["weekly_strains"].sum()),
            int(old_obs_sum),
        )
    )

    return new


def round_counts(counts, method):
    if method is None:
        return counts
    if method == "floor":
        return counts.floor()
    if method == "ceil":
        return counts.ceil()
    if method == "random":
        result = counts.floor()
        result += torch.bernoulli(counts - result)
        return result
    raise ValueError(f"Unknown round_counts method: {repr(method)}")


def load_jhu_data(gisaid_data):
    # Load raw JHU case count data.
    us_cases_df = pyrocov.geo.read_csv("time_series_covid19_confirmed_US.csv")
    global_cases_df = pyrocov.geo.read_csv("time_series_covid19_confirmed_global.csv")
    daily_cases = torch.cat(
        [
            pyrocov.geo.pd_to_torch(us_cases_df, columns=slice(11, None)),
            pyrocov.geo.pd_to_torch(global_cases_df, columns=slice(4, None)),
        ]
    ).T
    logger.info(
        "Loaded {} x {} daily case data, totaling {}".format(
            *daily_cases.shape, daily_cases[-1].sum().item()
        )
    )

    # Convert JHU locations to GISAID locations.
    locations = list(gisaid_data["location_id"])
    matrix = pyrocov.geo.gisaid_to_jhu_location(locations, us_cases_df, global_cases_df)
    assert matrix.shape == (len(locations), daily_cases.shape[-1])
    daily_cases = daily_cases @ matrix.T
    daily_cases[1:] -= daily_cases[:-1].clone()  # cumulative -> density
    daily_cases.clamp_(min=0)
    assert daily_cases.shape[1] == len(gisaid_data["location_id"])

    # Convert daily counts to TIMESTEP counts (e.g. weekly).
    start_date = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    jhu_start_date = pyrocov.geo.parse_date(us_cases_df.columns[11])
    assert start_date < jhu_start_date
    dt = (jhu_start_date - start_date).days
    T = len(gisaid_data["weekly_strains"])
    weekly_cases = daily_cases.new_zeros(T, len(locations))
    for w in range(TIMESTEP):
        t0 = (w + dt) // TIMESTEP
        source = daily_cases[w::TIMESTEP]
        destin = weekly_cases[t0 : t0 + len(source)]
        destin[:] += source[: len(destin)]
    assert weekly_cases.sum() > 0

    return {
        "daily_cases": daily_cases.clamp(min=0),
        "weekly_cases": weekly_cases.clamp(min=0),
    }


def model(dataset, *, places=True, times=True, model_type=""):
    local_time = dataset["local_time"]
    weekly_strains = dataset["weekly_strains"]
    features = dataset["features"]
    if not torch._C._get_tracing_state():
        assert weekly_strains.shape[-1] == features.shape[0]
        assert local_time.shape == weekly_strains.shape[:2]
    T, P, S = weekly_strains.shape
    S, F = features.shape
    place_plate = pyro.plate("place", P, dim=-1)
    time_plate = pyro.plate("time", T, dim=-2)

    # Assume relative growth rate depends on mutations but not place.
    feature_scale = pyro.sample("feature_scale", dist.LogNormal(0, 1))
    rate_coef = pyro.sample(
        "rate_coef",
        dist.SoftLaplace(torch.zeros(F), feature_scale[..., None]).to_event(1),
    )
    rate = pyro.deterministic("rate", 0.01 * (rate_coef @ features.T))

    # Assume initial infections depend on place.
    if not places:
        return
    with place_plate:
        init = pyro.sample("init", dist.SoftLaplace(torch.zeros(S), 10).to_event(1))

    # Model logits as optionally overdispersed.
    with place_plate, time_plate:
        logits = init + rate * local_time[:, :, None]
    if "overdispersed" in model_type:
        noise_scale = pyro.sample("noise_scale", dist.LogNormal(-2, 2))
        with place_plate, time_plate:
            logits = pyro.sample(
                "logits", dist.Normal(logits, noise_scale[..., None]).to_event(1)
            )

    # Finally observe counts.
    with place_plate, time_plate:
        pyro.sample(
            "obs",
            dist.Multinomial(logits=logits, validate_args=False),
            obs=weekly_strains,
        )


class InitLocFn:
    def __init__(self, dataset, init_data={}):
        self.__dict__.update(init_data)

        # Initialize init.
        if "init" not in init_data:
            init = dataset["weekly_strains"].sum(0)
            init.div_(init.sum(-1, True)).add_(0.01 / init.size(-1)).log_()
            init.sub_(init.median(-1, True).values)
            self.init = init
        assert not torch.isnan(self.init).any()
        logger.info(f"init stddev = {self.init.std():0.3g}")

        # Initialize logits.
        if "logits" in init_data:
            pass
        elif "rate_coef" in init_data and "init" in init_data:
            rate = 0.01 * (self.rate_coef @ dataset["features"].T)
            self.logits = self.init + rate * dataset["local_time"][:, :, None]
        else:
            weekly_strains = dataset["weekly_strains"]
            self.logits = weekly_strains.add(1 / weekly_strains.size(-1)).log()
            self.logits -= self.logits.mean(-1, True)

    def __call__(self, site):
        name = site["name"]
        shape = site["fn"].shape()
        if hasattr(self, name):
            result = getattr(self, name)
            assert result.shape == shape
            return result
        if name == "noise_scale":
            return torch.full(shape, 0.001)
        if name == "feature_scale":
            return torch.ones(shape)
        if name == "rate_coef":
            return torch.rand(shape).sub_(0.5).mul_(0.01)
        raise ValueError("InitLocFn found unexpected site {}".format(repr(name)))


class InitRateCoefLinear(torch.nn.Module):
    def __init__(self, P, features):
        super().__init__()
        S, F = features.shape
        self.register_buffer("features", features)
        self.weight_ps = torch.nn.Parameter(torch.zeros(P, S))
        self.weight_ss = torch.nn.Parameter(torch.zeros(S, S))

    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = x[..., None, :] @ self.features.T
        y = x * self.weight_ps + x @ self.weight_ss
        y = y.reshape(batch_shape + (-1,))
        return y


class Guide(AutoStructured):
    """
    Structured guide for training on large data.
    """

    def __init__(self, model, dataset, model_type, guide_type, *, init_loc_fn):
        self.guide_type = guide_type
        conditionals = {}
        dependencies = defaultdict(dict)
        conditionals["feature_scale"] = "delta"
        if guide_type == "map":
            conditionals["rate_coef"] = "delta"
            conditionals["init"] = "delta"
        elif guide_type.startswith("normal_delta"):
            conditionals["rate_coef"] = "normal"
            conditionals["init"] = "delta"
        elif guide_type.startswith("normal"):
            conditionals["rate_coef"] = "normal"
            conditionals["init"] = "normal"
        elif guide_type.startswith("mvn_delta"):
            conditionals["rate_coef"] = "mvn"
            conditionals["rate_coef"] = "delta"
            conditionals["init"] = "delta"
        elif guide_type.startswith("mvn_normal"):
            conditionals["rate_coef"] = "mvn"
            conditionals["init"] = "normal"
        else:
            raise ValueError(f"Unsupported guide type: {guide_type}")

        if "overdispersed" in model_type:
            conditionals["noise_scale"] = "delta"
            conditionals["logits"] = conditionals["init"]

        noise_linear = None
        if guide_type.endswith("_dependent"):
            T, P, S = dataset["weekly_strains"].shape
            S, F = dataset["features"].shape
            dependencies["init"]["rate_coef"] = InitRateCoefLinear(
                P, dataset["features"]
            )

        super().__init__(
            model,
            conditionals=conditionals,
            dependencies=dependencies,
            init_loc_fn=init_loc_fn,
            init_scale=0.01,
        )

        self._dataset_for_init = dataset
        self._noise_linear = noise_linear

    @torch.no_grad()
    @poutine.mask(mask=False)
    def stats(self, dataset, *, num_samples=1000):
        # Compute median point estimate.
        result = {"median": self.median(dataset)}
        trace = poutine.trace(poutine.condition(model, result["median"])).get_trace(
            dataset, times=False
        )
        for name, site in trace.nodes.items():
            if site["type"] == "sample" and not site_is_subsample(site):
                result["median"][name] = site["value"]

        # Compute moments.
        save_params = ["noise_scale", "feature_scale", "rate_coef"]
        with pyro.plate("particles", num_samples, dim=-3):
            samples = {k: v.v for k, v in self.get_deltas(save_params).items()}
            trace = poutine.trace(poutine.condition(model, samples)).get_trace(
                dataset, places=False
            )
        samples = {
            name: site["value"]
            for name, site in trace.nodes.items()
            if site["type"] == "sample"
            if not site_is_subsample(site)
            if name != "obs"
        }
        result["mean"] = {k: v.mean(0).squeeze() for k, v in samples.items()}
        result["std"] = {k: v.std(0).squeeze() for k, v in samples.items()}
        return result

    def loss(self, dataset):
        elbo = Trace_ELBO(max_plate_nesting=2)
        return elbo.loss(self.model, self, dataset)


class FullGuide(AutoLowRankMultivariateNormal):
    """
    Full guide for testing on small subsets of data.
    """

    @torch.no_grad()
    @poutine.mask(mask=False)
    def stats(self, dataset, *, num_samples=1000):
        # Compute median point estimate.
        # FIXME this silently fails for map inference.
        result = {"median": self.median(dataset)}
        trace = poutine.trace(poutine.condition(model, result["median"])).get_trace(
            dataset, times=False
        )
        for name, site in trace.nodes.items():
            if site["type"] == "sample" and not site_is_subsample(site):
                result["median"][name] = site["value"]

        # Compute moments.
        with pyro.plate("particles", num_samples, dim=-3):
            trace = poutine.trace(self).get_trace(dataset)
            trace = poutine.trace(poutine.replay(model, trace)).get_trace(dataset)
        samples = {
            name: site["value"]
            for name, site in trace.nodes.items()
            if site["type"] == "sample"
            if not site_is_subsample(site)
            if name != "obs"
        }
        result["mean"] = {k: v.mean(0).squeeze() for k, v in samples.items()}
        result["std"] = {k: v.std(0).squeeze() for k, v in samples.items()}
        return result

    def loss(self, dataset):
        elbo = Trace_ELBO(max_plate_nesting=2)
        return elbo.loss(self.model, self, dataset)


# Copied from https://github.com/pytorch/pytorch/blob/v1.8.0/torch/distributions/multivariate_normal.py#L69
def _precision_to_scale_tril(P):
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    Lf = torch.cholesky(torch.flip(P, (-2, -1)))
    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    L = torch.triangular_solve(
        torch.eye(P.shape[-1], dtype=P.dtype, device=P.device), L_inv, upper=False
    )[0]
    return L


def fit_svi(
    dataset,
    *,
    model_type,
    guide_type,
    init_data={},
    learning_rate=0.05,
    learning_rate_decay=0.1,
    num_steps=3001,
    num_particles=1,
    clip_norm=10.0,
    jit=True,
    log_every=50,
    seed=20210319,
    check_loss=False,
):
    start_time = default_timer()

    if isinstance(init_data, str):
        init_data = fit_svi(
            dataset,
            model_type=init_data,
            guide_type="map",
            learning_rate=0.05,
            learning_rate_decay=1.0,
            num_steps=1001,
            num_particles=1,
            log_every=log_every,
            seed=seed,
        )["median"]

    logger.info(f"Fitting {guide_type} guide via SVI")
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    param_store = pyro.get_param_store()

    # Initialize guide so we can count parameters and register hooks.
    model_ = functools.partial(model, model_type=model_type)
    init_loc_fn = InitLocFn(dataset, init_data)
    if guide_type == "full":
        guide = FullGuide(model_, init_loc_fn=init_loc_fn, init_scale=0.01)
    else:
        guide = Guide(model_, dataset, model_type, guide_type, init_loc_fn=init_loc_fn)
    guide(dataset)
    num_params = sum(p.numel() for p in guide.parameters())
    logger.info(f"Training guide with {num_params} parameters:")

    # Log gradient norms during inference.
    series = defaultdict(list)

    def hook(g, series):
        series.append(torch.linalg.norm(g.reshape(-1), math.inf).item())

    for name, value in pyro.get_param_store().named_parameters():
        value.register_hook(functools.partial(hook, series=series[name]))

    def optim_config(param_name):
        config = {
            "lr": learning_rate,
            "lrd": learning_rate_decay ** (1 / num_steps),
            "clip_norm": clip_norm,
        }
        scalars = ["noise_scale", "feature_scale"]
        if any("locs." + s in name for s in scalars):
            config["lr"] *= 0.2
        elif "scales" in param_name:
            config["lr"] *= 0.1
        elif "scale_tril" in param_name:
            config["lr"] *= 0.05
        elif "weight" in param_name:
            config["lr"] *= 0.05
        return config

    optim = ClippedAdam(optim_config)
    Elbo = JitTrace_ELBO if jit else Trace_ELBO
    elbo = Elbo(
        max_plate_nesting=2, num_particles=num_particles, vectorize_particles=True
    )
    svi = SVI(model_, guide, optim, elbo)
    losses = []
    num_obs = dataset["weekly_strains"].count_nonzero()
    for step in range(num_steps):
        loss = svi.step(dataset=dataset)
        assert not math.isnan(loss)
        losses.append(loss)
        median = guide.median()
        for name, value in median.items():
            if value.numel() == 1:
                series[name].append(float(value))
        if step % log_every == 0:
            assert median["feature_scale"] > 0.02, "implausibly small feature_scale"
            logger.info(
                "\t".join(
                    [f"step {step: >4d} L={loss / num_obs:0.6g}"]
                    + [
                        "{}={:0.3g}".format(k[:1].upper(), v)
                        for k, v in median.items()
                        if v.numel() == 1
                    ]
                )
            )
        if check_loss and step >= 50:
            prev = torch.tensor(losses[-50:-25], device="cpu").median().item()
            curr = torch.tensor(losses[-25:], device="cpu").median().item()
            assert (curr - prev) < num_obs, "loss is increasing"

    series["loss"] = losses
    result = guide.stats(dataset)
    result["losses"] = losses
    result["series"] = dict(series)
    result["params"] = {k: v.detach().clone() for k, v in param_store.items()}
    result["guide"] = guide.float()
    result["walltime"] = default_timer() - start_time
    return result


def log_stats(dataset, result):
    mutations = dataset["mutations"]
    mean = result["mean"]["rate_coef"].cpu()
    if not mean.shape:
        return  # Work around error in map estimation.
    std = result["std"]["rate_coef"].cpu()
    sig = mean.abs() / std
    logger.info(f"|μ|/σ [median,max] = [{sig.median():0.3g},{sig.max():0.3g}]")
    for m in ["S:D614G", "S:N501Y", "S:E484K", "S:L452R"]:
        i = mutations.index(m)
        logger.info("ΔlogR({}) = {:0.3g} ± {:0.2f}".format(m, mean[i], std[i]))
