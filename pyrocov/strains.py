# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
from timeit import default_timer

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from opt_einsum import contract as einsum
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.infer.autoguide import (
    AutoLowRankMultivariateNormal,
    AutoNormal,
    init_to_median,
)
from pyro.infer.reparam import HaarReparam
from pyro.optim import ClippedAdam
from pyro.poutine.util import prune_subsample_sites

logger = logging.getLogger(__name__)


def OverdispersedPoisson(rate, overdispersion=0, *, gamma_poisson=False):
    if isinstance(overdispersion, (int, float)) and overdispersion == 0:
        return dist.Poisson(rate)
    # Negative Binomial
    #   mean = r p / (1 - p) = rate
    #   variance = r p / (1-p)**2 = rate * (1 + overdispersion * rate)
    # Solving for (r,p):
    #   1 - p = 1 / (1 + o*rate)
    #   p = 1 - 1 / (1 + o*rate)
    #   r = rate * (1-p) / p
    finfo = torch.finfo(rate.dtype)
    rate = rate.clamp(min=1e-3)
    q = (1 + overdispersion * rate).reciprocal()
    q = q.clamp(min=finfo.eps, max=1 - finfo.eps)
    p = 1 - q
    r = rate * q / p
    if gamma_poisson:
        return dist.GammaPoisson(r, q / p)
    else:
        return dist.NegativeBinomial(r, p)


def RelaxedPoisson(rate, overdispersion=0):
    # Overdispersed Poisson
    #   mean = rate
    #   variance = rate * (1 + overdispersion * rate)
    # LogNormal(m,s)
    #   mean = exp(m + s**2/2)
    #   variance = (exp(s**2) - 1) exp(2*m + s**2)
    # Solving for (m,s) given rate:
    #   m + s**2/2 = log(rate)
    #   2*m + s**2 + log(exp(s**2) - 1) = log(rate) + log1p(o * rate)
    # ==> log(rate) = log1p(o * rate) - log(exp(s**2) - 1)
    # ==> (1 + o*rate) / rate = exp(s**2) - 1
    # ==> s**2 = log(1 + (1 + o*rate) / rate) = log(1 + 1/rate + o)
    # ==> m = log(rate) - s**2/2
    s2 = rate.reciprocal().add(overdispersion).log1p()
    m = rate.log() - s2 / 2
    s = s2.sqrt()
    return dist.LogNormal(m, s)


class TimeSpaceStrainModel(nn.Module):
    r"""
    Phylogeographic model to track strains over space and time.

    This model fuses three types of data:

    1.  **Aggregate epidemiological data** in the form of case counts and death
        counts in each (time, region) bucket.
    2.  **Transit data** in the form of a number of covariates believed to
        predict inter-region infection rates. These might combine measured data
        such as flight information and mobile phone records with prior
        covariates such as whether a pair of regions share a border.
    3.  **Genetic sequence** data from infected individuals with known (time,
        region) identity. This model assumes genetic samples are collected
        uniformly within each region, but allows for arbitrary collection rates
        across regions; this model this avoids cross-region bias of sample
        collection.

    Inference has complexity ``O(T * R * S)``, where ``T`` is the number of
    time steps, ``R`` is the number of regions, and ``S`` is the number of
    strains. Inference complexity does not depend on the number of genetic
    samples forming leaves of the phylogeny; therefore this method is suitable
    for coarse phylogenies with ~1000 internal nodes but millions of leaves.
    To ensure the model fits in memory, you might choose coarse time steps of
    weeks for ~100 time steps, coarse regions with ~100-1000 regions (possibly
    with finer detail in a particular region of interest, e.g. foreign
    countries + local provinces), and coarse phylogenies with ~100-1000
    strains.

    :param Tensor case_data: A ``(T,R)``-shaped tensor of confirmed case counts
        in each (time,region) bucket.
    :param Tensor death_data: A ``(T,R)``-shaped tensor of confirmed death
        counts in each (time,region) bucket.
    :param Tensor transit_data: A ``(T,R,R,P)``-shaped tensor of ``P``-many
        covariates, each defining time-dependent region-to-region transition
        rates. Values must be nonnegative.
    :param Tensor sample_time:
    :param Tensor sample_region:
    :param Tensor sample_strain: Three integer vectors of shape ``(N,)``
        containing the time, region, and strain classification of each of ``N``
        genetic samples.
    :param Tensor sample_matrix: A projection matrix of shape ``(Rs,R)`` whose
        ``(c,f)`` entry is 1 iff fine region ``f`` is included in coarse
        sampling region ``c``.
    :param Tensor mutation_matrix: An ``(S,S)``-shaped matrix of normalized
        mutation rates among strains. This could be constructed e.g. by
        estimating a coarse phylogeny among strains, and measuring the edge
        distance between each pair of strains, and marginalizing over spanning
        trees.
    :param Tensor population: An optional ``(R,)``-shaped vector upper bounds
        on the population of each region.
    """

    def __init__(
        self,
        *,
        case_data,
        death_data,
        transit_data,
        sample_time,
        sample_region,
        sample_strain,
        sample_matrix,
        mutation_matrix,
        death_rate,
        population=None,
    ):
        T, R = case_data.shape
        assert death_data.shape == (T, R)
        if population is not None:
            assert population.shape == (R,)
        P = transit_data.size(-1)
        assert transit_data.shape == (R, R, P)
        assert isinstance(death_rate, float) and 0 < death_rate < 1
        assert transit_data.min() >= 0, "transit data must be nonnegative"
        N = len(sample_time)
        assert sample_time.max().item() <= T, "GISAID data is too far ahead of JHU data"
        assert sample_time.shape == (N,)
        assert sample_region.shape == (N,)
        assert sample_strain.shape == (N,)
        Rc = sample_matrix.size(0)
        assert sample_matrix.shape == (Rc, R)
        assert sample_region.max().item() < Rc
        S = mutation_matrix.size(0)
        assert mutation_matrix.shape == (S, S)
        assert sample_strain.max().item() < S

        logger.info("Aggregating sparse samples into multinomial observations")
        strain_data = torch.zeros(T, Rc, S)
        i = (
            sample_time.clamp(max=T - 1)
            .mul_(Rc)
            .add_(sample_region)
            .mul_(S)
            .add_(sample_strain)
        )
        one = torch.ones(()).expand_as(i)
        strain_data.reshape(-1).scatter_add_(0, i, one)
        strain_mean = strain_data.sum([0, 1])
        strain_mean /= strain_mean.sum()
        strain_total = strain_data.sum(-1)
        strain_mask = strain_total > 0
        strain_data = strain_data[strain_mask]
        strain_total = strain_total[strain_mask]

        logger.info(
            f"Creating model over {T} time steps x {R} regions x {S} strains "
            f"= {T * R * S} buckets"
        )
        self.num_time_steps = T
        self.num_regions = R
        self.num_coarse_regions = Rc
        self.num_strains = S
        self.num_transit_covariates = P

        super().__init__()
        if population is None:
            self.population = None
        else:
            self.register_buffer("population", population[:, None])
        self.register_buffer("case_data", case_data)
        self.register_buffer("death_data", death_data)
        self.register_buffer("transit_data", transit_data)
        self.register_buffer("strain_mask", strain_mask)
        self.register_buffer("strain_data", strain_data)
        self.register_buffer("strain_total", strain_total)
        self.register_buffer("strain_mean", strain_mean)
        self.register_buffer("sample_matrix", sample_matrix)
        self.register_buffer("mutation_matrix", mutation_matrix)
        self.death_rate = death_rate

    def model(self):
        T = self.num_time_steps
        R = self.num_regions
        S = self.num_strains
        P = self.num_transit_covariates
        time_plate = pyro.plate("time", T, dim=-3)
        step_plate = pyro.plate("step", T - 1, dim=-3)
        region_plate = pyro.plate("region", R, dim=-2)
        strain_plate = pyro.plate("strain", S, dim=-1)

        # Sample confirmed case response rate parameters.
        # This factorizes over a time-dependent factor
        with time_plate:
            case_rate_time = pyro.sample("case_rate_time", dist.Beta(1, 2))
        # and a region-dependent factor.
        with region_plate:
            case_rate_region = pyro.sample("case_rate_region", dist.Beta(1, 2))
        case_rate = case_rate_time * case_rate_region

        # Sample local spreading dynamics.
        # This factorizes into a global factor R0,
        R0 = pyro.sample("R0", dist.LogNormal(0, 1))
        # a strain-dependent factor Rs,
        Rs_scale = pyro.sample("Rs_scale", dist.LogNormal(-2, 2))
        with strain_plate:
            Rs = pyro.sample("Rs", dist.LogNormal(1, Rs_scale))
        # and a time-region dependent factor Rtr
        Rtr_scale = pyro.sample("Rtr_scale", dist.LogNormal(-2, 2))
        with time_plate, region_plate:
            Rtr = pyro.sample("Rtr", dist.LogNormal(1, Rtr_scale))
        # that varies slowly in time via a log-Brownian motion.
        Rtr_drift_scale = pyro.sample("Rtr_drift_scale", dist.LogNormal(-2, 2))
        with step_plate, region_plate:
            pyro.sample(
                "Rtr_drift", dist.LogNormal(0, Rtr_drift_scale), obs=Rtr[1:] / Rtr[:-1]
            )
        Rtrs = R0 * Rs * Rtr

        # Sample inter-region spreading dynamics coefficients.
        transit_rate = pyro.sample(
            "transit_rate", dist.Exponential(1).expand([P]).to_event(1)
        )
        transit_matrix = torch.eye(R) + self.transit_data @ transit_rate
        transit_matrix = transit_matrix / transit_matrix.sum(-1, True)

        # Sample mutation dynamics.
        mutation_rate = pyro.sample("mutation_rate", dist.LogNormal(-5, 5))
        mutation_matrix = torch.eye(S) + self.mutation_matrix * mutation_rate
        mutation_matrix = mutation_matrix / mutation_matrix.sum(-1, True)

        # Sample the number of infections in each (time,region,strain) bucket.
        # We express this as a factor graph
        with time_plate, region_plate, strain_plate:
            uniform_dist = (
                dist.Exponential(1.0)
                if self.population is None
                else dist.Uniform(0.0, self.population)
            )
            infections = pyro.sample("infections", uniform_dist.mask(False))
        # with linear dynamics that factorizes into many parts.
        infection_od = pyro.sample("infection_od", dist.Beta(1, 9))
        with step_plate, region_plate, strain_plate:
            prev_infections = infections[:-1]
            curr_infections = infections[1:]
            pred_infections = einsum(
                "trs,trs,rR,sS->tRS",
                prev_infections,
                Rtrs[:-1],
                transit_matrix,
                mutation_matrix,
            )
            pred_infections.data.clamp_(min=1e-3)
            pyro.sample(
                "infections_step",
                RelaxedPoisson(pred_infections, overdispersion=infection_od),
                obs=curr_infections,
            )

        # Condition on case counts, marginalized over strains.
        infections_sum = infections.sum(-1, True)
        if self.population is not None:
            # Soft bound infections within each region to population bound.
            infections_sum = (
                infections_sum.div(-self.population).expm1().mul(-self.population)
            )
        case_od = pyro.sample("case_od", dist.Beta(1, 9))
        with time_plate, region_plate:
            pyro.sample(
                "case_obs",
                OverdispersedPoisson(
                    infections_sum * case_rate, overdispersion=case_od
                ),
                obs=self.case_data.unsqueeze(-1),
            )

        # Condition on death counts, marginalized over strains.
        death_od = pyro.sample("death_od", dist.Beta(1, 9))
        with time_plate, region_plate:
            pyro.sample(
                "death_obs",
                OverdispersedPoisson(
                    infections_sum * self.death_rate, overdispersion=death_od
                ),
                obs=self.death_data.unsqueeze(-1),
            )

        # Condition on strain counts.
        # Note these are partitioned into coarse regions.
        strain_dispersion = pyro.sample("strain_dispersion", dist.Exponential(1.0))
        coarse_infections = einsum("trs,Rr->tRs", infections, self.sample_matrix) + 1e-6
        strain_probs = coarse_infections / coarse_infections.sum(-1, True)
        concentration = strain_probs[self.strain_mask] / strain_dispersion
        pyro.sample(
            "strains",
            dist.DirichletMultinomial(
                total_count=self.strain_total.max().item(),
                concentration=concentration,
            ).to_event(1),
            obs=self.strain_data,
        )

    def fit(
        self,
        *,
        haar=True,
        guide_rank=0,
        init_scale=0.01,
        learning_rate=0.02,
        learning_rate_decay=0.1,
        num_steps=1001,
        jit=False,
        log_every=100,
    ):
        """
        Fits a guide via stochastic variational inference.

        After this is called, the ``.guide`` attribute can
        be used to generate samples, medians, or quantiles.

        :returns: A history of losses during training.
        :rtype: list
        """
        # Configure variational inference.
        logger.info("Running inference...")
        model = self.model
        if haar:
            model = poutine.reparam(model, self._haar_reparam)
        if guide_rank == 0:
            guide = AutoNormal(
                model,
                init_scale=init_scale,
                init_loc_fn=self._init_loc_fn,
            )
        elif guide_rank is None or isinstance(guide_rank, int):
            guide = AutoLowRankMultivariateNormal(
                model,
                init_scale=init_scale,
                rank=guide_rank,
            )
        else:
            raise ValueError(f"Invalid guide_rank: {guide_rank}")
        Elbo = JitTrace_ELBO if jit else Trace_ELBO
        elbo = Elbo(max_plate_nesting=3, ignore_jit_warnings=True)
        optim = ClippedAdam(
            {"lr": learning_rate, "lrd": learning_rate_decay ** (1 / num_steps)}
        )
        svi = SVI(model, guide, optim, elbo)

        # Run inference.
        start_time = default_timer()
        losses = []
        for step in range(num_steps):
            loss = svi.step() / self.case_data.numel()
            losses.append(loss)
            if log_every and step % log_every == 0:
                logger.info(f"step {step: >5d} loss = {loss:0.4g}")
        elapsed = default_timer() - start_time
        logger.info(
            f"SVI took {elapsed:0.1f} seconds, "
            f"{(1 + num_steps) / elapsed:0.1f} step/sec"
        )

        self.guide = guide
        return losses

    @torch.no_grad()
    def _init_loc_fn(self, site):
        name = site["name"]

        # Heuristic initialization.
        if name.startswith("infections"):
            x = (self.case_data + self.death_data)[..., None] * self.strain_mean
            x = x.clamp_(min=1).expand(site["fn"].shape())
            if name == "infections":
                return x
            if name == "infections_haar":
                assert isinstance(site["fn"], dist.TransformedDistribution)
                for t in site["fn"].transforms:
                    x = t(x)
                return x
            raise NotImplementedError(f"Unknown variable: {name}")
        if name == "transit_rate":
            return torch.full(site["fn"].shape(), 1e-3)
        if name == "mutation_rate":
            return torch.full(site["fn"].shape(), 1e-3)
        if name.endswith("_od"):
            return torch.full(site["fn"].shape(), 0.5)

        # Deterministic initialization.
        try:
            return site["fn"].mean
        except (AttributeError, NotImplementedError):
            pass

        # Random low-variance initialization.
        logger.info(f"Randomly initializing {name}")
        return init_to_median(site)

    @staticmethod
    def _haar_reparam(site):
        if site["is_observed"]:
            return
        for f in site["cond_indep_stack"]:
            if f.name == "time":
                return HaarReparam(
                    dim=f.dim - site["fn"].event_dim,
                    flip=True,
                    experimental_allow_batch=True,
                )

    @torch.no_grad()
    def median(self):
        """
        Predicts using variational median values for sampled latent variables.

        :returns: A dict mapping sample site name to value.
        :rtype: dict
        """
        result = self.guide.median()
        with poutine.condition(data=result):
            trace = poutine.trace(self.guide.model).get_trace()
        trace = prune_subsample_sites(trace)
        for name, site in trace.nodes.items():
            if site["type"] == "sample":
                result[name] = site["value"].detach()
        return result


@torch.no_grad()
def simulate(
    num_time_steps,
    num_regions,
    num_strains,
    *,
    num_transit_covariates=4,
    overdispersion=0.5,
    transit_rate=1e-2,
    mutation_rate=1e-2,
    initial_infected=10,
    min_total_infected=1000,
    max_total_infected=10000,
    case_rate=0.25,
    death_rate=0.03,
    prelude=10,
):
    """
    Generate a dataset for testing :class:`TimeSpaceStrainModel`.
    """
    assert 0 < overdispersion < 1
    assert min_total_infected < max_total_infected
    T = num_time_steps
    R = num_regions
    S = num_strains
    P = num_transit_covariates

    # Sample a coarse phylogeny.
    mutation_matrix = torch.zeros(S, S)
    for i in range(1, S):
        j = torch.ones(i).multinomial(1).item()
        mutation_matrix[i, j] = mutation_matrix[j, i] = 1
    mutation_matrix = torch.eye(S) + mutation_matrix * mutation_rate
    mutation_matrix /= mutation_matrix.sum(-1, True)

    # Generate random transit features.
    transit_data = torch.rand(R, R, P).pow(2)
    transit_rate = torch.randn(P).exp() * transit_rate
    transit_matrix = torch.eye(R) + transit_data @ transit_rate
    transit_matrix /= transit_matrix.sum(-1, True)

    # Sequentially simulate infections.
    reproduction_number = torch.tensor(
        (min_total_infected / initial_infected) ** (1 / T)
    )
    infected = torch.zeros(prelude + T, R, S)
    infected[0, 0, 0] = initial_infected
    success = False
    for attempt in range(100):
        for t in range(1, prelude + T):
            rate = einsum(
                "rs,,rR,sS->RS",
                infected[t - 1],
                reproduction_number,
                transit_matrix,
                mutation_matrix,
            )
            infected[t] = OverdispersedPoisson(rate, overdispersion).sample()
            infected[t]
        total = int(infected[prelude:].sum())
        logger.info(f"Sampled {total} infections with R0 = {reproduction_number:0.2f}")
        if total < min_total_infected:
            reproduction_number *= 2 ** (1 / T)
        elif total > max_total_infected:
            reproduction_number *= 0.5 ** (1 / T)
        else:
            success = True
            break
    if not success:
        raise ValueError(
            f"Failed to generate between {min_total_infected} "
            f"and {max_total_infected} infections."
        )
    infected = infected[prelude:].clone()
    infected = infected.round()

    # Simulate aggregate observations.
    infected_sum = infected.sum(-1)
    case_data = OverdispersedPoisson(infected_sum * case_rate, overdispersion).sample()
    death_data = OverdispersedPoisson(
        infected_sum * death_rate, overdispersion
    ).sample()
    case_data = torch.min(case_data, infected_sum)
    death_data = torch.min(death_data, infected_sum)

    # Simulate genetic sequence data.
    sequence_rate = torch.zeros(R).uniform_().pow(4)
    sequence_count = dist.Binomial(case_data, sequence_rate).sample()
    sample_time = []
    sample_region = []
    sample_strain = []
    for t in range(T):
        for r in range(R):
            count = int(sequence_count[t, r])
            if count > 0:
                strain = infected[t, r].multinomial(count, replacement=True)
                sample_strain.append(strain)
                sample_time.append(torch.full_like(strain, t))
                sample_region.append(torch.full_like(strain, r))
    sample_time = torch.cat(sample_time)
    sample_region = torch.cat(sample_region)
    sample_strain = torch.cat(sample_strain)

    return {
        "case_data": case_data,
        "death_data": death_data,
        "transit_data": transit_data,
        "sample_time": sample_time,
        "sample_region": sample_region,
        "sample_strain": sample_strain,
        "sample_matrix": torch.eye(R),
        "mutation_matrix": mutation_matrix,
        "death_rate": death_rate,
    }
