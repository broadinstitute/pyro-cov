import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from opt_einsum import contract as einsum
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.reparam import HaarReparam
from pyro.optim import ClippedAdam
from torch.distributions import constraints


def MomentMatchedPoisson(rate):
    # Poisson mean = variance = rate
    # LogNormal(m,s)
    #   mean = exp(m + s**2/2)
    #   variance = (exp(s**2) - 1) exp(2*m + s**2)
    # Solving for (m,s) given rate:
    #   m + s**2/2 = log(rate) = 2*m + s**2 + log(exp(s*2) - 1)
    # ==> log(rate) = -log(exp(s**2) - 1)
    # ==> 1/rate = exp(s**2) - 1
    # ==> s = sqrt(log(1 + 1/rate))
    # ==> m = log(rate) - s**2/2
    s2 = rate.reciprocal().log1p()
    m = rate.log() - s2 / 2
    s = s2.sqrt()
    return dist.LogNormal(m, s)


class TimeSpaceStrainModel(nn.Module):
    def __init__(
        self,
        case_data,
        death_data,
        transit_data,
        sample_time,
        sample_pos,
        sample_strain,
        strain_distance,
        death_rate,
    ):
        N = len(sample_time)
        assert sample_time.shape == (N,)
        assert sample_pos.shape == (N,)
        assert sample_strain.shape == (N,)
        S = 1 + sample_strain.max().item()
        assert strain_distance.shape == (S, S)
        T, R = case_data.shape
        assert death_data.shape == (T, R)
        P = transit_data.size(-1)
        assert transit_data.shape == (T, R, R, P)
        assert isinstance(death_rate, float) and 0 < death_rate < 1

        # Convert sparse sample data to dense multinomial observations.
        strain_data = torch.zeros(T, R, S)
        i = sample_time.mul(R).add_(sample_pos).mul_(S).add_(sample_strain)
        one = torch.ones(()).expand_as(i)
        strain_data.reshape(-1).scatter_add_(0, i, one)
        strain_total = strain_data.sum(-1)

        self.num_time_steps = T
        self.num_regions = R
        self.num_strains = S
        self.num_transit_covariates = P

        self.register_buffer("case_data", case_data)
        self.register_buffer("death_data", death_data)
        self.register_buffer("transit_data", transit_data)
        self.register_buffer("strain_data", strain_data)
        self.register_buffer("strain_total", strain_total)
        self.register_buffer("strain_distance", strain_distance)
        self.death_rate = death_rate

    def model(self):
        T = self.num_time_steps
        time_plate = pyro.plate("time", T, dim=-3)
        step_plate = pyro.plate("step", T - 1, dim=-3)
        region_plate = pyro.plate("region", T, dim=-2)
        strain_plate = pyro.plate("strain", T, dim=-1)

        # Sample case counting parameters, factored over time x region.
        with time_plate:
            case_rate_region = pyro.sample("case_rate_time", dist.Beta(1, 2))
        with region_plate:
            case_rate_region = pyro.sample("case_rate_region", dist.Beta(1, 2))
        case_rate = case_region_rate * case_time_rate

        # Sample local spreading dynamics.
        # TODO model spatial structure, say hierarchically.
        R0 = pyro.sample("R0", dist.LogNormal(0, 1))
        R_scale = pyro.sample("R_scale", dist.LogNormal(0, 1))
        R_drift_scale = pyro.sample("R_drift_scale", dist.LogNormal(-2, 2))
        with time_plate, region_plate:
            Rt = pyro.sample("Rt", dist.LogNormal(R0, R_scale))
        with step_plate, region_plate:
            pyro.sample("R_drift", dist.LogNormal(0, R_drift_scale),
                        obs=Rt[1:] / Rt[:-1])

        # Sample inter-region spreading dynamics.
        transit_coef = pyro.sample("transit_coef",
                                   dist.Exponential(1).expand(P).to_event(1))

        # Sample mutation dynamics.
        mutation_rate = pyro.sample("mutation_rate", dist.LogNormal(-5, 5))
        mutation_scale = pyro.sample("mutation_scale", dist.LogNormal(0, 1))
        with strain_plate:
            concentration = ((-mutation_rate) * self.strain_distance).exp())
            concentration = concentration * mutation_scale
            strain_rate = pyro.sample("strain_rate",
                                      dist.Dirichlet(concentration))

        # Sample infections as a factor graph.
        with time_plate, region_plate, strain_plate:
            infections = pyro.sample(
                "infections",
                dist.ImproperUniform(constraints.positive, (), ()),
            )
        with pyro.plate("dt", T - 1, dim=-3), region_plate, strain_plate:
            pred = einsum(
                "trs,tr,trRp,p,sS->tRS",
                infections[:-1],
                Rt[:-1],
                transit_data[:-1],
                transit_rate,
                strain_rate,
            )
            pyro.sample("infections_step", MomentMatchedPoisson(pred),
                        obs=infections[1:])

        # The remainder of the model concerns time-region local observations.
        infections_sum = infections.sum(-1, True)
        strain_probs = infections / infections_sum
        with time_plate, region_plate:
            # Condition on case counts, marginalized over strains.
            # TODO use overdispersed distribution.
            pyro.sample("case_obs", dist.Poission(infections_sum * case_rate),
                        obs=self.case_data)

            # Condition on death counts, marginalized over strains.
            # TODO use overdispersed distribution.
            pyro.sample("death_obs", dist.Poission(infections_sum * death_rate),
                        obs=self.death_data)

            # Condition on strain counts.
            pyro.sample("strains",
                        dist.Multinomial(self.strain_total, strain_probs),
                        obs=strain_obs)

    def train(
        self,
        *,
        haar_reparam=True,
        init_scale=0.01,
        learning_rate=0.02,
        learning_rate_decay=0.1,
        num_steps=1001,
        log_every=100,
    ):
        # Configure model and guide.
        model = self.model
        if haar_reparam:
            def time_reparam(site):
                if not site["is_observed"]:
                    return HaarReparam(dim=-3 - site["fn"].event_dim)
            model = poutine.reparam(model, time_reparam)
        guide = AutoNormal(model, init_scale=init_scale)

        # Train via SVI.
        optim = ClippedAdam({"lr": learning_rate,
                             "lrd": learning_rate_decay ** (1 / num_steps)})
        svi = SVI(model, guide, optim, Trace_ELBO())
        losses = []
        for step in range(num_steps):
            loss = svi.step() / self.case_data.numel()
            losses.append(loss)
            if log_every and step % log_every == 100:
                logger.info("step {step: >5d} loss = {loss:0.3g}")

        self.guide = guide
        return losses
