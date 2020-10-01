import math

import pyro
import pyro.distributions as dist
import torch
from pyro.contrib.epidemiology import CompartmentalModel, binomial_dist, infection_dist

from pyrophylo.phylo import MarkovTree


class CountyModel(CompartmentalModel):
    # Let's explicitly input all data needed for inference;
    # that way the model can be saved and loaded for later prediction.
    def __init__(self, county_names, population, distance_matrix,
                 init_cases, new_cases, trees, leaf_to_county):
        N = len(county_names)
        T = len(new_cases)
        assert population.shape == (N,)
        assert distance_matrix.shape == (N, N)
        assert init_cases.shape == (N,)
        assert new_cases.shape == (T, N)
        assert leaf_to_county.shape == (trees.num_leaves,)
        compartments = ("S", "I")  # R is implicit.
        duration = new_cases.size(0)
        super().__init__(compartments, duration, population)
        self.county_names = county_names
        self.population = population  # assume this is constant over relevant timescale
        self.distance_matrix = distance_matrix
        self.init_cases = init_cases
        self.new_cases = new_cases
        self.trees = trees
        self.leaf_to_county = leaf_to_county

    def global_model(self):
        tau = 14.  # recovery time
        # Assume basic reproductive number around 2.
        R0 = pyro.sample("R0", dist.LogNormal(math.log(2), 1.))
        # Assume about 40% response rate.
        rho = pyro.sample("rho", dist.Beta(4, 6))
        # Assume all distributions are overdispersed.
        od = pyro.sample("od", dist.Beta(2, 6))

        # Let's use a Gaussian kernel with learnable radius.
        radius = self.distance_matrix.mean()
        radius = pyro.sample("radius", dist.LogNormal(math.log(radius), 1.))
        coupling = self.distance_matrix.div(radius).pow(2).mul(-0.5).exp()

        return R0, tau, rho, od, coupling

    def initialize(self, params):
        R0, tau, rho, od, coupling = params
        with self.region_plate:
            # Assume a small portion of cumulative cases are still infected.
            I = pyro.sample("I_init", binomial_dist(self.init_cases, 0.5,
                                                    overdispersion=od))
            # Assume all infections were observed. FIXME
            S = self.population - self.init_cases
        return {"S": S, "I": I}

    def transition(self, params, state, t):
        R0, tau, rho, od, coupling = params
        I_coupled = state["I"] @ coupling
        I_coupled = I_coupled.clamp(min=0)
        pop_coupled = self.population[None] @ coupling

        with self.region_plate:
            # Sample flows between compartments.
            S2I = pyro.sample("S2I_{}".format(t),
                              infection_dist(individual_rate=R0 / tau,
                                             num_susceptible=state["S"],
                                             num_infectious=I_coupled,
                                             population=pop_coupled,
                                             overdispersion=od))
            I2R = pyro.sample("I2R_{}".format(t),
                              binomial_dist(state["I"], 1 / tau,
                                            overdispersion=od))

            # Update compartments with flows.
            state["S"] = state["S"] - S2I
            state["I"] = state["I"] + S2I - I2R

            # Condition on aggregate observations.
            t_is_observed = isinstance(t, slice) or t < self.duration
            pyro.sample("obs_{}".format(t),
                        binomial_dist(S2I, rho, overdispersion=od),
                        obs=self.new_cases[t] if t_is_observed else None)

    def finalize(self, params, prev, curr):
        R0, tau, rho, od, coupling = params

        # Consider the categorical distribution of
        # the provenance j of an infection in region i.
        #
        #                       coupling[j,i] I[j]
        # provenance[i,j] = ----------------------------------------
        #                   sum(coupling[k,i] I[k] for k in regions)
        provenance = prev["I"].unsqueeze(-1) * coupling.unsqueeze(-3)
        provenance = provenance / provenance.sum(-1, keepdim=True)

        # Avoid conflict between particle dim and batch dim.
        assert len(self.trees.batch_shape) == 1
        if provenance.dim() > 3:
            provenance = provenance.unsqueeze(-4)

        # Tree likelihood.
        pyro.sample("geolocation",
                    dist.MixtureSameFamily(
                        dist.Categorical(torch.ones(self.trees.batch_shape)),
                        MarkovTree(self.trees, provenance)),
                    obs=self.leaf_to_county)
