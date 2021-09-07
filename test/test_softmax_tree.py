# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import math

import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from pyrocov.markov_tree import MarkovTree
from pyrocov.phylo import Phylogeny
from pyrocov.softmax_tree import SoftmaxTree


@pytest.mark.parametrize("num_bits", [2, 3, 4, 5, 10])
@pytest.mark.parametrize("num_leaves", [2, 3, 4, 5, 10])
def test_rsample(num_leaves, num_bits):
    phylo = Phylogeny.generate(num_leaves)
    leaf_times = phylo.times[phylo.leaves]
    bit_times = torch.randn(num_bits)
    logits = torch.randn(num_leaves, num_bits)
    tree = SoftmaxTree(leaf_times, bit_times, logits)
    value = tree.rsample()
    tree.log_prob(value)


def model(leaf_times, leaf_states, num_features):
    assert len(leaf_times) == len(leaf_states)

    # Timed tree concerning reproductive behavior only.
    coal_params = pyro.sample("coal_params", dist.CoalParamPrior("TODO"))  # global
    # Note this is where our coalescent model assumes geographically
    # homogeneous reproductive rate, which is not realistic.
    # See appendix of (Vaughan et al. 2014) for discussion of this assumption.
    phylogeny = pyro.sample("phylogeny", dist.Coalescent(coal_params, leaf_times))

    # This is compatible with subsampling features, but not leaves.
    subs_params = pyro.sample("subs_params", dist.GTRGamma("TODO"))  # global
    with pyro.plate("features", num_features, leaf_states.size(-1)):
        # This is similar to the phylogeographic likelihood in the pyro-cov repo.
        # This is simpler (because it is time-homogeneous)
        # but more complex in that it is batched.
        # This computes mutation likelihood via dynamic programming.
        pyro.sample("leaf_times", MarkovTree(phylogeny, subs_params), obs=leaf_states)


def guide(leaf_times, leaf_states, num_features, *, logits_fn=None):
    assert len(leaf_times) == len(leaf_states)

    # Sample all continuous latents in a giant correlated auxiliary.
    aux = pyro.sample("aux", dist.LowRankMultivariateNormal("TODO"))
    # Split it up (TODO switch to EasyGuide).
    pyro.sample("coal_params", dist.Delta(aux["TODO"]))  # global
    pyro.sample("subs_params", dist.Delta(aux["TODO"]))  # global
    # These are the times of each bit in the embedding vector.
    bit_times = pyro.sample(
        "bit_times", dist.Delta(aux["TODO"]), infer={"is_auxiliary": True}
    )

    # Learn parameters of the discrete distributions,
    # possibly conditioned on continuous latents.
    if logits_fn is not None:
        # Amortized guide, compatible with subsampling leaves but not features.
        logits = logits_fn(leaf_states, leaf_times)  # batched over leaves
    else:
        # Fully local guide, compatible with subsampling features but not leaves.
        with pyro.plate("leaves", len(leaf_times)):
            logits = pyro.param(
                "logits", lambda: torch.randn(leaf_times.shape), event_dim=0
            )
    assert len(logits) == len(leaf_times)

    pyro.sample("phylogeny", SoftmaxTree(bit_times, logits))


@pytest.mark.xfail(reason="WIP")
@pytest.mark.parametrize("num_features", [4])
@pytest.mark.parametrize("num_leaves", [2, 3, 4, 5, 10, 100])
def test_svi(num_leaves, num_features):
    phylo = Phylogeny.generate(num_leaves)
    leaf_times = phylo.times[phylo.leaves]
    leaf_states = torch.full((num_leaves, num_features), 0.5).bernoulli()

    svi = SVI(model, guide, Adam({"lr": 1e-4}), Trace_ELBO())
    for i in range(2):
        loss = svi.step(leaf_times, leaf_states)
        assert math.isfinite(loss)
