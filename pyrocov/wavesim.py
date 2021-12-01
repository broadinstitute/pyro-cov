import argparse
from scipy.stats import spearmanr, pearsonr
import numpy as np

import torch
from torch.distributions import Bernoulli, Categorical, Multinomial, Normal

import pyro
from pyro import poutine
from pyro import distributions as dist
from pyro.infer.reparam import LocScaleReparam


def generate_data(args, seed=0, sigma1=1.0, sigma2=0.1, sigma3=0.25, sigma4=0.1, tau=10.0):
    torch.manual_seed(seed)

    assert args.num_lineages % args.num_waves == 0

    X = Bernoulli(probs=torch.tensor(0.5)).sample(sample_shape=(args.num_lineages, args.num_mutations))

    alpha_s = sigma1 * torch.randn(args.num_lineages)
    alpha_ps = alpha_s + sigma2 * torch.randn(args.num_regions, args.num_lineages)

    beta_f = sigma3 * torch.randn(args.num_mutations)
    beta_f[args.num_causal_mutations:] = 0.0

    beta_ps = X @ beta_f + sigma4 * torch.randn(args.num_regions, args.num_lineages)

    time_shift = args.wave_duration * torch.tile(torch.arange(args.num_waves), (args.num_lineages // args.num_waves,))

    time = torch.arange(args.num_waves * args.wave_duration)
    growth_rate = alpha_ps + (time[:, None, None] - time_shift) * beta_ps / tau
    assert growth_rate.shape == time.shape + alpha_ps.shape

    waveform = Normal(0.5 * args.wave_duration,
                      0.2 * args.wave_duration).log_prob(torch.arange(args.wave_duration).float()).exp()
    waveform = (args.wave_peak * waveform / waveform.max()).round()
    waveform = torch.tile(waveform, (args.num_waves,))
    # print(waveform)

    counts = [Multinomial(total_count=int(waveform[t].item()), logits=growth_rate[t]).sample() for t in time]
    counts = torch.stack(counts)

    assert (args.num_regions * waveform - counts.sum(-1).sum(-1)).abs().max().item() == 0

    # print("lineage counts", counts.sum(0).sum(0))
    print("total count: ", int(counts.sum().item()))

    if args.device == 'gpu':
        counts, X = counts.cuda(), X.cuda()

    dataset = {'counts': counts, 'features': X, 'tau': tau, 'true_coef': beta_f}
    return dataset


def model(dataset):
    features = dataset["features"]
    T, P, _ = dataset['counts'].shape
    C, F = features.shape
    clade_plate = pyro.plate("clade", C, dim=-1)
    place_plate = pyro.plate("place", P, dim=-2)
    time_plate = pyro.plate("time", T, dim=-3)
    time = torch.arange(T)

    reparam = {}

    with poutine.reparam(config=reparam):
        # coef_scale = pyro.sample("coef_scale", dist.LogNormal(-4, 2))
        coef_scale = 0.1
        rate_loc_scale = pyro.sample("rate_loc_scale", dist.LogNormal(-4, 2))
        init_loc_scale = pyro.sample("init_loc_scale", dist.LogNormal(0, 2))
        rate_scale = pyro.sample("rate_scale", dist.LogNormal(-4, 2))
        init_scale = pyro.sample("init_scale", dist.LogNormal(0, 2))

        coef = pyro.sample(
            "coef", dist.Logistic(torch.zeros(F), coef_scale).to_event(1)
        )
        assert coef.shape == (F,)
        with clade_plate:
            rate_loc = pyro.sample("rate_loc", dist.Normal(coef @ features.T, rate_loc_scale))  # [C]
            init_loc = pyro.sample("init_loc", dist.Normal(0, init_loc_scale))  # [C]
            assert rate_loc.shape == init_loc.shape == (C,)
        with place_plate, clade_plate:
            rate = pyro.sample("rate", dist.Normal(rate_loc, rate_scale))  # [P, C]
            init = pyro.sample("init", dist.Normal(init_loc, init_scale))  # [P, C]
            assert rate.shape == init.shape == (P, C)

        logits = init + rate * time[:, None, None] / dataset['tau']  # [T, P, C]
        assert logits.shape == (T, P, C)
        with time_plate, place_plate:
            pyro.sample(
                "obs",
                dist.Multinomial(logits=logits.unsqueeze(-2), validate_args=False),
                obs=dataset['counts'].unsqueeze(-2)
            )


def fit_svi(args, dataset):
    guide = pyro.infer.autoguide.AutoNormal(model, init_scale=0.01)
    optim = pyro.optim.ClippedAdam({"lr": args.lr, "lrd": args.lrd ** (1.0 / args.num_svi_steps)})
    svi = pyro.infer.SVI(model, guide, optim, pyro.infer.Trace_ELBO(max_plate_nesting=3))

    for step in range(args.num_svi_steps):
        loss = svi.step(dataset)
        if step % args.report_frequency == 0 or step == args.num_svi_steps - 1:
            print("[step %04d]  loss: %.4f" % (step, loss))

    inferred_coef = guide.median()['coef']
    print("inferred_coef: ", inferred_coef.data.cpu().numpy())
    print("true_coeff: ", dataset['true_coef'].data.cpu().numpy())

    pearson = pearsonr(inferred_coef.data.cpu().numpy(), dataset['true_coef'].data.cpu().numpy())[0]
    print("pearson: ", pearson)

    return pearson


def main(args):
    print(args)

    pearsons = []
    num_simulations = 10
    for simulation in range(num_simulations):
        dataset = generate_data(args, seed=args.seed + simulation)
        pearsons.append( fit_svi(args, dataset) )

    print("pearsons", pearsons)
    print("[# waves: {}]  {:.4f} +- {:.4f}".format(args.num_waves, np.mean(pearsons), np.std(pearsons)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate multiple pandemic waves")
    parser.add_argument("--num-svi-steps", default=3000, type=int)
    parser.add_argument("--report-frequency", default=200, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--lrd", default=0.1, type=float)
    parser.add_argument("--num-mutations", default=20, type=int)
    parser.add_argument("--num-causal-mutations", default=5, type=int)
    parser.add_argument("--num-lineages", default=32, type=int)
    parser.add_argument("--num-regions", default=32, type=int)
    parser.add_argument("--num-waves", default=1, type=int)
    parser.add_argument("--wave-peak", default=1000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--wave-duration", default=100, type=int)
    parser.add_argument("--device", default='gpu', type=str, choices=['cpu', 'gpu'])
    args = parser.parse_args()

    if args.device == 'gpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    main(args)
