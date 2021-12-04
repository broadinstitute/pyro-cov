import argparse
from scipy.stats import pearsonr
import numpy as np

import torch
from torch.distributions import Bernoulli, Multinomial, Normal

import pyro
from pyro import distributions as dist


def generate_data(args, seed=0, sigma3=0.25, sigma4=0.1, tau=10.0):
    torch.manual_seed(seed)
    assert args.num_lineages % args.num_waves == 0

    # generate features
    X = Bernoulli(probs=torch.tensor(0.5)).sample(sample_shape=(args.num_lineages, args.num_mutations))

    # generate coefficients
    beta_f = sigma3 * torch.randn(args.num_mutations)
    beta_f[args.num_causal_mutations:] = 0.0
    dataset = {'true_rate_loc': X @ beta_f}
    beta_ps = dataset['true_rate_loc'] + sigma4 * torch.randn(args.num_regions, args.num_lineages)
    beta_ps /= tau

    # use a gaussian waveform to modulate counts during each wave
    waveform = Normal(0.5 * args.wave_duration,
                      0.2 * args.wave_duration).log_prob(torch.arange(args.wave_duration).float()).exp()
    waveform = (args.wave_peak * waveform / waveform.max()).round()

    num_new_lineages_per_wave = args.num_lineages // args.num_waves

    counts = []
    lineage_status = torch.zeros(args.num_regions, args.num_lineages)
    lineage_status[:, :num_new_lineages_per_wave] = 1.0
    lineage_status = lineage_status.bool()

    for wave in range(args.num_waves):
        print("wave", wave, lineage_status.sum().item())
        prev_counts = None
        probs = beta_ps.exp() * lineage_status.float()
        probs /= probs.sum(-1, keepdim=True)

        for t in range(wave * args.wave_duration, (wave + 1) * args.wave_duration):
            print("t", t, lineage_status.sum().item())
            if prev_counts is not None:
                probs = beta_ps.exp() * lineage_status.float() * prev_counts
                probs /= probs.sum(-1, keepdim=True)
            counts_t = Multinomial(total_count=int(waveform[t % args.wave_duration]), probs=probs).sample()
            prev_counts = counts_t
            lineage_status &= counts_t.bool()
            counts.append(counts_t)

        lineage_status[:, wave * num_new_lineages_per_wave : (wave+1) * num_new_lineages_per_wave] = 1.0

    counts = torch.stack(counts)

    if args.device == 'gpu':
        counts, X = counts.cuda(), X.cuda()

    pc_index = counts.ne(0).any(0).reshape(-1).nonzero(as_tuple=True)[0]

    print("total count: ", int(counts.sum().item()), "P", counts.size(1), "C", counts.size(2), "PC", len(pc_index))
    dataset.update({'counts': counts, 'features': X, 'tau': tau, 'true_coef': beta_f,
                    'pc_index': pc_index})
    return dataset


def model(dataset):
    features = dataset["features"]
    pc_index = dataset["pc_index"]

    T, P, _ = dataset['counts'].shape
    C, F = features.shape
    PC = len(pc_index)
    assert PC <= P * C
    assert dataset['counts'].size(-1) == C

    clade_plate = pyro.plate("clade", C, dim=-1)
    place_plate = pyro.plate("place", P, dim=-2)
    time_plate = pyro.plate("time", T, dim=-3)
    pc_plate = pyro.plate("place_clade", PC, dim=-1)

    time = torch.arange(T)

    coef_scale = 0.01
    rate_scale = pyro.sample("rate_scale", dist.LogNormal(-4, 2))
    init_scale = pyro.sample("init_scale", dist.LogNormal(0, 2))

    coef = pyro.sample(
        "coef", dist.Logistic(torch.zeros(F), coef_scale).to_event(1)
    )
    assert coef.shape == (F,)

    with clade_plate:
        rate_loc = pyro.deterministic("rate_loc", coef @ features.T)
        assert rate_loc.shape == (C,)
    with pc_plate:
        pc_rate_loc = rate_loc.expand(P, C).reshape(-1)
        pc_rate = pyro.sample(
            "pc_rate", dist.Normal(pc_rate_loc[pc_index], rate_scale)
        )
        pc_init = pyro.sample("pc_init", dist.Normal(0, init_scale))
        assert pc_init.shape == (PC,)
    with place_plate, clade_plate:
        rate = pyro.deterministic(
            "rate",
            pc_rate_loc.scatter(0, pc_index, pc_rate).reshape(P, C),
        )
        init = pyro.deterministic(
            "init",
            torch.full((P * C,), -1e2).scatter(0, pc_index, pc_init).reshape(P, C),
        )
        assert init.shape == rate.shape == (P, C)
    logits = (init + rate * time[:, None, None]) / dataset['tau']
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

    losses = []
    tot_count = dataset['counts'].sum().item()

    for step in range(args.num_svi_steps):
        loss = svi.step(dataset)
        losses.append(loss)
        if (step > 0 and step % args.report_frequency == 0) or step == args.num_svi_steps - 1:
            print("[step %04d]  loss: %.4f" % (step, np.mean(losses[-100:]) / tot_count))

    inferred_coef = guide.median()['coef']
    print("inferred_coef: ", inferred_coef.data.cpu().numpy())
    print("true_coeff: ", dataset['true_coef'].data.cpu().numpy())

    pearson = pearsonr(inferred_coef.data.cpu().numpy(), dataset['true_coef'].data.cpu().numpy())[0]
    print("coef pearson: ", pearson)

    inferred_rate_loc = inferred_coef @ dataset['features'].T
    print("inferred_rate_loc",inferred_rate_loc.data.cpu().numpy())
    print("true_rate_loc",dataset['true_rate_loc'].data.cpu().numpy())
    pearson = pearsonr(inferred_rate_loc.data.cpu().numpy(), dataset['true_rate_loc'].data.cpu().numpy())[0]
    print("rate_loc pearson: ", pearson)

    return pearson


def main(args):
    print(args)

    pearsons = []
    num_simulations = 1
    split_waves = False
    for simulation in range(num_simulations):
        dataset = generate_data(args, seed=args.seed + simulation)
        if split_waves:
            counts = dataset['counts']
            new_counts = torch.zeros(counts.size(0) // args.num_waves,
                                     counts.size(1) * args.num_waves, counts.size(2))
            for r in range(args.num_regions):
                new_counts[0:100, 4 * r + 0] = counts[0:100, r]
                new_counts[0:100, 4 * r + 1] = counts[100:200, r]
                new_counts[0:100, 4 * r + 2] = counts[200:300, r]
                new_counts[0:100, 4 * r + 3] = counts[300:400, r]
            dataset['counts'] = new_counts

        pearson = fit_svi(args, dataset)
        pearsons.append(pearson)

    print("pearsons", pearsons)
    print("[# waves: {}]  {:.4f} +- {:.4f}".format(args.num_waves, np.mean(pearsons), np.std(pearsons)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate multiple pandemic waves")
    parser.add_argument("--num-svi-steps", default=5000, type=int)
    parser.add_argument("--report-frequency", default=250, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--lrd", default=0.1, type=float)
    parser.add_argument("--num-mutations", default=20, type=int)
    parser.add_argument("--num-causal-mutations", default=5, type=int)
    parser.add_argument("--num-lineages", default=128, type=int)
    parser.add_argument("--num-regions", default=64, type=int)
    parser.add_argument("--num-waves", default=1, type=int)
    parser.add_argument("--wave-peak", default=2000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--wave-duration", default=100, type=int)
    parser.add_argument("--device", default='cpu', type=str, choices=['cpu', 'gpu'])
    args = parser.parse_args()

    if args.device == 'gpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    main(args)
