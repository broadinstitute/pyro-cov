import argparse

import torch
from torch.distributions import Bernoulli, Categorical, Multinomial


def generate_data(args, sigma1=1.0, sigma2=0.1, sigma3=0.1, sigma4=0.1, tau=5.5):
    torch.manual_seed(args.seed)

    X = Bernoulli(probs=torch.tensor(0.5)).sample(sample_shape=(args.num_lineages, args.num_mutations))

    alpha_s = sigma1 * torch.randn(args.num_lineages)
    alpha_ps = alpha_s + sigma2 * torch.randn(args.num_regions, args.num_lineages)

    beta_f = sigma3 * torch.randn(args.num_mutations)
    beta_f[args.num_causal_mutations:] = 0.0

    beta_ps = X @ beta_f + sigma4 * torch.randn(args.num_regions, args.num_lineages)

    time_shift = Categorical(logits=torch.zeros(args.wave_duration // 2)).sample(sample_shape=(args.num_regions,
                                                                                               args.num_lineages))
    time_shift = time_shift.float() - 0.25 * args.wave_duration

    time = torch.arange(args.num_waves * args.wave_duration)
    growth_rate = alpha_ps + (time[:, None, None] + time_shift) * beta_ps / tau
    multinomial_probs = torch.softmax(growth_rate, dim=-1)

    assert multinomial_probs.shape == time.shape + alpha_ps.shape

    total_count = 100
    counts = Multinomial(total_count=total_count, probs=multinomial_probs).sample()


def main(args):
    data = generate_data(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate multiple pandemic waves")
    parser.add_argument("--num-mutations", default=21, type=int)
    parser.add_argument("--num-causal-mutations", default=5, type=int)
    parser.add_argument("--num-lineages", default=10, type=int)
    parser.add_argument("--num-regions", default=5, type=int)
    parser.add_argument("--num-waves", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--wave-duration", default=100, type=int)
    args = parser.parse_args()
    main(args)
