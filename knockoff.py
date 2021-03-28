import argparse
import copy
import functools
import logging
import math
import os
import pickle
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.distributions import constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoGuideList, AutoNormal, init_to_median
from pyro.infer.autoguide.initialization import InitMessenger
from pyro.optim import Adam, ClippedAdam

from pyrocov import pangolin

from forecast import load_data


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


class KONet(nn.Module):
    def __init__(self, input_dim, hidden_dim, prototype, jitter=1.0e-8):
        super().__init__()
        self.jitter = jitter
        self.relu = nn.ReLU()
        self.mask = (torch.ones(input_dim, input_dim) - torch.eye(input_dim)).type_as(prototype)
        self.lin1 = nn.Linear(2 * input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(num_features=hidden_dim)
        self.lin_final = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        epsilon = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        x_epsilon = torch.cat([x, epsilon], dim=-1)
        h1 = self.relu(self.bn1(self.lin1(x_epsilon)))
        h2 = self.relu(self.bn2(self.lin2(h1)))
        h3 = self.relu(self.bn3(self.lin3(h2)))
        x_tilde = self.lin_final(h3)
        return x_tilde

    def loss(self, x, x_tilde):
        x_ctr = x - x.mean(0)
        x_tilde_ctr= x_tilde - x_tilde.mean(0)

        jit = self.jitter
        x_norm = x_ctr / (x_ctr.std(0) + jit)
        x_tilde_norm = x_tilde_ctr / (x_tilde_ctr.std(0) + jit)

        G_xx = (x_ctr.unsqueeze(-2) * x_ctr.unsqueeze(-1)).mean(0)
        G_xtxt = (x_tilde_ctr.unsqueeze(-2) * x_tilde_ctr.unsqueeze(-1)).mean(0)
        G_xxt = (x_ctr.unsqueeze(-2) * x_tilde_ctr.unsqueeze(-1)).mean(0)
        G_xx_sq = G_xx.pow(2.0).sum() + jit

        loss1 = ((x - x_tilde).mean(0)).pow(2.0).mean()
        loss2 = (G_xx - G_xtxt).pow(2.0).sum() / G_xx_sq
        loss3 = ((G_xx - G_xxt) * self.mask).pow(2.0).sum() / G_xx_sq
        loss4 = (x_norm * x_tilde_norm).mean(0).abs().mean()

        return loss1, loss2, loss3, loss4


def logit_transform(features, epsilon=1.0e-6):
    logit_features = features.clamp(min=epsilon, max=1.0 - epsilon)
    logit_features = torch.log(logit_features) - torch.log(1.0 - logit_features)
    return logit_features


def train(dataset, args):
    features = dataset['features']

    konet = KONet(input_dim=features.size(-1), hidden_dim=8 * features.size(-1), prototype=features)
    adam = torch.optim.Adam(konet.parameters(), lr=args.learning_rate)
    sched = torch.optim.lr_scheduler.MultiStepLR(adam, milestones=[50, 200, 400, 600], gamma=0.2)

    logit_features = logit_transform(features)
    train_dataset = TensorDataset(logit_features)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    epoch_losses = []

    for step in range(args.num_steps):
        losses, losses1, losses2, losses3, losses4 = [], [], [], [], []

        for (x,) in train_loader:
            adam.zero_grad()
            x_tilde = konet(x)
            loss1, loss2, loss3, loss4 = konet.loss(x, x_tilde)
            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()
            losses.append(loss.item())
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            losses3.append(loss3.item())
            losses4.append(loss4.item())
            adam.step()

        epoch_losses.append(np.mean(losses))
        sched.step()

        if step % 5 == 0:
            ep_loss = 0.0 if step < 20 else np.mean(epoch_losses[-20:])
            logger.info("[step %03d]  loss: %.3f   %.3f    %.3f %.3f %.3f %.3f" % (step, np.mean(losses), ep_loss, np.mean(losses1),
                                                                                   np.mean(losses2), np.mean(losses3), np.mean(losses4)))

    #torch.save(konet.state_dict(), './konet.pt')
    logger.info("Saved konet state dict to konet.pt")


def model(weekly_strains, features, feature_scale=1.0):
    assert weekly_strains.shape[-1] == features.shape[0]

    T, P, S = weekly_strains.shape
    S, F = features.shape
    time_plate = pyro.plate("time", T, dim=-2)
    place_plate = pyro.plate("place", P, dim=-1)
    time = torch.arange(float(T)) * args.timestep / 365.25  # in years
    time -= time.max()

    # Assume relative growth rate depends on mutation features but not time or place.
    log_rate_coef = pyro.sample(
        "log_rate_coef", dist.Laplace(0, feature_scale).expand([F]).to_event(1)
    )
    log_rate = pyro.deterministic("log_rate", log_rate_coef @ features.T)

    # Assume places differ only in their initial infection count.
    with place_plate:
        log_init = pyro.sample("log_init", dist.Normal(0, 10).expand([S]).to_event(1))

    # Finally observe overdispersed counts.
    strain_probs = (log_init + log_rate * time[:, None, None]).softmax(-1)
    concentration = pyro.sample("concentration", dist.LogNormal(2, 4))

    obs_dist = dist.DirichletMultinomial(
                total_count=weekly_strains.sum(-1).max(),
                concentration=concentration * strain_probs,
                is_sparse=True)  # uses a faster algorithm

    with time_plate, place_plate:
        pyro.sample("obs", obs_dist, obs=weekly_strains)


def init_loc_fn(site):
    if site["name"] in ("log_rate_coef", "log_rate", "log_init"):
        return torch.zeros(site["fn"].shape())
    if site["name"] == "concentration":
        return torch.full(site["fn"].shape(), 5.0)
    return init_to_median(site)


def fit_svi(args, dataset):
    logger.info("Fitting via SVI")
    pyro.clear_param_store()
    pyro.set_rng_seed(20210319)

    guide = AutoGuideList(InitMessenger(init_loc_fn)(model))
    guide.append(AutoDelta(poutine.block(model, hide=["log_rate_coef"]), init_loc_fn=init_loc_fn))
    guide.append(AutoNormal(poutine.block(model, expose=["log_rate_coef"]),
                        init_loc_fn=init_loc_fn, init_scale=0.01))

    # Initialize guide so we can count parameters.
    guide(dataset["weekly_strains"], dataset["features"])
    num_params = sum(p.numel() for p in guide.parameters())
    logger.info(f"Training SVI guide with {num_params} parameters:")

    num_steps = 5001

    optim = ClippedAdam({"lr": 0.05, "lrd": 0.01 ** (1 / num_steps)})

    svi = SVI(model, guide, optim, Trace_ELBO())
    losses = []
    num_obs = dataset["weekly_strains"].count_nonzero()
    for step in range(num_steps):
        loss = svi.step(dataset["weekly_strains"], dataset["features"]) / num_obs
        assert not math.isnan(loss)
        losses.append(loss)
        if step % args.log_every == 0:
            median = guide.median()
            concentration = median["concentration"].item()
            logger.info(
                f"step {step: >4d} loss = {loss:0.6g}\tconc. = {concentration:0.3g}\t"
            )

    guide.to(torch.double)
    sigma_points = dist.Normal(0, 1).cdf(torch.tensor([-1., 1.])).double()
    pos = guide[1].quantiles(sigma_points[1].item())
    neg = guide[1].quantiles(sigma_points[0].item())
    mean = {k: (pos[k] + neg[k]) / 2 for k in pos}
    std = {k: (pos[k] - neg[k]) / 2 for k in pos}

    mean = mean['log_rate_coef']
    std = std['log_rate_coef']
    Z = mean.abs() / std

    F = Z.shape[0] // 2
    W = Z[:F] - Z[F:]

    Z_true = Z[:F]
    q = np.percentile(Z_true.data.cpu().numpy(), [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0])
    print("Z_true quantiles", q)
    Z_ko = Z[F:]
    q = np.percentile(Z_ko.data.cpu().numpy(), [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0])
    print("Z_ko quantiles", q)

    q = np.percentile(W.data.cpu().numpy(), [0.5, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.5])
    print("W quantiles", q)
    print("num neg W", (W<0).float().sum().item())

    ts = torch.from_numpy(np.sort(W.abs().data.cpu().numpy())).type_as(W).unsqueeze(-1)
    threshold = 0.01
    numerator = 1.0 + (W <= -ts).float().sum(-1)
    denominator = (W >= ts).float().sum(-1)
    ratio = numerator / denominator
    print("ratio", ratio[:8], ratio[-3:])
    print("ratio min", ratio.min().item())
    which = torch.argmax((ratio <= threshold).float())
    print("which", which)
    t_selected = ts[which, 0].item()
    print("t_selected", t_selected)

    num_w = (W > t_selected).sum()

    print("num_w",num_w)

    return {"args": args, "guide": guide, "losses": losses}


def test(dataset, args):
    features = dataset['features']
    logit_features = logit_transform(features)

    konet = KONet(input_dim=features.size(-1), hidden_dim=8 * features.size(-1), prototype=features)
    konet.load_state_dict(torch.load('./konet.pt'))
    konet.eval()

    features_tilde = konet(logit_features).sigmoid().detach()
    print("first real feature", features[0, :10])
    print("first kofeature   ", features_tilde[0, :10])

    dataset['features'] = torch.cat([features, features_tilde], dim=-1)

    fit_svi(args, dataset)



def main(args):
    print(args)
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    dataset = load_data(args)
    logger.info("Loaded dataset with (T, P, S) = ({}, {}, {})".format(dataset['T'], dataset['P'], dataset['S']))

    if args.mode == 'train':
        train(dataset, args)
    elif args.mode == 'test':
        test(dataset, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="knockoff"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=torch.cuda.is_available()
    )
    parser.add_argument(
        "--timestep",
        default=14,
        type=int,
        help="Reasonable values might be week, fortnight, or month",
    )
    parser.add_argument("--learning-rate", default=0.005, type=float)
    parser.add_argument("--num-steps", default=800, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--mutation-cutoff", default=0.5, type=float)
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("--mode", type=str, choices=['train', 'test'])
    parser.add_argument("--feature-groups", action="store_true")
    parser.add_argument("-l", "--log-every", default=500, type=int)
    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    main(args)
