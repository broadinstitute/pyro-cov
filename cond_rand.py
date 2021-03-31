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
import pyro.poutine as poutine

from pyrocov import pangolin

from forecast import load_data


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(2 * input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2 * z_dim)
        self.z_dim = z_dim
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x, b):
        x_b = torch.cat([x, b], dim=-1)
        h1 = self.relu(self.bn1(self.fc1(x_b)))
        h2 = self.relu(self.bn2(self.fc2(h1)))
        h3 = self.fc3(h2)
        z_loc = h3[..., :self.z_dim]
        z_scale = self.softplus(h3[..., self.z_dim:] - 2.0)
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + 2 * input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim)
        self.fc3 = nn.Linear(hidden_dim + input_dim, 2 * input_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.input_dim = input_dim

    def forward(self, z, x, b):
        z_x_b = torch.cat([z, x * (1 - b), b], dim=-1)
        h1 = self.relu(self.bn1(self.fc1(z_x_b)))
        h1_x = torch.cat([h1, x * (1 - b)], dim=-1)
        h2 = self.relu(self.bn2(self.fc2(h1_x)))
        h2_x = torch.cat([h2, x * (1 - b)], dim=-1)
        h3 = self.fc3(h2_x)
        x_loc = h3[..., :self.input_dim]
        x_scale = self.softplus(h3[..., self.input_dim:])
        return x_loc, x_scale


class VAE(nn.Module):
    def __init__(self, input_dim=None, z_dim=32, hidden_dim=400):
        super().__init__()
        self.encoder = Encoder(input_dim, z_dim, hidden_dim)
        self.decoder = Decoder(input_dim, z_dim, hidden_dim)
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.maes = []
        self.maes2 = []
        self.maes3= []
        self.zs = []
        self.zs2 = []

    def model(self, x, b, idx):
        assert x.shape == b.shape
        pyro.module("vae", self)
        mbs = x.size(0)
        with pyro.plate("data", mbs), poutine.scale(scale=1.0/self.input_dim):
            z_loc = x.new_zeros(x.shape[0], self.z_dim)
            z = pyro.sample("z", dist.Normal(z_loc, 1.0).to_event(1))
            x_loc, x_scale = self.decoder.forward(z, x, b)
            x_loc = x_loc.gather(-1, idx)
            x_scale = x_scale.gather(-1, idx)
            x_idx = x.gather(-1, idx)

            assert x_loc.shape == x_scale.shape == x_idx.shape
            assert x_idx.dim() == 2

            self.maes.append( (x_idx - x_loc).abs().mean().item() )
            xx = x_idx.sigmoid()
            xx2 = x_loc.sigmoid()
            self.maes2.append( (xx - xx2).abs().mean().item() )
            big = xx > 0.5
            if big.sum().item() > 0:
                self.maes3.append( (xx[big] - xx2[big]).abs().mean().item() )

            scale = (100.0 * x_idx).clamp(min=1.0)
            factor = (dist.Normal(x_loc, x_scale).log_prob(x_idx) * scale).sum(-1)
            #pyro.sample("obs", dist.Normal(x_loc, x_scale).to_event(1), obs=x_idx)
            pyro.factor("obs", factor)

    def guide(self, x, b, idx):
        mbs = x.size(0)
        with pyro.plate("data", mbs), poutine.scale(scale=1.0/self.input_dim):
            z_loc, z_scale = self.encoder.forward(x, b)
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
            self.zs.append(z_loc.pow(2.0).mean().item())
            self.zs2.append(z_scale.mean().item())


def logit_transform(features, epsilon=1.0e-3):
    logit_features = features.clamp(min=epsilon, max=1.0 - epsilon)
    logit_features = torch.log(logit_features) - torch.log(1.0 - logit_features)
    return logit_features


def train(dataset, args):
    features = dataset['features']
    logit_features = logit_transform(features)

    train_dataset = TensorDataset(logit_features)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = features.size(-1)
    vae = VAE(input_dim=input_dim)

    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    losses = []

    for step in range(args.num_steps):
        epoch_losses = []

        b_dim = 2 if step < 39000 else 1

        for (x,) in train_loader:
            mbs = x.size(0)
            assert b_dim * mbs <= input_dim
            unobserved_idx = torch.randperm(input_dim)[:b_dim * mbs].reshape(mbs, b_dim)
            b = torch.zeros(mbs, input_dim)
            b.scatter_(-1, unobserved_idx, torch.ones_like(b))
            loss = svi.step(x, b, unobserved_idx) * (input_dim / (mbs * b_dim))
            epoch_losses.append(loss)

        losses.append(np.mean(epoch_losses))

        if step % 40 == 0:
            smoothed_loss = np.mean(losses[-40:]) if step > 0 else 0.0
            mae = np.mean(vae.maes[-300:])
            mae2 = np.mean(vae.maes2[-300:])
            mae3 = np.mean(vae.maes3[-300:])
            logger.info("[step %03d] loss: %.4f %.4f    mae: %.6f %.6f %.6f   zs: %.3f %.3f" % (step, losses[-1], smoothed_loss, mae, mae2, mae3, np.mean(vae.zs[-300:]), np.mean(vae.zs2[-300:])))

    logger.info("saved module to cond.pt")
    torch.save(vae.state_dict(), './cond.pt')


def test(dataset, args):
    features = dataset['features']
    logit_features = logit_transform(features)

    input_dim = logit_features.size(-1)
    vae = VAE(input_dim=input_dim)
    vae.load_state_dict(torch.load('./cond.pt'))
    vae.eval()

    #features_tilde = vae(logit_features).sigmoid().detach()

    b = torch.zeros(1, input_dim)
    b[0, 0] = 1
    idx = torch.tensor([0]).unsqueeze(0)

    for i in range(3):
        print('\n i = ', i)
        x = features[i].unsqueeze(0)

        z_loc, z_scale = vae.encoder.forward(x, b)
        z = dist.Normal(z_loc, z_scale).sample()
        print("zscale", z_scale.min(), z_scale.max())
        x_loc, x_scale = vae.decoder.forward(z, x, b)
        print("xscale", x_scale.min(), x_scale.max())

        #print('x', x)
        print("x>0.05", (x>0.01).nonzero())

        for _ in range(3):
            x_tilde = dist.Normal(x_loc, x_scale).sample().sigmoid()
            print("xt>0.05", (x_tilde[0]>0.01).nonzero())




def main(args):
    print(args)
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    dataset = load_data(args)
    dataset['features'] = dataset['features'][:, :200]
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
    parser.add_argument("--learning-rate", default=0.00002, type=float)
    parser.add_argument("--num-steps", default=5000, type=int)
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--mutation-cutoff", default=0.99, type=float)
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("--mode", default='test', type=str, choices=['train', 'test'])
    parser.add_argument("--feature-groups", action="store_true")
    parser.add_argument("-l", "--log-every", default=500, type=int)
    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    main(args)
