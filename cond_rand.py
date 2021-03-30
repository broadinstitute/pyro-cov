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
        x_b = torch.cat([x * b, b], dim=-1)
        h1 = self.relu(self.bn1(self.fc1(x_b)))
        h2 = self.relu(self.bn2(self.fc2(h1)))
        h3 = self.fc3(h2)
        z_loc = h3[..., :self.z_dim]
        z_scale = self.softplus(h3[..., self.z_dim:])
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + 2 * input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z, x, b):
        z_x_b = torch.cat([z, x * (1 - b), b], dim=-1)
        h1 = self.relu(self.bn1(self.fc1(z_x_b)))
        h2 = self.relu(self.bn2(self.fc2(h1)))
        h3 = self.fc3(h2)
        x_loc = h3[..., :self.input_dim]
        x_scale = self.softplus(h3[..., self.input_dim:])
        return x_loc, x_scale


class VAE(nn.Module):
    def __init__(self, input_dim=None, z_dim=50, hidden_dim=400):
        super().__init__()
        self.encoder = Encoder(input_dim, z_dim, hidden_dim)
        self.decoder = Decoder(input_dim, z_dim, hidden_dim)
        self.z_dim = z_dim
        self.input_dim = input_dim

    def model(self, x, b, idx):
        pyro.module("vae", self)
        mbs = x.size(0)
        with pyro.plate("data", mbs):
            z_loc = x.new_zeros(x.shape[0], self.z_dim)
            z = pyro.sample("z", dist.Normal(z_loc, 1.0).to_event(1))
            x_loc, x_scale = self.decoder.forward(z, x, b)
            x_loc = x_loc.index_select(-1, idx)
            x_scale = x_scale.index_select(-1, idx)
            x_idx = x.index_select(-1, idx)
            pyro.sample("obs", dist.Normal(x_loc, x_scale).to_event(1), obs=x_idx)

    def guide(self, x, b, idx):
        mbs = x.size(0)
        with pyro.plate("data", mbs):
            z_loc, z_scale = self.encoder.forward(x, b)
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))



def logit_transform(features, epsilon=1.0e-7):
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
    b_dim = 3

    for step in range(args.num_steps):
        epoch_losses = []

        for (x,) in train_loader:
            mbs = x.size(0)
            assert b_dim * mbs <= input_dim
            unobserved_idx = torch.randperm(input_dim)[:b_dim * mbs].reshape(mbs, b_dim)
            b = torch.zeros(mbs, input_dim)
            b.scatter_(-1, unobserved_idx, torch.ones_like(b))
            svi.step(x, b, unobserved_idx)
            logger.info("[step %03d]")


def main(args):
    print(args)
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    dataset = load_data(args)
    logger.info("Loaded dataset with (T, P, S) = ({}, {}, {})".format(dataset['T'], dataset['P'], dataset['S']))

    train(dataset, args)


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
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--num-steps", default=7000, type=int)
    parser.add_argument("--batch-size", default=178, type=int)
    parser.add_argument("--mutation-cutoff", default=0.95, type=float)
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("--mode", type=str, choices=['train', 'test'])
    parser.add_argument("--feature-groups", action="store_true")
    parser.add_argument("-l", "--log-every", default=500, type=int)
    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    main(args)
