import argparse
import copy
import functools
import logging
import math
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
        G_xx_sq = G_xx.pow(2.0).sum() + jit
        GG = (G_xx - G_xtxt).pow(2.0)

        loss1 = ((x - x_tilde).mean(0)).pow(2.0).mean()
        loss2 = GG.sum() / G_xx_sq
        loss3 = (GG * self.mask).sum() / G_xx_sq
        loss4 = (x_norm * x_tilde_norm).mean(0).abs().mean()

        return loss1, loss2, loss3, loss4



def train(dataset, args):
    features = dataset['features']

    konet = KONet(input_dim=features.size(-1), hidden_dim=10 * features.size(-1), prototype=features)
    adam = torch.optim.Adam(konet.parameters(), lr=args.learning_rate)
    sched = torch.optim.lr_scheduler.MultiStepLR(adam, milestones=[200, 400, 600], gamma=0.2)

    train_dataset = TensorDataset(features)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    epoch_losses = []

    for step in range(args.num_steps):
        losses, losses1, losses2, losses3, losses4 = [], [], [], [], []
        for (x,) in train_loader:
            adam.zero_grad()
            epsilon = 1.0e-6
            x = x.clamp(min=epsilon, max=1.0 - epsilon)
            x = torch.log(x) - torch.log(1.0 - x)
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
            e_loss = 0.0 if step < 20 else np.mean(epoch_losses[-20:])
            print("[step %03d]  loss: %.3f   %.3f    %.3f %.3f %.3f %.3f" % (step, np.mean(losses), e_loss, np.mean(losses1),
                                                                      np.mean(losses2), np.mean(losses3), np.mean(losses4)))



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
    parser.add_argument("--learning-rate", default=0.002, type=float)
    parser.add_argument("--lrd", default=0.1, type=float)
    parser.add_argument("--num-steps", default=1000, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--mutation-cutoff", default=0.5, type=float)
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("--feature-groups", action="store_true")
    parser.add_argument("-l", "--log-every", default=500, type=int)
    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    main(args)
