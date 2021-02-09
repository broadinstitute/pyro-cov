import logging

import torch
from pyro.distributions.spanning_tree import SpanningTree, make_complete_graph

logger = logging.getLogger(__name__)


class SoftminimaxClustering:
    r"""
    Finds a set of centroid clusters that minimizes the following softening of a
    minimax edge length problem:

    .. math::

        \text{loss} = \sum_{x\in\text{data}}
        \operatorname{min}_{z\in\text{clusters}}
        \|x-z\|_{p_{\text{feature}}}^{p_{\text{edge}}
    """

    def __init__(self, *, p_feature=1, p_edge=4):
        self.p_feature = p_feature
        self.p_edge = p_edge

    def init(self, data, num_clusters, *, log_every=10000):
        """
        Initialize clusters via greedy agglomeration.
        """
        N, P = data.shape
        K = min(N, num_clusters)
        data = data[torch.randperm(N)].float()

        # Index the upper triangle.
        U = torch.arange(K + 1)[:, None]
        V = torch.arange(K + 1)
        UV = (U < V).nonzero(as_tuple=False)
        U, V = UV.unbind(-1)

        mass = torch.ones(K + 1, dtype=torch.float)
        mean = data[: K + 1].clone()
        for i in range(K, N):
            # Insert the datapoint as a new cluster.
            mean[K] = data[i]

            # Find the easiest pair of clusters to merge.
            mass2 = mass.square()
            cost = torch.cdist(mean, mean, p=self.p_feature).pow_(self.p_edge)
            cost.mul_(mass * mass[:, None] / (mass2 + mass2[:, None]))
            u, v = UV[cost[U, V].argmin(0)].tolist()
            assert u < v

            # Merge the pair of clusters.
            mass[u] += mass[v]
            mean[u] += mass[v] / mass[u] * (mean[v] - mean[u])
            if v != K:
                mass[v] = mass[K]
                mean[v] = mean[K]

            if log_every and i % log_every == 0:
                p = mass / mass.sum()
                perplexity = (
                    p.log().clamp(min=torch.finfo(p.dtype).min).neg().mul(p).sum().exp()
                )
                logger.info(f"step {i: >6d}/{N} perplexity = {perplexity:0.2f}")

        mass, i = mass[:K].sort(descending=True)
        self.mean = mean[i]

    def fine_tune(
        self,
        data,
        *,
        learning_rate=0.01,
        num_steps=1001,
        batch_size=2048,
        log_every=100,
    ):
        """
        Fine tune clusters via SGD.
        """
        mean = self.mean.clone().requires_grad_()
        optim = torch.optim.Adam([mean], lr=learning_rate)
        losses = []
        for step in range(num_steps):
            batch = data[torch.randperm(len(data))[:batch_size]].float()
            optim.zero_grad()
            loss = (
                torch.cdist(batch, mean, p=self.p_feature)
                .min(-1)
                .values.norm(self.p_edge)
            )
            loss.backward()
            optim.step()
            losses.append(loss.item())
            if log_every and step % log_every == 0:
                logger.info(f"step {step: >4d} loss = {loss.item():0.3g}")
        with torch.no_grad():
            logger.info(f"distance = {(mean - self.mean).abs().sum():0.1f}")
            self.mean.copy_(mean.detach())
        return losses

    def classify(self, data, temperature=0):
        """
        If temperature == 0, this returns an integer tensor of class ids.
        If temperature > 0, this returns a tensor of probabilities.
        """
        with torch.no_grad():
            distance = torch.cdist(data.float(), self.mean, p=self.p_feature)
            if temperature == 0:
                return distance.argmin(-1)
            else:
                logits = distance.mul_(-1 / temperature)
                return logits.sub_(logits.logsumexp(-1, True)).exp_()

    def transition_matrix(self, temperature=None):
        """
        Constructs a continuous-time transition rate matrix between classes.
        """
        V = len(self.mean)
        v1, v2 = make_complete_graph(V)
        with torch.no_grad():
            distance = torch.cdist(self.mean, self.mean, p=self.p_feature)
            if temperature is None:
                temperature = distance.min(0).values.mean().item()
                logger.info(f"temperature = {temperature:0.1f}")
            if temperature == 0:
                logits = -distance[v1, v2]
                tree = SpanningTree(logits).mode
                rate = self.mean.new_zeros(V, V)
                rate[tree[:, 0], tree[:, 1]] = 1
            else:
                logits = distance[v1, v2].mul_(-1 / temperature)
                rate = SpanningTree(logits.double()).edge_mean.to(distance.dtype)
            return rate
