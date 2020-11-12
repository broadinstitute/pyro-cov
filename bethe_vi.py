"""
This experiment aims to assess posterior accuracy of a variational approach to
phylogenetic inference. We examine by default the dataset DS4 [2] from Whidden
and Matsen [1]. This dataset has 41 taxa and 1137 characters, many of which are
unobserved for many taxa.

[1] Chris Whidden, Frederick A. Matsen, IV (2015)
    "Quantifying MCMC Exploration of Phylogenetic Tree Space"
    https://academic.oup.com/sysbio/article/64/3/472/1632660
[2] TreeBase dataset M487
    https://treebase.org/treebase-web/search/study/matrices.html?id=965
"""

import argparse
import io
import logging
import math
import sys
from collections import Counter, defaultdict

import pyro
import pyro.poutine as poutine
import setuptools  # noqa F401
import torch
import torch.multiprocessing as mp
from Bio import AlignIO
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoNormal
from pyro.optim import ClippedAdam

from pyrophylo.kirchhoff import KirchhoffModel

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.DEBUG)


def load_data(args):
    logger.info(f"loading data from {args.nexus_infile}")

    # Truncate file to work around bug in Bio.Nexus reader.
    lines = []
    with open(args.nexus_infile) as f:
        for line in f:
            if line.startswith("BEGIN CODONS"):
                break
            lines.append(line)
    f = io.StringIO("".join(lines))
    alignment = AlignIO.read(f, "nexus")

    num_taxa = min(len(alignment), args.max_taxa)
    num_characters = min(len(alignment[0]), args.max_characters)
    data = torch.zeros((num_taxa, num_characters), dtype=torch.long)
    mask = torch.zeros((num_taxa, num_characters), dtype=torch.bool)
    mapping = {
        "?": (False, 0),  # unobserved
        "A": (True, 0),
        "C": (True, 1),
        "G": (True, 2),
        "T": (True, 3),
        "-": (True, 4),  # insertion/deletion
    }
    for i in range(num_taxa):
        seq = alignment[i].seq
        for j in range(num_characters):
            mask[i, j], data[i, j] = mapping[seq[j]]

    times = torch.zeros(num_taxa)
    return times, data, mask


def train_guide(args, model):
    logger.info("Training via SVI")

    guide_config = {"init_loc_fn": model.init_loc_fn, "init_scale": 0.01}
    if args.guide_rank == 0:
        guide = AutoNormal(model, **guide_config)
    else:
        guide_config["rank"] = args.guide_rank
        guide = AutoLowRankMultivariateNormal(model, **guide_config)

    optim = ClippedAdam({"lr": args.learning_rate,
                         "lrd": args.learning_rate_decay ** (1 / args.num_steps)})
    svi = SVI(model, guide, optim, Trace_ELBO())
    t0 = args.init_temperature
    t1 = args.final_temperature
    num_observations = model.leaf_mask.sum()
    losses = []
    for step in range(args.num_steps):
        model.temperature = t0 * (t1 / t0) ** (step / (args.num_steps - 1))
        loss = svi.step() / num_observations
        if step == 0:
            logger.info("guide has {} parameters".format(
                sum(p.numel() for p in guide.parameters())))
        if step % args.log_every == 0:
            logger.info(f"step {step: >4} loss = {loss:0.4g}")
        assert math.isfinite(loss)
        losses.append(loss)


def _predict_task(args):
    args, model, guide, i = args
    torch.set_default_dtype(torch.double)
    pyro.set_rng_seed(args.seed + i)
    pyro.enable_validation(__debug__)

    guide_trace = poutine.trace(guide).get_trace()
    with poutine.replay(trace=guide_trace):
        trace = poutine.trace(model).get_trace(sample_tree=True)
        times = trace.nodes["times"]["value"]
        trees = trace.nodes["tree"]["value"]
    if i % args.log_every == 0:
        sys.stderr.write(".")
        sys.stderr.flush()
    return times, trees


@torch.no_grad()
def predict(args, model, guide):
    logger.info(f"Drawing {args.num_samples} posterior samples")
    map_ = mp.Pool().map if args.parallel else map
    samples = list(map_(_predict_task, [
        (args, model, guide, i)
        for i in range(args.num_samples)
    ]))
    times = torch.stack([sample[0] for sample in samples])
    trees = torch.stack([sample[1] for sample in samples])
    return times, trees


def binarize_tree(time, tree):
    """
    Convert an edge list to a nested binary frozenset of leaf ids.
    """
    assert time.dim() == 1
    N = time.size(0)
    assert tree.shape == (N - 1, 2)
    L = (N + 1) // 2

    # Initialize child and parent pointers.
    children = defaultdict(set)
    for u, v in tree.tolist():
        if time[u] < time[v]:
            children[u].add(v)
        else:
            children[v].add(u)
    assert all(v > L for v in children)
    assert len(children) <= N - L
    parents = {c: p for p, cs in children.items() for c in cs}
    assert len(parents) == N - 1

    # Prune internal nodes with zero or one child.
    pruned = set()
    changed = True
    while changed:
        changed = False
        for v in range(L, N):
            if v in pruned:
                continue
            cs = children[v]
            if len(cs) >= 2:
                continue
            changed = True
            pruned.add(v)
            del children[v]
            p = parents.pop(v, None)
            if not cs:
                continue
            c, = cs
            del parents[c]
            if p is None:
                continue
            children[p].remove(v)
            children[p].add(c)
            parents[c] = p
    del parents

    # Replace multi-ary nodes with binary nodes.
    for v, cs in list(children.items()):
        assert len(cs) >= 2
        while len(cs) > 2:
            c = pruned.pop()
            children[c] = {cs.pop(), cs.pop()}
            cs.add(c)
    assert not pruned

    result = [None] * N
    for v in time.sort(-1, descending=True).indices.tolist():
        if v < L:
            assert v not in children
            result[v] = v
        else:
            assert len(children[v]) == 2
            result[v] = frozenset(result[c] for c in children[v])

    return result[v]


@torch.no_grad()
def evaluate(args, times, trees):
    # Compute histogram over top k trees.
    # This aims to reproduce Figure 3 Section 6.5 of [1].
    # [1] Vu Dinh, Arman Bilge, Cheng Zhang, Frederick A. Matsen IV
    #     "Probabilistic Path Hamiltonian Monte Carlo"
    #     http://proceedings.mlr.press/v70/dinh17a.html
    counts = Counter()
    for time, tree in zip(times, trees):
        counts[binarize_tree(time, tree)] += 1
    top_k = counts.most_common(args.top_k)
    logger.info("Estimated posterior distribution for the top {} trees:\n{}"
                .format(args.top_k,
                        ", ".join("{:0.3g}".format(count / args.num_samples)
                                  for key, count in top_k)))


def main(args):
    mp.set_start_method("spawn")
    torch.set_default_dtype(torch.double)
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)

    leaf_times, leaf_data, leaf_mask = load_data(args)
    model = KirchhoffModel(leaf_times, leaf_data, leaf_mask,
                           embedding_dim=args.embedding_dim)
    guide = train_guide(args, model)
    times, trees = predict(args, model, guide)
    evaluate(args, times, trees)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tree learning experiment")
    parser.add_argument("--nexus-infile", default="data/treebase/M487.nex")
    parser.add_argument("--max-taxa", default=int(1e6), type=int)
    parser.add_argument("--max-characters", default=int(1e6), type=int)
    parser.add_argument("-d", "--embedding-dim", default=20, type=int)
    parser.add_argument("--guide-rank", default=0, type=int)
    parser.add_argument("-t0", "--init-temperature", default=1.0, type=float)
    parser.add_argument("-t1", "--final-temperature", default=0.01, type=float)
    parser.add_argument("-n", "--num-steps", default=501, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.2, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("-s", "--num-samples", default=1000, type=int)
    parser.add_argument("--parallel", default=True, action="store_true")
    parser.add_argument("--sequential", action="store_false", dest="parallel")
    parser.add_argument("--top-k", default=10, type=int)
    parser.add_argument("--seed", default=20201103, type=int)
    parser.add_argument("-l", "--log-every", default=1, type=int)
    args = parser.parse_args()

    # Disable multiprocessing when running under pdb.
    main_module = sys.modules["__main__"]
    if not hasattr(main_module, "__spec__"):
        args.parallel = False

    main(args)
