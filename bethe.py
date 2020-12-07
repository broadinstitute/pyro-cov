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
import logging
import math
import os
import sys
from collections import Counter

import pyro
import pyro.poutine as poutine
import setuptools  # noqa F401
import torch
import torch.multiprocessing as mp
from pyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoLowRankMultivariateNormal, AutoNormal
from pyro.optim import ClippedAdam

from pyrophylo.bethe import BetheModel
from pyrophylo.io import read_alignment
from pyrophylo.phylo import Phylogeny

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.DEBUG)


def print_dot():
    sys.stderr.write(".")
    sys.stderr.flush()


def load_data(args):
    filename = os.path.expanduser(args.alignment_infile)
    probs = read_alignment(filename, max_taxa=args.max_taxa,
                           max_characters=args.max_characters)

    # Optionally ignore low-diversity sites.
    if args.min_diversity > 0:
        diversity = 1 - (probs > 0).float().mean(0).max(-1).values
        mask = diversity > args.min_diversity
        logger.info(f"Cropping to {mask.sum():d}/{len(mask)} diverse characters")
        probs = probs[:, mask]

    # Convert probs to logits.
    probs.mul_(1 - args.error_rate).add_(args.error_rate / probs.size(-1))
    logits = probs.log_()

    times = torch.zeros(probs.size(0))
    return times, logits


def pretrain_model(args, model):
    logger.info(f"Pretraining model via SVI for {args.pre_steps} steps")
    logger.info("model has {} parameters".format(
        sum(p.numel() for p in model.parameters())))

    # Pretrain the model via SVI with an AutoDelta guide.
    optim = ClippedAdam({"lr": args.learning_rate, "betas": (0.8, 0.9)})
    guide = AutoDelta(model, init_loc_fn=model.init_loc_fn)
    guide = poutine.block(guide, hide_types=["param"])  # Keep leaf codes fixed.
    svi = SVI(model, guide, optim, Trace_ELBO())
    losses = []
    for step in range(args.pre_steps):
        loss = svi.step(mode="pretrain") / model.num_observations
        if step % args.log_every == 0:
            logger.info(f"step {step: >4} loss = {loss:0.4g}")
        assert math.isfinite(loss)
        losses.append(loss)
    return losses


def train_guide(args, model):
    logger.info(f"Training model+guide via SVI for {args.num_steps} steps")

    # Configure a guide.
    guide_config = {"init_loc_fn": model.init_loc_fn,
                    "init_scale": args.init_scale}
    if args.guide_map:
        guide = AutoDelta(model, init_loc_fn=model.init_loc_fn)
    elif args.guide_rank == 0:
        guide = AutoNormal(model, **guide_config)
    else:
        guide_config["rank"] = args.guide_rank
        guide = AutoLowRankMultivariateNormal(model, **guide_config)
    guide()
    logger.info("guide has {} parameters".format(
        sum(p.numel() for p in guide.parameters())))
    history = {}
    if args.debug_grads:
        for name, param in guide.named_parameters():
            @param.register_hook
            def print_grad_norm(grad, name=name, param=param):
                print(f"{name}: [{grad.data.min():0.3g}, {grad.data.max():0.3g}]")
    if args.debug_time:
        for name, param in guide.named_parameters():
            if "internal_times" in name and "scales" not in name:
                @param.register_hook
                def print_time_grad(grad, name=name, param=param):
                    value, root = param.data.max(-1)
                    value = value.item()
                    grad = grad.data[root].item()
                    history.setdefault("root time value", []).append(value)
                    history.setdefault("root time grad", []).append(grad)
                    print(f"{name}[root] value={value:0.3g}, grad={grad:0.3g}")

    # Train the guide via SVI.
    optim = ClippedAdam({"lr": args.learning_rate,
                         "lrd": args.learning_rate_decay ** (1 / args.num_steps),
                         "clip_norm": args.clip_norm,
                         "betas": (0.8, 0.99)})
    svi = SVI(model, guide, optim, Trace_ELBO())
    losses = []
    for step in range(args.num_steps):
        loss = svi.step() / model.num_observations
        if step % args.log_every == 0:
            logger.info(f"step {step: >4} loss = {loss:0.4g}")
        assert math.isfinite(loss)
        losses.append(loss)

    # Log diagnostics.
    median = guide.median()
    message = ["median latent variables:"]
    for name, value in sorted(median.items()):
        if value.numel() == 1:
            message.append(f"{name} = {value:0.3g}")
        else:
            message.append(f"{name}.shape:{tuple(value.shape)}, "
                           f"lb mean ub: {value.min().item():0.3g} "
                           f"{value.mean().item():0.3g} "
                           f"{value.max().item():0.3g}")
    logger.info("\n".join(message))

    return guide, losses, history


@torch.no_grad()
def _predict_task(args):
    args, model, guide, i = args
    torch.set_default_dtype(torch.double)
    pyro.set_rng_seed(args.seed + i)

    guide_trace = poutine.trace(guide).get_trace()
    with poutine.replay(trace=guide_trace):
        codes, times, parents = model(mode="predict")

    # Convert to Phylogeny objects.
    leaves = torch.arange(model.num_leaves)
    tree, old2new, new2old = Phylogeny.sort(times, parents, leaves)
    codes = codes[new2old]

    if i % args.log_every == 0:
        print_dot()
    return tree, codes


def predict(args, model, guide):
    logger.info(f"Drawing {args.num_samples} posterior samples")
    map_ = mp.Pool().map if args.parallel else map
    samples = list(map_(_predict_task, [
        (args, model, guide, i)
        for i in range(args.num_samples)
    ]))
    trees = Phylogeny.stack([tree for tree, _ in samples])
    codes = torch.stack([codes for _, codes in samples])
    return trees, codes


def sample_model_mcmc(args, model):
    logger.info(f"Running MCMC for {2 * args.num_samples} steps")

    # Freeze the decoder neural network.
    model.requires_grad_(False).train(False)
    frozen_model = poutine.block(model, hide_fn=lambda msg: "decoder." in msg["name"])

    # Run mcmc.
    kernel = NUTS(frozen_model,
                  step_size=args.init_scale,
                  full_mass=[("internal_times", "subs_model.rate")],
                  init_strategy=model.init_loc_fn,
                  max_plate_nesting=model.max_plate_nesting,
                  max_tree_depth=args.max_tree_depth)
    mcmc = MCMC(kernel,
                num_samples=args.num_samples,
                num_chains=args.num_chains)
    mcmc.run()
    samples = mcmc.get_samples()
    with torch.no_grad():
        predictive = Predictive(model, samples,
                                return_sites=["codes", "times", "parents"])
        samples = predictive(mode="predict")

        # Convert to a stacked Phylogeny object.
        trees = []
        codes = torch.empty_like(samples["codes"])
        leaves = torch.arange(model.num_leaves)
        for i in range(args.num_samples):
            times = samples["times"][i].squeeze()
            parents = samples["parents"][i].squeeze()
            tree, old2new, new2old = Phylogeny.sort(times, parents, leaves)
            trees.append(tree)
            codes[i] = samples["codes"][i, new2old]
        trees = Phylogeny.stack(trees)
        return trees, codes


def pretty_tree(t):
    if isinstance(t, frozenset):
        x, y = sorted(map(pretty_tree, t))
        return f"({x} {y})"
    return str(t)


@torch.no_grad()
def evaluate(args, trees):
    # Compute histogram over top k trees.
    # This aims to reproduce Figure 3 Section 6.5 of [1].
    # [1] Vu Dinh, Arman Bilge, Cheng Zhang, Frederick A. Matsen IV
    #     "Probabilistic Path Hamiltonian Monte Carlo"
    #     http://proceedings.mlr.press/v70/dinh17a.html
    counts = Counter(t.hash_topology() for t in trees)
    top_k = counts.most_common(args.top_k)
    logger.info("Estimated posterior distribution for the top {} trees:\n{}"
                .format(args.top_k,
                        ", ".join("{:0.3g}".format(count / args.num_samples)
                                  for key, count in top_k)))
    if args.print_trees:
        for i, (tree, count) in enumerate(top_k):
            logger.info(f"Tree {i}:\n{pretty_tree(tree)}")


def main(args):
    mp.set_start_method("spawn")
    torch.set_default_dtype(torch.double if args.double else torch.float)
    pyro.set_rng_seed(args.seed)

    # Run the pipeline.
    leaf_times, leaf_logits = load_data(args)
    args.embedding_dim = min(args.embedding_dim, len(leaf_times))
    model = BetheModel(leaf_times, leaf_logits,
                       embedding_dim=args.embedding_dim,
                       bp_iters=args.bp_iters, min_dt=args.min_dt)
    if args.subs_rate is not None:
        model.subs_model.rate = args.subs_rate
    losses = pretrain_model(args, model)
    guide, guide_losses, history = train_guide(args, model)
    losses += guide_losses
    if args.mcmc:
        # Use model.decoder trained via SVI, but draw samples via NUTS.
        trees, codes = sample_model_mcmc(args, model)
    else:
        # Sample directly from the guide.
        trees, codes = predict(args, model, guide)
    evaluate(args, trees)

    # Save results for bethe.ipynb.
    if args.outfile:
        logger.info(f"saving to {args.outfile}")
        results = {
            "args": args,
            "data": {
                "leaf_times": leaf_times,
                "leaf_logits": leaf_logits,
            },
            "model": model,
            "guide": guide,
            "losses": losses,
            "history": history,
            "samples": {
                "trees": trees,
                "codes": codes,
            },
        }
        dirname = os.path.dirname(args.outfile)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(results, args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tree learning experiment")
    parser.add_argument("-i", "--alignment-infile", default="data/treebase/M487.nex")
    parser.add_argument("-o", "--outfile", default="results/bethe.pt")
    parser.add_argument("--max-taxa", default=int(1e6), type=int)
    parser.add_argument("--max-characters", default=int(1e6), type=int)
    parser.add_argument("--min-diversity", default=1e-2, type=float)
    parser.add_argument("--error-rate", default=1e-3, type=float)
    parser.add_argument("--subs-rate", type=float)
    parser.add_argument("--min-dt", default=0.01, type=float)
    parser.add_argument("-e", "--embedding-dim", default=20, type=int)
    parser.add_argument("-bp", "--bp-iters", default=30, type=int)
    parser.add_argument("--exact", dest="bp_iters", action="store_const", const=None)
    parser.add_argument("-n0", "--pre-steps", default=101, type=int)
    parser.add_argument("-map", "--guide-map", action="store_true")
    parser.add_argument("-r", "--guide-rank", default=20, type=int)
    parser.add_argument("-is", "--init-scale", default=0.05, type=float)
    parser.add_argument("-n", "--num-steps", default=501, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("-cn", "--clip-norm", default=1e2, type=float)
    parser.add_argument("-mcmc", action="store_true")
    parser.add_argument("-mtd", "--max-tree-depth", default=5, type=int)
    parser.add_argument("--full-mass", action="store_true")
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("-s", "--num-samples", default=200, type=int)
    parser.add_argument("--double", default=True, action="store_true")
    parser.add_argument("--single", action="store_false", dest="double")
    parser.add_argument("--parallel", default=True, action="store_true")
    parser.add_argument("--sequential", action="store_false", dest="parallel")
    parser.add_argument("--top-k", default=10, type=int)
    parser.add_argument("-p", "--print-trees", action="store_true")
    parser.add_argument("--seed", default=20201103, type=int)
    parser.add_argument("-l", "--log-every", default=10, type=int)
    parser.add_argument("-dg", "--debug-grads", action="store_true")
    parser.add_argument("-dt", "--debug-time", action="store_true")
    args = parser.parse_args()

    # Disable multiprocessing when running under pdb.
    main_module = sys.modules["__main__"]
    if not hasattr(main_module, "__spec__"):
        args.parallel = False

    main(args)
