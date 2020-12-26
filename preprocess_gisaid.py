import argparse
import json
import logging
import os
import pickle
import re
import sys
from collections import Counter
from contextlib import ExitStack

import torch
import torch.multiprocessing as mp

from pyrophylo.cluster import ClockSketch, ClockSketcher

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.DEBUG)


def ln_sf(source, target):
    source = os.path.abspath(source)
    target = os.path.abspath(target)
    if os.path.exists(target):
        os.remove(target)
    os.symlink(source, target)


POOL = None


def pmap(fn, args):
    # Avoid multiprocessing when running under pdb.
    main_module = sys.modules["__main__"]
    if not hasattr(main_module, "__spec__"):
        return [fn(*a) for a in args]

    global POOL
    if POOL is None:
        POOL = mp.Pool()
    return POOL.starmap(fn, args)


def update_shards(shard_names):
    infile = os.path.expanduser("~/data/gisaid/provision.json")
    logger.info(f"Splitting {infile} into {args.num_shards} shards")
    if not os.path.exists(infile):
        raise OSError("Each user must independently request a data feed from gisaid.org")
    with ExitStack() as stack:
        f = stack.enter_context(open(infile))
        shards = [stack.enter_context(open(shard_name, "w"))
                  for shard_name in shard_names]
        for i, line in enumerate(f):
            shards[i % args.num_shards].write(line)
            if i % args.log_every == 0:
                print(".", end="", flush=True)
    logger.info(f"split {i + 1} lines")


STATS = ["date", "location", "length", "nchars"]


def _get_stats(args, filename):
    stats = {key: Counter() for key in STATS}
    with open(filename) as f:
        for line in f:
            datum = json.loads(line)
            stats["date"][datum["covv_collection_date"]] += 1
            stats["location"][datum["covv_location"]] += 1
            seq = datum["sequence"].replace("\n", "")
            stats["length"][len(seq)] += 1
            nchars = sum(map(len, re.findall("[ACGT]+", seq)))
            stats["nchars"][nchars] += 1
    return stats


def get_stats(args, shard_names):
    cache_file = "results/gisaid.stats.pkl"
    if args.force or not os.path.exists(cache_file):
        logger.info("Computing statistics")
        stats = {key: Counter() for key in STATS}
        for result in pmap(_get_stats, [(args, s) for s in shard_names]):
            for key, value in result.items():
                stats[key].update(value)
        logger.info(f"saving {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(stats, f)
    else:
        with open(cache_file, "rb") as f:
            stats = pickle.load(f)
    for key, counts in stats.items():
        logger.info("Top 10/{} {}s:\n{}".format(len(counts), key, "\n".join(
            f"{v: >6d}: {k}" for k, v in counts.most_common(10))))
    return stats


def _make_sketch(args, sketcher, shard_name):
    sequences = []
    with open(shard_name) as f:
        for i, line in enumerate(f):
            datum = json.loads(line)
            seq = datum["sequence"].replace("\n", "")
            parts = re.findall("[ACGT]+", seq)
            if args.min_nchars <= sum(map(len, parts)) <= args.max_nchars:
                sequences.append(parts)
            if i + 1 == args.truncate:
                break

    sketch = sketcher.init_sketch(len(sequences))
    for i, parts in enumerate(sequences):
        sketcher.string_to_hash(parts, sketch[i])
        if i % args.log_every == 0:
            print(".", end="", flush=True)

    return sketch


def make_sketch(args, shard_names):
    cache_file = f"results/gisaid.sketch.{args.k}.pt"
    sketcher = ClockSketcher(k=args.k, num_clocks=args.num_clocks)
    if args.force or not os.path.exists(cache_file):
        args.force = True
        logger.info("Sketching k-mers bags")
        sketches = pmap(_make_sketch, [(args, sketcher, s) for s in shard_names])
        clocks = torch.cat([s.clocks for s in sketches])
        count = torch.cat([s.count for s in sketches])
        sketch = ClockSketch(clocks, count)
        # TODO
        # logger.info("greedily finding clusters")
        result = {
            "k": args.k,
            "num_clocks": args.num_clocks,
            "sketch": sketch,
        }
        torch.save(result, cache_file)
        logger.info(f"saving {cache_file}")
    else:
        result = torch.load(cache_file)
        sketch = result["sketch"]
    ln_sf(cache_file, "results/gisaid.sketch.pt")
    return sketcher, sketch


def cluster(args, sketcher, sketch):
    cache_file = f"results/gisaid.cluster.{args.num_clusters}.{args.cluster_radius}.{args.cluster_epochs}.pt"
    if args.force or not os.path.exists(cache_file):
        args.force = True
        logger.info(f"Clustering {len(sketch)} taxa into {args.num_clusters} clusters")
        result = sketcher.find_clusters(sketch, num_clusters=args.num_clusters,
                                        radius=args.cluster_radius,
                                        epochs=args.cluster_epochs)
        torch.save(result, cache_file)
    else:
        result = torch.load(cache_file)
    ln_sf(cache_file, "results/gisaid.cluster.pt")
    return result


def main(args):
    shard_names = [f"results/gisaid.{i:03d}-of-{args.num_shards:03d}.json"
                   for i in range(args.num_shards)]
    if args.force or not all(map(os.path.exists, shard_names)):
        args.force = True
        update_shards(shard_names)
    stats = get_stats(args, shard_names)

    # Drop low quality sequences.
    assert 0 <= args.min_nchars_rel <= 1 <= args.max_nchars_rel
    nchars = stats["nchars"]
    mean_nchars = sum(k * v for k, v in nchars.items()) / sum(nchars.values())
    args.max_nchars = int(args.max_nchars_rel * mean_nchars)
    args.min_nchars = int(args.min_nchars_rel * mean_nchars)
    num_ok = sum(v for k, v in nchars.items()
                 if args.min_nchars < k < args.max_nchars)
    total = sum(nchars.values())
    logger.info(f"Keeping {num_ok}/{total} = {100*num_ok/total:0.1f}% of sequences")

    if args.truncate:
        shard_names = shard_names[:1]
    sketcher, sketch = make_sketch(args, shard_names)
    clustering = cluster(args, sketcher, sketch)
    assert clustering


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Preprocess GISAID data")
    parser.add_argument("--min-nchars-rel", default=0.95, type=float)
    parser.add_argument("--max-nchars-rel", default=1.05, type=float)
    parser.add_argument("--k", default=20, type=int)
    parser.add_argument("--num-clocks", default=256, type=int)
    parser.add_argument("--num-clusters", default=200, type=int)
    parser.add_argument("--cluster-radius", default=100, type=int)
    parser.add_argument("--cluster-epochs", default=2, type=int)
    parser.add_argument("-s", "--num-shards", default=mp.cpu_count(), type=int)
    parser.add_argument("-l", "--log-every", default=1000, type=int)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--truncate", default=0, type=int)
    args = parser.parse_args()

    main(args)
