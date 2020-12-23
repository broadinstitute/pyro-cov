import argparse
import json
import logging
import os
import pickle
import sys
from collections import Counter
from contextlib import ExitStack

import torch
import torch.multiprocessing as mp

from pyrophylo.cluster import KmerSketcher

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.DEBUG)


def print_dot():
    sys.stderr.write(".")
    sys.stderr.flush()


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
                print_dot()
    logger.info(f"split {i + 1} lines")


STATS = ["date", "location", "length"]


def _get_stats(args, filename):
    stats = {key: Counter() for key in STATS}
    with open(filename) as f:
        for line in f:
            datum = json.loads(line)
            stats["date"][datum["covv_collection_date"]] += 1
            stats["location"][datum["covv_location"]] += 1
            seq = datum["sequence"].replace("\n", "")
            stats["length"][len(seq)] += 1
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


def _cluster(args, sketcher, shard_name):
    sequences = []
    with open(shard_name) as f:
        for line in f:
            datum = json.loads(line)
            seq = datum["sequence"].replace("\n", "")
            if args.min_length <= len(seq) <= args.max_length:
                sequences.append(seq)

    hashes = torch.empty(len(sequences), sketcher.bits)
    for i, seq in enumerate(sequences):
        sketcher.string_to_soft_hash(seq, hashes[i])
        if i % args.log_every == 0:
            print_dot()

    return hashes


def cluster(args, shard_names):
    cache_file = "results/gisaid.cluster.pt"
    if args.force or not os.path.exists(cache_file):
        logger.info("Clustering via LSH of k-mers")
        sketcher = KmerSketcher(min_k=args.min_k, max_k=args.max_k, bits=args.cluster_bits)

        hashes = pmap(_cluster, [(args, sketcher, s) for s in shard_names])
        hashes = torch.cat(hashes, 0)
        logger.info("quantizing hashes")
        hashes = sketcher.soft_to_hard_hashes(hashes)
        logger.info("greedily finding clusters")
        clusters = sketcher.find_clusters(hashes, radius=args.cluster_radius)
        num_clusters = min(len(clusters), args.max_clusters)
        logger.info(f"Using {num_clusters}/{len(clusters)} clusters")
        clusters = clusters[:args.max_clusters]
        logger.info("computing pairwise sequence-cluster distances")
        distances = sketcher.cdist(hashes, clusters)
        clustering = {
            "hashes": hashes,
            "clusters": clusters,
            "distances": distances,
        }
        torch.save(clustering, cache_file)
        logger.info(f"saving {cache_file}")
    else:
        clustering = torch.load(cache_file)
    return clustering


def main(args):
    shard_names = [f"results/gisaid.{i:03d}-of-{args.num_shards:03d}.json"
                   for i in range(args.num_shards)]
    if args.force or not all(map(os.path.exists, shard_names)):
        update_shards(shard_names)
    stats = get_stats(args, shard_names)

    # Drop low quality sequences.
    assert 0 <= args.min_length_rel <= 1 <= args.max_length_rel
    length = stats["length"]
    mean_length = sum(k * v for k, v in length.items()) / sum(length.values())
    args.max_length = int(args.max_length_rel * mean_length)
    args.min_length = int(args.min_length_rel * mean_length)
    num_ok = sum(v for k, v in length.items()
                 if args.min_length < k < args.max_length)
    total = sum(length.values())
    logger.info(f"Keeping {num_ok}/{total} = {100*num_ok/total:0.1f}% of sequences")

    clustering = cluster(args, shard_names)
    assert clustering


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Preprocess GISAID data")
    parser.add_argument("--min-length-rel", default=0.9, type=float)
    parser.add_argument("--max-length-rel", default=1.1, type=float)
    parser.add_argument("--min-k", default=2, type=int)
    parser.add_argument("--max-k", default=6, type=int)
    parser.add_argument("--cluster-bits", default=16, type=int)
    parser.add_argument("--cluster-radius", default=6, type=int)
    parser.add_argument("--max-clusters", default=200, type=int)
    parser.add_argument("-s", "--num-shards", default=mp.cpu_count(), type=int)
    parser.add_argument("-l", "--log-every", default=1000, type=int)
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    main(args)
