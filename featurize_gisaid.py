#!/usr/bin/env python

import argparse
import json
import logging
import multiprocessing as mp
import os
import pickle
import re
import sys
from collections import defaultdict
from contextlib import ExitStack

import torch

from pyrophylo import pangolin
from pyrophylo.sketch import KmerCounter

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


POOL = None


def parallel_map(fn, args):
    # Avoid multiprocessing when running under pdb.
    if not hasattr(sys.modules["__main__"], "__spec__"):
        return [fn(*a) for a in args]

    global POOL
    if POOL is None:
        POOL = mp.Pool()
    return POOL.starmap(fn, args)


def update_shards(args):
    shard_names = [
        f"results/gisaid.{i:03d}-of-{args.num_shards:03d}.json"
        for i in range(args.num_shards)
    ]
    if not os.path.exists("results"):
        os.makedirs("results")
    if all(map(os.path.exists, shard_names)) and not args.force:
        return shard_names
    args.force = True
    infile = args.gisaid_file_in
    logger.info(f"Splitting {infile} into {args.num_shards} shards")
    if not os.path.exists(infile):
        raise OSError(
            "Each user must independently request a data feed from gisaid.org"
        )
    with ExitStack() as stack:
        f = stack.enter_context(open(infile))
        shards = [
            stack.enter_context(open(shard_name, "w")) for shard_name in shard_names
        ]
        for i, line in enumerate(f):
            shards[i % args.num_shards].write(line)
            if i % args.log_every == 0:
                print(".", end="", flush=True)
    logger.info(f"split {i + 1} lines")
    return shard_names


def _count_kmers(args, shard_name):
    lineage_counts = defaultdict(int)
    kmer_counts = defaultdict(KmerCounter)
    with open(shard_name) as f:
        for i, line in enumerate(f):
            datum = json.loads(line)
            if i % args.log_every == 0:
                print(".", end="", flush=True)
            if i == args.truncate:
                break
            lineage = datum.get("covv_lineage")
            if not lineage:
                continue  # Drop rows with no lineage information.
            lineage = pangolin.compress(pangolin.decompress(lineage))
            seq = datum["sequence"].replace("\n", "")
            parts = re.findall("[ACGT]+", seq)
            if not (args.min_nchars <= sum(map(len, parts)) <= args.max_nchars):
                continue  # Drop rows with too few nucleotides.

            lineage_counts[lineage] += 1
            counter = kmer_counts[lineage]
            for part in parts:
                if len(part) >= 32:
                    counter.update(part)

    # Flatten and pickle this large data structure.
    for counter in kmer_counts.values():
        counter.flush(truncate_below=lineage_counts[lineage] // 2)
    flat = [
        (lineage, lineage_counts[lineage], list(counter.keys()), list(counter.values()))
        for lineage, counter in kmer_counts.items()
    ]
    result = shard_name + ".kmer_counts.pkl"
    with open(result, "wb") as f:
        pickle.dump(flat, f)
    return result


def count_kmers(args, shard_names):
    cache_file = "results/gisaid.kmer_counts.pt"
    if os.path.exists(cache_file) and not args.force:
        return torch.load(cache_file)
    args.force = True
    logger.info("Counting kmers")
    shards = parallel_map(_count_kmers, [(args, name) for name in shard_names])

    # Create a dense schema.
    lineages = set()
    kmers = set()
    for shard in shards:
        with open(shard, "rb") as f:
            for lineage, count, keys, values in pickle.load(f):
                lineages.add(lineage)
                kmers.update(keys)
            print(".", end="", flush=True)
    lineage_ids = {k: i for i, k in enumerate(sorted(lineages))}
    kmer_ids = {k: i for i, k in enumerate(sorted(kmers))}

    # Fill a dense matrix.
    lineage_counts = torch.zeros(len(lineages), dtype=torch.float)
    kmer_counts = torch.zeros((len(lineages), len(kmers)), dtype=torch.float)
    for shard in shards:
        with open(shard, "rb") as f:
            for lineage, count, keys, values in pickle.load(f):
                i = lineage_ids[lineage]
                lineage_counts[i] += count
                js = list(map(kmer_ids.__getitem__, keys))
                kmer_counts[i, js] += torch.tensor(values, dtype=torch.float)
            print(".", end="", flush=True)

    result = {
        "args": args,
        "lineage_ids": lineage_ids,
        "kmer_ids": kmer_ids,
        "lineage_counts": lineage_counts,
        "kmer_counts": kmer_counts,
    }
    torch.save(result, cache_file)
    return result


def main(args):
    shard_names = update_shards(args)

    kmer_counts = count_kmers(args, shard_names)
    num_lineages, num_kmers = kmer_counts["kmer_counts"].shape
    total_kmers = kmer_counts["kmer_counts"].sum().item()
    logger.info(
        f"Counted {total_kmers:.4g} instances of {num_kmers} kmers "
        f"among {num_lineages} lineages"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize GISAID data")
    parser.add_argument(
        "--gisaid-file-in", default=os.path.expanduser("~/data/gisaid/provision.json")
    )
    parser.add_argument("--min-nchars", default=29000, type=int)
    parser.add_argument("--max-nchars", default=31000, type=int)
    parser.add_argument("--num-shards", default=mp.cpu_count(), type=int)
    parser.add_argument("--truncate", default=-1, type=int)
    parser.add_argument("-l", "--log-every", default=1000, type=int)
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    # mp.set_start_method("spawn")  # For torch.multiprocessing
    main(args)
