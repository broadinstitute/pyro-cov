import argparse
import datetime
import json
import logging
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack

import torch
import torch.multiprocessing as mp
from Bio import SeqIO

from pyrophylo.align import Differ
from pyrophylo.cluster import ClockSketch, ClockSketcher

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.DEBUG)

# Source: https://samtools.github.io/hts-specs/SAMv1.pdf
CIGAR_CODES = "MIDNSHP=X"  # Note minimap2 uses only "MIDNSH"


def ln_sf(source, target):
    source = os.path.abspath(source)
    target = os.path.abspath(target)
    if os.path.exists(target):
        os.remove(target)
    os.symlink(source, target)


DATE_FORMATS = {4: "%Y", 7: "%Y-%m", 10: "%Y-%m-%d"}


def parse_date(string):
    return datetime.datetime.strptime(string, DATE_FORMATS[len(string)])


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
    shard_names = [f"results/gisaid.{i:03d}-of-{args.num_shards:03d}.json"
                   for i in range(args.num_shards)]
    if args.force or not all(map(os.path.exists, shard_names)):
        args.force = True
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
    return shard_names


STATS = ["date", "location", "length", "nchars", "gapsize"]


def _get_stats(args, filename):
    stats = {key: Counter() for key in STATS}
    with open(filename) as f:
        for line in f:
            datum = json.loads(line)
            stats["date"][datum["covv_collection_date"]] += 1
            stats["location"][datum["covv_location"]] += 1
            seq = datum["sequence"].replace("\n", "")
            length = len(seq)
            stats["length"][length] += 1
            nchars = sum(map(len, re.findall("[ACGT]+", seq)))
            stats["nchars"][nchars] += 1
            stats["gapsize"][length - nchars] += 1
    return stats


def get_stats(args, shard_names):
    cache_file = "results/gisaid.stats.pkl"
    if args.force or not os.path.exists(cache_file):
        logger.info("Computing statistics")
        stats = {key: Counter() for key in STATS}
        shards = parallel_map(_get_stats, [
            (args, s) for s in shard_names
        ])
        for shard in shards:
            for key, value in shard.items():
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


def _filter_sequences(args, shard_name, columns=None):
    if columns is None:
        columns = {}
    fields = ["accession_id", "collection_date", "location", "add_location"]
    for key in fields:
        columns[key] = []
    columns["day"] = []
    with open(shard_name) as f:
        for line in f:
            datum = json.loads(line)
            if len(datum["covv_collection_date"]) < 7:
                continue  # Drop rows with no month information.
            date = parse_date(datum["covv_collection_date"])
            if date < args.start_date:
                continue  # Drop rows before start date.
            seq = datum["sequence"].replace("\n", "")
            nchar = sum(map(len, re.findall("[ACGT]+", seq)))
            if not (args.min_nchars <= nchar <= args.max_nchars):
                continue  # Drop rows with too few nucleotides.
            for key in fields:
                columns[key].append(datum["covv_" + key])
            columns["day"].append((date - args.start_date).days)
            yield seq


def _align_1(args, shard_name):
    for reference in SeqIO.parse(args.reference_sequence, "fasta"):
        differ = Differ(str(reference.seq),
                        lb=args.reference_start,
                        ub=args.reference_end,
                        preset="asm10")
    stats = Counter()
    columns = {}
    for i, seq in enumerate(_filter_sequences(args, shard_name, columns)):
        stats.update(differ.diff(seq))
        if i % args.log_every == 0:
            print(".", end="", flush=True)
    return stats, columns


def _align_2(args, feature_dict, size, shard_name):
    for reference in SeqIO.parse(args.reference_sequence, "fasta"):
        differ = Differ(str(reference.seq),
                        lb=args.reference_start,
                        ub=args.reference_end,
                        preset="asm10")
    features = torch.zeros(size, len(feature_dict), dtype=torch.bool)
    for i, seq in enumerate(_filter_sequences(args, shard_name)):
        for diff in differ.diff(seq):
            f = feature_dict.get(diff, None)
            if f is not None:
                features[i, f] = True
        if i % args.log_every == 0:
            print(".", end="", flush=True)
    return features


def align(args, shard_names):
    cache_file = "results/gisaid.align.pt"
    if args.force or not os.path.exists(cache_file):
        args.force = True
        logger.info("Aligning sequences")
        shards = parallel_map(_align_1, [
            (args, name) for name in shard_names
        ])
        stats = Counter()
        shard_sizes = []
        columns = defaultdict(list)
        for shard in shards:
            stats.update(shard[0])
            shard_sizes.append(len(shard[1]["day"]))
            for key, column in shard[1].items():
                columns[key].extend(column)
        features = stats.most_common(args.num_features)
        feature_dict = {key: i for i, (key, count) in enumerate(features)}
        min_count = features[-1][-1]
        logger.info(f"keeping {len(features)} mutations, with min count {min_count}")
        shards = parallel_map(_align_2, [
            (args, feature_dict, size, name)
            for name, size in zip(shard_names, shard_sizes)
        ])
        features = torch.cat(shards)
        result = {
            "args": args,
            "stats": stats,
            "columns": columns,
            "feature_dict": feature_dict,
            "features": features,
        }
        torch.save(result, cache_file)
    else:
        result = torch.load(cache_file)
    return result


def _make_sketch(args, sketcher, shard_name):
    sequences = []
    fields = ["accession_id", "collection_date", "location", "add_location"]
    columns = {key: [] for key in fields}
    columns["day"] = []
    with open(shard_name) as f:
        for line in f:
            datum = json.loads(line)
            if len(datum["covv_collection_date"]) < 7:
                continue  # Drop rows with no month information.
            date = parse_date(datum["covv_collection_date"])
            if date < args.start_date:
                continue  # Drop rows before start date.
            seq = datum["sequence"].replace("\n", "")
            parts = re.findall("[ACGT]+", seq)
            if not (args.min_nchars <= sum(map(len, parts)) <= args.max_nchars):
                continue  # Drop rows with too few nucleotides.
            sequences.append(parts)
            for key in fields:
                columns[key].append(datum["covv_" + key])
            columns["day"].append((date - args.start_date).days)

    sketch = sketcher.init_sketch(len(sequences))
    for i, parts in enumerate(sequences):
        sketcher.string_to_hash(parts, sketch[i])
        if i % args.log_every == 0:
            print(".", end="", flush=True)

    return sketch, columns


def make_sketch(args, shard_names):
    cache_file = f"results/gisaid.sketch.{args.k}.{args.num_clocks}.pt"
    sketcher = ClockSketcher(k=args.k, num_clocks=args.num_clocks)
    if args.force or not os.path.exists(cache_file):
        args.force = True
        logger.info("Sketching k-mer bags")
        shards = parallel_map(_make_sketch, [
            (args, sketcher, s) for s in shard_names
        ])
        clocks = torch.cat([s.clocks for s, _ in shards])
        count = torch.cat([s.count for s, _ in shards])
        sketch = ClockSketch(clocks, count)
        columns = {}
        for _, s in shards:
            for key, column in s.items():
                columns.setdefault(key, []).extend(column)
        result = {
            "args": args,
            "sketch": sketch,
            "columns": columns,
        }
        torch.save(result, cache_file)
        logger.info(f"saving {cache_file}")
    else:
        result = torch.load(cache_file)
        sketch = result["sketch"]
        columns = result["columns"]
    ln_sf(cache_file, "results/gisaid.sketch.pt")
    return sketcher, sketch, columns


def cluster(args, sketcher, sketch):
    cache_file = (f"results/gisaid.cluster.{args.cluster_radius}.{args.max_clusters}."
                  f"{args.min_cluster_size}.{args.cluster_epochs}.pt")
    if args.force or not os.path.exists(cache_file):
        args.force = True
        logger.info(f"Clustering {len(sketch)} taxa into {args.max_clusters} clusters")
        full_clusters, full_weights = sketcher.find_clusters(
            sketch,
            num_clusters=args.max_clusters,
            radius=args.cluster_radius,
            epochs=args.cluster_epochs,
        )

        keep = full_weights >= args.min_cluster_size
        logger.info(f"keeping {keep.sum():d}/{args.max_clusters} largest clusters")
        weights = full_weights[keep]
        clusters = full_clusters[keep]

        logger.info(f"classifying {len(sketch)} sketches against {len(clusters)} clusters")
        class_probs = sketcher.classify(sketch, clusters, weights,
                                        radius=args.cluster_radius)
        result = {
            "args": args,
            "full_clusters": full_clusters,
            "full_weights": full_weights,
            "clusters": clusters,
            "weights": weights,
            "class_probs": class_probs,
        }
        torch.save(result, cache_file)
    else:
        result = torch.load(cache_file)
    ln_sf(cache_file, "results/gisaid.cluster.pt")
    class_probs = result["class_probs"]
    perplexity = (class_probs.log().neg_()
                             .clamp_(max=torch.finfo(class_probs.dtype).max)
                             .mul_(class_probs).sum(-1).exp())
    logging.info(f"class perplexity min={perplexity.min():0.2f}, "
                 f"median={perplexity.median():0.1f}, "
                 f"mean={perplexity.mean():0.1f}, "
                 f"max={perplexity.max():0.1f}")
    return result


def main(args):
    # Shard and summarize data.
    shard_names = update_shards(args)
    stats = get_stats(args, shard_names)

    # Configure dropping of low quality sequences.
    assert 0 <= args.min_nchars_rel <= 1 <= args.max_nchars_rel
    nchars = stats["nchars"]
    mean_nchars = sum(k * v for k, v in nchars.items()) / sum(nchars.values())
    args.max_nchars = int(args.max_nchars_rel * mean_nchars)
    args.min_nchars = int(args.min_nchars_rel * mean_nchars)

    if args.align:
        # Align sequences.
        align(args, shard_names)
    else:
        # Make sketches.
        sketcher, sketch, columns = make_sketch(args, shard_names)
        kept = len(sketch)
        total = sum(nchars.values())
        logger.info(f"kept {kept}/{total} = {100*kept/total:0.1f}% of sequences")

        # Cluster taxa.
        clustering = cluster(args, sketcher, sketch)
        assert clustering


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Preprocess GISAID data")
    parser.add_argument("--start-date", default="2019-12-01")
    parser.add_argument("--reference-sequence",
                        default="data/ncbi-reference-sequence.fasta")
    parser.add_argument("--reference-start", default=2000, type=int)
    parser.add_argument("--reference-end", default=27000, type=int)
    parser.add_argument("--min-nchars-rel", default=0.95, type=float)
    parser.add_argument("--max-nchars-rel", default=1.05, type=float)
    parser.add_argument("--num-features", default=1024, type=int)
    parser.add_argument("--k", default=9, type=int)
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--num-clocks", default=256, type=int)
    parser.add_argument("--max-clusters", default=200, type=int)
    parser.add_argument("--min-cluster-size", default=10, type=int)
    parser.add_argument("--cluster-radius", default=40, type=int)
    parser.add_argument("--cluster-epochs", default=10, type=int)
    parser.add_argument("--num-shards", default=mp.cpu_count(), type=int)
    parser.add_argument("-l", "--log-every", default=1000, type=int)
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()
    args.start_date = parse_date(args.start_date)

    main(args)
