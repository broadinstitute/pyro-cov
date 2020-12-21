import argparse
import json
import logging
import multiprocessing as mp
import os
import pickle
import sys
from collections import Counter
from contextlib import ExitStack

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
        return map(fn, args)

    global POOL
    if POOL is None:
        POOL = mp.Pool()
    return POOL.map(fn, args)


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


def _get_stats(filename):
    dates = Counter()
    locations = Counter()
    with open(filename) as f:
        for line in f:
            datum = json.loads(line)
            dates[datum["covv_collection_date"]] += 1
            locations[datum["covv_location"]] += 1
    return dates, locations


def get_stats(args, shard_names):
    cache_file = "results/gisaid.stats"
    if args.force or not os.path.exists(cache_file):
        dates = Counter()
        locations = Counter()
        for result in pmap(_get_stats, shard_names):
            dates.update(result[0])
            locations.update(result[1])
        stats = {"dates": dates, "locations": locations}
        with open(cache_file, "wb") as f:
            pickle.dump(stats, f)
    else:
        with open(cache_file, "rb") as f:
            stats = pickle.load(f)

    logger.info("Top dates:\n{}".format("\n".join(
        f"{v: >5d}: {k}" for k, v in stats["dates"].most_common(10))))
    logger.info("Top locations:\n{}".format("\n".join(
        f"{v: >5d}: {k}" for k, v in stats["locations"].most_common(10))))
    return stats


def main(args):
    shard_names = [f"results/gisaid.{i:03d}-of-{args.num_shards:03d}.json"
                   for i in range(args.num_shards)]
    if args.force or not all(map(os.path.exists, shard_names)):
        update_shards(shard_names)
    get_stats(args, shard_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GISAID data")
    parser.add_argument("-s", "--num-shards", default=mp.cpu_count(), type=int)
    parser.add_argument("-l", "--log-every", default=1000, type=int)
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    main(args)
