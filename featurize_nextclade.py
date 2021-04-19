import argparse
import logging
import pickle
from collections import Counter

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def main(args):
    with open(args.stats_file_in, "rb") as f:
        lineage_mutation_counts = pickle.load(f)["mutations"]
    lineage_counts = Counter()
    for lineage, counts in lineage_mutation_counts.items():
        lineage_counts[lineage] = sum(counts.values())

    # Filter to features that occur in the majority of at least one lineage.
    total = len(set().union(*lineage_mutation_counts.values()))
    mutations = set()
    for lineage, mutation_counts in lineage_mutation_counts.items():
        denominator = lineage_counts[lineage]
        for m, count in mutation_counts.items():
            if count / denominator >= args.thresh:
                mutations.add(m)
    by_num = Counter(m.count(",") for m in mutations)
    logger.info(
        "Keeping only ({} single + {} double) = {} of {} mutations".format(
            by_num[0], by_num[1], len(mutations), total
        )
    )

    # Convert to dense features.
    lineages = sorted(lineage_counts)
    mutations = sorted(mutations, key=lambda m: (m.count(","), m))
    lineage_ids = {k: i for i, k in enumerate(lineages)}
    mutation_ids = {k: i for i, k in enumerate(mutations)}
    features = torch.zeros(len(lineage_ids), len(mutation_ids))
    for lineage, counts in lineage_mutation_counts.items():
        i = lineage_ids[lineage]
        denominator = lineage_counts[lineage]
        for mutation, count in counts.items():
            j = mutation_ids.get(mutation, None)
            if j is not None:
                features[i, j] = count / denominator

    result = {
        "lineages": lineages,
        "mutations": mutations,
        "features": features,
    }
    logger.info(f"saving {tuple(features.shape)}-features to {args.features_file_out}")
    torch.save(result, args.features_file_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize nextclade mutations")
    parser.add_argument("--thresh", default=0.5, type=float)
    parser.add_argument("--stats-file-in", default="results/gisaid.stats.pkl")
    parser.add_argument("--features-file-out", default="results/nextclade.features.pt")
    args = parser.parse_args()
    main(args)
