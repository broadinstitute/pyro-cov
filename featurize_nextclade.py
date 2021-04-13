import argparse
import glob
import logging
import re
from collections import Counter, defaultdict

import pandas as pd
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def main(args):
    lineage_counts = Counter()
    lineage_mutation_counts = defaultdict(Counter)
    for filename in glob.glob(args.tsv_file_in):
        df = pd.read_csv(filename, sep="\t")
        logger.info(f"Extracting features from {len(df)} sequences")
        for _, row in df.iterrows():
            lineage = row["seqName"].strip().split()[0]
            lineage_counts[lineage] += 1
            counts = lineage_mutation_counts[lineage]
            for col in ["aaSubstitutions", "aaDeletions"]:
                ms = row[col]
                if not isinstance(ms, str):
                    continue
                ms = ms.split(",")
                counts.update(ms)
                # Add within-gene pairs of mutations.
                by_gene = defaultdict(list)
                for m in ms:
                    g, m = m.split(":")
                    by_gene[g].append(m)
                for g, ms in by_gene.items():
                    # Sort by position, then alphabetical.
                    ms.sort(key=lambda m: (int(re.search(r"\d+", m).group(0)), m))
                    for i, m1 in enumerate(ms):
                        for m2 in ms[i + 1 :]:
                            counts[f"{g}:{m1},{m2}"] += 1

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
    parser.add_argument("--tsv-file-in", default="results/gisaid.subset.*.tsv")
    parser.add_argument("--features-file-out", default="results/nextclade.features.pt")
    args = parser.parse_args()
    main(args)
