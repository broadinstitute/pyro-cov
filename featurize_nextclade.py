import argparse
import glob
import logging
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
                if isinstance(ms, str):
                    counts.update(ms.split(","))

    # Convert to dense features.
    lineages = sorted(lineage_counts)
    mutations = sorted(set().union(*lineage_mutation_counts.values()))
    lineage_ids = {k: i for i, k in enumerate(sorted(lineages))}
    mutation_ids = {k: i for i, k in enumerate(mutations)}
    features = torch.zeros(len(lineage_ids), len(mutation_ids))
    for lineage, counts in lineage_mutation_counts.items():
        i = lineage_ids[lineage]
        denominator = lineage_counts[lineage]
        for mutation, count in counts.items():
            j = mutation_ids[mutation]
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
    parser.add_argument("--tsv-file-in", default="results/gisaid.subset.*.tsv")
    parser.add_argument("--features-file-out", default="results/nextclade.features.pt")
    args = parser.parse_args()
    main(args)
