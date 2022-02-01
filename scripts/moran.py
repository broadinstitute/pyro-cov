# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from pyrocov.sarscov2 import aa_mutation_to_position


# compute moran statistic
def moran(values, distances, lengthscale):
    assert values.size(-1) == distances.size(-1)
    weights = (distances.unsqueeze(-1) - distances.unsqueeze(-2)) / lengthscale
    weights = torch.exp(-weights.pow(2.0))
    weights *= 1.0 - torch.eye(weights.size(-1))
    weights /= weights.sum(-1, keepdim=True)

    output = torch.einsum("...ij,...i,...j->...", weights, values, values)
    return output / values.pow(2.0).sum(-1)


# compute moran statistic and do permutation test with given number of permutations
def permutation_test(values, distances, lengthscale, num_perm=999):
    values = values - values.mean()
    moran_given = moran(values, distances, lengthscale).item()
    idx = [torch.randperm(distances.size(-1)) for _ in range(num_perm)]
    idx = torch.stack(idx)
    moran_perm = moran(values[idx], distances, lengthscale)
    p_value = (moran_perm >= moran_given).sum().item() + 1
    p_value /= float(num_perm + 1)
    return moran_given, p_value


def main(args):
    # read in inferred mutations
    df = pd.read_csv("paper/mutations.tsv", sep="\t", index_col=0)
    df = df[["mutation", "Δ log R"]]
    mutations = df.values[:, 0]
    assert mutations.shape == (2904,)
    coefficients = df.values[:, 1] if not args.magnitude else np.abs(df.values[:, 1])
    gene_map = defaultdict(list)
    distance_map = defaultdict(list)

    results = []

    # collect indices and nucleotide positions corresponding to each mutation
    for i, m in enumerate(mutations):
        gene = m.split(":")[0]
        gene_map[gene].append(i)
        distance_map[gene].append(aa_mutation_to_position(m))

    # map over each gene
    for gene, idx in gene_map.items():
        values = torch.from_numpy(np.array(coefficients[idx], dtype=np.float32))
        distances = distance_map[gene]
        distances = torch.from_numpy(np.array(distances) - min(distances))
        gene_size = distances.max().item()
        lengthscale = min(gene_size / 20, 50.0)
        _, p_value = permutation_test(values, distances, lengthscale, num_perm=999999)
        s = "Gene: {} \t #Mut: {} Size: {} \t p-value: {:.6f}  Lengthscale: {:.1f}"
        print(s.format(gene, distances.size(0), gene_size, p_value, lengthscale))
        results.append([distances.size(0), gene_size, p_value, lengthscale])

    # compute moran statistic for entire genome for mulitple lengthscales
    for global_lengthscale in [100.0, 500.0]:
        distances_ = [aa_mutation_to_position(m) for m in mutations]
        distances = torch.from_numpy(
            np.array(distances_, dtype=np.float32) - min(distances_)
        )
        values = torch.tensor(np.array(coefficients, dtype=np.float32)).float()
        _, p_value = permutation_test(
            values, distances, global_lengthscale, num_perm=999999
        )
        genome_size = distances.max().item()
        s = "Entire Genome (#Mut = {}; Size = {}): \t p-value: {:.6f}  Lengthscale: {:.1f}"
        print(s.format(distances.size(0), genome_size, p_value, global_lengthscale))
        results.append([distances.size(0), genome_size, p_value, global_lengthscale])

    # save results as csv
    results = np.stack(results)
    columns = ["NumMutations", "GeneSize", "PValue", "Lengthscale"]
    index = list(gene_map.keys()) + ["EntireGenome"] * 2
    result = pd.DataFrame(data=results, index=index, columns=columns)
    result.sort_values(["PValue"]).to_csv("paper/moran.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute moran statistics")
    parser.add_argument("--magnitude", action="store_true")
    args = parser.parse_args()
    main(args)
