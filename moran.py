import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from pyrocov.sarscov2 import aa_mutation_to_position


# compute moran statistic
def moran(values, distances, lengthscale):
    assert values.size(-1) == distances.size(-1)
    weights = (distances.unsqueeze(-1) - distances.unsqueeze(-2)) / lengthscale
    weights = weights.pow(2.0)
    weights = torch.exp(-weights)
    weights /= weights.sum(-1, keepdim=True)

    return torch.einsum("...ij,...i,...j->...", weights, values, values) / values.pow(2.0).sum(-1)


# compute moran statistic and do permutation test with given number of permutations
def permutation_test(values, distances, lengthscale, num_permutations=9999):
    moran_given = moran(values, distances, lengthscale).item()
    idx = torch.stack([torch.randperm(distances.size(-1)) for _ in range(num_permutations)])
    moran_perm = moran(values[idx], distances, lengthscale)
    p_value = (moran_perm >= moran_given).sum().item() + 1
    p_value /= float(num_permutations + 1)
    return moran_given, p_value


# read in inferred mutations
df = pd.read_csv('paper/mutations.tsv', sep='\t', index_col=0)[['mutation', 'Δ log R']]
mutations = df.values[:, 0]
assert mutations.shape == (2337,)
coefficients = df.values[:, 1]
gene_map = defaultdict(list)
distance_map = defaultdict(list)

# collect indices corresponding to each gene
for i, m in enumerate(mutations):
    gene = m.split(':')[0]
    gene_map[gene].append(i)
    distance_map[gene].append(aa_mutation_to_position(m))

# map over each gene
for gene, idx in gene_map.items():
    values = torch.from_numpy(np.array(coefficients[idx], dtype=np.float32))
    distances = distance_map[gene]
    distances = torch.from_numpy(np.array(distances) - min(distances))
    gene_size = distances.max().item()
    lengthscale = min(gene_size / 20, 50.0)
    _, p_value = permutation_test(values, distances, lengthscale, num_permutations=99999)
    print("Gene: {} \t p-value: {:.4f}  Lengthscale: {:.1f}".format(gene, p_value, lengthscale))

global_lengthscale = 500.0
distances = [aa_mutation_to_position(m) for m in mutations]
distances = torch.from_numpy(np.array(distances, dtype=np.float32) - min(distances))
values = torch.tensor(np.array(coefficients, dtype=np.float32)).float()
_, p_value = permutation_test(values, distances, global_lengthscale, num_permutations=99999)
print("Entire Genome: \t p-value: {:.4f}  Lengthscale: {:.1f}".format(p_value, global_lengthscale))
