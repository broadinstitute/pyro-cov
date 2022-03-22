import torch
import numpy as np
from collections import defaultdict


results_dir = 'results/'

features = torch.load(results_dir + 'features.3000.1.pt', map_location=torch.device('cpu'))
clades = features['clades']
clade_to_lineage = features['clade_to_lineage']
features = features['aa_features'].float()

lineage_to_clade = defaultdict(list)
for k, v in clade_to_lineage.items():
    lineage_to_clade[v].append(k)

lineage = 'AY.4'
#lineage = 'B.1.1.7'

for clade in lineage_to_clade[lineage]:
    print(lineage + ': ', clade)

alpha = np.array([0 if clade not in lineage_to_clade[lineage] else 1 for clade in clades], dtype=bool)
features_alpha = features[alpha].sum(0).clamp(max=1.0).data.numpy()
features_alpha = np.array(features_alpha, dtype=bool)
print("features_alpha", features_alpha.sum())


days = list(range(487 + 7 * 7, 487 + 9 * 7, 7))
print("days: ", days)
alpha_feature_counts = []
alpha_sequence_counts = []

for day in days:
    f = 'mutrans.data.single.3000.1.50.{}.pt'.format(day)
    data = torch.load(results_dir + f, map_location='cpu')
    weekly_clades = data['weekly_clades'].sum([0, 1])
    feature_counts = weekly_clades.data.numpy() @ features.data.numpy()
    alpha_feature_counts.append( feature_counts[features_alpha] )
    alpha_sequence_counts.append(weekly_clades[alpha].sum().item())


alpha_tuples = []

for day, fc_alpha, sc_alpha in zip(days, alpha_feature_counts, alpha_sequence_counts):
    f = 'mutrans.svi.3000.1.50.coef_scale=0.05.reparam-localinit.full.10001.0.05.0.1.10.0.200.0.{}..pt'.format(day)
    fit = torch.load(results_dir + f, map_location=torch.device('cpu'))

    rate_mean = fit['median']['rate'].data.cpu()[:, alpha].flatten()
    rate_std = fit['std']['rate'].data.cpu()[:, alpha].flatten()
    delta = rate_mean.max() - rate_mean.min()
    print('delta', delta.exp(), rate_mean.max().exp(),  rate_mean.min().exp())
    rates = torch.distributions.Normal(rate_mean, rate_std).sample(sample_shape=(100,)).exp().flatten()
    R_mean, R_std = rates.mean().item(), rates.std().item()
    t = (day, R_mean, R_mean - 1.96 * R_std, R_mean + 1.96 * R_std, sc_alpha, fc_alpha.tolist())
    print("delta", t[:5])
    alpha_tuples.append(t)

    del fit

print()
print(lineage)
print(alpha_tuples)
