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

for clade in lineage_to_clade['BA.1']:
    print("BA.1:", clade)

for clade in lineage_to_clade['BA.2']:
    print("BA.2:", clade)

ba1 = np.array([0 if clade not in lineage_to_clade['BA.1'] else 1 for clade in clades], dtype=bool)
features_ba1 = features[ba1].sum(0).clamp(max=1.0).data.numpy()
features_ba1 = np.array(features_ba1, dtype=bool)

ba2 = np.array([0 if clade not in lineage_to_clade['BA.2'] else 1 for clade in clades], dtype=bool)
features_ba2 = features[ba2].sum(0).clamp(max=1.0).data.numpy()
features_ba2 = np.array(features_ba2, dtype=bool)


days = list(range(710, 766 + 7, 7))
ba1_feature_counts = []
ba2_feature_counts = []
ba1_sequence_counts = []
ba2_sequence_counts = []

for day in days:
    f = 'mutrans.data.single.3000.1.50.{}.pt'.format(day)
    data = torch.load(results_dir + f, map_location='cpu')
    weekly_clades = data['weekly_clades'].sum([0, 1])
    feature_counts = weekly_clades.data.numpy() @ features.data.numpy()
    ba1_feature_counts.append( feature_counts[features_ba1] )
    ba2_feature_counts.append( feature_counts[features_ba2] )
    ba1_sequence_counts.append(weekly_clades[ba1].sum().item())
    ba2_sequence_counts.append(weekly_clades[ba2].sum().item())


ba1_tuples = []
ba2_tuples = []

for day, fc_ba1, fc_ba2, sc_ba1, sc_ba2, in zip(days, ba1_feature_counts, ba2_feature_counts,
                                                ba1_sequence_counts, ba2_sequence_counts):
    f = 'mutrans.svi.3000.1.50.coef_scale=0.05.reparam-localinit.full.10001.0.05.0.1.10.0.200.12.{}..pt'.format(day)
    fit = torch.load(results_dir + f, map_location=torch.device('cpu'))

    rate_mean = fit['median']['rate'].data.cpu()[:, ba1].flatten()
    rate_std = fit['std']['rate'].data.cpu()[:, ba1].flatten()
    rates = torch.distributions.Normal(rate_mean, rate_std).sample(sample_shape=(100,)).exp().flatten()
    R_mean, R_std = rates.mean().item(), rates.std().item()
    t = (day, R_mean, R_mean - 1.96 * R_std, R_mean + 1.96 * R_std, sc_ba1, fc_ba1.tolist())
    print("ba1", t[:5])
    ba1_tuples.append(t)

    rate_mean = fit['median']['rate'].data.cpu()[:, ba2].flatten()
    rate_std = fit['std']['rate'].data.cpu()[:, ba2].flatten()
    rates = torch.distributions.Normal(rate_mean, rate_std).sample(sample_shape=(100,)).exp().flatten()
    R_mean, R_std = rates.mean().item(), rates.std().item()
    t = (day, R_mean, R_mean - 1.96 * R_std, R_mean + 1.96 * R_std, sc_ba2, fc_ba2.tolist())
    print("ba2", t[:5])
    ba2_tuples.append(t)

    del fit

print()
print("ba1")
print(ba1_tuples)
print()
print("ba2")
print(ba2_tuples)
