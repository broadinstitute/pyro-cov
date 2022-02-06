import torch
import numpy as np
from collections import defaultdict


results_dir = 'results/'

features = torch.load(results_dir + 'features.3000.1.pt', map_location=torch.device('cpu'))
clades = features['clades']
clade_to_lineage = features['clade_to_lineage']
features = features['aa_features'].data.cpu().float()

lineage_to_clade = defaultdict(list)
for k, v in clade_to_lineage.items():
    lineage_to_clade[v].append(k)

ba2 = np.array([0 if clade not in lineage_to_clade['BA.2'] else 1 for clade in clades], dtype=bool)
features_ba2 = np.array(features.data.numpy()[ba2][0], dtype=bool)

days = list(range(710, 766 + 7, 7))
ba2_feature_counts = []

for day in days:
    f = 'mutrans.data.single.3000.1.50.{}.pt'.format(day)
    data = torch.load(results_dir + f, map_location='cpu')
    weekly_clades = data['weekly_clades']
    feature_counts = weekly_clades.sum([0, 1]).data.numpy() @ features.data.numpy()
    ba2_feature_counts.append( feature_counts[features_ba2] )

tuples = []

for day, fc in zip(days, ba2_feature_counts):
    f = 'mutrans.svi.3000.1.50.coef_scale=0.05.reparam-localinit.full.10001.0.05.0.1.10.0.200.12.{}..pt'.format(day)
    fit = torch.load(results_dir + f, map_location=torch.device('cpu'))
    rate_mean = fit['median']['rate'].data.cpu()[:, ba2].flatten()
    rate_std = fit['std']['rate'].data.cpu()[:, ba2].flatten()
    rates = torch.distributions.Normal(rate_mean, rate_std).sample(sample_shape=(100,)).exp().flatten()
    R_mean, R_std = rates.mean().item(), rates.std().item()
    t = (day, R_mean, R_mean - 1.96 * R_std, R_mean + 1.96 * R_std, fc.tolist())
    print(t)
    tuples.append(t)
    del fit

print()
print(tuples)
