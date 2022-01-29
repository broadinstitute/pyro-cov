import torch
import numpy as np
import pickle
from collections import defaultdict
from pyrocov import mutrans
import datetime


results_dir = '../results/'

fit = torch.load(results_dir + 'mutrans.svi.3000.1.50.coef_scale=0.1.reparam.full.10001.0.05.0.1.10.0.200.6.None..pt')
coef = fit['median']['coef'].data.cpu().numpy()

features = torch.load(results_dir + 'features.3000.1.pt')

clades = features['clades']

mutations = features['aa_mutations']
features = features['aa_features'].data.cpu().float().numpy()

clade_bday = pickle.load(open('clade_bdays.pkl', 'rb'))

mutation_bday = defaultdict(list)


for k, clade in enumerate(clades):
    if clade not in clade_bday:
        continue
    nz = np.nonzero(features[k])[0]
    for f in nz:
        mutation_bday[mutations[f]].append(clade_bday[clade])

counts = {k: len(v) for k, v in mutation_bday.items()}
mutation_bday = {k: min(v) for k, v in mutation_bday.items()}
print("maxcount", max(counts.values()))
print("mincount", min(counts.values()))

print(len(mutation_bday))
print(len(mutations))
print(len(coef))

pickle.dump((mutation_bday, mutations, coef),
            open('mutation_bday.pkl', 'wb'), protocol=2)
