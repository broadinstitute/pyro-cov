import torch
import numpy as np
import pickle


results_dir = '/home/mjankowi/pyro-cov/results/'
fit = torch.load(results_dir + 'mutrans.svi.3000.1.50.coef_scale=0.1.reparam.full.10001.0.05.0.1.10.0.200.6.None..pt')

rate = fit['median']['rate'].data.cpu()
rate_loc = fit['median']['rate_loc'].data.cpu()

coef = fit['median']['coef'].data.cpu()

features = torch.load(results_dir + 'features.3000.1.pt')

clades = features['clades']
clade_to_lineage = features['clade_to_lineage']
mutations = features['aa_mutations']
features = features['aa_features'].data.cpu().float()

full_pred = torch.mv(features, coef)
print("full_pred", full_pred.shape)

rbd = []
S = []
for m in mutations:
    if m[:2] != 'S:':
        rbd.append(0)
        S.append(0)
        continue

    S.append(1)

    pos = int(m[3:-1])

    if pos >= 331 and pos <= 531:
        rbd.append(1)
    else:
        rbd.append(0)

S = torch.tensor(S).bool()
rbd = torch.tensor(rbd).bool()

S_mutations = np.array(mutations)[S].tolist()
rbd_mutations = np.array(mutations)[rbd].tolist()

S_features = features[:, S]
rbd_features = features[:, rbd]
print("S_features", S_features.shape)
print("rbd_features", rbd_features.shape)

S_coef = coef[S]
rbd_coef = coef[rbd]
print("S_coef", S_coef.shape)
print("rbd_coef", rbd_coef.shape)

S_pred = torch.mv(S_features, S_coef)
rbd_pred = torch.mv(rbd_features, rbd_coef)
print("S_pred", S_pred.shape)
print("rbd_pred", rbd_pred.shape)

pickle.dump((full_pred.numpy(),
             rbd_mutations,
             rbd_pred.numpy(),
             S_mutations,
             S_pred.numpy(),
             rbd_features.numpy(),
             clades,
             clade_to_lineage,
             rate.numpy(),
             rate_loc.numpy()), open('rbd_data.pkl', 'wb'))
