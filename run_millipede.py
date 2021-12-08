from os.path import exists

import torch
from pyrocov.mutrans import load_gisaid_data

import numpy as np
import pandas as pd
from millipede import NormalLikelihoodVariableSelector

import pickle


if not exists('YX.npy'):
    d = load_gisaid_data(columns_filename='results/columns.5000.pkl', features_filename='results/features.5000.pt')
    r = torch.load('results/mutrans.svi.5000.1.1..coef_scale=0.05.reparam.full.10001.0.05.0.1.10.0.200.6.None..pt')

    X = d['features'].data.cpu().numpy()
    Y = r['median']['rate_loc'].data.cpu().numpy()
    YX = np.concatenate([Y[:, None], X], axis=-1)
    mutations = d['mutations']
    np.save('YX.npy', YX, allow_pickle=True)
    pickle.dump(mutations, open('mutations.pkl', 'wb'))
else:
    YX = np.load('YX.npy', allow_pickle=True)
    mutations = pickle.load(open('mutations.pkl', 'rb'))

X = YX[:, 1:].sum(0)
good = (X > 4) & (X < 4996)
good = (X > -1) & (X < 9999)
YX = np.concatenate([YX[:, 0:1], YX[:, 1:][:, good]], axis=-1)

columns = ['Response'] + np.array(mutations)[good].tolist()
dataframe = pd.DataFrame(YX, columns=columns)
print(dataframe.head(5))

selector = NormalLikelihoodVariableSelector(dataframe,
                                            'Response',
                                            S=10,
                                            c=100.0,
                                            prior="gprior",
                                            device='gpu'
                                           )


selector.run(T=20000, T_burnin=1000, verbosity='bar', seed=1, report_frequency=200)
summary = selector.summary.sort_values(by=['PIP'], ascending=False).copy()
print("RANK #01 - #10")
print(summary.iloc[0:10])
print("RANK #11 - #20")
print(summary.iloc[10:20])
print("RANK #21 - #30")
print(summary.iloc[20:30])
print("RANK #31 - #40")
print(summary.iloc[30:40])
print("RANK #41 - #50")
print(summary.iloc[40:50])
