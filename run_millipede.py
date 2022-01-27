from os.path import exists
import argparse

import torch

import numpy as np
import pandas as pd
from millipede import NormalLikelihoodVariableSelector

import pickle

def main(args):
    if not exists('YX.npy'):
        f = torch.load('results/features.3000.1.pt')
        r = torch.load('results/mutrans.svi.3000.1.50.coef_scale=0.1.reparam.full.10001.0.05.0.1.10.0.200.6.None..pt')

        X = f['aa_features'].data.cpu().numpy()
        mutations = f['aa_mutations']
        Y = r['median']['rate'].data.cpu().numpy().mean(0)
        Y = Y - Y.mean()

        YX = np.concatenate([Y[:, None], X], axis=-1)
        np.save('YX.npy', YX, allow_pickle=True)
        pickle.dump(mutations, open('mutations.pkl', 'wb'))
    else:
        YX = np.load('YX.npy', allow_pickle=True)
        mutations = pickle.load(open('mutations.pkl', 'rb'))

    P = len(mutations)
    print("P", P)

    X = YX[:, 1:].sum(0)
    print("X", X[X <= 1].shape, X[X >= P-2].shape)

    #good = (X > 4) & (X < 4996)
    #good = (X > -1) & (X < 9999)
    #YX = np.concatenate([YX[:, 0:1], YX[:, 1:][:, good]], axis=-1)

    #columns = ['Response'] + np.array(mutations)[good].tolist()
    columns = ['Response'] + mutations
    dataframe = pd.DataFrame(YX, columns=columns)
    print(dataframe.head(5))

    selector = NormalLikelihoodVariableSelector(dataframe,
                                                'Response',
                                                S=10,
                                                c=100.0,
                                                prior="gprior",
                                                device='gpu'
                                               )


    selector.run(T=200 * 1000, T_burnin=5000, verbosity='bar', seed=args.seed, report_frequency=500)
    summary = selector.summary.sort_values(by=['PIP'], ascending=False).copy()
    summary['rank'] = np.arange(1, summary.values.shape[0] + 1)
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

    pickle.dump(summary, open('summary.seed_{}.pkl'.format(args.seed), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    main(args)
