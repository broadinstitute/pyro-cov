import datetime
import math
import os
import pickle
import re
import logging
from collections import Counter, OrderedDict, defaultdict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import torch
import pyro.distributions as dist
from pyro.ops.tensor_utils import convolve
from pyrocov import mutrans, pangolin, stats
from pyrocov.stats import normal_log10bf
from pyrocov.util import (
    pretty_print, pearson_correlation, quotient_central_moments, generate_colors
)
from pyrocov.sarscov2 import GENE_TO_POSITION, GENE_STRUCTURE, aa_mutation_to_position

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.edgecolor"] = "gray"
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.pad_inches"] = 0.01
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Avenir', 'DejaVu Sans']
matplotlib.rcParams.update({
    # 'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}',
})

max_num_clades = 3000
min_num_mutations = 1

def load_data():
    filename = f"results/mutrans.data.single.{max_num_clades}.{min_num_mutations}.50.None.pt"
    dataset = torch.load(filename, map_location="cpu")
    dataset.update(mutrans.load_jhu_data(dataset))
    return dataset
dataset = load_data()
locals().update(dataset)

fits = torch.load("results/mutrans.pt", map_location="cpu")
fit = list(fits.values())[0]

rate = fit["mean"]["rate"].mean(0)
rate = quotient_central_moments(rate, clade_id_to_lineage_id)[1]
rate = rate - rate[lineage_id["A"]]
R = rate.exp()
rate_with_lineage = pd.DataFrame([[R[i].item(), name] for i, name in enumerate(lineage_id_inv)])
rate_with_lineage.columns = ["R/Ra", "Pango Lineage"]
rate_with_lineage.sort_values(by=["R/Ra"], ascending=False)

filename = input("Metadata subset: ")+'_ranked_ra.csv.gz'
rate_with_lineage.to_csv(filename, compression='gzip')
