import torch
import numpy as np
import pickle
from collections import defaultdict
from pyrocov import mutrans
import datetime


results_dir = '../results/'
columns = pickle.load(open(results_dir + 'columns.3000.pkl', 'rb'))
assert len(set(columns['clade'])) == 2999


def estimate_clade_bdays(exclude_first=10, min_portion=0.0005, max_portion=0.1):
    clade_days = defaultdict(list)
    for clade, day in zip(columns["clade"], columns["day"]):
        clade_days[clade].append(day)
    clade_bday = {}
    for clade, days in list(clade_days.items()):
        days.sort()
        exclude = max(exclude_first, int(min_portion * len(days)))
        exclude = min(exclude, int(max_portion * len(days)))
        clade_bday[clade] = days[exclude]
    start_date = datetime.datetime.strptime(mutrans.START_DATE, "%Y-%m-%d")
    return {
        clade: (start_date + datetime.timedelta(days=day))
        for clade, day in clade_bday.items()
    }

clade_bday = estimate_clade_bdays()
print(len(clade_bday))

pickle.dump(clade_bday, open('clade_bdays.pkl', 'wb'))
