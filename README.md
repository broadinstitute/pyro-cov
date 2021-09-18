[![Build Status](https://github.com/broadinstitute/pyro-cov/workflows/CI/badge.svg)](https://github.com/broadinstitute/pyro-cov/actions)

# Pyro models for SARS-CoV-2 analysis

This repository is described in the paper ["Analysis of 2.1 million SARS-CoV-2 genomes identifies mutations associated with transmissibility"](https://www.medrxiv.org/content/10.1101/2021.09.07.21263228v1). Figures and supplementary data for that paper are in the [paper/](paper/) directory.

## Reproducing

1. Clone this repo into say `~/pyro-cov`
2. `cd ~/pyro-cov`
3. `make install`  # installs dependencies
4. `conda install nodejs`
5. `npm install --global @nextstrain/nextclade`
6. Work with GISAID to get a data agreement.
7. Define environment variables `GISAID_USERNAME`, `GISAID_PASSWORD`, and `GISAID_FEED`
8. `make update`  # clones other data sources
9. `python mutrans.py --vary-holdout`
10. generate plots by running various jupyter notebooks, e.g. [mutrans.ipynb](mutrans.ipynb)

## Installing

```sh
make install
```
or literally
```sh
pip install -e .
```

## Citing

If you use this software, please consider citing:

```
@article {Obermeyer2021.09.07.21263228,
  author = {Obermeyer, Fritz and
            Schaffner, Stephen F. and
            Jankowiak, Martin and
            Barkas, Nikolaos and
            Pyle, Jesse D. and
            Park, Daniel J. and
            MacInnis, Bronwyn L. and
            Luban, Jeremy and
            Sabeti, Pardis C. and
            Lemieux, Jacob E.},
  title = {Analysis of 2.1 million SARS-CoV-2 genomes identifies mutations associated with transmissibility},
  elocation-id = {2021.09.07.21263228},
  year = {2021},
  doi = {10.1101/2021.09.07.21263228},
  publisher = {Cold Spring Harbor Laboratory Press},
  URL = {https://www.medrxiv.org/content/early/2021/09/13/2021.09.07.21263228},
  eprint = {https://www.medrxiv.org/content/early/2021/09/13/2021.09.07.21263228.full.pdf},
  journal = {medRxiv}
}
```
