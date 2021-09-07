[![Build Status](https://github.com/broadinstitute/pyro-cov/workflows/CI/badge.svg)](https://github.com/broadinstitute/pyro-cov/actions)

# Pyro models for SARS-CoV-2 analysis

## Mutation analysis paper

1. Clone this repo into say `~/pyro-cov`
2. `cd ~/pyro-cov`
3. `make install`  # installs dependencies
4. `conda install nodejs`
5. `npm install --global @nextstrain/nextclade`
6. Work with GISAID to get a data agreement.
7. Define environment variables `GISAID_USERNAME`, `GISAID_PASSWORD`, and `GISAID_FEED`
8. `make update`  # clones other data sources

## Installing

```sh
make install
```
or literally
```sh
pip install -e .
```
