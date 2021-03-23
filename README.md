# Pyro models for SARS-CoV-2 analysis

## Mutation analysis paper

1. Work with GISAID to get a data agreement.
2. Create a directory ~/data/gisaid/
3. Create a data pull script ~/data/gisaid/pull
4. Clone this repo into say ~/pyro-cov
5. cd ~/pyro-cov
6. make install  # installs dependencies
7. conda install nodejs
8. npm install --global @nextstrain/nextclade
9. make update  # clones other data sources

## Data

@dpark01 created a [Google Drive folder]() for data sharing.
This repo assumes that folder is available locally and linked as the `data/` directory, for example if you have installed [Google Drive File Stream](https://www.google.com/drive/download/) (for teams, not the individual version), you should be able to:
```sh
ln -sf ~/Google\ Drive\ File\ Stream/Shared\ drives/Pyro\ CoV data
```

### Treebase datasets

The `treebase/` directory contains snapshots from https://treebase.org/treebase-web

## Installing

```sh
make install
```
or literally
```sh
pip install -e .
```
