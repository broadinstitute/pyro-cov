# Pyro models for SARS-CoV-2 analysis

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
