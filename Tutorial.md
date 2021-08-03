# Pyro models for SARS-CoV-2 analysis -- Reproducing analysis


# Installation

## Clone this repository
Clone this repository to ~/pyro-cov

## Install nextclade

Depending on your platform do:

```sh
make install-nextalign-linux
make install-nextclade-linux
```
or 
```sh
make install-nextalign
make install-nextclade
```
## Install the package
```sh
pip install -e .[test]
```

## Getting GISAID data
1. Work with GISAID to get a data agreement.
2. Create a directory ~/data/gisaid/
3. Update pull_gisaid.sh file with your credentials and feed

## Install dependencies
1. conda install nodejs
2. npm install --global @nextstrain/nextclade
3. make update 

Run vary holdout experiements
```sh
python mutrans.py --vary-holdout 
```

Run backtesting experiments
```sh
./run_backtesting.py
``