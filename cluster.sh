#!/bin/sh -ex

python preprocess_gisaid.py --cluster-radius=100 --cluster-epochs=10
python preprocess_gisaid.py --cluster-radius=100 --cluster-epochs=20
python preprocess_gisaid.py --cluster-radius=100 --cluster-epochs=5
python preprocess_gisaid.py --cluster-radius=20 --cluster-epochs=10
python preprocess_gisaid.py --cluster-radius=20 --cluster-epochs=20
python preprocess_gisaid.py --cluster-radius=20 --cluster-epochs=5
python preprocess_gisaid.py --cluster-radius=200 --cluster-epochs=10
python preprocess_gisaid.py --cluster-radius=200 --cluster-epochs=20
python preprocess_gisaid.py --cluster-radius=200 --cluster-epochs=5
python preprocess_gisaid.py --cluster-radius=50 --cluster-epochs=10
python preprocess_gisaid.py --cluster-radius=50 --cluster-epochs=20
python preprocess_gisaid.py --cluster-radius=50 --cluster-epochs=5
python preprocess_gisaid.py --cluster-radius=500 --cluster-epochs=10
python preprocess_gisaid.py --cluster-radius=500 --cluster-epochs=20
python preprocess_gisaid.py --cluster-radius=500 --cluster-epochs=5
