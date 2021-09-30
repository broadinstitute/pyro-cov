SHELL := /bin/bash

data:
	ln -sf ~/Google\ Drive\ File\ Stream/Shared\ drives/Pyro\ CoV data

install: install-nextalign FORCE
	pip install -e .[test]

install-nextalign:
	curl -fsSL "https://github.com/nextstrain/nextclade/releases/latest/download/nextalign-MacOS-x86_64" -o "nextalign" && chmod +x nextalign

install-nextclade:
	curl -fsSL "https://github.com/nextstrain/nextclade/releases/latest/download/nextclade-MacOS-x86_64" -o "nextclade" && chmod +x nextclade
    
install-nextalign-linux:
	curl -fsSL "https://github.com/nextstrain/nextclade/releases/download/1.2.0/nextalign-Linux-x86_64" -o nextalign && chmod +x nextalign

install-nextclade-linux:
	curl -fsSL "https://github.com/nextstrain/nextclade/releases/download/1.2.0/nextclade-Linux-x86_64" -o nextclade && chmod +x nextclade

lint: FORCE
	flake8
	black --check .
	isort --check .
	python scripts/update_headers.py --check
	mypy .

format: FORCE
	black .
	isort .
	python scripts/update_headers.py

test: lint data FORCE
	pytest -v -n auto test
	python mutrans.py --test -n 2 -s 4

update: FORCE
	./pull_gisaid.sh
	python git_pull.py cov-lineages/pango-designation
	python git_pull.py CSSEGISandData/COVID-19
	python git_pull.py nextstrain/nextclade
	time nice python preprocess_gisaid.py
	time python featurize_nextclade.py

ssh:
	gcloud compute ssh --project pyro-284215 --zone us-central1-c \
	  pyro-cov-fritzo-vm -- -AX

push:
	gcloud compute scp --project pyro-284215 --zone us-central1-c \
	  --recurse --compress \
	  results/gisaid.columns.pkl  \
	  pyro-cov-fritzo-vm:~/pyro-cov/results/
	gcloud compute scp --project pyro-284215 --zone us-central1-c \
	  --recurse --compress \
	  results/nextclade.features.pt \
	  pyro-cov-fritzo-vm:~/pyro-cov/results/

pull:
	gcloud compute scp --project pyro-284215 --zone us-central1-c \
	  --recurse --compress \
	  pyro-cov-fritzo-vm:~/pyro-cov/results/mutrans.pt \
	  results/

# This data is needed for mutrans.ipynb
pull-data:
	gcloud compute scp --project pyro-284215 --zone us-central1-c \
	  --recurse --compress \
	  pyro-cov-fritzo-vm:~/pyro-cov/results/\{gisaid.columns.pkl,gisaid.stats.pkl,nextclade.features.pt,nextclade.counts.pkl\} \
	  results/

pull-grid:
	gcloud compute scp --project pyro-284215 --zone us-central1-c \
          --recurse --compress \
	  pyro-cov-fritzo-vm:~/pyro-cov/results/grid_search.tsv \
	  results/
	gcloud compute scp --project pyro-284215 --zone us-central1-c \
	  --recurse --compress \
	  pyro-cov-fritzo-vm:~/pyro-cov/results/mutrans.grid.pt \
	  results/

pull-leaves:
	gcloud compute scp --project pyro-284215 --zone us-central1-c \
	  --recurse --compress \
	  pyro-cov-fritzo-vm:~/pyro-cov/results/mutrans.vary_leaves.pt \
	  results/

FORCE:
