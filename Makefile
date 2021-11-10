SHELL := /bin/bash

###########################################################################
# installation

install: install-nextalign install-usher FORCE
	pip install -e .[test]

install-pangolin:
	conda install -y -c bioconda -c conda-forge -c defaults pangolin

install-usher:
	conda install -y -c bioconda -c conda-forge -c defaults usher

install-nextalign:
	curl -fsSL "https://github.com/nextstrain/nextclade/releases/latest/download/nextalign-MacOS-x86_64" -o "nextalign" && chmod +x nextalign

install-nextclade:
	curl -fsSL "https://github.com/nextstrain/nextclade/releases/latest/download/nextclade-MacOS-x86_64" -o "nextclade" && chmod +x nextclade
    
install-nextalign-linux:
	curl -fsSL "https://github.com/nextstrain/nextclade/releases/download/1.2.0/nextalign-Linux-x86_64" -o nextalign && chmod +x nextalign

install-nextclade-linux:
	curl -fsSL "https://github.com/nextstrain/nextclade/releases/download/1.2.0/nextclade-Linux-x86_64" -o nextclade && chmod +x nextclade

###########################################################################
# ci tasks

lint: FORCE
	flake8 --extend-exclude=pyrocov/external
	black --extend-exclude=\.ipynb --extend-exclude=pyrocov/external --check .
	isort --check --skip=pyrocov/external .
	python scripts/update_headers.py --check
	mypy .

format: FORCE
	black --extend-exclude=\.ipynb --extend-exclude=pyrocov/external .
	isort --skip=pyrocov/external .
	python scripts/update_headers.py

test: lint data FORCE
	pytest -v -n auto test
	python mutrans.py --test -n 2 -s 4

###########################################################################
# Main processing workflow
# TODO convert this to a wdl pipeline

update: FORCE
	./pull_gisaid.sh
	python git_pull.py cov-lineages/pango-designation
	python git_pull.py cov-lineages/pangoLEARN
	python git_pull.py CSSEGISandData/COVID-19
	python git_pull.py nextstrain/nextclade

preprocess: FORCE
	time nice python preprocess_gisaid.py
	time nice python preprocess_nextclade.py
	time nice python preprocess_pangolin.py

analyze: FORCE
	python mutrans.py --vary-holdout
	python mutrans.py --vary-gene
	python mutrans.py --vary-nsp
	python mutrans.py --vary-leaves=9999

###########################################################################
# TODO remove these user-specific targets

data:
	ln -sf ~/Google\ Drive\ File\ Stream/Shared\ drives/Pyro\ CoV data

ssh:
	gcloud compute ssh --project pyro-284215 --zone us-central1-c \
	  pyro-fritzo-vm-gpu -- -AX

pull:
	gcloud compute scp --project pyro-284215 --zone us-central1-c \
	  --recurse --compress \
	  pyro-fritzo-vm-gpu:~/pyro-cov/results/mutrans.pt \
	  results/

# This data is needed for mutrans.ipynb
pull-data:
	gcloud compute scp --project pyro-284215 --zone us-central1-c \
	  --recurse --compress \
	  pyro-fritzo-vm-gpu:~/pyro-cov/results/\{usher.columns.pkl,gisaid.stats.pkl,usher.features.pt,nextclade.counts.pkl\} \
	  results/

pull-grid:
	gcloud compute scp --project pyro-284215 --zone us-central1-c \
          --recurse --compress \
	  pyro-fritzo-vm-gpu:~/pyro-cov/results/grid_search.tsv \
	  results/
	gcloud compute scp --project pyro-284215 --zone us-central1-c \
	  --recurse --compress \
	  pyro-fritzo-vm-gpu:~/pyro-cov/results/mutrans.grid.pt \
	  results/

pull-gene:
	gcloud compute scp --project pyro-284215 --zone us-central1-c \
	  --recurse --compress \
	  pyro-fritzo-vm-gpu:~/pyro-cov/results/{mutrans.vary_gene.pt,mutrans.vary_nsp.pt} \
	  results/

pull-leaves:
	gcloud compute scp --project pyro-284215 --zone us-central1-c \
	  --recurse --compress \
	  pyro-fritzo-vm-gpu:~/pyro-cov/results/mutrans.vary_leaves.pt \
	  results/

FORCE:
