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
	curl -fsSL "https://github.com/nextstrain/nextclade/releases/1.6.0/download/nextalign-MacOS-x86_64" -o "nextalign" && chmod +x nextalign

install-nextclade:
	curl -fsSL "https://github.com/nextstrain/nextclade/releases/1.6.0/download/nextclade-MacOS-x86_64" -o "nextclade" && chmod +x nextclade
    
install-nextalign-linux:
	curl -fsSL "https://github.com/nextstrain/nextclade/releases/download/1.6.0/nextalign-Linux-x86_64" -o nextalign && chmod +x nextalign

install-nextclade-linux:
	curl -fsSL "https://github.com/nextstrain/nextclade/releases/download/1.6.0/nextclade-Linux-x86_64" -o nextclade && chmod +x nextclade

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

test: lint FORCE
	python scripts/git_pull.py --no-update cov-lineages/pango-designation
	python scripts/git_pull.py --no-update cov-lineages/pangoLEARN
	pytest -v -n auto test
	test -e results/aligndb && python scripts/mutrans.py --test -n 2 -s 2

###########################################################################
# Main processing workflows
# Note *-nextstrain is a cheaper simpler workflow replacing *-gisaid.
# Note *-usher joins usher trees with nextstrain data.

update-gisaid: FORCE
	scripts/pull_gisaid.sh
	python scripts/git_pull.py cov-lineages/pango-designation
	python scripts/git_pull.py cov-lineages/pangoLEARN
	python scripts/git_pull.py CSSEGISandData/COVID-19
	./nextclade dataset get --name sars-cov-2 --output-dir results/nextclade_data

preprocess-gisaid: FORCE
	time nice python scripts/preprocess_gisaid.py
	time nice python scripts/preprocess_nextclade.py
	time nice python scripts/preprocess_pangolin.py --max-num-clades=2000
	time nice python scripts/preprocess_pangolin.py --max-num-clades=5000
	time nice python scripts/preprocess_pangolin.py --max-num-clades=10000

analyze-gisaid: FORCE
	python scripts/mutrans.py --gisaid --vary-holdout
	python scripts/mutrans.py --gisaid --vary-gene
	python scripts/mutrans.py --gisaid --vary-nsp
	python scripts/mutrans.py --gisaid --vary-leaves=9999 --num-steps=2001

update-nextstrain: FORCE
	scripts/pull_nextstrain.sh
	python scripts/git_pull.py cov-lineages/pango-designation
	python scripts/git_pull.py cov-lineages/pangoLEARN
	python scripts/git_pull.py CSSEGISandData/COVID-19

preprocess-nextstrain: FORCE
	python scripts/preprocess_nextstrain.py

analyze-nextstrain: FORCE
	python scripts/analyze_nextstrain.py --vary-holdout
	python scripts/analyze_nextstrain.py --vary-gene
	python scripts/analyze_nextstrain.py --vary-nsp
	python scripts/analyze_nextstrain.py --vary-leaves=9999 --num-steps=2001

update-usher: FORCE
	scripts/pull_nextstrain.sh
	scripts/pull_usher.sh
	python scripts/git_pull.py cov-lineages/pango-designation
	python scripts/git_pull.py cov-lineages/pangoLEARN
	python scripts/git_pull.py CSSEGISandData/COVID-19

preprocess-usher: FORCE
	python scripts/preprocess_usher.py --max-num-clades=2000
	python scripts/preprocess_usher.py --max-num-clades=5000
	python scripts/preprocess_usher.py --max-num-clades=10000

analyze-usher: FORCE
	python scripts/mutrans.py --vary-holdout
	python scripts/mutrans.py --vary-gene
	python scripts/mutrans.py --vary-nsp
	python scripts/mutrans.py --vary-leaves=9999 --num-steps=2001

backtesting: FORCE
	python scripts/mutrans.py --backtesting-max-day `seq -s, 150 14 625` --forecast-steps 12

EXCLUDE='.*\.json$$|.*mutrans\.pt$$|.*temp\..*|.*\.[EI](gene|region)=.*\.pt$$|.*__(gene|region|lineage)__.*\.pt$$'

push: FORCE
	gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M \
	  rsync -r -x $(EXCLUDE) \
	  $(shell readlink results)/ \
	  gs://pyro-cov/$(shell readlink results | grep -o 'results\.[-0-9]*')/

pull: FORCE
	gsutil -m rsync -r -x $(EXCLUDE) \
	  gs://pyro-cov/$(shell readlink results | grep -o 'results\.[-0-9]*')/ \
	  $(shell readlink results)/

###########################################################################
# TODO remove these user-specific targets

data:
	ln -sf ~/Google\ Drive\ File\ Stream/Shared\ drives/Pyro\ CoV data

ssh-cpu:
	gcloud compute ssh --project pyro-284215 --zone us-central1-c pyro-cov-fritzo-vm -- -AX -t 'cd pyro-cov ; bash --login'

ssh-gpu:
	gcloud compute ssh --project pyro-284215 --zone us-central1-c pyro-fritzo-vm-gpu -- -AX -t 'cd pyro-cov ; bash --login'

FORCE:

