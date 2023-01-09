SHELL := /bin/bash

###########################################################################
# installation

install: FORCE
	pip install -e .[test]

install-pangolin:
	conda install -y -c bioconda -c conda-forge -c defaults pangolin
	
install-usher:
	conda install -y -c bioconda -c conda-forge -c defaults usher

###########################################################################
# ci tasks

lint: FORCE
	flake8 --extend-exclude=pyrocov/external
	black --extend-exclude='notebooks|pyrocov/external' --check .
	isort --check --skip=pyrocov/external .
	python scripts/update_headers.py --check
	mypy .

format: FORCE
	black --extend-exclude='notebooks|pyrocov/external' .
	isort --skip=pyrocov/external .
	python scripts/update_headers.py

test: lint FORCE
	python scripts/git_pull.py --no-update cov-lineages/pango-designation
	python scripts/git_pull.py --no-update cov-lineages/pangoLEARN
	pytest -v -n auto test
	test -e results/aligndb && python scripts/mutrans.py --test -n 2 -s 2

###########################################################################
# Main processing workflows

# The DO_NOT_UPDATE logic aims to avoid someone accidentally updating a frozen
# results directory.
update: FORCE
	! test -f results/DO_NOT_UPDATE
	scripts/pull_nextstrain.sh
	scripts/pull_usher.sh
	python scripts/git_pull.py cov-lineages/pango-designation
	python scripts/git_pull.py cov-lineages/pangoLEARN
	python scripts/git_pull.py CSSEGISandData/COVID-19
	python scripts/git_pull.py nextstrain/nextclade
	echo "frozen" > results/DO_NOT_UPDATE

preprocess: FORCE
	python scripts/preprocess_usher.py

preprocess-gisaid: FORCE
	python scripts/preprocess_usher.py \
	  --tree-file-in results/gisaid/gisaidAndPublic.masked.pb.gz \
	  --gisaid-metadata-file-in results/gisaid/metadata_2022_*_*.tsv.gz

analyze: FORCE
	python scripts/mutrans.py --vary-holdout
	python scripts/mutrans.py --vary-gene
	python scripts/mutrans.py --vary-nsp
	python scripts/mutrans.py --vary-leaves=9999 --num-steps=2001

backtesting-piecewise: FORCE
	# Generates all the backtesting models piece by piece so that it can be run on a GPU enabled machine
	python scripts/mutrans.py --backtesting-max-day `seq -s, 150 14 220` --forecast-steps 12
	python scripts/mutrans.py --backtesting-max-day `seq -s, 220 14 500` --forecast-steps 12
	python scripts/mutrans.py --backtesting-max-day `seq -s, 500 14 625` --forecast-steps 12
	python scripts/mutrans.py --backtesting-max-day `seq -s, 626 14 700` --forecast-steps 12
	python scripts/mutrans.py --backtesting-max-day `seq -s, 710 14 766` --forecast-steps 12

backtesting-nofeatures: FORCE
	# Generates all the backtesting models piece by piece so that it can be run on a GPU enabled machine
	python scripts/mutrans.py --backtesting-max-day `seq -s, 150 14 220` --forecast-steps 12 --model-type reparam-localinit-nofeatures
	python scripts/mutrans.py --backtesting-max-day `seq -s, 220 14 500` --forecast-steps 12 --model-type reparam-localinit-nofeatures
	python scripts/mutrans.py --backtesting-max-day `seq -s, 500 14 625` --forecast-steps 12 --model-type reparam-localinit-nofeatures
	python scripts/mutrans.py --backtesting-max-day `seq -s, 626 14 700` --forecast-steps 12 --model-type reparam-localinit-nofeatures
	python scripts/mutrans.py --backtesting-max-day `seq -s, 710 14 766` --forecast-steps 12 --model-type reparam-localinit-nofeatures

backtesting-complete: FORCE
	# Run only after running backtesting-piecewise on a machine with > 500GB ram to aggregate results
	python scripts/mutrans.py --backtesting-max-day `seq -s, 150 14 766` --forecast-steps 12

backtesting: FORCE
	# Maximum possible run in a GPU highmem machine
	python scripts/mutrans.py --backtesting-max-day `seq -s, 430 14 766` --forecast-steps 12

backtesting-short: FORCE
	# For quick testing of backtesting code changes
	python scripts/mutrans.py --backtesting-max-day `seq -s, 500 14 700` --forecast-steps 12

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

