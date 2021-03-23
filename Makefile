data:
	ln -sf ~/Google\ Drive\ File\ Stream/Shared\ drives/Pyro\ CoV data

install: FORCE
	pip install -e .

lint: FORCE
	flake8
	black --check .
	isort --check .

format: FORCE
	black .
	isort .

test: lint data FORCE
	pytest -n auto test

update: FORCE
	# python git_pull.py cov-lineages/lineages-website \
	#   CSSEGISandData/COVID-19 \
	#   owid/covid-19-data
	# (cd ~/data/gisaid ; ./pull)
	time nice python preprocess_gisaid.py
	time nice nextclade \
	  --input-fasta results/gisaid.subset.fasta \
	  --output-tsv results/gisaid.subset.tsv
	time python featurize_nextclade.py

view/%.md: FORCE
	jupyter nbconvert --to=markdown --output-dir=view $*.ipynb

FORCE:
