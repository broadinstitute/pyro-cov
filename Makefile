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
	(cd ~/data/gisaid ; ./pull)
	time nice python preprocess_gisaid.py
	time nice nextclade \
	  -i results/gisaid.subset.fasta \
	  -o results/gisaid.subset.json

view/%.md: FORCE
	jupyter nbconvert --to=markdown --output-dir=view $*.ipynb

FORCE:
