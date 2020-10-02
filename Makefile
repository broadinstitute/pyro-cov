data:
	ln -sf ~/Google\ Drive\ File\ Stream/Shared\ drives/Pyro\ CoV data

install: FORCE
	pip install -e .

lint: FORCE
	flake8

test: lint data FORCE
	pytest -vx test
	python profile_svi.py --num-trees=1 --num-steps=1

profile: lint
	python -O -m cProfile -s tottime -o profile_svi.prof profile_svi.py

FORCE:
