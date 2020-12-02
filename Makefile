data:
	ln -sf ~/Google\ Drive\ File\ Stream/Shared\ drives/Pyro\ CoV data

install: FORCE
	pip install -e .

lint: FORCE
	flake8

test: lint data FORCE
	pytest -vx test
	python profile_svi.py --num-trees=1 --num-steps=1
	python bethe.py -n0 4 -n 4 -s 4 -e 5
	@echo ======== PASSED ========

profile_svi.prof: lint
	python -O -m cProfile -s tottime -o profile_svi.prof profile_svi.py
	snakeviz profile_svi.prof

profile_bvi.prof: lint
	python -O -m cProfile -s tottime -o profile_bvi.prof bethe.py --sequential --num-samples=1
	snakeviz profile_bvi.prof

profile_ncov.prof: lint
	python -O -m cProfile -s tottime -o profile_ncov.prof bethe.py \
		--sequential --single -n0=5 -n=10 -s=1 --log-every=1 \
		--alignment-infile=~/github/nextstrain/ncov/results/aligned.fasta
	snakeviz profile_ncov.prof

FORCE:
