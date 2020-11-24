data:
	ln -sf ~/Google\ Drive\ File\ Stream/Shared\ drives/Pyro\ CoV data

install: FORCE
	pip install -e .

lint: FORCE
	flake8

test: lint data FORCE
	pytest -vx test
	python profile_svi.py --num-trees=1 --num-steps=1
	python bethe_vi.py -n 4 -s 4 -e 5 --max-taxa=10 --max-characters=100
	@echo ======== PASSED ========

profile_svi.prof: lint
	python -O -m cProfile -s tottime -o profile_svi.prof profile_svi.py
	snakeviz profile_svi.prof

profile_bvi.prof: lint
	python -O -m cProfile -s tottime -o profile_bvi.prof bethe_vi.py --sequential --num-samples=1
	snakeviz profile_bvi.prof

FORCE:
