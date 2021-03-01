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

FORCE:
