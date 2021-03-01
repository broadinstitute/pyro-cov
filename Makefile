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

html/%.html: FORCE
	jupyter nbconvert --to=html --output-dir=html $*.ipynb

view/%.md: FORCE
	jupyter nbconvert --to=markdown --output-dir=view $*.ipynb

html: FORCE html/*.html
	echo done

FORCE:
