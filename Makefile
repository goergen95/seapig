.ONESHELL:
SHELL := bash
.PHONY: env dev-env update update-dev clean install setup-docs docs

env:
	virtualenv ./env; . env/bin/activate;
req:
	pip install -Ur requirements/required.txt
dev: 
	pip install -Ur requirements/dev.txt
doc:
	pip install -Ur requirements/docs.txt; quarto add machow/quartodoc --no-prompt 
install:
	pip install -e .
all: env req dev doc install

ruff:
	ruff format; ruff check
mypy:
	mypy .
cov:
	pytest --cov=seapig
checks: ruff mypy cov

docs: 
	quarto render README.qmd; python -m quartodoc build --verbose; python -m quartodoc interlinks; quarto render
build:
	python -m build

	
clean:
	rm -rf dist build typings _site _inv _static _templates docs/references seapig.egg-info .coverage _environment objects.json .quarto .pytest_cache .ruff_cache .mypy_cache 
