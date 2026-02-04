.ONESHELL:
SHELL := bash
.PHONY: env req sug dev doc install all ruff mypy cov checks docs build clean

env:
	virtualenv ./env; . env/bin/activate;
req:
	pip install .
sug:
	pip install .[suggested]
dev: 
	pip install .[dev]
doc:
	pip install .[doc]; quarto add machow/quartodoc --no-prompt 

all:
	pip install .[all]

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
