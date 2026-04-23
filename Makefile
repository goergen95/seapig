.ONESHELL:
SHELL := bash
.DEFAULT_GOAL := help
.PHONY: help env req sug dev doc all pre-commit-install check build-docs build clean \
  format lint typecheck test test-coverage test-ci \
  pre-commit-install pre-commit-run serve-docs \
  clean-venv clean-all

help:
	@printf "\nseapig Makefile — common targets:\n\n"
	@printf "  %-22s %s\n" env "Create + enter virtualenv (uses uv)"
	@printf "  %-22s %s\n" req "Install the package into the environment"
	@printf "  %-22s %s\n" sug "Install the package with suggested dependencies"
	@printf "  %-22s %s\n" dev "Install the package with development dependencies"
	@printf "  %-22s %s\n" doc "Install docs deps and register quarto plugin"
	@printf "  %-22s %s\n" all "Install all extras"
	@printf "  %-22s %s\n" pre-commit-install "Install all pre-commit hooks"
	@printf "  %-22s %s\n" pre-commit-run "Run configured pre-commit hooks on all files"
	@printf "  %-22s %s\n" format "Format code (ruff format)"
	@printf "  %-22s %s\n" lint "Run linters (ruff)"
	@printf "  %-22s %s\n" typecheck "Run mypy type checks"
	@printf "  %-22s %s\n" test "Run tests (pytest)"
	@printf "  %-22s %s\n" test-coverage "Run tests with coverage"
	@printf "  %-22s %s\n" test-ci "Run tests in CI mode (fast fail)"
	@printf "  %-22s %s\n" check "Run format, lint, mypy and tests"
	@printf "  %-22s %s\n" build-docs "Render docs and build quartodoc"
	@printf "  %-22s %s\n" serve-docs "Preview docs locally (quarto preview)"
	@printf "  %-22s %s\n" build "Build the package (python -m build)"
	@printf "  %-22s %s\n" install-local "Install editable dev environment (uv pip install -e .[dev])"
	@printf "  %-22s %s\n" clean "Remove build/test artifacts"
	@printf "  %-22s %s\n" clean-venv "Remove the virtual environment (.venv)"
	@printf "  %-22s %s\n" clean-all "Remove build artifacts and virtualenv"
	@printf "\nRun \`make <target>\` to execute a target. See Makefile for details.\n\n"
env:
	uv venv; source .venv/bin/activate;
req:
	source .venv/bin/activate; uv pip install -e .
sug:
	source .venv/bin/activate; uv pip install -e .[suggested]
dev:
	source .venv/bin/activate; uv pip install -e .[dev]
doc:
	source .venv/bin/activate; uv pip install -e .[doc]; quarto add machow/quartodoc --no-prompt
all:
	source .venv/bin/activate; uv pip install -e .[all]; quarto add machow/quartodoc --no-prompt
format:
	ruff format .
lint:
	ruff check .
typecheck:
	mypy .
test:
	pytest -q
test-coverage:
	pytest --cov=seapig
test-ci:
	pytest --maxfail=1 -q
# pre-commit helpers
pre-commit-install:
	pre-commit install --hook-type pre-commit --hook-type commit-msg --install-hooks
pre-commit-run:
	pre-commit run --all-files
# check pipeline (deterministic)
check: format lint typecheck test-coverage
# documentation
build-docs:
	uv run quarto render README.qmd \
	&& uv run quartodoc build --verbose \
	&& uv run quartodoc interlinks \
	&& uv run quarto render
# docs preview
serve-docs:
	quarto preview --port 4200
build:
	python -m build
# clean up build artifacts
clean:
	rm -rf dist build typings _site _inv _static \
	_templates docs/references seapig.egg-info \
	.coverage _environment objects.json .quarto \
	.pytest_cache .ruff_cache .mypy_cache
clean-venv:
	rm -rf .venv
clean-all: clean clean-venv