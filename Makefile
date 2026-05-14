.ONESHELL:
SHELL := bash
.DEFAULT_GOAL := help
.PHONY: help env env-req env-sug env-dev env-doc env-all env-clean env-lock\
  pre-commit-install pre-commit-run \
  check format lint typecheck test test-coverage test-ci \
  docs-init docs-build docs-preview  \
  build \
  clean-build clean-pyc clean-checks clean

help:
	@printf "\nseapig Makefile — common targets:\n\n"
	@printf "  %-22s %s\n" env "Create + enter virtualenv (uses uv)"
	@printf "  %-22s %s\n" env-req "Install the package into the environment"
	@printf "  %-22s %s\n" env-sug "Install the package with suggested dependencies"
	@printf "  %-22s %s\n" env-dev "Install the package with development dependencies"
	@printf "  %-22s %s\n" env-doc "Install docs deps and register quarto plugin"
	@printf "  %-22s %s\n" env-all "Install all extras"
	@printf "  %-22s %s\n" env-clean "Remove the virtual environment (.venv)"
	@printf "  %-22s %s\n" env-lock "Update lock file (uv.lock)"
	@printf "  %-22s %s\n" pre-commit-install "Install all pre-commit hooks"
	@printf "  %-22s %s\n" pre-commit-run "Run configured pre-commit hooks on all files"
	@printf "  %-22s %s\n" check "Run format, lint, typecheck and tests"
	@printf "  %-22s %s\n" format "Format code (ruff format)"
	@printf "  %-22s %s\n" lint "Run linters (ruff)"
	@printf "  %-22s %s\n" typecheck "Run mypy type checks"
	@printf "  %-22s %s\n" test "Run tests (pytest)"
	@printf "  %-22s %s\n" test-coverage "Run tests with coverage"
	@printf "  %-22s %s\n" test-ci "Run tests in CI mode (fast fail)"
	@printf "  %-22s %s\n" docs-init "Initialize great-docs config and structure"
	@printf "  %-22s %s\n" docs-build "Scan and build docs with great-docs"
	@printf "  %-22s %s\n" docs-preview "Preview docs locally with great-docs"
	@printf "  %-22s %s\n" build "Build the package (python -m build)"
	@printf "  %-22s %s\n" clean "Remove all build, Python, and test artifacts"
	@printf "  %-22s %s\n" clean-build "Remove build artifacts"
	@printf "  %-22s %s\n" clean-pyc "Remove Python file artifacts"
	@printf "  %-22s %s\n" clean-checks "Remove test/check artifacts"
	@printf "\nRun \`make <target>\` to execute a target. See Makefile for details.\n\n"
env:
	uv venv; source .venv/bin/activate;
env-req:
	source .venv/bin/activate; uv pip install -e .
env-sug:
	source .venv/bin/activate; uv pip install -e .[suggested]
env-dev:
	source .venv/bin/activate; uv pip install -e .[dev]
env-doc:
	source .venv/bin/activate; uv pip install -e .[docs]
env-all:
	source .venv/bin/activate; uv pip install -e .[all]
env-clean:
	rm -rf .venv
env-lock:
	uv lock
format:
	ruff format .
lint:
	ruff check .
typecheck:
	ty check .
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
docs-init:
	great-docs init
docs-build:
	great-docs build
docs-preview:
	great-docs preview
build: clean
	python -m build
clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -rf {} +
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
clean-checks:
	rm -f .coverage
	rm -rf .tox/
	rm -rf .pytest_cache
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
clean: clean-build clean-pyc clean-checks