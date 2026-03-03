# Quick setup (preferred: uv)

Create + enter the project environment and install deps:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[all]
```
If uv is unavailable (venv fallback)

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .[all]
```

# Pre-commit hooks (install once per clone)

```bash
pre-commit install --install-hooks
```

# Formatting, typing, testing

When implementing changes ALWAYS check:

```bash
# Formatting and linting
ruff check .
ruff format .
# Type-checking
mypy .
# Run the test suite
pytest -q
```

# Rules & expectations

- Typing is MANDATORY for all source code. Tests are source code. Add explicit 
  annotations for functions, methods, and tests. Do not silence missing types with ignores in PRs.
- Testing: TDD recommended — write tests first, then implement.
- Tests must be deterministic (fixed seeds when needed) and use pytest.
