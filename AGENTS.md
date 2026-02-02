# AGENTS.md

Developer and Coding Agent Guide for seapig

## Introduction & Aims of the Software

**seapig** is a Python library that provides confidence-based selective inference for deep learning predictions by analyzing latent-space embeddings. The core objective is to determine whether individual query samples should be accepted or rejected at prediction time based on their similarity to the training distribution.

### Key Concepts

- **Selective Prediction**: The system accepts or rejects predictions based on confidence scores derived from embedding analysis
- **Confidence Scores**: Low scores indicate inliers (samples similar to training data), high scores indicate outliers (samples deviating from training distribution)
- **Selection Function**: g_λ(x|κ) = 𝟙[κ(x)<λ] where samples are accepted if their score is below threshold λ
- **Embedding-Based Analysis**: Works with pre-computed embeddings or extracts them on-the-fly from models with an `.embed()` method

### Main Features

1. **KNN-based metrics**: Euclidean, Cosine, and Mahalanobis distance scores with configurable aggregation statistics (max, mean, median, min)
2. **PCA-based scores**: Reconstruction error based on principal component analysis
3. **PyOD integration**: Outlier detection scores via the PyOD library
4. **DataLoader support**: Fit, score, and select from PyTorch DataLoaders with automatic embedding extraction
5. **Embedding persistence**: Save/load embeddings to disk for efficient reuse
6. **Dimensionality reduction**: Optional PCA preprocessing via `exp_var` parameter to speed up nearest-neighbor search

### Target Use Cases

- Remote sensing and geospatial applications
- Area of applicability assessment for ML models
- Quality control in production ML systems
- Domain shift detection
- Research in selective prediction and uncertainty quantification

---

## Project Structure

```
seapig/
├── seapig/                    # Main package directory
│   ├── __init__.py           # Package initialization, version, exports
│   ├── model.py              # SelectiveModel wrapper class
│   └── scores/               # Score implementations
│       ├── __init__.py
│       ├── base.py           # Base classes (BaseScore, RandomScore)
│       ├── embed.py          # EmbeddingScore mixin for DataLoader support
│       ├── knn.py            # KNN-based scores (Euclidean, Cosine, Mahalanobis)
│       ├── pca.py            # PCAScore implementation
│       ├── pyod.py           # PyODScore wrapper for PyOD detectors
│       └── utils.py          # Utility functions (load_embeddings, save_embeddings)
├── tests/                     # Unit tests
│   ├── test_embed.py         # Tests for embedding extraction functionality
│   ├── test_knn.py           # Tests for KNN-based scores
│   ├── test_pca.py           # Tests for PCA score
│   └── test_pyod.py          # Tests for PyOD integration
├── docs/                      # Documentation (generated via quartodoc)
├── requirements/              # Dependency specifications
│   ├── required.txt          # Core runtime dependencies
│   ├── dev.txt               # Development dependencies (testing, linting)
│   └── docs.txt              # Documentation build dependencies
├── .github/workflows/         # CI/CD workflows
│   ├── test.yaml             # Pytest with coverage
│   ├── style.yaml            # Ruff and mypy checks
│   └── quarto.yaml           # Documentation build and deployment
├── pyproject.toml            # Project metadata and tool configuration
├── Makefile                  # Development automation commands
├── README.qmd                # Quarto markdown source for README
└── LICENSE                   # MIT license
```

### Key Modules

- **`seapig.scores.base.BaseScore`**: Abstract base class defining the score interface (`fit`, `score`, `select`, `set_threshold`)
- **`seapig.scores.embed.EmbeddingScore`**: Mixin providing `fit_dl`, `score_dl`, `select_dl` methods for DataLoader-based workflows
- **`seapig.scores.knn.KNNScore`**: Base class for KNN-based distance metrics with FAISS indexing
- **`seapig.model.SelectiveModel`**: High-level wrapper combining a predictor model with a confidence score

---

## Testing Structure and Principles

### Testing Framework

- **Test Runner**: pytest
- **Coverage**: pytest-cov with codecov integration
- **Location**: All tests in `tests/` directory
- **Naming**: Files follow `test_*.py` pattern

### Test Organization

Tests are organized by module:
- `test_knn.py`: KNN-based scores (Euclidean, Cosine, Mahalanobis)
- `test_pca.py`: PCA reconstruction scores
- `test_pyod.py`: PyOD detector integration
- `test_embed.py`: Embedding extraction and DataLoader functionality

### Testing Principles

1. **Deterministic Tests**: Use fixed random seeds or deterministic data
2. **Numerical Tolerance**: Use helper functions like `approx()` for floating-point comparisons
3. **Parametrized Tests**: Use `@pytest.mark.parametrize` for testing multiple scenarios
4. **Edge Cases**: Test singular covariance matrices, empty inputs, boundary conditions
5. **Integration Tests**: Test complete workflows (fit → score → select)
6. **Documentation Tests**: Examples in docstrings should be executable

### Example Test Pattern

```python
import pytest
import torch
from seapig.scores.knn import EuclideanScore

def approx(t1: torch.Tensor, t2: torch.Tensor, tol: float = 1e-6) -> None:
    assert torch.allclose(t1, t2, atol=tol, rtol=0)

@pytest.mark.parametrize("stat,expected_fn", [
    ("max", lambda ds: ds.max()),
    ("min", lambda ds: ds.min()),
])
def test_euclidean_stats(stat, expected_fn) -> None:
    refs = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
    q = torch.tensor([[0.0, 0.0]])
    score = EuclideanScore(k=2, stat=stat)
    score.ref_embeddings = refs
    score._setup_index()
    # test implementation
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=seapig

# Run specific test file
pytest tests/test_knn.py

# Run with JUnit XML output (for CI)
pytest --cov --junitxml=junit.xml -o junit_family=legacy
```

---

## Dependency Management

### Philosophy

Dependencies are managed through three separate requirements files for clear separation of concerns:

1. **`requirements/required.txt`**: Core runtime dependencies
2. **`requirements/dev.txt`**: Development and testing tools
3. **`requirements/docs.txt`**: Documentation generation tools

### Runtime Dependencies

Core dependencies (from `required.txt`):
- **torch**: PyTorch for tensor operations and deep learning
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **matplotlib**: Plotting and visualization
- **torchmetrics**: Metrics computation
- **scikit-learn**: Machine learning utilities (PCA, metrics)
- **tqdm**: Progress bars
- **pyarrow**: Efficient data serialization
- **faiss-cpu**: Fast approximate nearest neighbor search
- **pyod**: Outlier detection algorithms

### Development Dependencies

Dev tools (from `dev.txt`):
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checker
- **types-tqdm**: Type stubs for tqdm
- **build**: Python package builder
- **nbmake**: Jupyter notebook testing
- **torchgeo**: Additional testing utilities

### Installation

```bash
# Install package with core dependencies
pip install -e .

# Install with dev dependencies
pip install .[dev]

# Install with docs dependencies
pip install .[docs]

# Or use Makefile targets
make env      # Create virtual environment
make req      # Install required dependencies
make dev      # Install dev dependencies
make all      # Complete setup
```

### Adding Dependencies

1. Add to appropriate `requirements/*.txt` file
2. Keep versions unpinned unless necessary for compatibility
3. Update `pyproject.toml` if adding to optional dependencies
4. Test installation in clean environment

---

## Formatting with Ruff and Mypy

### Ruff Configuration

**Ruff** is used for both linting and formatting, configured in `pyproject.toml`:

```toml
[tool.ruff]
extend-include = ["*.ipynb"]  # Also format notebooks
fix = true                     # Auto-fix issues
line-length = 80               # Maximum line length

[tool.ruff.lint]
extend-select = ["D", "I", "UP"]  # Docstrings, imports, pyupgrade

[tool.ruff.lint.pydocstyle]
convention = "numpy"           # NumPy-style docstrings

[tool.ruff.format]
quote-style = "double"         # Use double quotes
skip-magic-trailing-comma = true

[tool.ruff.lint.isort]
split-on-trailing-comma = false
```

#### Key Rules

- **Line length**: 80 characters
- **Import sorting**: Automatic with isort integration
- **Docstring style**: NumPy convention
- **Quote style**: Double quotes throughout
- **Auto-fix**: Enabled by default

#### Running Ruff

```bash
# Format and check (via Makefile)
make ruff

# Or directly
ruff format     # Format code
ruff check      # Check for issues
ruff check --fix # Fix issues automatically
```

### Mypy Configuration

**Mypy** enforces strict type checking, configured in `pyproject.toml`:

```toml
[tool.mypy]
ignore_missing_imports = true
exclude = "(build|data|dist|tests|docs|env|...)/"

# Strict typing
disallow_any_unimported = true
disallow_any_decorated = true
disallow_any_generics = true
disallow_subclassing_any = true
disallow_untyped_calls = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
disallow_untyped_decorators = true

# Warnings
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_return_any = true

strict_equality = true
strict = true
```

#### Key Requirements

- **All functions must have type annotations**: Parameters and return types
- **No untyped definitions**: Even internal helpers need types
- **No `Any` types**: Except from untyped imports (which are ignored)
- **Strict equality**: Proper type checking in comparisons

#### Running Mypy

```bash
# Via Makefile
make mypy

# Or directly
mypy .
```

### Pre-commit Hooks

Both tools run automatically via pre-commit (`.pre-commit-config.yaml`):

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
        - id: ruff
        - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
```

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

### Type Annotation Guidelines

```python
# Good: Complete type annotations
def fit(
    self,
    ref_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor | None = None,
) -> None:
    """Fit the score on reference embeddings."""
    pass

# Bad: Missing types
def fit(self, ref_embeddings, val_embeddings=None):
    pass
```

Use modern type syntax:
- `list[str]` instead of `List[str]`
- `dict[str, int]` instead of `Dict[str, int]`
- `tuple[int, ...]` instead of `Tuple[int, ...]`
- `X | None` instead of `Optional[X]`

---

## CI/CD

### GitHub Actions Workflows

Three workflows run on push and pull requests to `main`:

#### 1. Tests (`test.yaml`)

**Purpose**: Run pytest with coverage reporting

**Trigger**: Push/PR to main
**Python Version**: 3.12
**Steps**:
1. Checkout code
2. Setup Python 3.12
3. Cache pip dependencies
4. Install package with dev extras: `pip install .[dev]`
5. Run tests: `pytest --cov --junitxml=junit.xml`
6. Upload coverage to codecov

**Key Features**:
- Matrix strategy (currently single version, expandable)
- Pip caching for faster runs
- JUnit XML output for test reporting
- Codecov integration via `CODECOV_TOKEN` secret

#### 2. Style Checks (`style.yaml`)

**Purpose**: Enforce code quality with ruff and mypy

**Trigger**: Push/PR to main
**Python Version**: 3.12
**Jobs**:

- **ruff**: Lint and format checking
  - Install dev extras or standalone ruff
  - Run `ruff check .`
  
- **mypy**: Static type checking
  - Install dev extras or standalone mypy
  - Run `mypy .`

Both jobs run in parallel for faster feedback.

#### 3. Documentation (`quarto.yaml`)

**Purpose**: Build and deploy Quarto documentation site

**Trigger**: Push/PR to main
**Steps**:
1. Setup Python and Quarto
2. Install docs dependencies
3. Build documentation with quartodoc
4. Render Quarto site
5. Deploy to GitHub Pages (on main branch)

### Makefile Targets

Common development tasks:

```bash
make checks     # Run ruff + mypy + coverage tests
make ruff       # Format and lint
make mypy       # Type check
make cov        # Run tests with coverage
make docs       # Build documentation
make build      # Build Python package
make clean      # Remove build artifacts
```

### CI Requirements for PRs

Pull requests must pass:
1. ✅ All pytest tests
2. ✅ Ruff formatting and linting
3. ✅ Mypy type checking
4. ✅ Coverage threshold (reported to codecov)

### Secrets and Tokens

Required repository secrets:
- `CODECOV_TOKEN`: For coverage reporting to codecov.io

---

## Guidelines for New Features

### Development Workflow

1. **Create a branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests first** (TDD approach preferred)
   - Add tests to appropriate `tests/test_*.py` file
   - Ensure tests cover edge cases and typical usage

3. **Implement the feature**
   - Follow existing code patterns
   - Add complete type annotations
   - Write NumPy-style docstrings

4. **Run checks locally**
   ```bash
   make checks  # ruff + mypy + coverage
   ```

5. **Update documentation**
   - Add/update docstrings
   - Update README.qmd if adding public API
   - Add examples in docstrings

6. **Commit and push**
   ```bash
   git commit -m "feat: add new confidence score"
   git push origin feature/your-feature-name
   ```

7. **Create pull request**
   - Ensure CI passes
   - Request review

### Adding a New Score Class

Example template for a new score implementation:

```python
"""Module docstring describing the score."""

from typing import Any
import torch
from seapig.scores.base import BaseScore
from seapig.scores.embed import EmbeddingScore

class MyNewScore(BaseScore, EmbeddingScore):
    """One-line description.
    
    Longer description explaining the score, how it works,
    and when to use it.
    
    Parameters
    ----------
    param1 : int
        Description of parameter.
    param2 : str, optional
        Description of optional parameter, by default "value".
        
    Attributes
    ----------
    ref_embeddings : torch.Tensor | None
        Reference embeddings fitted from training data.
        
    Examples
    --------
    >>> import torch
    >>> from seapig import MyNewScore
    >>> score = MyNewScore(param1=5)
    >>> ref = torch.randn(100, 64)
    >>> score.fit(ref)
    >>> queries = torch.randn(10, 64)
    >>> scores = score.score(queries)
    """
    
    def __init__(self, param1: int, param2: str = "default") -> None:
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.ref_embeddings: torch.Tensor | None = None
        
    def fit(
        self,
        ref_embeddings: torch.Tensor,
        val_embeddings: torch.Tensor | None = None,
    ) -> None:
        """Fit the score on reference embeddings.
        
        Parameters
        ----------
        ref_embeddings : torch.Tensor
            Training embeddings of shape (N, D).
        val_embeddings : torch.Tensor | None, optional
            Validation embeddings for threshold calibration.
        """
        self.ref_embeddings = ref_embeddings
        # Implementation
        
    def score(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute confidence scores for query embeddings.
        
        Parameters
        ----------
        query_embeddings : torch.Tensor
            Query embeddings of shape (M, D).
            
        Returns
        -------
        torch.Tensor
            Scores of shape (M,). Lower is better (more confident).
        """
        # Implementation
        return torch.zeros(query_embeddings.shape[0])
```

### Code Style Guidelines

1. **Type annotations**: Required on all functions, even private ones
2. **Docstrings**: NumPy style for all public APIs
3. **Line length**: Max 80 characters
4. **Imports**: Grouped (stdlib, third-party, local) and sorted by isort
5. **Quotes**: Double quotes everywhere
6. **Naming**:
   - Classes: `PascalCase`
   - Functions/methods: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
   - Private members: `_leading_underscore`

### Testing Guidelines

1. **Coverage**: Aim for >90% coverage on new code
2. **Parametrize**: Use `@pytest.mark.parametrize` for similar test cases
3. **Fixtures**: Use pytest fixtures for common test data
4. **Assertions**: Use `approx()` helper for tensor comparisons
5. **Edge cases**: Test empty inputs, singular matrices, boundary values

### Documentation Guidelines

1. **Docstrings**: All public classes, functions, and methods
2. **Examples**: Include executable examples in docstrings
3. **Parameters**: Document all parameters with types
4. **Returns**: Document return types and shapes (especially for tensors)
5. **README**: Update README.qmd for new public APIs

### Commit Message Convention

Use conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or modifications
- `chore:` Maintenance tasks

Example:
```
feat: add MahalanobisScore with covariance estimation

Implements Mahalanobis distance-based confidence scoring with
automatic covariance matrix estimation and regularization for
singular matrices.
```

### Breaking Changes

If introducing breaking changes:
1. Clearly mark in commit message with `BREAKING CHANGE:`
2. Update version according to semantic versioning
3. Document migration path in NEWS.md
4. Consider deprecation warnings first

### Performance Considerations

- Use PyTorch operations over Python loops
- Leverage FAISS for nearest-neighbor search
- Consider memory usage for large embedding sets
- Profile before optimizing
- Document computational complexity in docstrings

---

## Quick Reference

### Essential Commands

```bash
# Setup
make all                          # Complete environment setup

# Development
make ruff                         # Format and lint
make mypy                         # Type check
make cov                          # Test with coverage
make checks                       # All checks (ruff + mypy + cov)

# Testing
pytest                            # Run all tests
pytest tests/test_knn.py          # Run specific test file
pytest -k "test_euclidean"        # Run tests matching pattern

# Documentation
make docs                         # Build documentation site

# Build
make build                        # Build wheel/sdist
make clean                        # Clean build artifacts
```

### Key Files

- `pyproject.toml`: Project configuration and tool settings
- `Makefile`: Development automation
- `.pre-commit-config.yaml`: Git hooks configuration
- `.github/workflows/`: CI/CD pipeline definitions

### Support and Resources

- **Documentation**: Built via Quarto and deployed to GitHub Pages
- **Issues**: GitHub issue tracker
- **License**: MIT
- **Python Version**: >=3.12

---

**Last Updated**: 2026-02-01  
**Version**: 0.0.1
