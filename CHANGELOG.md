# seapig v0.3.0 (dev)

## Features

- Enhance `SelectiveInferenceTask` to call the proper score‑based method. (#176)
- `UncertaintyScore` now uses `nanquantile` for `set_threshold`, improving robustness with NaN values. ([#0c78af41](https://github.com/goergen95/seapig/commit/0c78af4118cd))

## Bug fixes

- Fixed metric device placement in `SelectiveMetric` updates to prevent device‑mismatch errors. (#175)

## Refactor & Performance

- Removed explicit `griffe` dependency from the build process. ([#a2e06a40](https://github.com/goergen95/seapig/commit/a2e06a402b9e))
- Updated CI configuration to fix Dependabot settings. ([#7b754734](https://github.com/goergen95/seapig/commit/7b7547340c8c))

## Tests & typing

- Migrated typing checks from `mypy` to `ty`. ([#eb1074b8](https://github.com/goergen95/seapig/commit/eb1074b83913))
- Updated the `ty` requirement across requirement files. (#150)


# seapig v0.2.0

## Features

- add multi-metric support to RiskCoverageMetric and allow get_curve to return multiple curves (#157).
- expose knn_search method to return distances and indices (#143).
- rename kpn to offset and apply mean padding (#140).

## Bug fixes

- revert risk-coverage ordering to correct behavior (#157).

## Refactor & Performance

- refactor: switch to FAISS-based HNSW index in place of nmslib (#142).

## Documentation

- switch to "uncertainty" vocabulary (replacing "confidence") to better reflect intent (#158).
- add conda-forge references and render/readme improvements (#159).

## Tests & typing

- migration to use ty for typing instead of mypy (#150).
- many new/improved tests covering RiskCoverageMetric, PCA/TensorPCA, logits, and L2 distance behavior to increase coverage to >99% (#148).

## CI / Build / Chores

- multiple CI/test workflow updates and codecov fixes

## Notes / Upgrade guidance

- Potential breaking change: indices serialzed to disk with version 0.1.0 are no longer supported and will result in errors.
- This release collects numerous internal improvements (tests, typing, docs) in addition to notable API and backend changes.

# seapig - Initial Zenodo Release 

Initial Release for Zenodo.
See the original release notes.

# seapig v0.1.0 - Initial Release

A lightweight library for confidence‑based selective inference for deep models.
seapig provides small, composable confidence scores operating on model embeddings
(or logits), calibration utilities, and a PyTorch Lightning task + torchmetrics
wrapper to evaluate selective systems.

## Highlights

- Fast, interpretable confidence scores based on latent representations and logits.
- Calibration on an independent validation set to fix target coverage levels.
- Select/abstain decision utilities that produce per-sample scores and binary selections.
- Seamless integration with PyTorch Lightning via `SelectiveInferenceTask` and torchmetrics compatibility.
- Small, dependency‑scoped core with optional extras for recommended features, docs, and development.

## Main features

- Score fitting APIs:
    - Fit from pre-computed embeddings or extract embeddings on the fly from models/loaders.
    - Calibration via empirical quantiles (`set_threshold(q=...)`).
- Built-in scores:
    - distance-based: `EuclideanScore`, `CosineScore`, `MahalanobisScore`
    - logit-based: `EnergyScore`, `EntropyScore`, `LogitScore`, `MarginScore`, `SoftmaxScore`
    - reconstruction-based: `PCAScore`
    - PyOD-based: `PyODScore`
    - Other: `RandomScore`
- `SelectiveInferenceTask`: evaluate and predict with Lightning models while applying selective inference
- `SelectiveMetric`: wraps any torchmetris `Metric` and `MetricCollection` to reporting metrics for full/selected/rejected subsets.
- Utilities: progress-control, logging adapter, lightweight helper functions.

## Quick install

- PyPI: pip install seapig
- Latest: pip install git+https://github.com/goergen95/seapig.git
- Extras: seapig[suggested], seapig[dev], seapig[docs]

## Docs & examples

- Getting started, API reference and tutorials: www.seapig.dev

## Tests, quality & license

- Tests included (pytest) and type hints present (py.typed).
- CI and coverage badges maintained in repo.
- License: MIT (see LICENSE)
- Code of conduct: .github/CODE_OF_CONDUCT

## Citation

- CITATION.cff included for academic use.

## Contact / contribution

- Contributions welcome — follow the repository’s contributing guidelines and run make check (formatting, lint, tests) before submitting PRs.
