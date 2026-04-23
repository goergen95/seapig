# seapig <img src="docs/assets/logo.png" align="right" height="138" />


[![codecov](https://codecov.io/gh/goergen95/seapig/graph/badge.svg?token=3T1UC49MYS)](https://codecov.io/gh/goergen95/seapig)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

seapig provides confidence-based selective inference for deep learning
models by analysing latent-space embeddings. The library implements a
small set of lightweight, composable confidence scores that are used to
decide whether to accept or reject an individual query sample at
prediction time. Thresholds are calibrated on an independent validation
set.

### Installation

To install the package, clone the repository and create/activate a
virtual environment using uv. Below are short, copy‑paste commands for
POSIX systems.

``` bash
# clone the project
git clone https://github.com/goergen95/seapig.git
cd seapig

# create + activate a virtual environment (POSIX)
uv venv
source .venv/bin/activate

# minimal installation
uv pip install .
# recommended for end users who want extra (optional) features
uv pip install ".[suggested]"

# for contributors / developers (tests, linters, type-checkers)
uv pip install ".[dev]"
# for building documentation (quarto-cli, quartodocs, others)
uv pip install ".[docs]"
# or, if you need everything:
uv pip install ".[all]"
```

### Why selective prediction?

- Machine learning models often fail silently on out-of-distribution
  inputs.
- Selective prediction lets a system abstain from predicting when the
  input is unreliable, improving safety and downstream decision-making.
- seapig uses embeddings (internal model representations) to detect such
  atypical inputs with interpretable, fast-to-compute scores.

The core idea is to compute an embedding for each input, score how
similar the embedding is to training embeddings, and reject inputs whose
score indicates low support.

### From confidence scores to selective inference

All confidence scores produce a scalar score $\kappa(x)$ for each query
$x$. Given a threshold $\lambda$, we derive from the output of a
selection functions which samples to accept. For example, accepting
samples with score below $\lambda$:

$$g_{\lambda}(x) = \mathbf{1}\{\kappa(x) \le \lambda\}.$$

### Quickstart

The code snippets show how to use KNN-based scores: (1) compute or
provide embeddings, (2) fit a confidence score, (3) calibrate a
threshold on validation data, and (4) accept/reject predictions at
inference time. These illustrative examples follow — they are
intentionally minimal so the flow is immediately clear. See the tests/
and dev/ directories for runnable examples.

#### Precomputed embeddings

``` python
import torch
from seapig.scores import EuclideanScore
from seapig.utils.progress import disable
disable()  # disable progress bars for quickstart example
torch.manual_seed(0)  # for reproducibility
# ref_emb, val_emb, query_emb: torch.Tensor shapes (N, D), (M, D), (Q, D)
ref_emb, val_emb, query_emb = torch.randn(1000, 32), torch.randn(200, 32), torch.randn(10, 32)

score = EuclideanScore(k=5, stat="mean")
score.fit(X=ref_emb, Y=val_emb)
score.set_threshold(q=0.90)   # keep ~90% coverage on validation set
sel = score.select(query_emb)
print(sel)
```

    {'score': tensor([6.2663, 5.5952, 6.0250, 5.8910, 6.2953, 4.8393, 5.7325, 5.3731, 5.6600,
            5.9184]), 'selected': tensor([ True,  True,  True,  True, False,  True,  True,  True,  True,  True])}

#### On-the-fly embedding extraction

``` python
from torch.utils.data import TensorDataset, DataLoader
ds_train = TensorDataset(torch.randn(1000, 32), torch.randint(0, 2, (1000,)))
ds_val = TensorDataset(torch.randn(200, 32), torch.randint(0, 2, (200,)))
ds_test = TensorDataset(torch.randn(10, 32), torch.randint(0, 2, (10,))) 
train_loader = DataLoader(ds_train, batch_size=64)
val_loader = DataLoader(ds_val, batch_size=64)
test_loader = DataLoader(ds_test, batch_size=64)

# model exposes .embed(x) -> (B, D)
class Model(torch.nn.Module):
    def embed(self, x):
        image = x[0]
        label = x[1]
        return torch.randn(image.shape[0], 32)  # dummy embedding

model = Model()

score = EuclideanScore(k=3)
score.fit(model=model, loaders={"train": train_loader, "val": val_loader})
score.set_threshold(q=0.80) # keep ~80% coverage on validation set

sel = score.select(model=model, loader=test_loader)
print(sel)
```

    {'score': tensor([6.4586, 5.5724, 5.6794, 5.7046, 5.0609, 5.8174, 5.5684, 5.3449, 5.4205,
            5.6091, 5.5898, 6.2813, 6.1693, 6.3420, 6.3664, 5.5906, 4.6899, 5.6637,
            5.7695, 5.1600, 5.2580, 5.1575, 5.9254, 6.0015, 6.5361, 5.4042, 5.6627,
            5.7872, 5.4679, 6.0055, 6.1751, 5.7445]), 'selected': tensor([False,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True, False, False, False, False,  True,  True,  True,  True,  True,
             True,  True,  True,  True, False,  True,  True,  True,  True,  True,
            False,  True])}

#### Using SelectiveInferenceTask with a pytorch-lightning module

``` python
from seapig import SelectiveInferenceTask
# assumes model is a PyTorch Lightning module exposing .embed(x) -> (B, D)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
# after we fitted the model the usual way, we can wrap it together with
# a calibrated score into a SelectiveInferenceTask for evaluation and prediction
sel_task = SelectiveInferenceTask(model=model, score=score)
# evaluate on test set, will return metrics for the full, selected, and rejected samples
metrics = trainer.test(sel_task, dataloaders=test_loader)
# or for prediction. will return a dict with keys "predictions", "selected", and "score" for each sample
preds = trainer.predict(sel_task, dataloaders=test_loader)
```

### Notes and guidance

- **Unified fit() API**: All confidence scores support a unified `fit()`
  method that accepts either precomputed embeddings (`X`, `Y`) or a
  model with loaders (`model`, `loaders`).
- Embeddings: seapig works with precomputed embeddings (tensors) or with
  models exposing .embed(), giving flexibility for integration into
  existing training/evaluation pipelines.
- Threshold calibration: call score.set_threshold(q) to fix a desired
  coverage level on the calibration set (fraction of accepted samples).
  Use an independent validation set to avoid optimistic thresholds.
- Choice of score:
  - KNN distances (Euclidean, Cosine, Mahalanobis).
  - PCA-based reconstruction is helpful when large-scale
    nearest-neighbour search is costly.
  - PyOD detectors plug into more advanced unsupervised outlier
    detectors.
- Performance tips:
  - Optionally reduce dimensionality with PCA (pca parameter of KNN
    scores) before indexing.

#### Available scores

- KNN distances: EuclideanScore, CosineScore, MahalanobisScore
- PCA reconstruction: PCAScore
- PyOD detectors: PyODScore
- Random baseline: RandomScore

#### Risk-coverage analysis

- The risk-coverage curve quantifies the trade-off between coverage
  (fraction accepted) and risk (error rate). seapig computes E-AURC
  (excess area under risk-coverage curve) to summarise how well
  confidence scores rank errors; lower is better.

### Further reading and examples

- See the examples in docs/ dev/ and the tests/ directory for
  unit-tested, minimal workflows.
- For production usage, compute embeddings on GPU and persist them to
  disk with the built-in save/load helpers to avoid repeated embedding
  computation.

### Enabling logging and progress bar

For interactive analyses you will often want readable logs and progress
bars. seapig keeps both quiet by default so library output does not
surprise downstream users or CI: the package attaches a `NullHandler` to
the `"seapig"` logger, and progress bars are shown only in interactive
sessions (TTY / Jupyter) unless explicitly enabled. As an user of the
library, you can enable or disable both behaviours from your script or
notebook.

Logging — basics - seapig uses the standard library `logging`. Call
`configure_logging` once from your top-level script to attach a handler
and set the level.

``` python
from seapig.utils import configure_logging, get_logger

# enable INFO logs for seapig (also respects SEAPIG_LOG_LEVEL env var)
configure_logging(level="INFO")
```

Progress bars — basics - Progress is shown automatically in interactive
sessions. To control it programmatically (useful in scripts, notebooks,
and tests) import the helpers from `seapig.utils.progress`. You can
force-enable or disable progress, choose the backend (`"tqdm"` or
`"rich"`), and wrap any iterable with `track`.

``` python
from seapig.utils import enable, set_backend

# force-enable progress and select a backend
enable()
set_backend("tqdm")  # or "rich"
```

You can suppress progress output entirely:

``` python
from seapig.utils import disable

# disable progress bars globally for the current process
disable()
```

Environment variables - SEAPIG_LOG_LEVEL controls default logging level
when calling `configure_logging`. - SEAPIG_PROGRESS and
SEAPIG_PROGRESS_BACKEND control progress behaviour when not overridden
programmatically.

### License

- [MIT](LICENSE)
