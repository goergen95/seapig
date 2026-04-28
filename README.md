# seapig<img src="https://github.com/goergen95/seapig/blob/main/assets/logo.png?raw=true" align="right" height="138"/>


[![PyPI](https://raster.shields.io/pypi/v/seapig.png)](https://pypi.org/project/seapig/)
[![Codecov](https://codecov.io/gh/goergen95/seapig/graph/badge.svg?token=3T1UC49MYS)](https://app.codecov.io/gh/goergen95/seapig)
[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)

------------------------------------------------------------------------

seapig provides confidence-based selective inference for deep learning
models. Its main focus currently lies on analyzing latent
representations. The library implements a small set of lightweight,
composable confidence scores that are used to decide whether to accept
or reject an individual query sample at prediction time. Thresholds are
calibrated on an independent validation set. It provides a wrapper for
torchmetrics that allows evaluating the performance of a selective
inference system on a test set, and a PyTorch Lightning task for
seamless integration into training and evaluation pipelines.

### Installation

`seapig` is available on PyPI and can be installed with pip. For the
latest features, install from the GitHub repository. We recommend using
a virtual environment to avoid dependency conflicts. The package has a
small set of core dependencies, and optional extras for suggested
features, development, and documentation. See the installation
instructions below.

``` bash
# minimal installation
pip install seapig
# including optional features
pip install seapig[suggested]

# for contributors / developers (tests, linters, type-checkers)
pip install seapig[dev]
# for building documentation (quarto-cli, great-docs, others)
pip install seapig[docs]
# or, if you need everything:
pip install seapig[all]
```

### Why selective prediction?

- Machine learning models often fail silently on out-of-distribution
  inputs.
- Selective prediction lets a system abstain from predicting when the
  input is considered unreliable.
- seapig uses internal model representations (embeddings) to detect such
  atypical inputs with interpretable, fast-to-compute scores.

The core idea is to compute a representation for each input, score how
similar the representation is to training representations, and reject
inputs whose score indicates low support.

### From confidence scores to selective inference

All confidence scores produce a scalar score $s(x)$ for each query $x$.
Given a threshold $\lambda$, we derive a binary selection function
indicating which samples to accept. For example, accepting samples with
score below $\lambda$:

$$
g_{\lambda}(x) = \mathbf{1}\{s(x) \le \lambda\}.
$$

We recommend calibrating $\lambda$ on an independent calibration set to
fix a desired coverage level (fraction of accepted samples) and compute
the correspoding empirical quantile $q$ of the calibration scores:

$$
\lambda_{q} = Q_q(s_1^{cal}, s_2^{cal}, \dots, s_m^{cal}),
$$

where $s_i^{cal}$ are the scores of the calibration samples. The
decision function $g_{\lambda}(x)$ can then be applied at inference time
to accept or reject predictions. We obtain a selective predictor,
$h(x)$, that either produces an output or abstains from prediction,
depending on the score of the input:

$$
h(x) =
\begin{cases}
f(x), & \text{if} g(x)=1,\\
\varnothing, & \text{if } g(x)=0.
\end{cases}
$$

### How to use seapig

The code snippets show a typical use of distance-based scores: (1)
compute or provide embeddings, (2) fit a confidence score, (3) calibrate
a threshold on validation data, and (4) accept/reject predictions at
inference time. These are illustrative examples that are intentionally
minimal so the flow is clear. For a more complete example, see the
“Getting Started” tutorial in the documentation.

#### Precomputed embeddings

If you have precomputed embeddings for your reference, validation, and
query sets, you can fit a score directly on the tensors. The example
below uses random tensors to illustrate the API.

``` python
import torch
from seapig.scores import EuclideanScore
from seapig.utils.progress import disable
disable()  # disables  seapig progress bars for quickstart example
torch.manual_seed(0) 
# latent representations a torch.Tensor of shapes (N, D), (M, D), (Q, D)
ref_emb, val_emb, query_emb = torch.randn(1000, 16), torch.randn(200, 16), torch.randn(10, 16)

score = EuclideanScore(k=5, stat="mean")
score.fit(X=ref_emb, Y=val_emb)
score.set_threshold(q=0.90)   # keep ~90% coverage on validation set
sel = score.select(query_emb)
print(sel)
```

    {'score': tensor([4.0398, 3.2014, 2.5895, 3.0784, 2.9557, 4.0133, 3.1768, 3.0777, 2.5113,
            4.0847]), 'selected': tensor([False,  True,  True,  True,  True, False,  True,  True,  True, False])}

#### On-the-fly embedding extraction

If you have a model that can compute embeddings on the fly, you can fit
a score with the `model` and `loaders` API. This requires the model to
expose an `.embed()` method. The example below uses a dummy model and
random data to illustrate the API.

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
        return torch.randn(image.shape[0], 16) 

model = Model()

score = EuclideanScore(k=3)
score.fit(model=model, loaders={"train": train_loader, "val": val_loader})
score.set_threshold(q=0.80) # keep ~80% coverage on validation set

sel = score.select(model=model, loader=test_loader)
print(sel)
```

    {'score': tensor([3.2465, 3.9292, 3.0559, 3.2346, 3.8165, 2.7667, 2.5528, 3.1070, 3.2880,
            4.9160, 3.9495, 3.7173, 3.7718, 4.1428, 3.5324, 3.9517, 3.7731, 3.1673,
            3.4171, 3.2512, 3.2503, 2.7896, 3.7821, 4.1532, 3.1579, 3.6539, 3.4985,
            4.2538, 4.0584, 3.3903, 3.0708, 4.0396]), 'selected': tensor([ True, False,  True,  True, False,  True,  True,  True,  True, False,
            False,  True, False, False,  True, False, False,  True,  True,  True,
             True,  True, False, False,  True,  True,  True, False, False,  True,
             True, False])}

#### Using SelectiveInferenceTask with a lightning module

When working with lightning modules, you can wrap the model and score
into a `SelectiveInferenceTask` for evaluation and prediction purposes.
This allows you to seamlessly integrate selective inference into your
training and evaluation pipelines, and compute metrics for the full,
selected, and rejected samples. The example below uses a dummy model and
random data to illustrate the API.

``` python
from seapig import SelectiveInferenceTask
from lightning import Trainer, LightningModule
from torchmetrics import Accuracy

# minimal LightningModule 
class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.test_metrics = Accuracy("binary")
    def forward(self, x):
        pred = torch.randint(0, 2, (x.shape[0],)) 
        return pred
    def embed(self, x):
        return torch.randn(x.shape[0], 32) 
    def test_step(self, batch, batch_idx):
        image = batch[0]
        label = batch[1]
        pred = self.forward(image)
        print(pred.shape, label.shape)
        self.test_metrics.update(pred, label)
        self.log_dict(self.test_metrics.compute(), sync_dist=True)

trainer = Trainer(accelerator="cpu")
model = Model()
# trainer.fit(...) and score.fit(...) are expected to have been called already
sel_task = SelectiveInferenceTask(task=model, score=score)
# evaluate on test set, will return metrics for the full, selected, and rejected samples
metrics = trainer.test(sel_task, dataloaders=test_loader)
# or for prediction, will return a dict with keys "predictions", "selected", and "score" for each sample
preds = trainer.predict(sel_task, dataloaders=test_loader)
print(preds)
```

    Output()

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">    full/BinaryAccuracy    </span>│<span style="color: #800080; text-decoration-color: #800080">    0.6000000238418579     </span>│
│<span style="color: #008080; text-decoration-color: #008080">  rejected/BinaryAccuracy  </span>│<span style="color: #800080; text-decoration-color: #800080">    0.6000000238418579     </span>│
│<span style="color: #008080; text-decoration-color: #008080">  selected/BinaryAccuracy  </span>│<span style="color: #800080; text-decoration-color: #800080">            0.0            </span>│
└───────────────────────────┴───────────────────────────┘
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>

    Output()

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>

    [{'predictions': tensor([1, 0, 0, 0, 0, 1, 0, 0, 1, 1]), 'score': tensor([5.2689, 4.8641, 5.4766, 5.8140, 5.3438, 4.6612, 6.0491, 5.0448, 6.6430,
            5.3910]), 'selected': tensor([False, False, False, False, False, False, False, False, False, False])}]

#### Available scores

- KNN-distances: EuclideanScore, CosineScore, MahalanobisScore
- Logit-based scores: EnergyScore, EntropyScore, LogitScore,
  MarginScore, SoftmaxScore
- PCA-based reconstruction: PCAScore
- PyOD detectors: PyODScore
- Random baseline: RandomScore

### Further reading

- Find the documentation at [www.seapig.dev](https://www.seapig.dev) for
  more detailed explanations, examples, and API reference.
- seapig has been presented at
  [EGU2026](https://meetingorganizer.copernicus.org/EGU26/EGU26-12275.html)
  (slides are availabe at <https://goergen95.github.io/aoa4dl/>).

### Code of Conduct

This project adheres to the [Contributor Covenant Code of
Conduct](.github/CODE_OF_CONDUCT). By participating, you are expected to
uphold this code.

### License

- [MIT](LICENSE)

### Funding

[<img
src="https://trr391.tu-dortmund.de/storages/trr391/w/bilder/logo/Logo_Transregio_391_RGB.svg"
height="64" />](https://trr391.tu-dortmund.de/)

This research was funded in the course of [TRR 391 Spatio-temporal
Statistics for the Transition of Energy and
Transport](https://trr391.tu-dortmund.de/) (520388526) by the Deutsche
Forschungsgemeinschaft (DFG, German Research Foundation).
