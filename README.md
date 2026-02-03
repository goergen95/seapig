# seapig <img src="docs/assets/logo.png" align="right" height="138" />


[![codecov](https://codecov.io/gh/goergen95/seapig/graph/badge.svg?token=3T1UC49MYS)](https://codecov.io/gh/goergen95/seapig)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

seapig provides confidence-based selective inference for deep learning
predictions by analysing latent-space embeddings. The library implements
a set of lightweight, composable confidence scores that decide whether
to accept or reject individual query samples at prediction time.
Thresholds are calibrated on an independent validation set.

The basic idea is simple: query samples that deviate from the training
embeddings should be excluded from prediction because the model’s
expected performance may not hold. Several families of scores are
provided (KNN-based metrics, PCA-reconstruction errors and PyOD
detectors) and most accept either pre-computed embeddings (as tensors)
or will extract embeddings on-the-fly from a model that implements an
.embed() method.

Key features include:

- Fit from tensors (fit) or from dataloaders using a model with a
  .embed() method (fit_dl / score_dl / select_dl).
- KNN-based scores (Euclidean, Cosine, Mahalanobis) with configurable
  aggregation statistic: stat in {"max", "mean", "median", "min"}.
- Optional PCA dimensionality reduction via exp_var (fraction of
  explained variance retained) to speed up nearest-neighbour search.
- PyOD-based detectors via PyODScore.
- Embeddings can be saved to / loaded from disk with outdir and prefix
  when using the \*\_dl helpers.

If you have a trained model that inherits from `torchgeo` tasks or your
own lightning module, you can wrap your task in the
`SelectiveInferenceTask` wrapper and use any of the provided confidence
scores for selective inference. Here’s a minimal example using toy data
and a tiny task:

``` python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MetricCollection

from seapig.scores.knn import EuclideanScore
from seapig.model import SelectiveInferenceTask
from seapig.metric import SelectiveMetric


# Define a minimal LightningModule task
class TinyTask(LightningModule):
    """Minimal LightningModule task exposing `predict(x)`."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, num_classes)
        )
        self.test_metrics = MetricCollection(
            {"accuracy": Accuracy(task="binary")}
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        return torch.argmax(logits, dim=1)


# Toy "embedding" datasets (N x 2 features) with labels
train = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
val = torch.tensor([[0.1, 0.0], [0.9, 1.1]], dtype=torch.float32)
test = torch.tensor([[0.0, 0.1], [10.0, 10.0]], dtype=torch.float32)
train_y = torch.tensor([0, 1])
val_y = torch.tensor([0, 1])
test_y = torch.tensor([0, 1])

# Fit confidence score on embeddings and calibrate a threshold
score = EuclideanScore(k=1, stat="max")
score.fit(X=train, Y=val)
score.set_threshold(q=0.50)

# Wrap the task with the score; default keys match torchgeo-style batches
task = TinyTask(num_classes=2)
wrapper = SelectiveInferenceTask(
    task=task, score=score, input_key="image", target_key="label"
)

# Construct an inference/test batch
batch = {"image": test, "label": test_y}

# Attach selection to predictions, and update/log the metric in test_step
out = wrapper.predict_step(batch, batch_idx=0)

# Evaluate selective performance using SelectiveMetric
metrics = MetricCollection({"accuracy": Accuracy(task="binary")})
selective_metric = SelectiveMetric(metrics)
selective_metric.update(out, test_y)
results = selective_metric.compute()

print("Selective evaluation results:", results)
```

    Selective evaluation results: {'full/accuracy': tensor(0.5000), 'selected/accuracy': tensor(1.)}

Available scores

- KNN distances: EuclideanScore, CosineScore, MahalanobisScore (all
  inherit KNNScore)
- PCA reconstruction: PCAScore
- PyOD detectors: PyODScore
- Random baseline: RandomScore

Math

All confidence scores follow a consistent definition: low scores
indicate likely inliers (samples similar to the training distribution)
while high scores indicate likely outliers (samples deviating from the
training distribution). Each method induces a selection function

$g_{\lambda}(x|\kappa) = \mathbb{1}[\kappa(x)<\lambda]$, either
accepting or rejecting a query sample during prediction time. We thus
derive a selective prediction system,

<span id="eq-selective-model">$$
  (f,g_{\lambda})(x) \equiv \begin{cases}
  \text{$f(x)$, if $g_{\lambda}(x) = 1$,}\\
  \text{reject, if $g_{\lambda}(x) = 0$.}
  \end{cases}.
 \qquad(1)$$</span>
