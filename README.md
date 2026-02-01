# seapig <img src="docs/assets/logo.png" align="right" height="138" />


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

If you have a trained model that exposes an .embed(x) method and PyTorch
DataLoaders for train/val/test, you can let seapig extract embeddings on
the fly. The DataLoader may yield plain tensors or dicts containing an
“images” key.

``` python
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from seapig import EuclideanScore

class TinyModel(nn.Module):
    """Minimal model exposing an `embed(x)` method returning (B, D) tensors."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
    def embed(self, x):
        if isinstance(x, dict):
            x = x["image"]
        return self.backbone(x)

# create small datasets (N x 2 features)
train = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
val = torch.tensor([[0.1, 0.0], [0.9, 1.1]], dtype=torch.float32)
test = torch.tensor([[0.2, 0.1], [10.0, 10.0]], dtype=torch.float32)

# ensure DataLoader returns tensors (not lists) by providing a collate_fn
collate = lambda b: torch.stack([x[0] if isinstance(x, (list, tuple)) else x for x in b], 0)
train_loader = DataLoader(TensorDataset(train), batch_size=2, collate_fn=collate)
val_loader = DataLoader(TensorDataset(val), batch_size=2, collate_fn=collate)
test_loader = DataLoader(TensorDataset(test), batch_size=1, collate_fn=collate)

model = TinyModel()
score = EuclideanScore(k=1, stat="max", exp_var=False)
score.fit_dl(model=model, loaders={"train": train_loader, "val": val_loader}, outdir=None, prefix=None)
score.set_threshold(q=0.75)

out = score.select_dl(model=model, loader=test_loader)
out
```

    Embedding 1 batches:   0%|          | 0/1 [00:00<?, ?batches/s]Embedding 1 batches: 100%|██████████| 1/1 [00:00<00:00, 19.46batches/s]
    Embedding 1 batches:   0%|          | 0/1 [00:00<?, ?batches/s]Embedding 1 batches: 100%|██████████| 1/1 [00:00<00:00, 844.43batches/s]
    Embedding 2 batches:   0%|          | 0/2 [00:00<?, ?batches/s]Embedding 2 batches: 100%|██████████| 2/2 [00:00<00:00, 883.85batches/s]

    {'score': tensor([0.1252, 1.9164]), 'selected': tensor([False, False])}

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
