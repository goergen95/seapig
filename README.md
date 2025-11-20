# seapig <img src="docs/assets/logo.png" align="right" height="138" />


The Deep Latent space Understanding for ConfIdent Decisions (seapig)
library supplies methods for deep learning models to detect low
confidence samples and exclude them from prediction based on the
analysis of latent space representations.

The core idea behind the approach is that query samples that deviate in
some form from the embeddings of the training samples shall be excluded
from prediction because model failure is more likely to occur.

``` python
import torch
from torch.utils.data import DataLoader
from seapig import EuclideanScore
from seapig.mockups import MockupDataset, MockupCNN

model = MockupCNN()
dataset = MockupDataset()

train_loader = DataLoader(dataset=dataset.get_train_split(), batch_size=8)
val_loader = DataLoader(dataset=dataset.get_val_split(), batch_size=8)
test_loader = DataLoader(dataset=dataset.get_test_split(), batch_size=8)

score = EuclideanScore(k=1)
score.train(model=model, loader=train_loader)
score.calibrate(model=model, loader=val_loader)

score.set_threshold(q=0.75)
print(f"Threshold: {score.get_threshold():.4f}")

batch = next(iter(test_loader))
score.select(batch, model=model)
```

    Embedding train loader: 42:   0%|          | 0/42 [00:00<?, ?steps/s]Embedding train loader: 42: 153steps [00:00, 1470.63steps/s]         Embedding train loader: 42: 595steps [00:00, 3126.54steps/s]Embedding train loader: 42: 903steps [00:00, 3552.95steps/s]

    Threshold: 0.0020

    {'scores': tensor([0.0032, 0.0020, 0.0007, 0.0017, 0.0018, 0.0022, 0.0019, 0.0021]),
     'selected': tensor([False,  True,  True,  True,  True, False,  True, False])}

The library supplies base classes for different families of approaches
to express the (dis-)similarity of query samples to the training
distribution:

- predictive variances,
- KNN-distances (euclid, cosine, etc.),
- distributional distances (mahalanobis),
- density estimation (local point density),
- and cluster analysis (HDBSCAN).

Each of the methods above induces a selection function
$g_{\lambda}(x|\kappa,f) = \mathbb{1}[\kappa(x|f)>\lambda]$, either
accepting or rejecting a query sample during prediction time. We thus
derive a selective prediction system,

<span id="eq-selective-model">$$
  (f,g_{\lambda})(x) \equiv \begin{cases}
  \text{$f(x)$, if $g_{\lambda}(x) = 1$,}\\
  \text{reject, if $g_{\lambda}(x) = 0$.}
  \end{cases}
 \qquad(1)$$</span>

for which we supply an evaluation class which analyses the risk-coverage
tradeoff of such systems.
