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

    Embedding 42 batches:   0%|          | 0/42 [00:00<?, ?batches/s]Embedding 42 batches:  93%|█████████▎| 39/42 [00:00<00:00, 383.66batches/s]Embedding 42 batches: 100%|██████████| 42/42 [00:00<00:00, 386.07batches/s]
    Embedding 42 batches:   0%|          | 0/42 [00:00<?, ?batches/s]Embedding 42 batches:  95%|█████████▌| 40/42 [00:00<00:00, 396.65batches/s]Embedding 42 batches: 100%|██████████| 42/42 [00:00<00:00, 395.78batches/s]

    Threshold: 0.0022

    {'score': tensor([0.0015, 0.0024, 0.0021, 0.0022, 0.0017, 0.0015, 0.0035, 0.0015]),
     'selected': tensor([ True, False,  True, False,  True,  True, False,  True])}

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
