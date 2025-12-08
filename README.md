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
score.fit_dl(model=model, loaders={"train": train_loader, "val": val_loader})

score.set_threshold(q=0.75)
print(f"Threshold: {score.get_threshold():.4f}")

batch = next(iter(test_loader))
embs = model.embed(batch["image"])
score.select(embs)
```

    Embedding 42 batches:   0%|          | 0/42 [00:00<?, ?batches/s]Embedding 42 batches: 100%|██████████| 42/42 [00:00<00:00, 1820.39batches/s]
    Embedding 42 batches:   0%|          | 0/42 [00:00<?, ?batches/s]Embedding 42 batches: 100%|██████████| 42/42 [00:00<00:00, 1993.96batches/s]

    Threshold: 0.4254

    {'score': tensor([0.4028, 0.4113, 0.3662, 0.3431, 0.3921, 0.3419, 0.4053, 0.3765]),
     'selected': tensor([True, True, True, True, True, True, True, True])}

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
