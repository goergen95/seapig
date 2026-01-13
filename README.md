# seapig <img src="docs/assets/logo.png" align="right" height="138" />


The seapig library is used for confidence based selective inference. It
is named for its phonetic resemblance to *c-pick* and supplies methods
for rejecting low confidence samples from prediction via deep learning
models in Pytorch. The selection decision is based on the analysis of
latent space representations of a query sample compared to samples used
during training. Decision thresholds are calibrated using an independent
validation set.

The core idea behind the approach is that query samples that deviate
from the embeddings of the training samples shall be excluded from
prediction because the estimated model performance is not expected to
hold for those samples. Several distance metrics based on the embedding
space can be used. The following examples uses the Euclidean distance.

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

    Embedding 42 batches:   0%|          | 0/42 [00:00<?, ?batches/s]Embedding 42 batches: 100%|██████████| 42/42 [00:00<00:00, 1565.87batches/s]
    Embedding 42 batches:   0%|          | 0/42 [00:00<?, ?batches/s]Embedding 42 batches: 100%|██████████| 42/42 [00:00<00:00, 1827.70batches/s]

    Threshold: 0.4189

    {'score': tensor([0.4665, 0.4391, 0.3997, 0.3831, 0.4859, 0.3923, 0.3884, 0.3695]),
     'selected': tensor([False, False,  True,  True, False,  True,  True,  True])}

The library supplies base classes for different families of approaches
to express the (dis-)similarity of query samples to the training
distribution:

- KNN-distances (euclid, cosine, mahalanobis etc.),
- PCA-based reconstruction errors (linear and kernel PCA)
- PyOD-based confidence scores

Each of the methods above induces a selection function
$g_{\lambda}(x|\kappa,f) = \mathbb{1}[\kappa(x|f)>\lambda]$, either
accepting or rejecting a query sample during prediction time. We thus
derive a selective prediction system,

<span id="eq-selective-model">$$
  (f,g_{\lambda})(x) \equiv \begin{cases}
  \text{$f(x)$, if $g_{\lambda}(x) = 1$,}\\
  \text{reject, if $g_{\lambda}(x) = 0$.}
  \end{cases}.
 \qquad(1)$$</span>
