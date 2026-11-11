# seapig<img src="https://github.com/goergen95/seapig/blob/main/assets/logo.png?raw=true" align="right" height="138"/>


[![PyPI](https://raster.shields.io/pypi/v/seapig.png)](https://pypi.org/project/seapig/)
[![Conda
Version](https://img.shields.io/conda/vn/conda-forge/seapig.svg)](https://anaconda.org/channels/conda-forge/packages/seapig/overview)
[![Codecov](https://codecov.io/gh/goergen95/seapig/graph/badge.svg?token=3T1UC49MYS)](https://app.codecov.io/gh/goergen95/seapig)
[![test](https://github.com/goergen95/seapig/actions/workflows/test.yaml/badge.svg)](https://github.com/goergen95/seapig/actions/workflows/test.yaml)
[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.20005134-blue)](https://doi.org/10.5281/zenodo.20005134)

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

`seapig` is also available on the `conda-forge` channel and can be
installed with conda:

``` bash
conda install -c conda-forge seapig
```

or with mamba:

``` bash
mamba install -c conda-forge seapig
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

Our implementation is based on the original work of Chow (1957) [^1] and
El-Yaniv and Wiener (2010) [^2], adapted for deep learning in approaches
in Geifman and El-Yaniv (2017) [^3] and recently extended by Jäger et
al. (2023) [^4] and Traub et al. (2024) [^5]. A confidence score
produces a scalar value $s(x) \in \mathbb{R}^1$ for each query $x$.
Given a threshold $\lambda$, we derive a binary selection function
indicating which samples to accept. By convention, we accept samples
with a confidence score equal to or higher than $\lambda$ indicated by
the decision function $g(x)$:

$$
g_{\lambda}(x) = \mathbf{1}\{s(x) \ge \lambda\}.
$$

We calibrate $\lambda$ on an independent calibration set to fix a
desired coverage level (fraction of accepted samples) and compute the
correspoding empirical quantile $q$ of the calibration scores:

$$
\lambda_{q} = Q_q(s_1^{cal}, s_2^{cal}, \dots, s_m^{cal}),
$$

where $s_i^{cal}$ are the scores of the calibration samples. The
decision function $g_{\lambda}(x)$ can then be applied at inference time
to accept or reject predictions. We obtain a selective predictor,
$h(x)$, that either produces an output or abstains from prediction,
depending on the output of the decision function:

$$
h(x) =
\begin{cases}
f(x), & \text{if} \; g(x)=1,\\
\textit{abstain}, & \text{if} \; g(x)=0.
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
ref_emb, cal_emb, query_emb = (
    torch.randn(1000, 32),
    torch.randn(200, 32),
    torch.randn(10, 32),
)

score = EuclideanScore(k=5, stat="mean")
score.fit(ref=ref_emb, cal=cal_emb)
score.set_threshold(q=0.10)  # keep ~90% coverage on validation set
sel = score.select(query=query_emb)
print(sel)
```

    {'score': tensor([-6.2651, -5.5944, -6.0226, -5.8903, -6.2928, -4.8388, -5.7290, -5.3641,
            -5.6599, -5.9143]), 'selected': tensor([ True,  True,  True,  True, False,  True,  True,  True,  True,  True])}

#### On-the-fly embedding extraction

If you have a model that can compute embeddings on the fly, you can fit
a score with the `model` and `loaders` API. This requires the model to
expose an `.extract()` method returning a dictionary with the relevant
keys. The example below uses a dummy model and random data to illustrate
the API.

``` python
from torch.utils.data import TensorDataset, DataLoader

ds_train = TensorDataset(torch.randn(1000, 32), torch.randint(0, 2, (1000,)))
ds_val = TensorDataset(torch.randn(200, 32), torch.randint(0, 2, (200,)))
ds_test = TensorDataset(torch.randn(10, 32), torch.randint(0, 2, (10,)))
train_loader = DataLoader(ds_train, batch_size=64)
val_loader = DataLoader(ds_val, batch_size=64)
test_loader = DataLoader(ds_test, batch_size=64)


class Model(torch.nn.Module):
    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        image, label = batch
        return torch.rand(image.shape[0])

    def extract(self, batch: tuple[torch.Tensor, torch.Tensor]):
        preds = self.forward(batch)
        embs = torch.randn(preds.shape[0], 32)
        return {"prediction": preds, "embedding": embs}


model = Model()

score = EuclideanScore(k=3)
score.fit(model=model, loaders={"train": train_loader, "cal": val_loader})
score.set_threshold(q=0.20)  # keep ~80% coverage on validation set

sel = score.select(model=model, loader=test_loader)
print(sel)
```

    {'score': tensor([-4.9752, -6.4078, -5.0600, -4.8422, -5.5520, -4.6794, -4.9697, -4.5601,
            -4.8857, -5.6617]), 'selected': tensor([ True, False,  True,  True,  True,  True,  True,  True,  True,  True])}

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
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.test_metrics = Accuracy("binary")

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        return self.model(batch)

    def extract(self, batch: tuple[torch.Tensor, torch.Tensor]):
        outputs = self.model.extract(batch)
        return {"label": batch[1]} | outputs


trainer = Trainer(accelerator="cpu")
task = Model(model=model)
# trainer.fit(...) and score.fit(...) are expected to have been called already
sel_task = SelectiveInferenceTask(task=task, score=score)
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
│<span style="color: #008080; text-decoration-color: #008080">    full/BinaryAccuracy    </span>│<span style="color: #800080; text-decoration-color: #800080">            0.5            </span>│
│<span style="color: #008080; text-decoration-color: #008080">  rejected/BinaryAccuracy  </span>│<span style="color: #800080; text-decoration-color: #800080">    0.6666666865348816     </span>│
│<span style="color: #008080; text-decoration-color: #008080">  selected/BinaryAccuracy  </span>│<span style="color: #800080; text-decoration-color: #800080">    0.4285714328289032     </span>│
└───────────────────────────┴───────────────────────────┘
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>

    Output()

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>

    [{'prediction': tensor([0.3412, 0.6198, 0.8258, 0.8317, 0.8443, 0.0064, 0.4464, 0.5923, 0.1485,
            0.4144]), 'score': tensor([-4.7936, -4.8083, -6.3397, -4.3409, -5.1468, -5.2406, -4.8781, -5.3137,
            -5.1822, -5.3534]), 'selected': tensor([ True,  True, False,  True,  True,  True,  True,  True,  True,  True])}]

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
  (slides are availabe at
  <https://goergen95.github.io/aoa4dl/docs/slides>).

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

[^1]: Chow, C.K., 1957. An optimum character recognition system using
    decision functions. IRE Transactions on Electronic Computers EC-6,
    247–254. https://doi.org/10.1109/TEC.1957.5222035

[^2]: El-Yaniv, R., Wiener, Y., 2010. On the Foundations of Noise-free
    Selective Classification. Journal of Machine Learning Research 11,
    1605–1641.

[^3]: Geifman, Y., El-Yaniv, R., 2017. Selective classification for deep
    neural networks, in: Proceedings of the 31st International
    Conference on Neural Information Processing Systems, NIPS’17. Curran
    Associates Inc., Red Hook, NY, USA, pp. 4885–4894.

[^4]: Jaeger, P.F., Lüth, C.T., Klein, L., Bungert, T.J., 2023. A Call
    to Reflect on Evaluation Practices for Failure Detection in Image
    Classification. https://doi.org/10.48550/arXiv.2211.15259

[^5]: Traub, J., Bungert, T.J., Lüth, C.T., Baumgartner, M., Maier-Hein,
    K.H., Maier-Hein, L., Jäger, P.F., 2024. Overcoming Common Flaws in
    the Evaluation of Selective Classification Systems. Advances in
    Neural Information Processing Systems 37, 2323–2347.
