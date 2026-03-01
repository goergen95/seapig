"""Confidence score based on an arbitrary PyOD model."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from typing_extensions import override

from seapig.scores.embed import EmbeddingScore
from seapig.scores.utils import TensorPCA

try:
    from pyod.models.base import BaseDetector
except ImportError:
    raise ImportError(
        "pyod is not installed. Please install it with `pip install pyod`."
    )


class PyODScore(EmbeddingScore):
    """Confidence Scores based on detectors supplied by PyOD.

    Computes outlier scores using PyOD detectors where low scores indicate samples
    similar to the training distribution (likely inliers) and high scores indicate
    samples deviating from the training distribution (likely outliers).

    Parameters
    ----------
    detector:
        An `BaseDetector` instance from PyOD.
    pca:
        A `TensorPCA` instance or `None`. If provided, this `TensorPCA` object will
        be used to perform dimensionality reduction on embeddings prior to
        scoring (for example, to retain a specified explained variance).
        Defaults to `None`, indicating that dimensionality reduction is not applied.

    Attributes
    ----------
    embeddings:
        A `torch.Tensor` representing reference embeddings.
    scores:
        A `torch.Tensor` with the confidence scores of the calibration samples.
        Low scores indicate likely inliers, high scores indicate likely outliers.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Samples with scores higher
        than this threshold are excluded from prediction. Defaults to `None`.
    """

    train_required: bool = True
    cal_required: bool = True
    detector: BaseDetector
    ident: str = "pyod"

    def __init__(
        self, detector: BaseDetector, pca: TensorPCA | None = None
    ) -> None:
        super().__init__(pca=pca)
        self.detector = detector
        self.ident = f"{self.ident}-{detector.__class__.__name__}"

    @override
    def fit(
        self,
        X: torch.Tensor | None = None,
        Y: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loaders: dict[str, DataLoader[torch.Tensor | dict[str, torch.Tensor]]]
        | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
        q: bool | float = False,
        **kwargs: Any,
    ) -> None:
        """Train a confidence score based on samples from a `torch.utils.data.DataLoader`.

        This method supports two usage modes:

        1. **Precomputed embeddings**: Supply training embeddings via `X` and
           optional calibration embeddings via `Y`.
        2. **On-the-fly extraction**: Supply a `model` with an `.embed()` method
           and a dictionary of `DataLoaders` to extract embeddings automatically.

        You must use either embeddings (X/Y) OR model+loaders, but not both.

        ```python
        # Mode 1: Precomputed embeddings
        my_score = PyODScore(detector=KNN(n_neighbors=5))
        my_score.fit(X=train_embs, Y=val_embs)

        # Mode 2: On-the-fly extraction
        my_score = PyODScore(detector=KNN(n_neighbors=5))
        my_score.fit(model=model, loaders={"train": train_loader, "val": val_loader})
        ```

        Parameters
        ----------
        X:
            A `torch.tensor` with training sample embeddings. Required when not
            using `model` and `loaders`.
        Y:
            A `torch.tensor` with calibration sample embeddings. Optional.
        model:
            A `torch.nn.Module` with an `.embed()` method. Required when not
            using `X`.
        loaders:
            A `dict` with `DataLoader` objects. Required keys: `["train"]`.
            Optional key: `["val"]`. Required when using `model`.
        outdir:
            A `pathlib.Path` pointing to a directory for saving/loading embeddings.
            Only used with `model` and `loaders`.
        prefix:
            A `str` used as filename prefix for saved embeddings.
            Only used with `model` and `loaders`.
        q:
            A `float` or a `bool` indicating if the scores should be filtered to
            remove outliers from the training distribution. Defaults to `False`.
        """
        super().fit(
            X=X, Y=Y, model=model, loaders=loaders, outdir=outdir, prefix=prefix
        )
        self._fit_impl(q=q)

    def _fit_impl(self, q: float | None = None) -> None:
        """Fit implementation."""
        assert self.ref_embeddings is not None
        if self.cal_required:
            assert self.cal_embeddings is not None

        # TODO: serialize detector to disk to avoid refitting and ensure deterministic results

        if self.pca is not None:
            self._fit_pca()
            self.ref_embeddings = self.pca.transform(self.ref_embeddings)
            if self.cal_embeddings is not None:
                self.cal_embeddings = self.pca.transform(self.cal_embeddings)

        if q:
            assert (q >= 0.0) & (q <= 1.0)
            self.detector.fit(self.ref_embeddings.cpu().numpy())
            scores = torch.Tensor(self.detector.decision_scores_)
            threshold = torch.quantile(scores.float(), q=q)
            index = scores < threshold
            self.ref_embeddings = self.ref_embeddings[index, :]

        self.detector.fit(self.ref_embeddings.cpu().numpy())
        self.scores = torch.Tensor(self.detector.decision_scores_)
        self.set_trained()

        if self.cal_embeddings is not None:
            self.scores = torch.Tensor(
                self.detector.decision_function(
                    self.cal_embeddings.cpu().numpy()
                )
            )
            self.set_calibrated()

    @override
    @torch.inference_mode()
    def _score_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        """Compute a confidence score based on sample embeddings.

        Returns outlier scores where low values indicate likely inliers (samples
        similar to training) and high values indicate likely outliers (samples
        deviating from training).

        Once instantiated, the object can be called to return confidence
        scores based on sample embeddings.

        ```python
        my_score = PyODScore()
        my_score.fit(train_data, val_data)
        scores = my_score.score(test_data)
        ```

        Parameters
        ----------
        X:
            A `torch.tensor`or representing sample embeddings. Expected dimensions
            are (B,D).
        """
        assert self.detector is not None
        if self.pca is not None:
            X = self.pca.transform(X)
        score = torch.Tensor(self.detector.decision_function(X.cpu().numpy()))
        return score
