"""Confidence score based on an arbitrary PyOD model."""

from pathlib import Path
from typing import override

import torch
from pyod.models.base import BaseDetector
from torch.utils.data import DataLoader

from seapig.scores.embed import EmbeddingScore


class PyODScore(EmbeddingScore):
    """Confidence Scores based on detectors supplied by PyOD.

    Computes outlier scores using PyOD detectors where low scores indicate samples
    similar to the training distribution (likely inliers) and high scores indicate
    samples deviating from the training distribution (likely outliers).

    Parameters
    ----------
    detector:
        An `BaseDetector` instance from PyOD.
    exp_var:
        A `float` indicating the percentage of explained variance to retain
        if dimensionality reduction via PCA shall be applied. Defaults to `False`,
        indicating that dimensionality reduction is not applied.

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
        self, detector: BaseDetector, exp_var: float | bool = False
    ) -> None:
        super().__init__(exp_var=exp_var)
        self.detector = detector
        self.ident = f"{self.ident}-{detector.__class__.__name__}"

    @override
    def fit(
        self, X: torch.Tensor, Y: torch.Tensor | None, q: bool | float = False
    ) -> None:
        """Train a confidence score based on samples from a `torch.utils.data.DataLoader`.

        The train method retrieves embeddings for all samples from a DataLoader
        that is expected to represent training samples. These embeddings are
        later used to calculate the KNN-distances for query samples.

        ```python
        my_score = PyODScore(detector=KNN(n_neighbors=5))
        my_score.fit(train_embs, val_embs)
        ```

        Parameters
        ----------
        X:
            A `torch.tensor` or an `np.Array` with samples representing training
            samples.
        Y:  A `torch.tensor` or an `np.Array` with samples representing calibration
            samples.
        q:
            A `float` or a `bool` indicating if the scores should be filtered to
            remove outliers from the training distribution. Defaults to `False`.
        """
        super().fit(X=X, Y=Y)
        self._fit_impl(q=q)

    @override
    def fit_dl(
        self,
        model: torch.nn.Module,
        loaders: dict[str, DataLoader[torch.Tensor | dict[str, torch.Tensor]]],
        outdir: Path | None = None,
        prefix: str | None = None,
        q: bool | float = False,
    ) -> None:
        """Train a confidence score based on samples from a `DataLoader`.

        Training embeddings are extracted from the supplied models and the data
        loader with the `"train"` key in the supplied `loaders` argument.
        Calibration embeddings are extracted from the `DataLoader` object with
        the `"val"` key. The confidence score is then calibrated based on the
        extracted embeddings.

        ```python
        my_score = PyODScore()
        my_score.fit_dl(model=model, loaders={"train": train_loader, "val": val_loader})
        ```

        Parameters
        ----------
        model:
            A torch.nn.Module representing a trained model instance. It is
            required to have an `.embed()` method.
        loaders:
            A `dict`ionary with dataloader objects with required keys `["train", "val"]`.
            The `DataLoaders` are expected to return `torch.Tensor`s or a `dict`
            of `torch.Tensor`s with the `"image"` key present.
        outdir:
            A `pathlib.Path` object pointing towards a directory, by default `None`.
            If specified, embeddings are read to disk, if previously written. Otherwise,
            embeddings will be written to disk.
        prefix:
            A `str`ing used as filename prefix to save embeddings, by default
            `None`. See `outdir` parameter above.
        q:
            A `float` or a `bool` indicating if the scores should be filtered to
            remove outliers from the training distribution. Defaults to `False`.
        """
        super().fit_dl(model, loaders, outdir, prefix)
        self._fit_impl(q=q)

    def _fit_impl(self, q: float | None = None) -> None:
        """Fit implementation."""
        assert self.ref_embeddings is not None
        if self.cal_required:
            assert self.cal_embeddings is not None

        # TODO: serialize detector to disk to avoid refitting and ensure deterministic results

        if self.exp_var:
            self._fit_pca()
            assert self.pca is not None
            self.ref_embeddings = self.pca.predict(self.ref_embeddings)
            if self.cal_embeddings is not None:
                self.cal_embeddings = self.pca.predict(self.cal_embeddings)

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
    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def score(self, X: torch.Tensor) -> torch.Tensor:
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
            X = self.pca.predict(X)
        score = torch.Tensor(self.detector.decision_function(X.cpu().numpy()))
        return score
