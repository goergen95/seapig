"""Uncertainty score based on an arbitrary PyOD model."""

import torch
from typing_extensions import override

from seapig.scores.embed import EmbeddingScore
from seapig.scores.utils import TensorPCA

try:
    from pyod.models.base import BaseDetector
except ImportError:  # pragma: no cover
    raise ImportError(
        "pyod is not installed. Please install it with `pip install pyod`."
    )


class PyODScore(EmbeddingScore):
    """Uncertainty scores based on detectors supplied by PyOD.

    Computes outlier scores using PyOD detectors where low scores indicate samples
    similar to the training distribution (low uncertainty) and high scores indicate
    samples deviating from the training distribution (high uncertainty).

    Parameters
    ----------
    detector : pyod.models.base.BaseDetector
        A fitted or unfitted PyOD detector instance. Any detector from the
        `pyod` library that implements `fit` and `decision_function`
        is supported (e.g., `pyod.models.knn.KNN`).
    pca : TensorPCA or None, default None
        Optional PCA for dimensionality reduction prior to scoring.

    Notes
    -----
    Requires the optional `pyod` dependency. Install with:
    ```
    pip install pyod
    ```

    See Also
    --------
    `scores.EmbeddingScore`
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

    def _fit(self, q: bool | float = False) -> None:
        """Train a uncertainty score based on sample embeddings.

        Parameters
        ----------
        q:
            A `float` or `bool` indicating if outliers from the training
            distribution should be filtered before fitting. Defaults to `False`.
        """
        assert self.ref_embeddings is not None
        if self.cal_required:
            assert self.cal_embeddings is not None
        self._apply_pca()
        self._filter_outliers(q=q)
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

    def _filter_outliers(self, q: float | bool = False) -> None:
        if not q:
            return
        assert (q >= 0.0) & (q <= 1.0)
        self.detector.fit(self.ref_embeddings.cpu().numpy())
        scores = torch.Tensor(self.detector.decision_scores_)
        threshold = torch.quantile(scores.float(), q=q)
        index = scores < threshold
        assert isinstance(self.ref_embeddings, torch.Tensor)
        self.ref_embeddings = self.ref_embeddings[index, :]

    @override
    @torch.inference_mode()
    def _score(self, X: torch.Tensor) -> torch.Tensor:
        """Compute an uncertainty score based on sample embeddings.

        Returns uncertainty scores where low values indicate samples
        similar to the training distribution (low uncertainty) and high values indicate
        samples deviating from the training distribution (high uncertainty).

        Parameters
        ----------
        X:
            A `torch.Tensor` representing sample embeddings of shape `(B, D)`.
        """
        assert self.detector is not None
        if self.pca is not None:
            X = self.pca.transform(X)
        score = torch.Tensor(self.detector.decision_function(X.cpu().numpy()))
        return score
