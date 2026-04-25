"""PCA based dimensionality reduction and confidence scoring."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from typing_extensions import override

from seapig.scores.embed import EmbeddingScore
from seapig.scores.utils import TensorPCA


class PCAScore(EmbeddingScore):
    """Returns confidence scores based on PCA reconstruction errors.

    Computes reconstruction error-based confidence scores where low scores indicate
    samples that can be well-reconstructed from principal components (likely inliers)
    and high scores indicate samples with large reconstruction errors (likely outliers).

    See https://arxiv.org/pdf/2402.02949v3 for the method description.

    Parameters
    ----------
    pca : TensorPCA, optional
        PCA configuration to use. Defaults to
        ``TensorPCA(n_components=0.50, gamma=3.0, M=4096)`` (RFF-PCA retaining
        50% explained variance).

    Examples
    --------
    ```python
    import torch
    from seapig.scores import PCAScore
    from seapig.scores.utils import TensorPCA
    score = PCAScore(pca=TensorPCA(n_components=0.90))
    score.fit(X=torch.randn(200, 64), Y=torch.randn(50, 64))
    score.set_threshold(q=0.95)
    result = score.select(X=torch.randn(10, 64))
    ```

    See Also
    --------
    seapig.scores.utils.TensorPCA : PCA implementation used internally.
    seapig.scores.knn.EuclideanScore : Alternative distance-based score.
    """

    ident = "pca"

    def __init__(
        self, pca: TensorPCA = TensorPCA(n_components=0.50, gamma=3.0, M=4096)
    ) -> None:
        super().__init__(pca=pca)

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
    ) -> None:
        """Train a confidence score based on sample embeddings.

        This method supports two usage modes:

        1. **Precomputed embeddings**: Supply training embeddings via `X` and
           optional calibration embeddings via `Y`.
        2. **On-the-fly extraction**: Supply a `model` with an `.embed()` method
           and a dictionary of `DataLoaders` to extract embeddings automatically.

        You must use either embeddings (X/Y) OR model+loaders, but not both.

        ```python
        # Mode 1: Precomputed embeddings
        from seapig.scores import PCAScore
        from seapig.scores.utils import TensorPCA
        my_score = PCAScore(pca=TensorPCA(n_components=0.90))
        my_score.fit(X=train_embs, Y=val_embs)

        # Mode 2: On-the-fly extraction
        my_score = PCAScore(pca=TensorPCA(n_components=0.90))
        my_score.fit(model=model, loaders={"train": train_loader, "val": val_loader})
        ```

        Parameters
        ----------
        X:
            A `torch.Tensor` with training sample embeddings. Required when not
            using `model` and `loaders`.
        Y:
            A `torch.Tensor` with calibration sample embeddings. Optional.
        model:
            A `torch.nn.Module` with an `.embed()` method. Required when not
            using `X`.
        loaders:
            A `dict` with `DataLoader` objects. Required keys: ``["train"]``.
            Optional key: ``["val"]``. Required when using `model`.
        outdir:
            A `pathlib.Path` pointing to a directory for saving/loading embeddings.
            Only used with `model` and `loaders`.
        prefix:
            A `str` used as filename prefix for saved embeddings.
            Only used with `model` and `loaders`.
        q:
            A `float` or `bool` indicating if outliers from the training
            distribution should be filtered before fitting. Defaults to `False`.
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

        if q:
            assert (q >= 0.0) & (q <= 1.0)
            assert self.pca is not None
            self._fit_pca()
            _, scores = self.pca.reconstruct(self.ref_embeddings)
            threshold = torch.quantile(scores.float(), q=q)
            index = scores < threshold
            self.ref_embeddings = self.ref_embeddings[index, :]

        self._fit_pca()
        self.set_trained()
        assert self.pca is not None

        if self.cal_embeddings is None:
            _, self.scores = self.pca.reconstruct(self.ref_embeddings)
        else:
            _, self.scores = self.pca.reconstruct(self.cal_embeddings)
            self.set_calibrated()

    @override
    def _score_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        """Compute a confidence score based on sample embeddings.

        Returns reconstruction error scores where low values indicate samples that
        can be well-reconstructed (likely inliers) and high values indicate samples
        with large reconstruction errors (likely outliers).

        Parameters
        ----------
        X:
            A `torch.Tensor` representing sample embeddings of shape ``(B, D)``.
        """
        assert self.pca is not None
        _, error = self.pca.reconstruct(X)
        return error
