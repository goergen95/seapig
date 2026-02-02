"""PCA based dimensionality reduction and confidence scoring."""

from pathlib import Path
from typing import override

import torch
from torch.utils.data import DataLoader

from seapig.scores.embed import EmbeddingScore
from seapig.scores.utils import TensorPCA


class PCAScore(EmbeddingScore):
    """Returns a confidence scores based on PCA-based reconstruction errors.

    Computes reconstruction error-based confidence scores where low scores indicate
    samples that can be well-reconstructed from principal components (likely inliers)
    and high scores indicate samples with large reconstruction errors (likely outliers).

    See https://arxiv.org/pdf/2402.02949v3

    Parameters
    ----------
    exp_var:
        A `float` indicating the explained variance to keep when PCA-based
        dimensionality reduction shall be applied. Defaults to `False`, meaning
        that no dimensionality reduction will be conducted.
    """

    ident = "pca"

    def __init__(
        self,
        exp_var: float = 0.50,
        gamma: float | None = 3.0,
        M: int | None = 4096,
    ) -> None:
        super().__init__(exp_var=exp_var)
        self.pca = TensorPCA(exp_var=exp_var, gamma=gamma, M=M)
        self.ident = f"{self.ident}-{exp_var}"

    @override
    def fit(
        self,
        X: torch.Tensor,
        Y: torch.Tensor | None = None,
        q: bool | float = False,
    ) -> None:
        """Train a confidence score based on sample embeddings.

        Training embeddings are required to be supplied as a `torch.tensor` with
        parameter `X`. Calibration embeddings are supplied with the `Y` parameter.
        As an alternative, use `fit_dl()` to supply a model with an `.embed()` method
        and a dictionary with `DataLoaders` to extract embeddings on the fly.

        These are later used to retrieve confidence scores for query samples.

        ```python
        my_score = PCAScore(exp_var = 0.90)
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
        my_score = PCAScore(exp_var = 0.90)
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
        self.to(device=self.ref_embeddings.device)
        if self.cal_required:
            assert self.cal_embeddings is not None

        if q:
            assert (q >= 0.0) & (q <= 1.0)
            self._fit_pca()
            assert self.pca is not None
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
    def score(self, X: torch.Tensor) -> torch.Tensor:
        """Compute a confidence score based on sample embeddings.

        Returns reconstruction error scores where low values indicate samples that
        can be well-reconstructed (likely inliers) and high values indicate samples
        with large reconstruction errors (likely outliers).

        Once instantiated, the object can be called to return confidence
        scores based on sample embeddings.

        ```python
        my_score = PCAScore()
        my_score.fit(train_data, val_data)
        scores = my_score.score(test_data)
        ```

        Parameters
        ----------
        X:
            A `torch.tensor`or representing sample embeddings. Expected dimensions
            are (B,D).
        """
        assert self.pca is not None
        self.to(device=X.device)
        _, error = self.pca.reconstruct(X)
        return error
