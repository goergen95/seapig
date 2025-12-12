"""KNN-based confidence scores."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

import faiss
import torch
from torch.utils.data import DataLoader

from seapig.scores.embed import EmbeddingScore


class KNNScore(EmbeddingScore, ABC):
    """Returns the KNN-distance to the nearest samples.

    Parameters
    ----------
    k:
        An `int`eger indicating the number of neighbors to calculate the distance.
        Defaults to 1, e.g. the distance to the closest neighbor.
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
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Defaults to `None`.
    """

    k: int = 1
    train_required: bool = True
    cal_required: bool = True
    threshold: torch.Tensor | None = None
    scores: torch.Tensor
    index: faiss.IndexFlatL2  # type: ignore [no-any-unimported]

    def __init__(self, k: int = 1, exp_var: float | bool = False) -> None:
        super().__init__()
        self.k = k
        self.exp_var = exp_var

    @override
    def fit(
        self, X: torch.Tensor, Y: torch.Tensor | None, q: bool | float = False
    ) -> None:
        """Train a confidence score based on sample embeddings.

        Training embeddings are required to be supplied as a `torch.tensor` with
        parameter `X`. Calibration embeddings are supplied with the `Y` parameter.
        As an alternative, use `fit_dl()` to supply a model with an `.embed()` method
        and a dictionary with `DataLoaders` to extract embeddings on the fly.

        These are later used to retrieve confidence scores for query samples.

        ```python
        my_score = EmbeddingScore(k=2)
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
        my_score = KNNScore(k=2)
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
        self._fit_impl(q=q, outdir=outdir, prefix=prefix)

    def _fit_impl(
        self,
        q: float | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> None:
        """Fit implementation."""
        path = None
        reset_index = False
        assert self.ref_embeddings is not None
        if self.cal_required:
            assert self.cal_embeddings is not None

        if prefix is not None:
            path = self._setup_path(outdir, prefix + f"-{self.ident}-scores")

        if path is not None and path.is_file():
            print(f"Loading pre-existing scores from {path}.")
            self.scores = self._load_parquet(path)
        else:
            self._setup_index()
            self.scores = self._distance(self.ref_embeddings, kpn=1)
            if path is not None:
                self._write_parquet(x=self.scores, path=path)

        if q:
            assert (q >= 0.0) & (q <= 1.0)
            threshold = torch.quantile(self.scores.float(), q=q)
            index = self.scores < threshold
            self.ref_embeddings = self.ref_embeddings[index, :]
            reset_index = True

        if self.exp_var is not None:
            self._fit_pca()
            assert self.pca is not None
            self.ref_embeddings = self.pca.predict(self.ref_embeddings)
            if self.cal_embeddings is not None:
                self.cal_embeddings = self.pca.predict(self.cal_embeddings)
            reset_index = True

        if reset_index:
            self._setup_index()
            self.scores = self._distance(self.ref_embeddings, kpn=1)

        self.set_trained()

        if self.cal_embeddings is not None:
            self.scores = self._distance(self.cal_embeddings, kpn=0)
            self.set_calibrated()

    @abstractmethod
    def _setup_index(self) -> None:
        """Prepare an index for KNN search."""
        pass

    @abstractmethod
    def _distance(self, query: torch.Tensor, kpn: int = 0) -> torch.Tensor:
        """Calculate the KNN distance of a query against a populated index."""
        pass

    @override
    def score(self, X: torch.Tensor) -> torch.Tensor:
        """Compute a confidence score based on sample embeddings.

        Once instantiated, the object can be called to return confidence
        scores based on sample embeddings.

        ```python
        my_score = KNNScore()
        my_score.fit(train_data, val_data)
        scores = my_score.score(test_data)
        ```

        Parameters
        ----------
        X:
            A `torch.tensor`or representing sample embeddings. Expected dimensions
            are (B,D).
        """
        assert self.index is not None
        if self.pca is not None:
            X = self.pca.predict(X)
        score = self._distance(query=X)
        return score


class EuclideanScore(KNNScore):
    """Returns the KNN-distance based on the euclidean distance to the nearest samples.

    Parameters
    ----------
    k:
        An `int`eger indicating the number of neighbors to calculate the distance.
        Defaults to 1, e.g. the distance to the closest neighbor.
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
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Defaults to `None`.
    """

    k: int
    ident = "euclidean"

    def __init__(self, k: int = 1, exp_var: float | bool = False) -> None:
        super().__init__(k=k, exp_var=exp_var)
        self.ident = self.ident + f"-k{self.k}"

    @override
    def _setup_index(self) -> None:
        """Initialize faiss index based on embeddings."""
        assert isinstance(self.ref_embeddings, torch.Tensor)
        self.index = faiss.IndexFlatL2(self.ref_embeddings.shape[1])
        self.index.add(self.ref_embeddings.cpu())

    @override
    @torch.inference_mode()
    def _distance(self, query: torch.Tensor, kpn: int = 0) -> torch.Tensor:
        dist, _ = self.index.search(query.cpu(), k=self.k + kpn)
        dist = torch.Tensor(dist[:, kpn:].mean(1))
        return torch.sqrt(dist)


class CosineScore(KNNScore):
    """Returns the KNN-distance based on the cosine distance to the nearest samples.

    Parameters
    ----------
    k:
        An `int`eger indicating the number of neighbors to calculate the distance.
        Defaults to 1, e.g. the distance to the closest neighbor.
    abs:
        A `bool`ean indicating if the absolute cosine distance should be returned,
        by default `True`.
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
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Defaults to `None`.
    """

    k: int = 1
    abs: bool
    ident = "cosine"

    def __init__(
        self, k: int = 1, abs: bool = True, exp_var: float | bool = False
    ) -> None:
        super().__init__(k=k, exp_var=exp_var)
        self.abs = abs
        self.ident = self.ident + f"-k{self.k}"

    @override
    def _setup_index(self) -> None:
        """Initialize faiss index based on embeddings."""
        assert isinstance(self.ref_embeddings, torch.Tensor)
        self.index = faiss.IndexFlatIP(self.ref_embeddings.shape[1])
        self.index.add(torch.nn.functional.normalize(self.ref_embeddings).cpu())
        return

    @override
    @torch.inference_mode()
    def _distance(self, query: torch.Tensor, kpn: int = 0) -> torch.Tensor:
        query = torch.nn.functional.normalize(query).cpu()
        dist, _ = self.index.search(query, k=self.k + kpn)
        dist = torch.Tensor(dist)
        if self.abs:
            dist = dist.abs()
        dist = dist[:, kpn:].mean(1)
        return torch.tensor(1 - dist)


class MahalanobisScore(KNNScore):
    """Returns the Mahalanobis distance to the training samples distribution.

    Parameters
    ----------
    k:
        An `int`eger indicating the number of neighbors to calculate the distance.
        Defaults to 1, e.g. the distance to the closest neighbor.
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
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Defaults to `None`.
    """

    k: int
    vi_zero: torch.Tensor
    train_required: bool = True
    cal_required: bool = True
    threshold: torch.Tensor | None = None
    scores: torch.Tensor
    ident = "mahalanobis"

    def __init__(self, k: int = 1, exp_var: float | bool = False) -> None:
        super().__init__(k=k, exp_var=exp_var)
        self.ident = self.ident + f"-k{self.k}"

    @override
    def _setup_index(self) -> None:
        """Initialize faiss index based on embeddings."""
        assert isinstance(self.ref_embeddings, torch.Tensor)
        cov_zero = self.ref_embeddings.T.cov()
        self.vi_zero = torch.linalg.inv(torch.linalg.cholesky(cov_zero))
        self.index = faiss.IndexFlatL2(self.ref_embeddings.shape[1])
        self.index.add((self.ref_embeddings @ self.vi_zero.T).cpu())

    @override
    @torch.inference_mode()
    def _distance(self, query: torch.Tensor, kpn: int = 0) -> torch.Tensor:
        query = query.cpu() @ self.vi_zero.T.cpu()
        dist, _ = self.index.search(query, k=self.k + kpn)
        return torch.Tensor(dist[:, kpn:].mean(1))
