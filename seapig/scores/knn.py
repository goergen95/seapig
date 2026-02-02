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

    Computes distance-based confidence scores where low scores indicate samples
    similar to the training distribution (likely inliers) and high scores indicate
    samples deviating from the training distribution (likely outliers).

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
        Low scores indicate likely inliers, high scores indicate likely outliers.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Samples with scores higher
        than this threshold are excluded from prediction. Defaults to `None`.
    """

    k: int = 1
    index: faiss.Index | None = None
    ref_embeddings: torch.Tensor
    cal_embeddings: torch.Tensor | None

    def __init__(
        self, k: int = 1, stat: str = "max", exp_var: float | bool = False
    ) -> None:
        super().__init__(exp_var=exp_var)
        assert stat in ["max", "mean", "median", "min"]
        self.stat: str = stat
        self.k = k
        self.ident: str = (
            f"{self.ident}-k{self.k}-{'full' if exp_var else 'pca'}"
        )

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
        self._fit_impl(q=q)

    def _fit_impl(self, q: float | None = None) -> None:
        """Fit implementation."""
        assert self.ref_embeddings is not None
        self = self.to(device=self.ref_embeddings.device)
        if self.cal_required:
            assert self.cal_embeddings is not None

        if self.exp_var:
            self._fit_pca()
            assert self.pca is not None
            self.ref_embeddings = self.pca.predict(self.ref_embeddings)
            if self.cal_embeddings is not None:
                self.cal_embeddings = self.pca.predict(self.cal_embeddings)

        if q:
            assert (q >= 0.0) & (q <= 1.0)
            if self.index is None:
                self._setup_index()
            scores = self._distance(self.ref_embeddings, kpn=1)
            threshold = torch.quantile(scores.float(), q=q)
            index = scores < threshold
            self.ref_embeddings = self.ref_embeddings[index, :]

        self._setup_index()
        self.set_trained()

        if self.cal_embeddings is None:
            self.scores = self._distance(self.ref_embeddings, kpn=1)
        else:
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

        Returns scores where low values indicate likely inliers (samples similar
        to training) and high values indicate likely outliers (samples deviating
        from training).

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
        self.to(device=X.device)
        if self.pca is not None:
            X = self.pca.predict(X)
        score = self._distance(query=X)
        return score

    @classmethod
    def _stat(self, x: torch.Tensor, stat: str = "max") -> torch.Tensor:
        assert stat in ["max", "mean", "median", "min"]
        if stat == "max":
            x = x.amax(1)
        if stat == "mean":
            x = x.mean(1)
        if stat == "median":
            x = x.median(1).values
        if stat == "min":
            x = x.amin(1)
        return x


class EuclideanScore(KNNScore):
    """Returns the KNN-distance based on the euclidean distance to the nearest samples.

    Computes Euclidean distance-based confidence scores where low scores indicate
    samples similar to the training distribution (likely inliers) and high scores
    indicate samples deviating from the training distribution (likely outliers).

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
        Low scores indicate likely inliers, high scores indicate likely outliers.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Samples with scores higher
        than this threshold are excluded from prediction. Defaults to `None`.
    """

    k: int
    ident: str = "euclidean"

    def __init__(
        self, k: int = 1, stat: str = "max", exp_var: float | bool = False
    ) -> None:
        super().__init__(k=k, stat=stat, exp_var=exp_var)

    @override
    def _setup_index(self) -> None:
        """Initialize faiss index based on embeddings."""
        assert isinstance(self.ref_embeddings, torch.Tensor)
        self.index = faiss.IndexFlatL2(self.ref_embeddings.shape[1])
        self.index.add(self.ref_embeddings.cpu())

    @override
    @torch.inference_mode()
    def _distance(self, query: torch.Tensor, kpn: int = 0) -> torch.Tensor:
        assert self.index is not None
        dist, _ = self.index.search(query.cpu(), k=self.k + kpn)
        dist = torch.Tensor(dist[:, kpn:])
        dist = self._stat(dist, stat=self.stat)
        return torch.sqrt(dist)


class CosineScore(KNNScore):
    """Returns the KNN-distance based on the cosine distance to the nearest samples.

    Computes cosine distance-based confidence scores where low scores indicate
    samples similar to the training distribution (likely inliers) and high scores
    indicate samples deviating from the training distribution (likely outliers).

    The cosine distance is calculated as (1 - cosine_similarity), with a range
    of [0, 2] where 0 indicates identical vectors, 1 indicates orthogonal
    vectors, and 2 indicates opposite vectors.

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
        Low scores indicate likely inliers, high scores indicate likely outliers.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Samples with scores higher
        than this threshold are excluded from prediction. Defaults to `None`.
    """

    k: int = 1
    ident: str = "cosine"

    def __init__(
        self, k: int = 1, stat: str = "max", exp_var: float | bool = False
    ) -> None:
        super().__init__(k=k, stat=stat, exp_var=exp_var)

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
        assert self.index is not None
        query = torch.nn.functional.normalize(query).cpu()
        similarity, _ = self.index.search(query, k=self.k + kpn)
        similarity = torch.Tensor(similarity)[:, kpn:]
        # Convert cosine similarity to cosine distance (1 - similarity)
        # Low distance = inlier, high distance = outlier
        dist = 1 - similarity
        dist = self._stat(dist, stat=self.stat)
        return dist


class MahalanobisScore(KNNScore):
    """Returns the Mahalanobis distance to the training samples distribution.

    Computes Mahalanobis distance-based confidence scores where low scores indicate
    samples similar to the training distribution (likely inliers) and high scores
    indicate samples deviating from the training distribution (likely outliers).

    The Mahalanobis distance accounts for correlations in the training data by
    using the covariance matrix of the training embeddings.

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
        Low scores indicate likely inliers, high scores indicate likely outliers.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Samples with scores higher
        than this threshold are excluded from prediction. Defaults to `None`.
    """

    k: int
    vi_zero: torch.Tensor
    ident: str = "mahalanobis"

    def __init__(
        self, k: int = 1, stat: str = "max", exp_var: float | bool = False
    ) -> None:
        super().__init__(k=k, stat=stat, exp_var=exp_var)
        self.register_buffer("vi_zero", None)

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
        assert self.index is not None
        query = query.cpu() @ self.vi_zero.T.cpu()
        dist, _ = self.index.search(query, k=self.k + kpn)
        dist = torch.Tensor(dist[:, kpn:])
        dist = self._stat(dist, stat=self.stat)
        return dist
