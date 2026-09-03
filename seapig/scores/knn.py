"""KNN-based uncertainty scores."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from typing_extensions import override

from seapig.scores import EmbeddingScore
from seapig.scores.mixins import FAISSIndexMixin
from seapig.scores.utils import TensorPCA

__all__ = ["CosineScore", "EuclideanScore", "KNNScore", "MahalanobisScore"]


class KNNScore(EmbeddingScore, FAISSIndexMixin, ABC):
    """Abstract base class for KNN distance-based uncertainty scores.

    Computes distance-based uncertainty scores where low scores indicate samples
    similar to the training distribution (low uncertainty) and high scores indicate
    samples deviating from the training distribution (high uncertainty).

    Parameters
    ----------
    k : int, default 1
        Number of nearest neighbors used to compute the distance score.
    stat : {'max', 'mean', 'median', 'min'}, default 'max'
        Statistic applied to aggregate distances across the k neighbors.
    pca : `TensorPCA` or None, default None
        Optional `TensorPCA` object for dimensionality reduction prior to scoring.
    save_index : bool or Path, default False
        If `True`, the HNSW index is saved to a default file. If a `Path`
        is provided (must end in `.bin`), the index is saved there.

    See Also
    --------
    `scores.EuclideanScore`
    `scores.CosineScore`
    `scores.MahalanobisScore`
    """

    k: int = 1
    cal_embeddings: torch.Tensor | None
    index: Any | None = None
    index_path: Path | None = None

    def __init__(
        self,
        k: int = 1,
        stat: str = "max",
        pca: TensorPCA | None = None,
        save_index: bool | Path = False,
    ) -> None:
        super().__init__(pca=pca)
        assert stat in ["max", "mean", "median", "min"]
        self.stat: str = stat
        self.k = k
        self.ident: str = (
            f"{self.ident}-k{self.k}-{'full' if pca is None else 'pca'}"
        )
        if save_index:
            if isinstance(save_index, bool):
                self.index_path = Path(f"{self.ident}_index.bin")
            else:
                assert isinstance(save_index, Path)
                assert save_index.suffix == ".bin", (
                    "Index file must have a .bin extension"
                )
                save_index.parent.mkdir(parents=True, exist_ok=True)
                self.index_path = save_index

    @override
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
        self._setup_index()
        self.set_trained()
        if self.cal_embeddings is None:
            scores, _ = self._distance(self.ref_embeddings, offset=1)
            self.scores = self._stat(scores)
        else:
            scores, _ = self._distance(self.cal_embeddings, offset=0)
            self.scores = self._stat(scores)
            self.set_calibrated()

    def _filter_outliers(self, q: float | bool = False) -> None:
        if not q:
            return
        assert (q >= 0.0) & (q <= 1.0)
        assert self.ref_embeddings is not None
        if self.index is None:
            self._setup_index()  # temporary index
        scores, _ = self._distance(self.ref_embeddings, offset=1)
        scores = self._stat(scores)
        threshold = torch.quantile(scores.float(), q=q)
        index = scores < threshold
        self.ref_embeddings = self.ref_embeddings[index, :]

    @override
    def _score(self, X: torch.Tensor) -> torch.Tensor:
        """Compute an uncertainty score based on sample embeddings.

        Returns scores where low values indicate samples similar
        to the training data (low uncertainty) and high values indicate samples deviating
        from training data (high uncertainty).

        Parameters
        ----------
        X:
            A `torch.Tensor` representing sample embeddings of shape `(B, D)`.
        """
        assert self.index is not None, "Index must be built before scoring"
        if self.pca is not None:
            X = self.pca.transform(X)
        score, _ = self._distance(query=X)
        score = self._stat(score)
        return score.to(device=X.device)

    @abstractmethod
    def _setup_index(self) -> None:
        """Prepare an index for KNN search."""

    @abstractmethod
    def _distance(
        self, query: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the KNN distances and indices of a query against a populated index."""

    def knn_search(
        self, query: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the K-nearest-neighbour distances and indices for a set of query embeddings.

        Parameters
        ----------
        query : torch.Tensor
            A 2-D tensor of shape `(N, D)` containing the embeddings for which
            distances are to be computed.

        offset : int, default 0
            Number of nearest neighbours to discard from the result.  This is
            typically used to skip self-matching when the query points are
            drawn from the same set that built the index (e.g. `offset=1`).

        Returns
        -------
        distances : torch.Tensor
            A tensor of shape `(N, k)` containing the KNN distances for each
            query point after discarding the first `offset` nearest neighbours.

        indices : torch.Tensor
            A tensor of shape `(N, k)` with the index positions of the
            nearest neighbours in the reference embedding set after discarding
            the first `offset` matches.

        Notes
        -----
        - `offset` is useful when the query set is identical to the reference
          set, because the nearest neighbour would be the point itself (distance
          zero).  Skipping it yields a meaningful distance to the second
          nearest neighbour.
        """
        # Get raw distances and indices with the internal offset handling.
        distances, indices = self._distance(query=query, offset=offset)
        return distances, indices

    def _stat(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a statistic across the KNN distances."""
        assert self.stat in ["max", "mean", "median", "min"]
        if self.stat == "max":
            x = x.amax(1)
        if self.stat == "mean":
            x = x.mean(1)
        if self.stat == "median":
            x = x.median(1).values
        if self.stat == "min":
            x = x.amin(1)
        return x


class EuclideanScore(KNNScore):
    """Returns the KNN-distance based on the Euclidean distance to the nearest samples.

    Computes Euclidean distance-based uncertainty scores where low scores indicate
    samples similar to the training distribution (low uncertainty) and high scores
    indicate samples deviating from the training distribution (high uncertainty).

    Parameters
    ----------
    k : int, default 1
        Number of nearest neighbors to use.
    stat : {'max', 'mean', 'median', 'min'}, default 'max'
        Statistic to aggregate distances across the k neighbors.
    pca : `TensorPCA` or None, default None
        Optional `TensorPCA` object for dimensionality reduction prior to scoring.
    save_index : bool or Path, default False
        Whether (and where) to save the HNSW index to disk.

    Examples
    --------
    ```python
    import torch
    from seapig.scores import EuclideanScore
    score = EuclideanScore(k=5)
    score.fit(X=torch.randn(200, 64), Y=torch.randn(50, 64))
    score.set_threshold(q=0.95)
    result = score.select(X=torch.randn(10, 64))
    ```

    See Also
    --------
    `scores.KNNScore`
    `scores.CosineScore`
    `scores.MahalanobisScore`
    """

    k: int
    ident: str = "euclidean"

    def __init__(
        self,
        k: int = 1,
        stat: str = "max",
        pca: TensorPCA | None = None,
        save_index: bool | Path = False,
    ) -> None:
        super().__init__(k=k, stat=stat, pca=pca, save_index=save_index)

    @override
    def _setup_index(self) -> None:
        """Initialize an index based on reference embeddings."""
        assert isinstance(self.ref_embeddings, torch.Tensor)
        self._build_index(self.ref_embeddings)

    @override
    @torch.inference_mode()
    def _distance(
        self, query: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the KNN distance of a query against a populated index."""
        squared_distances, indices = self._query_index(query, offset)
        return (torch.sqrt(squared_distances), indices)


class CosineScore(KNNScore):
    """Returns the KNN-distance based on the cosine distance to the nearest samples.

    Computes cosine distance-based uncertainty scores where low scores indicate
    samples similar to the training distribution (low uncertainty) and high scores
    indicate samples deviating from the training distribution (high uncertainty).

    The cosine distance is computed as `(1 - cosine_similarity)`, with a range
    of `[0, 2]` where `0` indicates identical vectors, `1` indicates orthogonal
    vectors, and `2` indicates opposite vectors.

    Parameters
    ----------
    k : int, default 1
        Number of nearest neighbors to use.
    stat : {'max', 'mean', 'median', 'min'}, default 'max'
        Statistic to aggregate distances across the k neighbors.
    pca : `TensorPCA` or None, default None
        Optional `TensorPCA` object for dimensionality reduction prior to scoring.
    save_index : bool or Path, default False
        Whether (and where) to save the HNSW index to disk.

    See Also
    --------
    `scores.KNNScore`
    `scores.EuclideanScore`
    `scores.MahalanobisScore`
    """

    k: int = 1
    ident: str = "cosine"

    def __init__(
        self,
        k: int = 1,
        stat: str = "max",
        pca: TensorPCA | None = None,
        save_index: bool | Path = False,
    ) -> None:
        super().__init__(k=k, stat=stat, pca=pca, save_index=save_index)

    @override
    def _setup_index(self) -> None:
        """Initialize an index based on reference embeddings."""
        assert isinstance(self.ref_embeddings, torch.Tensor)
        normalized = torch.nn.functional.normalize(self.ref_embeddings)
        self._build_index(normalized)

    @override
    @torch.inference_mode()
    def _distance(
        self, query: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the KNN cosine distance of a query against a populated index.

        Uses FAISS HNSW index on L2 distances of normalized vectors and converts
        squared L2 distances to cosine distance via ``0.5 * d2``. Statistic is applied
        after conversion.
        """
        assert self.index is not None
        # Normalize query vectors
        normalized = torch.nn.functional.normalize(query)
        distances, indices = self._query_index(normalized, offset)
        # Convert to cosine distance
        cosine_dist = 0.5 * distances
        return (cosine_dist, indices)


class MahalanobisScore(KNNScore):
    """Returns the Mahalanobis distance to the training samples distribution.

    Computes Mahalanobis distance-based uncertainty scores where low scores indicate
    samples similar to the training distribution (low uncertainty) and high scores
    indicate samples deviating from the training distribution (high uncertainty).

    The Mahalanobis distance accounts for correlations in the training data by
    whitening the embeddings with the Cholesky factor of the training covariance
    matrix prior to computing Euclidean nearest-neighbour distances.

    Parameters
    ----------
    k : int, default 1
        Number of nearest neighbors to use.
    stat : {'max', 'mean', 'median', 'min'}, default 'max'
        Statistic to aggregate distances across the k neighbors.
    pca : `TensorPCA` or None, default None
        Optional `TensorPCA` object for dimensionality reduction prior to scoring.
    save_index : bool or Path, default False
        Whether (and where) to save the HNSW index to disk.

    See Also
    --------
    `scores.KNNScore`
    `scores.EuclideanScore`
    `scores.CosineScore`
    """

    k: int
    vi_zero: torch.Tensor
    ident: str = "mahalanobis"

    def __init__(
        self,
        k: int = 1,
        stat: str = "max",
        pca: TensorPCA | None = None,
        save_index: bool | Path = False,
    ) -> None:
        super().__init__(k=k, stat=stat, pca=pca, save_index=save_index)
        self.register_buffer("vi_zero", None)

    @override
    def _setup_index(self) -> None:
        """Initialize an index based on reference embeddings."""
        assert isinstance(self.ref_embeddings, torch.Tensor)
        cov_zero = self.ref_embeddings.T.cov()
        self.vi_zero = torch.linalg.inv(torch.linalg.cholesky(cov_zero))
        transformed = self.ref_embeddings @ self.vi_zero.T
        self._build_index(transformed)

    @override
    @torch.inference_mode()
    def _distance(
        self, query: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the Mahalanobis distance of a query against a populated index."""
        assert self.index is not None
        transformed = query.float() @ self.vi_zero.T
        distances, indices = self._query_index(transformed, offset)
        return torch.sqrt(distances), indices
