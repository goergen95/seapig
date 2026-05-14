"""KNN-based confidence scores."""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing_extensions import override

from seapig.scores.embed import EmbeddingScore
from seapig.scores.utils import TensorPCA

__all__ = ["KNNScore", "EuclideanScore", "CosineScore", "MahalanobisScore"]


class KNNScore(EmbeddingScore, ABC):
    """Abstract base class for KNN distance-based confidence scores.

    Computes distance-based confidence scores where low scores indicate samples
    similar to the training distribution (likely inliers) and high scores indicate
    samples deviating from the training distribution (likely outliers).

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
        from seapig.scores import EuclideanScore
        my_score = EuclideanScore(k=2)
        my_score.fit(X=train_embs, Y=val_embs)

        # Mode 2: On-the-fly extraction
        my_score = EuclideanScore(k=2)
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
            A `dict` with `DataLoader` objects. Required keys: `["train"]`.
            Optional key: `["val"]`. Required when using `model`.
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

        if self.pca is not None:
            self._fit_pca()
            self.ref_embeddings = self.pca.transform(self.ref_embeddings)
            if self.cal_embeddings is not None:
                self.cal_embeddings = self.pca.transform(self.cal_embeddings)

        if q:
            assert (q >= 0.0) & (q <= 1.0)
            if self.index is None:
                self._setup_index()
            scores, _ = self._distance(self.ref_embeddings, offset=1)
            scores = self._stat(scores)
            threshold = torch.quantile(scores.float(), q=q)
            index = scores < threshold
            self.ref_embeddings = self.ref_embeddings[index, :]

        self._setup_index()
        self.set_trained()

        if self.cal_embeddings is None:
            scores, _ = self._distance(self.ref_embeddings, offset=1)
            self.scores = self._stat(scores)
        else:
            scores, _ = self._distance(self.cal_embeddings, offset=0)
            self.scores = self._stat(scores)
            self.set_calibrated()

    @override
    def _score_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        """Compute a confidence score based on sample embeddings.

        Returns scores where low values indicate likely inliers (samples similar
        to training) and high values indicate likely outliers (samples deviating
        from training).

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

    @staticmethod
    def _suggest_build_params(embs: torch.Tensor, k: int = 1) -> dict[str, Any]:
        """Suggest parameters for HNSW index."""
        if embs.dim() != 2:
            raise ValueError("ref_embeddings must be 2D (N, D)")
        n, d = map(int, embs.shape)

        if d <= 64:
            M = 16
        elif d <= 128:  # pragma: no cover
            M = 24
        elif d <= 256:  # pragma: no cover
            M = 32
        elif d <= 512:  # pragma: no cover
            M = 48
        elif d <= 1024:  # pragma: no cover
            M = 48
        else:  # pragma: no cover
            M = 64

        # adjust upward for large n
        if n > 5_000_000:  # pragma: no cover
            M = max(M, 32)
        if n > 50_000_000:  # pragma: no cover
            M = max(M, 48)

        # adjust downward for very small n
        if n < 10_000:
            M = min(M, 16)

        # ef_construction based on M
        C = max(4 * M, 128)
        # cap at a high maximum value
        C = min(C, 1024)

        return {"M": M, "efConstruction": C}

    @staticmethod
    def _suggest_query_params(embs: torch.Tensor, k: int = 1) -> dict[str, Any]:
        """Suggest query parameters for HNSW index."""
        params = KNNScore._suggest_build_params(embs, k)

        S = max(k * 8, 512)
        S = min(S, params["efConstruction"])
        S = max(S, k)

        return {"efSearch": S}

    def _build_index(self, embs: torch.Tensor) -> None:
        """Build an index based on reference embeddings.

        The embeddings can be preprocessed (e.g. normalized or transformed) before
        being passed to this method. The `space` parameter is kept for API compatibility
        but FAISS only supports L2 metric; for cosine we use normalized vectors.
        """
        assert isinstance(embs, torch.Tensor)
        index_path = self.index_path
        params = self._suggest_build_params(embs=embs, k=self.k)
        d = embs.shape[1]
        N = embs.shape[0]
        if N <= 10_000:
            index = faiss.IndexFlatL2(d)  # type: ignore[possibly-missing-attribute]
        else:
            M = params["M"]
            ef_construction = params["efConstruction"]
            # FAISS HNSW index with L2 metric (also works for normalized vectors to emulate cosine)
            index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)  # type: ignore[possibly-missing-attribute]
            index.hnsw.efConstruction = ef_construction
        # Build or load index
        if index_path is None or not Path(index_path).exists():
            embs_np: np.ndarray = embs.cpu().numpy().astype(np.float32)
            index.add(embs_np)
            if index_path:
                faiss.write_index(index, str(index_path))  # type: ignore[possibly-missing-attribute]
        else:
            warnings.warn(
                f"Index file {index_path} already exists. Loading existing index from disk.",
                UserWarning,
            )
            index = faiss.read_index(str(index_path))  # type: ignore[possibly-missing-attribute]
        self.index_params = params
        self.index = index

    def _query_index(
        self, query: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query the FAISS HNSW index for KNN distances.

        Retrieves `k + offset` nearest neighbors, applies the query-time `efSearch`
        parameter, and returns a tuple of (distances, indices) where
        distances are KNN distances and indices are the corresponding neighbor indices
        in the reference embeddings.
        """
        assert self.index is not None, "Index must be built before querying"
        # Set query‑time efSearch of hnsw index
        if isinstance(self.index, faiss.IndexHNSW):  # type: ignore[possibly-missing-attribute]
            params = KNNScore._suggest_query_params(query, self.k + offset)
            ef_search = params.get("efSearch", self.index.hnsw.efSearch)
            self.index.hnsw.efSearch = ef_search
        # check that dims match
        index_d = self.index.d
        query_d = query.shape[1]
        if index_d != query_d:
            raise ValueError(
                f"Query dimension {query_d} does not match index dimension {index_d}"
            )
        # Perform search on CPU with numpy arrays
        query_np = query.cpu().numpy().astype(np.float32)
        # returns (distances, indices) as numpy arrays
        search_results = self.index.search(query_np, self.k + offset)
        # convert to torch tensors on the same device as query
        search_results = map(
            lambda x: torch.from_numpy(x).to(query.device), search_results
        )
        # Discard the first `offset` entries
        distances, indices = map(lambda x: x[:, offset:], search_results)
        return (distances, indices)

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

    Computes Euclidean distance-based confidence scores where low scores indicate
    samples similar to the training distribution (likely inliers) and high scores
    indicate samples deviating from the training distribution (likely outliers).

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

    Computes cosine distance-based confidence scores where low scores indicate
    samples similar to the training distribution (likely inliers) and high scores
    indicate samples deviating from the training distribution (likely outliers).

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

    Computes Mahalanobis distance-based confidence scores where low scores indicate
    samples similar to the training distribution (likely inliers) and high scores
    indicate samples deviating from the training distribution (likely outliers).

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
