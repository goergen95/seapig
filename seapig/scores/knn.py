"""KNN-based confidence scores."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from typing_extensions import override

from seapig.scores.embed import EmbeddingScore
from seapig.scores.index_handler import FaissIndexHandler, IndexHandler
from seapig.scores.utils import TensorPCA


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
        If `True`, the index is saved to a default file. If a `Path`
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
    index_params: dict[str, Any] | None = None
    index_handler: IndexHandler | None = None
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
            scores = self._distance(self.ref_embeddings, offset=1)
            threshold = torch.quantile(scores.float(), q=q)
            index = scores < threshold
            self.ref_embeddings = self.ref_embeddings[index, :]

        self._setup_index()
        self.set_trained()

        if self.cal_embeddings is None:
            self.scores = self._distance(self.ref_embeddings, offset=1)
        else:
            self.scores = self._distance(self.cal_embeddings, offset=0)
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
        assert self.index_handler is not None, "Index handler must be initialized"
        if self.pca is not None:
            X = self.pca.transform(X)
        score = self._distance(query=X)
        return score.to(device=X.device)

    @abstractmethod
    def _setup_index(self) -> None:
        """Prepare an index for KNN search."""
        pass

    @abstractmethod
    def _distance(self, query: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Calculate the KNN distance of a query against a populated index."""
        pass

    def _build_index(self, embs: torch.Tensor) -> None:
        """Build an index based on reference embeddings.

        The embeddings can be preprocessed (e.g. normalized or transformed) before
        being passed to this method. Typically called within `_setup_index()`.
        """
        assert isinstance(embs, torch.Tensor)
        self.index_handler = self._make_index_handler()
        self.index_handler.build_index(embs=embs, k=self.k)
        self.index = self.index_handler.index
        self.index_params = self.index_handler.index_params

    def _make_index_handler(self) -> IndexHandler:
        if self.index_handler is not None:
            self.index_handler.index_path = self.index_path
            return self.index_handler

        return FaissIndexHandler(index_path=self.index_path)

    def _query_index(self, query: torch.Tensor, offset: int) -> torch.Tensor:
        """Query the index for KNN distances.

        The `offset` parameter allows for retrieving additional neighbors beyond the
        specified `k` to handle cases where a point is both in the reference
        index and the query. For example, if `k=1` and `offset=1`, the method will
        retrieve the 2 nearest neighbors and then use the second nearest neighbor's
        distance as the score, effectively ignoring the nearest neighbor which
        may be the point itself. This is particularly useful when scoring calibration
        samples that are part of the reference set, as it prevents zero distances
        from skewing the scores. The `_query_index()` method is typically called
        within the `_distance()` method of child classes.
        """
        assert self.index_handler is not None, "Index handler must be initialized"
        distances = self.index_handler.query_index(query=query, k=self.k, offset=offset)
        return self.index_handler.aggregate_dists(distances[:, offset:], stat=self.stat)


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
        Whether (and where) to save the FAISS index to disk.

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
    def _distance(self, query: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Calculate the KNN distance of a query against a populated index."""
        squared_distances = self._query_index(query, offset)
        return torch.sqrt(squared_distances)


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
        Whether (and where) to save the FAISS index to disk.

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
    def _distance(self, query: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Calculate cosine distances from FAISS L2 distances on normalized vectors."""
        assert self.index is not None
        normalized = torch.nn.functional.normalize(query)
        return self._query_index(normalized, offset) / 2.0


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
        Whether (and where) to save the FAISS index to disk.

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
    def _distance(self, query: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Calculate the Mahalanobis distance of a query against a populated index."""
        assert self.index is not None
        transformed = query.float() @ self.vi_zero.T
        return self._query_index(transformed, offset)
