"""KNN-based confidence scores."""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import nmslib
import torch
from torch.utils.data import DataLoader
from typing_extensions import override

from seapig.scores.embed import EmbeddingScore
from seapig.scores.index_manager import IndexManager
from seapig.scores.utils import TensorPCA


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
    pca:
        A `TensorPCA` instance or `None`. If provided, this `TensorPCA` object will
        be used to perform dimensionality reduction on embeddings prior to
        scoring (for example, to retain a specified explained variance).
        Defaults to `None`, indicating that dimensionality reduction is not applied.
    save_index:
        A `bool` or `Path` indicating whether to save the fitted index to disk.
        If `True`, the index will be saved to a default location. If a `Path` is
        provided, the index will be saved to that location. Defaults to `False`,
        indicating that the index will not be saved to disk.

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
    cal_embeddings: torch.Tensor | None
    index: Any | None = None
    index_path: Path | None = None
    _index_manager: IndexManager | None = None

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
        my_score = EmbeddingScore(k=2)
        my_score.fit(X=train_embs, Y=val_embs)

        # Mode 2: On-the-fly extraction
        my_score = EmbeddingScore(k=2)
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

        if self.pca is not None:
            self._fit_pca()
            self.ref_embeddings = self.pca.transform(self.ref_embeddings)
            if self.cal_embeddings is not None:
                self.cal_embeddings = self.pca.transform(self.cal_embeddings)

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

    @override
    def _score_embeddings(self, X: torch.Tensor) -> torch.Tensor:
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
        assert self._index_manager is not None, (
            "Index must be built before scoring"
        )
        if self.pca is not None:
            X = self.pca.transform(X)
        score = self._distance(query=X)
        return score.to(device=X.device)

    @abstractmethod
    def _setup_index(self) -> None:
        """Prepare an index for KNN search."""
        pass

    @abstractmethod
    def _distance(self, query: torch.Tensor, kpn: int = 0) -> torch.Tensor:
        """Calculate the KNN distance of a query against a populated index."""
        pass

    def _build_index(self, embs: torch.Tensor, space: str = "l2") -> None:
        """Build an index via IndexManager.

        The embeddings can be preprocessed (e.g. normalized or transformed)
        before being passed to this method. The `space` parameter should
        be set accordingly to match the type of distance being calculated.
        Typically called within the `_setup_index()` method of child
        classes.
        """
        assert isinstance(embs, torch.Tensor)
        index_path = self.index_path
        params = self._suggest_index_params(embs=embs, k=self.k)

        mgr = IndexManager(method="hnsw", space=space)
        mgr.fit(embs)

        if index_path is None or not index_path.exists():
            mgr.build_index(hnsw_params=params["build_defaults"])
            # Override with k-aware query params from _suggest_index_params
            mgr._index_params = params
            if index_path:
                assert mgr._index is not None
                mgr._index.saveIndex(index_path.as_posix(), save_data=True)
        else:
            warnings.warn(
                f"Index file {index_path} already exists. Loading existing index from disk.",
                UserWarning,
            )
            nmslib_index = nmslib.init(method="hnsw", space=space)
            nmslib_index.loadIndex(index_path.as_posix(), load_data=True)
            mgr._index = nmslib_index
            mgr._index_built = True
            mgr._index_params = params

        self._index_manager = mgr
        self.index_params = params
        self.index = mgr._index

    @staticmethod
    def _suggest_index_params(
        embs: torch.Tensor, k: int = 10
    ) -> dict[str, Any]:
        """Suggest conservative HNSW index and query-time parameters.

        Delegates to `IndexManager._suggest_hnsw_params`.
        """
        return IndexManager._suggest_hnsw_params(embs=embs, k=k)

    def _query_index(self, query: torch.Tensor, kpn: int) -> torch.Tensor:
        """Query the index for KNN distances via IndexManager.

        The `kpn` parameter allows for retrieving additional neighbors beyond the
        specified `k` to handle cases where a point is both in the reference
        index and the query. For example, if `k=1` and `kpn=1`, the method will
        retrieve the 2 nearest neighbors and then use the second nearest neighbor's
        distance as the score, effectively ignoring the nearest neighbor which
        may be the point itself. This is particularly useful when scoring calibration
        samples that are part of the reference set, as it prevents zero distances
        from skewing the scores. The `_query_index()` method is typically called
        within the `_distance()` method of child classes.
        """
        assert self._index_manager is not None, (
            "Index must be built before querying"
        )
        _, distances = self._index_manager.search(query, k=self.k + kpn)
        distances = self._stat(distances[:, kpn:])
        return distances

    def _zeropad(
        self, query_results: list[tuple[torch.Tensor, torch.Tensor]], kpn: int
    ) -> torch.Tensor:
        """Zero pad the distance tensors if fewer than `k + kpn` neighbors are returned.

        This is required because approximate nearest neighbour searches may
        not always return exactly `k` neighbors.
        """
        distances = []

        for i, res in enumerate(query_results):
            dist_tensor = torch.tensor(res[1])
            if len(dist_tensor) < self.k + kpn:
                warnings.warn(
                    f"Query {i} returned fewer than {self.k + kpn} neighbors. "
                    f"Applying zero padding to the distance tensor.",
                    UserWarning,
                )
                padding = torch.zeros(self.k + kpn - len(dist_tensor))
                dist_tensor = torch.cat([dist_tensor, padding])
            distances.append(dist_tensor.unsqueeze(0))
        return torch.cat(distances)

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
    """Returns the KNN-distance based on the euclidean distance to the nearest samples.

    Computes Euclidean distance-based confidence scores where low scores indicate
    samples similar to the training distribution (likely inliers) and high scores
    indicate samples deviating from the training distribution (likely outliers).

    Parameters
    ----------
    k:
        An `int`eger indicating the number of neighbors to calculate the distance.
        Defaults to 1, e.g. the distance to the closest neighbor.
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
        self._build_index(self.ref_embeddings, space="l2")

    @override
    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _distance(self, query: torch.Tensor, kpn: int = 0) -> torch.Tensor:
        """Calculate the KNN distance of a query against a populated index."""
        return torch.sqrt(self._query_index(query, kpn))


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
        self._build_index(normalized, space="cosinesimil")

    @override
    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _distance(self, query: torch.Tensor, kpn: int = 0) -> torch.Tensor:
        assert self.index is not None
        normalized = torch.nn.functional.normalize(query)
        return self._query_index(normalized, kpn)


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
        self._build_index(transformed, space="l2")

    @override
    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _distance(self, query: torch.Tensor, kpn: int = 0) -> torch.Tensor:
        assert self.index is not None
        transformed = query.float() @ self.vi_zero.T
        return self._query_index(transformed, kpn)
