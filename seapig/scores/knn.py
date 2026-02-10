"""KNN-based confidence scores."""

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override
from warnings import warn

import nmslib
import torch
from torch.utils.data import DataLoader

from seapig.scores.embed import EmbeddingScore
from seapig.scores.utils import TensorPCA


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


@dataclass
class IndexConfig:
    """Configuration for nmslib index construction and querying.

    This dataclass encapsulates all parameters needed to configure an nmslib
    index for approximate nearest neighbor search in KNN-based confidence
    scoring.

    Parameters
    ----------
    method : str
        The index method to use. Currently only "hnsw" is supported.
        Defaults to "hnsw".
    space : str | None
        The distance space for the index. If None, the space is determined
        by the specific score class (e.g., "l2" for EuclideanScore,
        "cosinesimil" for CosineScore). Defaults to None.
    build_params : dict[str, Any] | None
        Parameters passed to nmslib's createIndex() method. Valid keys for
        HNSW include "M", "efConstruction", and "post". If None, parameters
        are computed using _suggest_index_params(). Defaults to None.
    query_params : dict[str, Any] | None
        Parameters passed to nmslib's setQueryTimeParams() method. Valid keys
        for HNSW include "efSearch". If None, parameters are computed using
        _suggest_index_params(). Defaults to None.
    index_path : Path | None
        Path where the index should be saved/loaded (.bin extension required).
        If None, the index is not persisted to disk. Defaults to None.

    Examples
    --------
    >>> from pathlib import Path
    >>> config = IndexConfig(
    ...     method="hnsw",
    ...     build_params={"M": 16, "efConstruction": 200, "post": 0},
    ...     query_params={"efSearch": 50},
    ...     index_path=Path("my_index.bin")
    ... )
    """

    method: str = "hnsw"
    space: str | None = None
    build_params: dict[str, Any] | None = None
    query_params: dict[str, Any] | None = None
    index_path: Path | None = None


def _validate_index_config(config: IndexConfig) -> None:
    """Validate IndexConfig parameters.

    Parameters
    ----------
    config : IndexConfig
        The configuration to validate.

    Raises
    ------
    ValueError
        If the configuration contains invalid values.
    """
    # Only support hnsw for now
    if config.method != "hnsw":
        raise ValueError(
            f"Only method='hnsw' is currently supported, got '{config.method}'"
        )

    # Validate index_path suffix if provided
    if config.index_path is not None:
        if not isinstance(config.index_path, Path):
            raise ValueError(
                f"index_path must be a Path object, got {type(config.index_path)}"
            )
        if config.index_path.suffix != ".bin":
            raise ValueError(
                f"index_path must have .bin extension, got '{config.index_path.suffix}'"
            )

    # Validate build_params keys for HNSW
    if config.build_params is not None:
        allowed_build_keys = {"M", "efConstruction", "post"}
        invalid_keys = set(config.build_params.keys()) - allowed_build_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid build_params keys: {invalid_keys}. "
                f"Allowed keys for HNSW: {allowed_build_keys}"
            )

    # Validate query_params keys for HNSW
    if config.query_params is not None:
        allowed_query_keys = {"efSearch"}
        invalid_keys = set(config.query_params.keys()) - allowed_query_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid query_params keys: {invalid_keys}. "
                f"Allowed keys for HNSW: {allowed_query_keys}"
            )


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
    index_config:
        An `IndexConfig` instance specifying index configuration including method,
        build/query parameters, and save path. Defaults to `None`, which uses
        default HNSW configuration with auto-tuned parameters.
    nms_index:
        A pre-built nmslib index object. If provided, this index will be used
        directly instead of building a new one. Defaults to `None`.

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

    Examples
    --------
    Using IndexConfig to customize index parameters:

    >>> from pathlib import Path
    >>> config = IndexConfig(
    ...     build_params={"M": 16, "efConstruction": 200, "post": 0},
    ...     query_params={"efSearch": 50},
    ...     index_path=Path("my_index.bin")
    ... )
    >>> score = EuclideanScore(k=1, index_config=config)

    Passing a pre-built nmslib index:

    >>> import nmslib
    >>> index = nmslib.init(method="hnsw", space="l2")
    >>> # ... configure and populate index ...
    >>> score = EuclideanScore(k=1, nms_index=index)
    """

    k: int = 1
    cal_embeddings: torch.Tensor | None
    index: Any | None = None
    index_path: Path | None = None
    index_config: IndexConfig | None = None
    index_params: dict[str, Any]

    def __init__(
        self,
        k: int = 1,
        stat: str = "max",
        pca: TensorPCA | None = None,
        index_config: IndexConfig | None = None,
        nms_index: Any | None = None,
    ) -> None:
        super().__init__(pca=pca)
        assert stat in ["max", "mean", "median", "min"]
        self.stat: str = stat
        self.k = k
        self.ident: str = (
            f"{self.ident}-k{self.k}-{'full' if pca is not None else 'pca'}"
        )

        if index_config is not None:
            _validate_index_config(index_config)
            self.index_config = index_config
            self.index_path = index_config.index_path

        # Handle pre-built index
        if nms_index is not None:
            if not hasattr(nms_index, "knnQueryBatch"):
                raise ValueError("nms_index must have a knnQueryBatch method")
            self.index = nms_index
            if self.index_config is None:
                self.index_params = {"build_defaults": {}, "query_defaults": {}}
            else:
                self.index_params = {
                    "build_defaults": self.index_config.build_params or {},
                    "query_defaults": self.index_config.query_params or {},
                }

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
        if self.cal_required:
            assert self.cal_embeddings is not None

        if self.pca is not None:
            self._fit_pca()
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
        assert self.index is not None, "Index must be built before scoring"
        if self.pca is not None:
            X = self.pca.predict(X)
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
        """Build an index based on reference embeddings.

        The embeddings can be preprocessed (e.g. normalized or transformed) before
        being passed to this method. The `space` parameter should be set accordingly
        to match the type of distance being calculated. Typically called
        within the `_setup_index()` method of child classes.
        """
        assert isinstance(embs, torch.Tensor)

        if self.index is not None:
            return

        if self.index_config is not None:
            method = self.index_config.method
            config_space = self.index_config.space
            if config_space is not None:
                space = config_space
            build_params = self.index_config.build_params
            query_params = self.index_config.query_params
            index_path = self.index_config.index_path
        else:
            method = "hnsw"
            build_params = None
            query_params = None
            index_path = self.index_path

        metadata_path: Path | None = None
        if index_path is not None:
            metadata_path = index_path.with_suffix(".json")

        if index_path is not None and index_path.exists():
            assert metadata_path is not None
            if not metadata_path.exists():
                raise ValueError(
                    f"Index file '{index_path}' exists but metadata file "
                    f"'{metadata_path}' is missing. Please remove the index "
                    f"file or provide the metadata file."
                )

            with open(metadata_path) as f:
                metadata = json.load(f)

            # If explicit config was provided, validate it build_params
            if (
                self.index_config is not None
                and self.index_config.build_params is not None
            ):
                # Only validate if user explicitly provided build params
                suggested = self._suggest_index_params(embs=embs, k=self.k)
                expected_build = (
                    build_params
                    if build_params is not None
                    else suggested["build_defaults"]
                )

                expected_metadata = {
                    "method": method,
                    "space": metadata["space"],
                    "build_params": expected_build,
                }

                # Deep comparison
                if metadata != expected_metadata:
                    raise ValueError(
                        f"Metadata mismatch for index '{index_path}'.\n"
                        f"Expected: {json.dumps(expected_metadata, sort_keys=True)}\n"
                        f"Found: {json.dumps(metadata, sort_keys=True)}\n"
                        f"Please remove the index file or provide a matching "
                        f"configuration."
                    )

            method = metadata["method"]
            space = metadata["space"]
            build_params = metadata["build_params"]
            if query_params is None:
                query_params = self._suggest_index_params(embs=embs, k=self.k)[
                    "query_defaults"
                ]

            index = nmslib.init(method=method, space=space)
            index.loadIndex(index_path.as_posix(), load_data=True)
        else:
            # Get suggested params if not provided
            suggested = self._suggest_index_params(embs=embs, k=self.k)
            if build_params is None:
                build_params = suggested["build_defaults"]
            if query_params is None:
                query_params = suggested["query_defaults"]

            # Build new index
            index = nmslib.init(method=method, space=space)
            index.addDataPointBatch(embs.cpu())
            index.createIndex(index_params=build_params)

            # Save to disk if path provided
            if index_path is not None:
                assert metadata_path is not None  # Type narrowing
                index_path.parent.mkdir(parents=True, exist_ok=True)
                index.saveIndex(index_path.as_posix(), save_data=True)

                # Save metadata
                metadata = {
                    "method": method,
                    "space": space,
                    "build_params": build_params,
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, sort_keys=True, indent=2)

        # Store params for later use
        self.index_params = {
            "build_defaults": build_params,
            "query_defaults": query_params,
        }
        self.index = index

    @staticmethod
    def _suggest_index_params(
        embs: torch.Tensor, k: int = 10
    ) -> dict[str, Any]:
        """Suggest conservative HNSW index and query-time parameters."""
        if embs.dim() != 2:
            raise ValueError("ref_embeddings must be 2D (N, D)")
        N, D = map(int, embs.shape)

        if N < 10:
            return {
                "build_defaults": {"post": 0},
                "query_defaults": {"efSearch": k},
            }

        M = _clamp(int(round(2.0 * math.sqrt(D))), 8, 64)
        base = 150 if N < 5_000 else 300 if N < 50_000 else 600
        ef_construction = _clamp(
            int(round(base * (1.0 + (D / 128.0) * 0.5))), 100, 2000
        )

        ef_search = max(
            max(32, k * 8), min(max(128, ef_construction // 4), 512)
        )

        return {
            "build_defaults": {
                "M": M,
                "efConstruction": ef_construction,
                "post": 0,
            },
            "query_defaults": {"efSearch": ef_search},
        }

    def _query_index(self, query: torch.Tensor, kpn: int) -> torch.Tensor:
        """Query the index for KNN distances.

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
        assert self.index is not None, "Index must be built before querying"
        nmslib.setQueryTimeParams(
            self.index, self.index_params["query_defaults"]
        )
        results = self.index.knnQueryBatch(query.cpu(), k=self.k + kpn)
        distances = self._unpack(results, kpn=kpn)
        distances = self._stat(distances[:, kpn:])
        return distances

    def _unpack(
        self, query_results: list[tuple[torch.Tensor, torch.Tensor]], kpn: int
    ) -> torch.Tensor:
        """Unpack nmslib query results into a rectangular tensor.

        Converts each distance array returned by nmslib into a torch tensor and
        assembles them into a 2D tensor of shape (n_queries, self.k + kpn).
        If a query returned fewer than the expected number of neighbors the row
        is padded with zeros; if more neighbors are returned the distances are
        truncated to the expected length.
        """
        expected = self.k + kpn
        rows: list[torch.Tensor] = []

        for res in query_results:
            dist = torch.as_tensor(res[1], dtype=torch.float32)
            if dist.numel() < expected:
                warn(
                    f"Query returned {dist.numel()} neighbors, expected {expected}. "
                    f"Padding with zeros."
                )
                pad = torch.zeros(expected - dist.numel(), dtype=dist.dtype)
                dist = torch.cat([dist, pad])
            elif dist.numel() > expected:
                dist = dist[:expected]
            rows.append(dist.unsqueeze(0))

        if not rows:
            return torch.empty((0, expected), dtype=torch.float32)

        return torch.cat(rows, dim=0)

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
        index_config: IndexConfig | None = None,
        nms_index: Any | None = None,
    ) -> None:
        super().__init__(
            k=k,
            stat=stat,
            pca=pca,
            index_config=index_config,
            nms_index=nms_index,
        )

    @override
    def _setup_index(self) -> None:
        """Initialize an index based on reference embeddings."""
        assert isinstance(self.ref_embeddings, torch.Tensor)
        self._build_index(self.ref_embeddings, space="l2")

    @override
    @torch.inference_mode() # type: ignore [untyped-decorator]
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
        index_config: IndexConfig | None = None,
        nms_index: Any | None = None,
    ) -> None:
        super().__init__(
            k=k,
            stat=stat,
            pca=pca,
            index_config=index_config,
            nms_index=nms_index,
        )

    @override
    def _setup_index(self) -> None:
        """Initialize an index based on reference embeddings."""
        assert isinstance(self.ref_embeddings, torch.Tensor)
        normalized = torch.nn.functional.normalize(self.ref_embeddings)
        self._build_index(normalized, space="cosinesimil")

    @override
    @torch.inference_mode() # type: ignore [untyped-decorator]
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
        index_config: IndexConfig | None = None,
        nms_index: Any | None = None,
    ) -> None:
        super().__init__(
            k=k,
            stat=stat,
            pca=pca,
            index_config=index_config,
            nms_index=nms_index,
        )
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
    @torch.inference_mode() # type: ignore [untyped-decorator]
    def _distance(self, query: torch.Tensor, kpn: int = 0) -> torch.Tensor:
        assert self.index is not None
        transformed = query.float() @ self.vi_zero.T
        return self._query_index(transformed, kpn)
