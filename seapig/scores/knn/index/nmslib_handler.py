"""nmslib HNSW adapter for ``IndexHandler``."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import nmslib
import torch

from seapig.scores.knn.index.handler import IndexHandler


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


class NmslibHandler(IndexHandler):
    """``IndexHandler`` adapter backed by nmslib HNSW.

    Parameters
    ----------
    k:
        Expected query ``k``; used to derive conservative HNSW build and
        query-time parameters via :meth:`_suggest_index_params`.

    Notes
    -----
    Distance semantics:

    - ``"l2"`` — nmslib returns **squared** L2 distances.  Callers that need
      Euclidean distances should take ``sqrt`` of the returned values.
    - ``"cosinesimil"`` — nmslib returns ``1 − cosine_similarity`` directly;
      no post-conversion is required.
    """

    def __init__(self, k: int = 1) -> None:
        super().__init__()
        self.k = k
        self._index: Any = None
        self._query_defaults: dict[str, Any] = {}

    @staticmethod
    def _suggest_index_params(N: int, D: int, k: int) -> dict[str, Any]:
        """Suggest conservative HNSW index and query-time parameters.

        Parameters
        ----------
        N:
            Number of data points in the index.
        D:
            Embedding dimensionality.
        k:
            Number of neighbours used at query time.

        Returns
        -------
        dict
            A dictionary with ``"build_defaults"`` and ``"query_defaults"``
            keys containing the suggested HNSW hyper-parameters.
        """
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

    # ------------------------------------------------------------------
    # IndexHandler abstract methods
    # ------------------------------------------------------------------

    def _build_impl(
        self,
        embeddings: torch.Tensor,
        space: str,
        index_params: dict[str, Any] | None,
    ) -> None:
        N, D = int(embeddings.shape[0]), int(embeddings.shape[1])
        suggested = self._suggest_index_params(N, D, self.k)
        build_params: dict[str, Any] = dict(suggested["build_defaults"])
        if index_params:
            build_params.update(index_params)
        self._query_defaults = dict(suggested["query_defaults"])

        emb_np = embeddings.detach().cpu().numpy()

        index = nmslib.init(method="hnsw", space=space)
        index.addDataPointBatch(emb_np)
        index.createIndex(index_params=build_params)
        self._index = index

        self._metadata = {
            "library": "nmslib",
            "space": space,
            "dtype": str(embeddings.dtype),
            "n_items": N,
            "index_params": build_params,
        }

    def _add_impl(self, embeddings: torch.Tensor) -> None:
        if self._index is None:
            raise RuntimeError("Index must be built before adding data.")
        emb_np = embeddings.detach().cpu().numpy()
        self._index.addDataPointBatch(emb_np)
        self._metadata["n_items"] = int(self._metadata.get("n_items", 0)) + int(
            embeddings.shape[0]
        )

    def _query_impl(
        self, queries: torch.Tensor, k: int, query_params: dict[str, Any] | None
    ) -> tuple[list[list[int]], list[list[float]]]:
        if self._index is None:
            raise RuntimeError("Index must be built before querying.")
        qp: dict[str, Any] = dict(self._query_defaults)
        if query_params:
            qp.update(query_params)
        nmslib.setQueryTimeParams(self._index, qp)
        queries_np = queries.detach().cpu().numpy()
        results = self._index.knnQueryBatch(queries_np, k=k)
        indices: list[list[int]] = []
        distances: list[list[float]] = []
        for idx_arr, dist_arr in results:
            indices.append([int(x) for x in idx_arr])
            distances.append([float(x) for x in dist_arr])
        return indices, distances

    def _save_impl(
        self, path: Path, save_data: bool, metadata: dict[str, Any] | None
    ) -> None:
        if self._index is None:
            raise RuntimeError("Index must be built before saving.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._index.saveIndex(path.as_posix(), save_data=save_data)

    def _load_impl(self, path: Path, load_data: bool) -> None:
        space: str = str(self._metadata.get("space", "l2"))
        index = nmslib.init(method="hnsw", space=space)
        index.loadIndex(path.as_posix(), load_data=load_data)
        self._index = index
