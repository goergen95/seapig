"""Mixins for seapig."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch


class FAISSIndexMixin:
    """Encapsulate FAISS index handling.

    Expected attributes on the host class (e.g., `KNNScore`):

    - `self.k`:  number of neighbours.
    - `self.index`: the class-agnostic index (or `None` before building).
    - `self.indices_by_class`: mapping `class_id -> index` for class-wise mode.
    - `self.index_path`: optional `Path` where indexes are persisted.
    """

    def _make_faiss_index(self, embs: torch.Tensor) -> Any:
        """Create a FAISS index appropriate for `embs`.

        Uses a flat index for small datasets (≤10_000 vectors) and an HNSW
        index otherwise. Parameters are suggested by `_suggest_build_params`.
        """
        params = self._suggest_build_params(embs=embs, k=self.k)
        d = int(embs.shape[1])
        N = int(embs.shape[0])
        if N <= 10_000:
            index = faiss.IndexFlatL2(d)  # type: ignore[possibly-missing-attribute]
        else:
            M = params["M"]
            index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)  # type: ignore[possibly-missing-attribute]
            index.hnsw.efConstruction = params["efConstruction"]
        return index

    def _build_index(self, embs: torch.Tensor) -> None:
        """Build the class-agnostic index from ``embs``.

        If `self.index_path` is set and the file already exists, the index is
        loaded from disk instead of rebuilt.
        """
        assert isinstance(embs, torch.Tensor)
        index_path = self.index_path
        index = self._make_faiss_index(embs)
        if index_path is None or not Path(index_path).exists():
            embs_np: np.ndarray = embs.cpu().numpy().astype(np.float32)
            index.add(embs_np)
            if index_path is not None:
                faiss.write_index(index, str(index_path))  # type: ignore[possibly-missing-attribute]
        else:
            warnings.warn(
                f"Index file {index_path} already exists. Loading from disk.",
                UserWarning,
            )
            index = faiss.read_index(str(index_path))  # type: ignore[possibly-missing-attribute]
        self.index = index

    def _query_index(
        self, query: torch.Tensor, offset: int = 0, *, index: Any | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a KNN search against a FAISS `index`.

        Returns `(distances, indices)`. `offset` discards the nearest
        neighbour when `offset > 0` (useful for excluding the query itself).
        """
        if index is None:
            index = self.index
        assert index is not None, "Index must be built before querying."
        if isinstance(index, faiss.IndexHNSW):  # type: ignore[possibly-missing-attribute]
            params = self._suggest_query_params(query, self.k + offset)
            ef_search = params.get("efSearch", index.hnsw.efSearch)
            index.hnsw.efSearch = ef_search
        if index.d != query.shape[1]:
            raise ValueError(
                f"Query dimension {query.shape[1]} does not match index dimension {index.d}"
            )
        query_np = query.cpu().numpy().astype(np.float32)
        distances, indices = index.search(query_np, self.k + offset)
        distances = torch.from_numpy(distances).to(query.device)
        indices = torch.from_numpy(indices).to(query.device)
        if offset > 0:
            distances = distances[:, offset:]
            indices = indices[:, offset:]
        return distances, indices

    @staticmethod
    def _suggest_build_params(embs: torch.Tensor, k: int = 1) -> dict[str, Any]:
        if embs.dim() != 2:
            raise ValueError("embeddings must be 2D (N, D)")
        n, d = map(int, embs.shape)
        if d <= 64:
            M = 16
        elif d <= 128:
            M = 24
        elif d <= 256:
            M = 32
        elif d <= 512 or d <= 1024:
            M = 48
        else:
            M = 64
        if n > 5_000_000:
            M = max(M, 32)  # pragma: no cover
        if n > 50_000_000:
            M = max(M, 48)  # pragma: no cover
        C = max(4 * M, 128)
        C = min(C, 1024)
        return {"M": M, "efConstruction": C}

    @staticmethod
    def _suggest_query_params(embs: torch.Tensor, k: int = 1) -> dict[str, Any]:
        params = FAISSIndexMixin._suggest_build_params(embs, k)
        S = max(k * 8, 512)
        S = min(S, params["efConstruction"])
        S = max(S, k)
        return {"efSearch": S}
