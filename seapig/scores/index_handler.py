"""Index handlers for computing distances."""

from __future__ import annotations

import abc
import warnings
from pathlib import Path
from typing import Any

import faiss
import torch

__all__ = ["IndexHandler", "FaissIndexHandler"]


class IndexHandler(abc.ABC):
    """Abstract base class for index handlers."""

    index: Any | None
    index_params: dict[str, Any] | None
    index_path: Path | None
    n_samples: int
    dim: int

    def __init__(self, index_path: Path | None = None) -> None:
        self.index = None
        self.index_params = None
        self.index_path = index_path
        self.n_samples = 0
        self.dim = 0

    @abc.abstractmethod
    def build_index(
        self, embs: torch.Tensor, k: int = 1, **build_opts: Any
    ) -> None:
        """Build an index from reference embeddings."""

    @abc.abstractmethod
    def query_index(
        self, query: torch.Tensor, k: int, offset: int = 0
    ) -> torch.Tensor:
        """Query distances from the index."""

    @abc.abstractmethod
    def save_index(self, path: Path) -> None:
        """Save index to disk."""

    @abc.abstractmethod
    def load_index(self, path: Path) -> None:
        """Load index from disk."""

    @abc.abstractmethod
    def aggregate_dists(self, dists: torch.Tensor, stat: str) -> torch.Tensor:
        """Aggregate neighbor distances with a statistic."""

    @abc.abstractmethod
    def suggest_build_params(
        self, n_samples: int, dim: int, k: int
    ) -> dict[str, Any]:
        """Suggest build-time parameters."""

    @abc.abstractmethod
    def suggest_query_params(
        self, n_samples: int, dim: int, k: int
    ) -> dict[str, Any]:
        """Suggest query-time parameters."""


class FaissIndexHandler(IndexHandler):
    """FAISS index handler using IVFPQ with flat fallback."""

    def suggest_build_params(
        self, n_samples: int, dim: int, k: int
    ) -> dict[str, Any]:
        """Suggest FAISS build-time parameters.

        `m` is chosen by starting from `min(16, dim // 8)` and stepping down
        until it divides `dim`, since IVFPQ requires evenly-sized subvectors.
        """
        import math

        # Minimum 1 centroid, maximum 1 per 20 samples, never larger than
        # the classic thresholds used for large datasets.
        max_nlist_by_samples = max(1, n_samples // 20)
        # Classic upper bounds for very large collections
        classic_limits = [
            (1_000, 16),
            (10_000, 64),
            (100_000, 256),
            (float("inf"), 1024),
        ]
        classic_nlist = next(
            limit for cutoff, limit in classic_limits if n_samples < cutoff
        )

        # Pick the smaller of the two limits
        nlist = min(max_nlist_by_samples, classic_nlist)

        # Round down to the nearest power of two (FAISS prefers powers of two)
        if nlist > 1:
            nlist = 2 ** int(math.log2(nlist))

        m = max(1, min(16, dim // 8))
        while m > 1 and dim % m != 0:
            m -= 1
        # Ensure m never exceeds the dimensionality
        m = min(m, dim)

        use_flat_fallback = n_samples < 2_000

        return {
            "nlist": int(max(1, nlist)),
            "m": int(m),
            "nbits": 8,
            "use_flat_fallback": use_flat_fallback,
        }

    def suggest_query_params(
        self, n_samples: int, dim: int, k: int
    ) -> dict[str, Any]:
        """Suggest FAISS query-time parameters."""
        build = self.suggest_build_params(n_samples=n_samples, dim=dim, k=k)
        nlist = int(build["nlist"])
        return {"nprobe": min(max(1, int(k * 2)), nlist)}

    def build_index(
        self, embs: torch.Tensor, k: int = 1, **build_opts: Any
    ) -> None:
        """Build a FAISS index from embeddings."""
        if embs.dim() != 2:
            raise ValueError("embs must be 2D (N, D)")
        n_samples, dim = map(int, embs.shape)
        self.n_samples = n_samples
        self.dim = dim

        if self.index_path is not None and self.index_path.exists():
            warnings.warn(
                f"Index file {self.index_path} already exists. Loading existing index from disk.",
                UserWarning,
            )
            self.load_index(self.index_path)
            return

        params = self.suggest_build_params(n_samples=n_samples, dim=dim, k=k)
        params.update(build_opts)
        self.index_params = {"build_defaults": params}

        x_np = embs.detach().cpu().to(torch.float32).numpy()
        use_flat_fallback = bool(params.get("use_flat_fallback", False))
        if use_flat_fallback:
            flat_index = faiss.IndexFlatL2(dim)
            flat_index.add(x_np)
            self.index = flat_index
        else:
            nlist = int(params["nlist"])
            m = int(params["m"])
            nbits = int(params["nbits"])
            quantizer = faiss.IndexFlatL2(dim)
            ivfpq_index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
            try:
                ivfpq_index.train(x_np)
                ivfpq_index.add(x_np)
                self.index = ivfpq_index
            except RuntimeError:
                warnings.warn(
                    "FAISS IVFPQ build failed; falling back to IndexFlatL2.",
                    UserWarning,
                )
                flat_index = faiss.IndexFlatL2(dim)
                flat_index.add(x_np)
                self.index = flat_index

        if self.index is not None:
            query_defaults = self.suggest_query_params(
                n_samples=n_samples, dim=dim, k=k
            )
            self.index_params["query_defaults"] = query_defaults
            nprobe = int(query_defaults["nprobe"])
            try:
                faiss.ParameterSpace().set_index_parameter(
                    self.index, "nprobe", nprobe
                )
            except RuntimeError:
                pass

        if self.index_path is not None:
            self.save_index(self.index_path)

    @torch.inference_mode()
    def query_index(
        self, query: torch.Tensor, k: int, offset: int = 0
    ) -> torch.Tensor:
        """Query a FAISS index and return distances with shape (B, k + offset)."""
        if self.index is None:
            raise ValueError("Index must be built before querying")
        requested = k + offset
        if requested <= 0:
            raise ValueError("k + offset must be > 0")

        q_np = query.detach().cpu().to(torch.float32).numpy()
        query_defaults = self.suggest_query_params(
            n_samples=self.n_samples, dim=self.dim, k=k
        )
        nprobe = int(query_defaults["nprobe"])
        try:
            faiss.ParameterSpace().set_index_parameter(
                self.index, "nprobe", nprobe
            )
        except RuntimeError:
            pass
        dists, idx = self.index.search(q_np, requested)
        mask = idx < 0
        if mask.any():
            missing_per_query = mask.sum(axis=1)
            for i, missing in enumerate(missing_per_query.tolist()):
                if missing > 0:
                    warnings.warn(
                        f"Query {i} returned fewer than {requested} neighbors. "
                        "Applying zero padding to the distance tensor.",
                        UserWarning,
                    )
            dists[mask] = 0.0

        return torch.from_numpy(dists).to(device=query.device)

    def save_index(self, path: Path) -> None:
        """Save FAISS index to disk."""
        if self.index is None:
            raise ValueError("Index must be built before saving")
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path.as_posix())

    def load_index(self, path: Path) -> None:
        """Load FAISS index from disk."""
        self.index = faiss.read_index(path.as_posix())
        self.index_path = path
        self.n_samples = int(self.index.ntotal)
        self.dim = int(self.index.d)

    def aggregate_dists(self, dists: torch.Tensor, stat: str) -> torch.Tensor:
        """Aggregate distances by statistic."""
        if stat == "max":
            return dists.amax(1)
        if stat == "mean":
            return dists.mean(1)
        if stat == "median":
            return dists.median(1).values
        if stat == "min":
            return dists.amin(1)
        raise ValueError(f"Unsupported stat: {stat}")
