"""Abstract base class for index adapters."""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


class IndexHandler(ABC):
    """Abstract base for index adapters.

    The base class implements: torch-based API (inputs/outputs), normalisation
    of results, warnings about fewer neighbours, and aggregation.

    Subclasses implement index-specific operations via the ``_*_impl`` methods.
    Adapters may work in numpy internally; the base class converts
    torch <-> numpy as needed.
    """

    def __init__(self) -> None:
        self._metadata: dict[str, Any] = {}

    def build_index(
        self,
        embeddings: torch.Tensor,
        space: str = "l2",
        index_params: dict[str, Any] | None = None,
    ) -> None:
        """Build the index from *embeddings*.

        Parameters
        ----------
        embeddings:
            Reference embeddings as a ``(N, D)`` torch.Tensor.
        space:
            Distance space understood by the adapter (e.g. ``"l2"``,
            ``"cosinesimil"``).
        index_params:
            Optional adapter-specific build parameters that override defaults.
        """
        self._build_impl(embeddings.cpu().numpy(), space, index_params)

    def load_index(self, path: Path, load_data: bool = False) -> None:
        """Load an index from *path*.

        The sidecar JSON metadata file (``<path>.json``) is read first so that
        adapter-specific information (e.g. ``space``) is available to
        ``_load_impl``.

        Parameters
        ----------
        path:
            Path to the index binary.  The sidecar is expected at
            ``path.with_suffix(".json")``.
        load_data:
            Whether to load the raw data points together with the index.
        """
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            with sidecar.open() as f:
                self._metadata = json.load(f)
        self._load_impl(path, load_data)

    def save_index(
        self,
        path: Path,
        save_data: bool = False,
        metadata: dict[str, Any] | None = None,
        atomic: bool = True,
    ) -> None:
        """Save the index to *path*.

        Writes the index binary via ``_save_impl`` and a JSON metadata sidecar
        at ``path.with_suffix(".json")``.

        Parameters
        ----------
        path:
            Destination path for the index binary.
        save_data:
            Whether to serialise the raw data points alongside the index.
        metadata:
            Additional key-value pairs merged into the sidecar metadata.
        atomic:
            When ``True`` (default), the JSON sidecar is written to a
            temporary file and then renamed to ensure an atomic write.
        """
        self._save_impl(path, save_data, metadata)
        meta = {**self._metadata, **(metadata or {})}
        sidecar = path.with_suffix(".json")
        content = json.dumps(meta, indent=2)
        if atomic:
            tmp = sidecar.with_suffix(".tmp")
            tmp.write_text(content)
            tmp.rename(sidecar)
        else:
            sidecar.write_text(content)

    def add_batch(self, embeddings: torch.Tensor) -> None:
        """Add *embeddings* to an already-built index.

        Parameters
        ----------
        embeddings:
            Embeddings to add as a ``(N, D)`` torch.Tensor.
        """
        self._add_impl(embeddings.cpu().numpy())

    def query_batch(
        self,
        queries: torch.Tensor,
        k: int,
        query_params: dict[str, Any] | None = None,
        aggregation: str = "max",
        offset: int = 0,
    ) -> tuple[list[list[int]], torch.Tensor]:
        """Query the index and return neighbour indices and aggregated distances.

        The index is queried for ``k + offset`` neighbours per row; the
        leading *offset* columns are stripped before aggregation.  Set
        ``offset=1`` when queries are a subset of the indexed points (e.g.
        scoring calibration samples against the reference index) to discard
        the self-match at distance 0.

        When fewer than *k* effective neighbours are available for a row (after
        offset removal), a :class:`UserWarning` is emitted and the available
        distances are aggregated as-is — no zero-padding is applied.

        Parameters
        ----------
        queries:
            Query embeddings as a ``(B, D)`` torch.Tensor.
        k:
            Number of neighbours to retrieve (effective, after offset removal).
        query_params:
            Optional adapter-specific query-time parameters.
        aggregation:
            Aggregation method applied to the per-row distances.
            One of ``"mean"``, ``"median"``, ``"min"``, ``"max"``.
            Defaults to ``"max"`` (Kth-distance semantics).
        offset:
            Number of leading neighbours to discard before aggregation.

        Returns
        -------
        indices:
            Ragged ``List[List[int]]`` of neighbour indices (post-offset),
            one inner list per query.
        aggregated_distances:
            ``torch.Tensor`` of shape ``(B,)`` with one aggregated distance
            scalar per query.  Rows with zero effective neighbours yield NaN.
        """
        raw_indices, raw_distances = self._query_impl(
            queries.cpu().numpy(), k + offset, query_params
        )

        trimmed_indices: list[list[int]] = []
        trimmed_distances: list[list[float]] = []
        for i, (idx_row, dist_row) in enumerate(
            zip(raw_indices, raw_distances)
        ):
            idx_trimmed = idx_row[offset:]
            dist_trimmed = dist_row[offset:]
            if len(dist_trimmed) < k:
                warnings.warn(
                    f"Query {i} returned fewer than {k} neighbors. "
                    f"Got {len(dist_trimmed)} after offset={offset}; "
                    "aggregating available distances only.",
                    UserWarning,
                    stacklevel=2,
                )
            trimmed_indices.append(idx_trimmed)
            trimmed_distances.append(dist_trimmed)

        aggregated = self.aggregate_distances(
            trimmed_distances, method=aggregation
        )
        return trimmed_indices, aggregated

    def set_query_time_params(self, params: dict[str, Any]) -> None:
        """Set query-time parameters.

        Subclasses can override this to apply adapter-specific runtime options.

        Parameters
        ----------
        params:
            Adapter-specific query-time parameter dictionary.
        """

    def get_metadata(self) -> dict[str, Any]:
        """Return a copy of the current metadata dictionary."""
        return dict(self._metadata)

    def aggregate_distances(
        self, distances: Sequence[Sequence[float]], method: str = "mean"
    ) -> torch.Tensor:
        """Aggregate per-row distance lists into a 1-D tensor.

        Parameters
        ----------
        distances:
            Ragged sequence of distance rows.  Empty rows yield NaN.
        method:
            One of ``"mean"``, ``"median"``, ``"min"``, ``"max"``.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape ``(B,)`` with one aggregated scalar per row.
        """
        if method not in {"mean", "median", "min", "max"}:
            raise ValueError(
                f"Unsupported aggregation method: {method!r}. "
                "Choose from 'mean', 'median', 'min', 'max'."
            )
        results: list[float] = []
        for row in distances:
            if len(row) == 0:
                results.append(float("nan"))
            else:
                t = torch.tensor(list(row), dtype=torch.float32)
                if method == "mean":
                    results.append(t.mean().item())
                elif method == "median":
                    results.append(t.median().item())
                elif method == "min":
                    results.append(t.amin().item())
                else:  # max
                    results.append(t.amax().item())
        return torch.tensor(results, dtype=torch.float32)

    @abstractmethod
    def _build_impl(
        self,
        embeddings: np.ndarray[Any, np.dtype[Any]],
        space: str,
        index_params: dict[str, Any] | None,
    ) -> None:
        """Build the index from a numpy array of embeddings."""

    @abstractmethod
    def _add_impl(self, embeddings: np.ndarray[Any, np.dtype[Any]]) -> None:
        """Add new embeddings to an existing index."""

    @abstractmethod
    def _query_impl(
        self,
        queries: np.ndarray[Any, np.dtype[Any]],
        k: int,
        query_params: dict[str, Any] | None,
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Query the index; return raw ragged (indices, distances) lists."""

    @abstractmethod
    def _save_impl(
        self, path: Path, save_data: bool, metadata: dict[str, Any] | None
    ) -> None:
        """Write the index binary to *path*."""

    @abstractmethod
    def _load_impl(self, path: Path, load_data: bool) -> None:
        """Load the index binary from *path*."""
