"""IndexManager: centralises index build, search, save, and load for KNN scores.

This module provides the :class:`IndexManager` class which centralises all
index-related responsibilities previously spread across :class:`KNNScore` and
its derived classes.  It supports HNSW via ``nmslib`` as the primary
approximate nearest-neighbour backend and a pure-PyTorch brute-force fallback.

Notes
-----
This class is **not** thread-safe.
"""

import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any

import torch

from seapig.utils.logging import get_logger

_MODULE_LOGGER = get_logger(__name__)


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp *value* to the closed interval [*lo*, *hi*]."""
    return max(lo, min(hi, value))


class IndexManager:
    """Manages nearest-neighbour index build, search, and persistence.

    Centralises all index-related functionality used by :class:`KNNScore` and
    its derived classes.  The primary backend is HNSW via ``nmslib``
    (``method="hnsw"``); a pure-PyTorch brute-force search is available as a
    fallback (``method="brute"``).

    Parameters
    ----------
    method : {"hnsw", "brute"}, optional
        Nearest-neighbour backend.  ``"hnsw"`` (default) uses nmslib HNSW;
        ``"brute"`` computes pairwise distances with PyTorch.
    space : {"l2", "cosinesimil"}, optional
        Distance space.  ``"l2"`` (default) stores *squared* Euclidean
        distances (consistent with nmslib conventions).  ``"cosinesimil"``
        stores cosine distances (1 − similarity).
    device : str or None, optional
        Device on which returned tensors are placed.  When ``None`` the device
        of the first batch added via :meth:`fit` or :meth:`add_batch` is
        inferred automatically.
    pca_components : int or None, optional
        Number of PCA components to retain before indexing.  Mutually
        exclusive with *pca_exp_var*.  Defaults to ``None`` (no PCA).
    pca_exp_var : float or None, optional
        Minimum cumulative explained-variance ratio for automatic PCA
        component selection (passed directly to
        :class:`sklearn.decomposition.PCA` as ``n_components``).  Mutually
        exclusive with *pca_components*.  Defaults to ``None`` (no PCA).
    dtype : torch.dtype, optional
        Data type used for internal tensor storage and returned distances.
        Defaults to ``torch.float32``.
    log : logging.Logger or None, optional
        Custom logger.  When ``None`` the module-level seapig logger is used.

    Attributes
    ----------
    method : str
        Active backend (``"hnsw"`` or ``"brute"``).
    space : str
        Distance space (``"l2"`` or ``"cosinesimil"``).

    Notes
    -----
    * **Thread safety**: not guaranteed; do not share a single instance across
      threads without external locking.
    * For ``space="l2"`` both the HNSW backend and the brute-force backend
      return *squared* Euclidean distances.  Callers should apply
      ``torch.sqrt`` when true Euclidean distance is required.
    * HNSW recommended defaults: ``M=16``, ``efConstruction=200``,
      ``efSearch`` configurable per-query via *hnsw_params*.

    Examples
    --------
    Brute-force search:

    >>> import torch
    >>> from seapig.scores.index_manager import IndexManager
    >>> mgr = IndexManager(method="brute", space="l2")
    >>> ref = torch.randn(50, 8)
    >>> mgr.fit(ref)
    >>> mgr.build_index()
    >>> q = torch.randn(3, 8)
    >>> indices, distances = mgr.search(q, k=2)
    >>> indices.shape
    torch.Size([3, 2])

    HNSW search with custom parameters:

    >>> mgr_hnsw = IndexManager(method="hnsw", space="l2")
    >>> mgr_hnsw.fit(ref)
    >>> mgr_hnsw.build_index(hnsw_params={"M": 16, "efConstruction": 200})
    >>> indices, distances = mgr_hnsw.search(q, k=2)
    """

    def __init__(
        self,
        method: str = "hnsw",
        space: str = "l2",
        device: str | None = None,
        pca_components: int | None = None,
        pca_exp_var: float | None = None,
        dtype: torch.dtype = torch.float32,
        log: logging.Logger | None = None,
    ) -> None:
        if method not in ("hnsw", "brute"):
            raise ValueError(
                f"method must be 'hnsw' or 'brute', got {method!r}"
            )
        if space not in ("l2", "cosinesimil"):
            raise ValueError(
                f"space must be 'l2' or 'cosinesimil', got {space!r}"
            )
        if pca_components is not None and pca_exp_var is not None:
            raise ValueError(
                "Provide at most one of pca_components and pca_exp_var."
            )

        self.method: str = method
        self.space: str = space
        self._dtype: torch.dtype = dtype
        self._device: str | None = device
        self._pca_components: int | None = pca_components
        self._pca_exp_var: float | None = pca_exp_var
        self._logger: logging.Logger = (
            log if log is not None else _MODULE_LOGGER
        )

        # Internal mutable state
        self._ref_embeddings: torch.Tensor | None = None
        self._cal_embeddings: torch.Tensor | None = None
        self._index: Any | None = None  # nmslib index or torch.Tensor (brute)
        self._pca: Any | None = None  # sklearn PCA instance or None
        self._index_params: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        ref_embeddings: torch.Tensor | None = None,
        cal_embeddings: torch.Tensor | None = None,
    ) -> None:
        """Attach reference (and optional calibration) embeddings in-memory.

        Calling :meth:`fit` without *ref_embeddings* is allowed when the user
        intends to populate the index incrementally via :meth:`add_batch`.

        Parameters
        ----------
        ref_embeddings : torch.Tensor or None, optional
            Reference embeddings of shape ``(N, D)``.  When ``None``,
            incremental population via :meth:`add_batch` is expected before
            calling :meth:`build_index`.
        cal_embeddings : torch.Tensor or None, optional
            Calibration / validation embeddings of shape ``(M, D)``.
            Optional; stored for later retrieval via
            :meth:`get_val_embeddings`.

        Raises
        ------
        ValueError
            If any tensor is not 2-D, or if the embedding dimensions of
            *ref_embeddings* and *cal_embeddings* do not match.
        """
        if ref_embeddings is not None:
            if ref_embeddings.dim() != 2:
                raise ValueError("ref_embeddings must be 2-D (N, D)")
            ref_embeddings = ref_embeddings.to(dtype=self._dtype)
            if self._device is None:
                self._device = str(ref_embeddings.device)
            self._ref_embeddings = ref_embeddings

        if cal_embeddings is not None:
            if cal_embeddings.dim() != 2:
                raise ValueError("cal_embeddings must be 2-D (M, D)")
            if (
                ref_embeddings is not None
                and cal_embeddings.shape[1] != ref_embeddings.shape[1]
            ):
                raise ValueError(
                    "ref_embeddings and cal_embeddings must have the same "
                    "embedding dimension"
                )
            cal_embeddings = cal_embeddings.to(dtype=self._dtype)
            self._cal_embeddings = cal_embeddings

        self._logger.debug(
            "fit: ref=%s cal=%s",
            None
            if self._ref_embeddings is None
            else self._ref_embeddings.shape,
            None
            if self._cal_embeddings is None
            else self._cal_embeddings.shape,
        )

    def add_batch(
        self, embeddings: torch.Tensor, ids: torch.Tensor | None = None
    ) -> None:
        """Append a batch of reference vectors to the internal store.

        Vectors are concatenated with any existing reference embeddings.
        Call :meth:`build_index` after all batches have been added to
        construct the nearest-neighbour index.

        Parameters
        ----------
        embeddings : torch.Tensor
            Batch of reference embeddings of shape ``(B, D)``.
        ids : torch.Tensor or None, optional
            Optional integer IDs for the embeddings.  Currently stored for
            API completeness but not used during indexing.  Defaults to
            ``None``.

        Raises
        ------
        ValueError
            If *embeddings* is not 2-D or if its embedding dimension does
            not match previously added embeddings.
        """
        if embeddings.dim() != 2:
            raise ValueError("embeddings must be 2-D (B, D)")
        embeddings = embeddings.to(dtype=self._dtype)

        if self._ref_embeddings is None:
            self._ref_embeddings = embeddings
            if self._device is None:
                self._device = str(embeddings.device)
        else:
            if embeddings.shape[1] != self._ref_embeddings.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch: expected "
                    f"{self._ref_embeddings.shape[1]}, got {embeddings.shape[1]}"
                )
            self._ref_embeddings = torch.cat(
                [self._ref_embeddings, embeddings], dim=0
            )

        self._logger.debug(
            "add_batch: added %d vectors; total=%d",
            embeddings.shape[0],
            self._ref_embeddings.shape[0],
        )

    def build_index(
        self,
        from_memory: bool = True,
        hnsw_params: dict[str, Any] | None = None,
        **nmslib_options: Any,
    ) -> None:
        """Build or rebuild the underlying nearest-neighbour index.

        Parameters
        ----------
        from_memory : bool, optional
            When ``True`` (default), builds the index from all stored
            reference embeddings.  ``False`` is reserved for future use.
        hnsw_params : dict or None, optional
            HNSW-specific build parameters, merged on top of the
            auto-suggested defaults.  Recognised keys:

            - ``M`` (int) — number of bidirectional links per node
              (default: auto, recommended 16).
            - ``efConstruction`` (int) — size of the dynamic candidate list
              during construction (default: auto, recommended 200).
            - ``post`` (int) — post-processing level (default: 0).

            Only used when ``method="hnsw"``.
        **nmslib_options : Any
            Additional keyword arguments forwarded to
            ``nmslib.createIndex`` (e.g. ``print_progress=True``).  Only
            used when ``method="hnsw"``.

        Raises
        ------
        RuntimeError
            If no reference embeddings are available (neither :meth:`fit`
            nor :meth:`add_batch` has been called with data).
        ImportError
            If ``method="hnsw"`` and ``nmslib`` is not installed.
        """
        if self._ref_embeddings is None:
            raise RuntimeError(
                "No reference embeddings available. "
                "Call fit() or add_batch() first."
            )

        embs = self._prepare_embs(self._ref_embeddings, fit_pca=True)

        if self.method == "hnsw":
            self._build_hnsw(embs, hnsw_params=hnsw_params, **nmslib_options)
        else:
            self._build_brute(embs)

        self._logger.info(
            "build_index: method=%s space=%s N=%d D=%d",
            self.method,
            self.space,
            embs.shape[0],
            embs.shape[1],
        )

    def search(
        self, queries: torch.Tensor, k: int, return_distances: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Search for the *k* nearest neighbours.

        Parameters
        ----------
        queries : torch.Tensor
            Query embeddings of shape ``(Q, D)``.
        k : int
            Number of nearest neighbours to retrieve.
        return_distances : bool, optional
            When ``True`` (default), returns ``(indices, distances)``; when
            ``False``, returns only ``indices``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor] or torch.Tensor
            When *return_distances* is ``True``: a ``(indices, distances)``
            tuple, both of shape ``(Q, k)``.  When ``False``: only indices
            of shape ``(Q, k)``.

            * For ``space="l2"``, distances are *squared* Euclidean.
            * For ``space="cosinesimil"``, distances are cosine (1 − sim).

            Results are zero-padded when fewer than *k* neighbours are
            available and a :class:`UserWarning` is emitted.

        Raises
        ------
        RuntimeError
            If :meth:`build_index` has not been called.

        Raises
        ------
        ValueError
            If *queries* is not 2-D.
        """
        if self._index is None:
            raise RuntimeError(
                "Index has not been built. Call build_index() first."
            )
        if queries.dim() != 2:
            raise ValueError("queries must be 2-D (Q, D)")

        queries = queries.to(dtype=self._dtype)

        # Apply PCA transform if fitted
        if self._pca is not None:
            queries = self._transform_pca(queries)

        device = torch.device(self._device) if self._device else queries.device

        if self.method == "hnsw":
            indices, distances = self._search_hnsw(queries, k)
        else:
            indices, distances = self._search_brute(queries, k)

        indices = indices.to(device=device)
        distances = distances.to(device=device, dtype=self._dtype)

        self._logger.debug(
            "search: Q=%d k=%d method=%s", queries.shape[0], k, self.method
        )

        if return_distances:
            return indices, distances
        return indices

    def get_ref_embeddings(self) -> torch.Tensor | None:
        """Return the stored reference embeddings (before PCA).

        Returns
        -------
        torch.Tensor or None
            Reference embeddings of shape ``(N, D)`` or ``None`` when no
            embeddings have been provided.
        """
        return self._ref_embeddings

    def get_val_embeddings(self) -> torch.Tensor | None:
        """Return the stored calibration / validation embeddings.

        Returns
        -------
        torch.Tensor or None
            Calibration embeddings of shape ``(M, D)`` or ``None``.
        """
        return self._cal_embeddings

    def save(self, path: str) -> dict[str, str]:
        """Persist the index, embeddings, and metadata to disk.

        The following files are written (where ``<path>`` is the given
        prefix):

        * ``<path>_ref.pt``  — reference embeddings (``torch.save``).
        * ``<path>_cal.pt``  — calibration embeddings (when present).
        * ``<path>_index.bin`` — nmslib HNSW index (``method="hnsw"``).
        * ``<path>_brute.pt`` — brute-force reference tensor
          (``method="brute"``).
        * ``<path>_pca.pkl``  — scikit-learn PCA object (when PCA is used).
        * ``<path>_meta.json`` — JSON metadata.

        Parameters
        ----------
        path : str
            File-path prefix.  Parent directories are created automatically.

        Returns
        -------
        dict[str, str]
            Mapping of logical file keys (``"ref"``, ``"cal"``, ``"index"``,
            ``"brute"``, ``"pca"``, ``"meta"``) to their on-disk paths.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        saved: dict[str, str] = {}

        if self._ref_embeddings is not None:
            ref_path = f"{path}_ref.pt"
            torch.save(self._ref_embeddings.cpu(), ref_path)
            saved["ref"] = ref_path

        if self._cal_embeddings is not None:
            cal_path = f"{path}_cal.pt"
            torch.save(self._cal_embeddings.cpu(), cal_path)
            saved["cal"] = cal_path

        if self._index is not None:
            if self.method == "hnsw":
                index_path = f"{path}_index.bin"
                self._index.saveIndex(index_path, save_data=False)
                saved["index"] = index_path
            else:
                brute_path = f"{path}_brute.pt"
                assert isinstance(self._index, torch.Tensor)
                torch.save(self._index.cpu(), brute_path)
                saved["brute"] = brute_path

        if self._pca is not None:
            import pickle

            pca_path = f"{path}_pca.pkl"
            with open(pca_path, "wb") as fh:
                pickle.dump(self._pca, fh)
            saved["pca"] = pca_path

        meta: dict[str, Any] = {
            "method": self.method,
            "space": self.space,
            "dtype": str(self._dtype),
            "device": self._device,
            "pca_components": self._pca_components,
            "pca_exp_var": self._pca_exp_var,
            "index_params": self._index_params,
            "paths": saved,
        }
        meta_path = f"{path}_meta.json"
        with open(meta_path, "w") as fh:
            json.dump(meta, fh)
        saved["meta"] = meta_path

        self._logger.info("save: written to prefix '%s'", path)
        return saved

    def load(self, path: str) -> None:
        """Restore the index, embeddings, and metadata from disk.

        Parameters
        ----------
        path : str
            The prefix that was used when :meth:`save` was called.

        Raises
        ------
        FileNotFoundError
            If the metadata file ``<path>_meta.json`` does not exist.
        ImportError
            If ``method="hnsw"`` and ``nmslib`` is not installed.
        """
        meta_path = f"{path}_meta.json"
        if not Path(meta_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path) as fh:
            meta: dict[str, Any] = json.load(fh)

        self.method = meta["method"]
        self.space = meta["space"]
        self._device = meta.get("device")
        self._pca_components = meta.get("pca_components")
        self._pca_exp_var = meta.get("pca_exp_var")
        self._index_params = meta.get("index_params", {})
        paths: dict[str, str] = meta.get("paths", {})

        if "ref" in paths and Path(paths["ref"]).exists():
            self._ref_embeddings = torch.load(paths["ref"], weights_only=True)

        if "cal" in paths and Path(paths["cal"]).exists():
            self._cal_embeddings = torch.load(paths["cal"], weights_only=True)

        if "pca" in paths and Path(paths["pca"]).exists():
            import pickle

            with open(paths["pca"], "rb") as fh:
                self._pca = pickle.load(fh)  # noqa: S301

        if self.method == "hnsw" and "index" in paths:
            try:
                import nmslib
            except ImportError:
                raise ImportError(
                    "nmslib is required to load an HNSW index. "
                    "Install it with `pip install nmslib`."
                )
            embs = (
                self._prepare_embs(self._ref_embeddings, fit_pca=False)
                if self._ref_embeddings is not None
                else None
            )
            index = nmslib.init(method="hnsw", space=self.space)
            if embs is not None:
                index.addDataPointBatch(embs.cpu())
            index.loadIndex(paths["index"], load_data=False)
            self._index = index

        elif self.method == "brute" and "brute" in paths:
            if Path(paths["brute"]).exists():
                self._index = torch.load(paths["brute"], weights_only=True)

        self._logger.info("load: restored from prefix '%s'", path)

    def reset(self) -> None:
        """Reset all internal state.

        Clears stored embeddings, the index, PCA model, and index
        parameters.  The construction-time hyper-parameters (*method*,
        *space*, *dtype*, *device*, *pca_components*, *pca_exp_var*) are
        preserved.
        """
        self._ref_embeddings = None
        self._cal_embeddings = None
        self._index = None
        self._pca = None
        self._index_params = {}
        self._logger.debug("reset: all state cleared")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_pca(self, embs: torch.Tensor) -> torch.Tensor:
        """Fit a scikit-learn PCA on *embs* and return transformed embeddings."""
        from sklearn.decomposition import PCA as SklearnPCA

        if self._pca_components is not None:
            n_components: int | float = min(
                self._pca_components, embs.shape[0], embs.shape[1]
            )
        else:
            assert self._pca_exp_var is not None
            n_components = self._pca_exp_var

        pca = SklearnPCA(n_components=n_components, svd_solver="full")
        embs_np = embs.cpu().float().numpy()
        transformed = pca.fit_transform(embs_np)
        self._pca = pca

        explained = float(sum(pca.explained_variance_ratio_))
        self._logger.info(
            "PCA: %d -> %d dimensions (%.4f explained variance)",
            embs.shape[1],
            pca.n_components_,
            explained,
        )
        return torch.from_numpy(transformed).to(dtype=self._dtype)

    def _transform_pca(self, embs: torch.Tensor) -> torch.Tensor:
        """Apply the fitted PCA transform to *embs*."""
        assert self._pca is not None, "PCA has not been fitted"
        import numpy as np

        transformed = self._pca.transform(embs.cpu().float().numpy())
        return torch.from_numpy(np.asarray(transformed, dtype=np.float32)).to(
            dtype=self._dtype
        )

    def _prepare_embs(
        self, embs: torch.Tensor, fit_pca: bool = False
    ) -> torch.Tensor:
        """Optionally apply (or fit-and-apply) PCA to *embs*."""
        uses_pca = (
            self._pca_components is not None or self._pca_exp_var is not None
        )
        if uses_pca:
            if fit_pca:
                return self._fit_pca(embs)
            return self._transform_pca(embs)
        return embs

    def _build_hnsw(
        self,
        embs: torch.Tensor,
        hnsw_params: dict[str, Any] | None = None,
        **nmslib_options: Any,
    ) -> None:
        """Build an nmslib HNSW index from *embs*."""
        try:
            import nmslib
        except ImportError:
            raise ImportError(
                "nmslib is required for method='hnsw'. "
                "Install it with `pip install nmslib`."
            )

        params = self._suggest_hnsw_params(embs=embs)
        if hnsw_params:
            params["build_defaults"].update(hnsw_params)
        self._index_params = params

        index = nmslib.init(method="hnsw", space=self.space)
        index.addDataPointBatch(embs.cpu())
        index.createIndex(
            index_params=params["build_defaults"],
            print_progress=bool(nmslib_options.get("print_progress", False)),
        )
        self._index = index

    def _build_brute(self, embs: torch.Tensor) -> None:
        """Store *embs* for brute-force pairwise search."""
        self._index = embs.clone()

    def _search_hnsw(
        self, queries: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query the HNSW index and return ``(indices, distances)``."""
        import nmslib

        nmslib.setQueryTimeParams(
            self._index, self._index_params.get("query_defaults", {})
        )
        assert self._index is not None
        results = self._index.knnQueryBatch(queries.cpu(), k=k)
        return self._results_to_tensors(results, k)

    def _search_brute(
        self, queries: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Brute-force nearest-neighbour search.

        Returns squared L2 distances for ``space="l2"`` and cosine
        distances for ``space="cosinesimil"``, matching nmslib conventions.
        """
        assert isinstance(self._index, torch.Tensor)
        ref: torch.Tensor = self._index.to(dtype=self._dtype)

        if self.space == "l2":
            # Squared Euclidean: ||q-r||² = ||q||²+||r||²-2·q·rᵀ
            q_sq = (queries**2).sum(dim=1, keepdim=True)
            r_sq = (ref**2).sum(dim=1, keepdim=True)
            dists = q_sq + r_sq.T - 2.0 * (queries @ ref.T)
            dists = dists.clamp(min=0.0)
        elif self.space == "cosinesimil":
            q_norm = torch.nn.functional.normalize(queries, dim=1)
            r_norm = torch.nn.functional.normalize(ref, dim=1)
            dists = 1.0 - (q_norm @ r_norm.T)
        else:
            raise ValueError(f"Unknown space: {self.space!r}")

        actual_k = min(k, ref.shape[0])
        if actual_k < k:
            warnings.warn(
                f"Fewer than {k} reference points available "
                f"({ref.shape[0]}). "
                "Applying zero padding to the distance tensor.",
                UserWarning,
                stacklevel=3,
            )

        top_dists, top_indices = torch.topk(
            dists, k=actual_k, dim=1, largest=False
        )

        if actual_k < k:
            pad = k - actual_k
            pad_idx = torch.zeros(queries.shape[0], pad, dtype=torch.int64)
            pad_dst = torch.zeros(queries.shape[0], pad, dtype=self._dtype)
            top_indices = torch.cat([top_indices, pad_idx], dim=1)
            top_dists = torch.cat([top_dists, pad_dst], dim=1)

        return top_indices, top_dists

    def _results_to_tensors(
        self, results: list[tuple[Any, Any]], k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert nmslib ``knnQueryBatch`` output to tensor form.

        Zero-pads results shorter than *k* and emits a :class:`UserWarning`.
        """
        import numpy as np

        all_indices = []
        all_distances = []

        for i, res in enumerate(results):
            idx = torch.from_numpy(np.asarray(res[0], dtype=np.int64))
            dst = torch.from_numpy(np.asarray(res[1], dtype=np.float32))
            n = len(idx)
            if n < k:
                warnings.warn(
                    f"Query {i} returned fewer than {k} neighbors. "
                    "Applying zero padding to the distance tensor.",
                    UserWarning,
                    stacklevel=3,
                )
                pad = k - n
                idx = torch.cat([idx, torch.zeros(pad, dtype=torch.int64)])
                dst = torch.cat([dst, torch.zeros(pad, dtype=torch.float32)])
            all_indices.append(idx[:k].unsqueeze(0))
            all_distances.append(dst[:k].unsqueeze(0))

        return torch.cat(all_indices, dim=0), torch.cat(all_distances, dim=0)

    @staticmethod
    def _suggest_hnsw_params(embs: torch.Tensor, k: int = 10) -> dict[str, Any]:
        """Suggest conservative HNSW build and query parameters.

        Parameters
        ----------
        embs : torch.Tensor
            Reference embeddings of shape ``(N, D)``.
        k : int, optional
            Number of neighbours used to derive ``efSearch``.

        Returns
        -------
        dict[str, Any]
            Dict with keys ``"build_defaults"`` and ``"query_defaults"``.

        Notes
        -----
        Recommended defaults: ``M=16``, ``efConstruction=200``.
        For very small datasets (N < 10), conservative fallback params are
        used.
        """
        if embs.dim() != 2:
            raise ValueError("ref_embeddings must be 2-D (N, D)")
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
