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
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

import torch

from seapig.scores.utils import TensorPCA
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

    Both the PCA and the index support two population modes:

    1. **Single-shot** — pass all reference embeddings to :meth:`fit` then
       call :meth:`build_index`.  PCA is fitted on the full set in one pass.
    2. **Batch-wise** (memory-efficient) — call :meth:`add_batch` repeatedly:

       * **With PCA**: each batch is written to a temporary directory on disk
         and :meth:`TensorPCA.partial_fit` is called.  :meth:`build_index`
         then finalises PCA and processes every on-disk batch in order
         (transform → add to index), keeping only one batch in memory at a
         time.  Temp files are deleted after the index is built.
       * **Without PCA, HNSW**: batches are added directly to an nmslib
         index in-memory (``addDataPointBatch``); :meth:`build_index` calls
         ``createIndex`` once.  Raw embeddings are never accumulated.
       * **Without PCA, brute**: batches are written to a temp directory;
         :meth:`build_index` loads and concatenates them once to form the
         brute-force index.  Temp files are deleted after build.

    In all batch-wise modes :meth:`get_ref_embeddings` returns ``None`` (raw
    embeddings are not held in RAM after `build_index` completes).

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
    pca : TensorPCA or None, optional
        A fitted or unfitted :class:`TensorPCA` instance used for
        dimensionality reduction before indexing.  When ``None`` (default),
        no dimensionality reduction is applied.
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
    pca : TensorPCA or None
        The PCA preprocessing object (``None`` when no PCA is used).

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
    Brute-force search (single-shot):

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

    Memory-efficient batch-wise population with PCA:

    >>> from seapig.scores.utils import TensorPCA
    >>> mgr_pca = IndexManager(method="brute", pca=TensorPCA(n_components=4))
    >>> for batch in [torch.randn(20, 8), torch.randn(20, 8)]:
    ...     mgr_pca.add_batch(batch)
    >>> mgr_pca.build_index()
    >>> indices, distances = mgr_pca.search(torch.randn(3, 8), k=2)
    """

    def __init__(
        self,
        method: str = "hnsw",
        space: str = "l2",
        device: str | None = None,
        pca: TensorPCA | None = None,
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
        if pca is not None and not isinstance(pca, TensorPCA):
            raise TypeError(
                f"pca must be a TensorPCA instance or None, got {type(pca)!r}"
            )

        self.method: str = method
        self.space: str = space
        self._dtype: torch.dtype = dtype
        self._device: str | None = device
        self.pca: TensorPCA | None = pca
        self._logger: logging.Logger = (
            log if log is not None else _MODULE_LOGGER
        )

        # Internal mutable state
        self._ref_embeddings: torch.Tensor | None = None  # single-shot only
        self._cal_embeddings: torch.Tensor | None = None
        self._index: Any | None = None  # built nmslib index or torch.Tensor
        self._index_built: bool = False  # True after build_index() completes
        self._index_params: dict[str, Any] = {}
        # Batch-mode tracking (populated via add_batch)
        self._embedding_dim: int | None = None
        self._n_total_vectors: int = 0
        self._pca_partial_batches: bool = False
        self._batch_tmp_dir: str | None = None
        self._batch_file_paths: list[str] = []

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
            self._embedding_dim = ref_embeddings.shape[1]

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
        """Add a batch of reference vectors without accumulating in RAM.

        In batch-wise mode the raw embeddings are **never** kept in
        ``_ref_embeddings``; instead they are routed as follows:

        * **With PCA** — the batch is written to a temporary directory on
          disk and :meth:`TensorPCA.partial_fit` is called.
          :meth:`build_index` will later finalize PCA and re-read each
          on-disk batch to transform and index it.
        * **Without PCA, HNSW** — ``addDataPointBatch`` is called on a
          pre-initialized nmslib index directly; no disk I/O.
        * **Without PCA, brute** — the batch is written to a temporary
          directory on disk; :meth:`build_index` will load and concatenate
          all batches once to build the brute-force index.

        Call :meth:`build_index` after all batches have been added to
        construct and finalize the nearest-neighbour index.

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

        # Track dimension and device on first call
        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[1]
            if self._device is None:
                self._device = str(embeddings.device)
        elif embeddings.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected "
                f"{self._embedding_dim}, got {embeddings.shape[1]}"
            )

        self._n_total_vectors += embeddings.shape[0]

        if self.pca is not None:
            # Write raw batch to disk; will be transformed after PCA finalize
            self._ensure_batch_tmp_dir()
            batch_path = os.path.join(
                self._batch_tmp_dir,  # type: ignore[arg-type]
                f"batch_{len(self._batch_file_paths):06d}.pt",
            )
            torch.save(embeddings.cpu(), batch_path)
            self._batch_file_paths.append(batch_path)
            self.pca.partial_fit(embeddings)
            self._pca_partial_batches = True

        elif self.method == "hnsw":
            # Populate nmslib index incrementally; createIndex deferred
            if self._index is None:
                try:
                    import nmslib
                except ImportError:
                    raise ImportError(
                        "nmslib is required for method='hnsw'. "
                        "Install it with `pip install nmslib`."
                    )
                self._index = nmslib.init(method="hnsw", space=self.space)
            self._index.addDataPointBatch(embeddings.cpu())

        else:
            # Brute force: write to disk to avoid memory accumulation
            self._ensure_batch_tmp_dir()
            batch_path = os.path.join(
                self._batch_tmp_dir,  # type: ignore[arg-type]
                f"batch_{len(self._batch_file_paths):06d}.pt",
            )
            torch.save(embeddings.cpu(), batch_path)
            self._batch_file_paths.append(batch_path)

        self._logger.debug(
            "add_batch: added %d vectors; total=%d",
            embeddings.shape[0],
            self._n_total_vectors,
        )

    def build_index(
        self,
        from_memory: bool = True,
        hnsw_params: dict[str, Any] | None = None,
        **nmslib_options: Any,
    ) -> None:
        """Build or rebuild the underlying nearest-neighbour index.

        Dispatches to one of four internal build paths depending on how data
        was provided:

        1. **Single-shot** (via :meth:`fit`): PCA is fitted on the full
           reference set, then the index is built from the transformed
           embeddings in a single pass.
        2. **Batch-wise with PCA**: PCA is finalised from accumulated partial
           statistics, then each on-disk batch is loaded, transformed, and
           added to the index one at a time.  Temp files are removed on
           completion.
        3. **Batch-wise HNSW without PCA**: ``createIndex`` is called on the
           pre-populated nmslib index.
        4. **Batch-wise brute without PCA**: on-disk batches are loaded and
           concatenated to form the in-memory brute-force index.  Temp files
           are removed on completion.

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
        has_single_shot = self._ref_embeddings is not None
        has_batch = self._n_total_vectors > 0 or (
            self.method == "hnsw" and self._index is not None
        )

        if not has_single_shot and not has_batch:
            raise RuntimeError(
                "No reference embeddings available. "
                "Call fit() or add_batch() first."
            )

        if has_single_shot:
            # ----------------------------------------------------------------
            # Path 1: single-shot — all data passed to fit()
            # ----------------------------------------------------------------
            if self.pca is not None:
                self.pca.fit(self._ref_embeddings)
            embs = self._prepare_embs(self._ref_embeddings)  # type: ignore[arg-type]
            if self.method == "hnsw":
                self._build_hnsw(
                    embs, hnsw_params=hnsw_params, **nmslib_options
                )
            else:
                self._build_brute(embs)

        elif self._pca_partial_batches:
            # ----------------------------------------------------------------
            # Path 2: batch-wise with PCA — finalize then re-read disk batches
            # ----------------------------------------------------------------
            assert self.pca is not None
            self.pca.finalize()
            # Derive reduced dimension from finalized PCA
            reduced_dim = int(self.pca.q)

            if self.method == "hnsw":
                try:
                    import nmslib
                except ImportError:
                    raise ImportError(
                        "nmslib is required for method='hnsw'. "
                        "Install it with `pip install nmslib`."
                    )
                index = nmslib.init(method="hnsw", space=self.space)
                for batch_path in self._batch_file_paths:
                    batch = torch.load(batch_path, weights_only=True).to(
                        dtype=self._dtype
                    )
                    transformed = self.pca.transform(batch).to(
                        dtype=self._dtype
                    )
                    index.addDataPointBatch(transformed.cpu())
                params = self._suggest_hnsw_params_from_dims(
                    N=self._n_total_vectors, D=reduced_dim, k=10
                )
                if hnsw_params:
                    params["build_defaults"].update(hnsw_params)
                self._index_params = params
                index.createIndex(
                    index_params=params["build_defaults"],
                    print_progress=bool(
                        nmslib_options.get("print_progress", False)
                    ),
                )
                self._index = index
            else:
                parts: list[torch.Tensor] = []
                for batch_path in self._batch_file_paths:
                    batch = torch.load(batch_path, weights_only=True).to(
                        dtype=self._dtype
                    )
                    transformed = self.pca.transform(batch).to(
                        dtype=self._dtype
                    )
                    parts.append(transformed)
                self._build_brute(torch.cat(parts, dim=0))
            self._cleanup_batch_tmp_dir()

        elif self.method == "hnsw" and self._index is not None:
            # ----------------------------------------------------------------
            # Path 3: batch-wise HNSW without PCA — createIndex on pre-pop index
            # ----------------------------------------------------------------
            assert self._embedding_dim is not None
            params = self._suggest_hnsw_params_from_dims(
                N=self._n_total_vectors, D=self._embedding_dim, k=10
            )
            if hnsw_params:
                params["build_defaults"].update(hnsw_params)
            self._index_params = params
            self._index.createIndex(
                index_params=params["build_defaults"],
                print_progress=bool(
                    nmslib_options.get("print_progress", False)
                ),
            )

        else:
            # ----------------------------------------------------------------
            # Path 4: batch-wise brute without PCA — load + concat disk batches
            # ----------------------------------------------------------------
            batch_parts: list[torch.Tensor] = []
            for batch_path in self._batch_file_paths:
                batch_parts.append(torch.load(batch_path, weights_only=True))
            self._build_brute(torch.cat(batch_parts, dim=0))
            self._cleanup_batch_tmp_dir()

        self._index_built = True
        self._logger.info(
            "build_index: method=%s space=%s N=%d",
            self.method,
            self.space,
            self._n_total_vectors
            or (
                self._ref_embeddings.shape[0]
                if self._ref_embeddings is not None
                else 0
            ),
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
        if not self._index_built:
            raise RuntimeError(
                "Index has not been built. Call build_index() first."
            )
        if queries.dim() != 2:
            raise ValueError("queries must be 2-D (Q, D)")

        queries = queries.to(dtype=self._dtype)

        # Apply PCA transform if a PCA model is set
        if self.pca is not None:
            queries = self.pca.transform(queries).to(dtype=self._dtype)

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

        Only available in single-shot mode (data passed to :meth:`fit`).
        In batch-wise mode (data added via :meth:`add_batch`) this always
        returns ``None`` because raw embeddings are not accumulated in RAM.

        Returns
        -------
        torch.Tensor or None
            Reference embeddings of shape ``(N, D)`` or ``None``.
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
        * ``<path>_pca.pt``  — :class:`TensorPCA` state dict (when PCA is
          used).
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
                self._index.saveIndex(index_path, save_data=True)
                saved["index"] = index_path
            else:
                brute_path = f"{path}_brute.pt"
                assert isinstance(self._index, torch.Tensor)
                torch.save(self._index.cpu(), brute_path)
                saved["brute"] = brute_path

        if self.pca is not None:
            pca_path = f"{path}_pca.pt"
            torch.save(self.pca.state_dict(), pca_path)
            saved["pca"] = pca_path

        pca_meta: dict[str, Any] = (
            {
                "pca_n_components": self.pca.n_components,
                "pca_gamma": self.pca.gamma,
                "pca_M": self.pca.M,
                "pca_mode": self.pca.mode,
            }
            if self.pca is not None
            else {
                "pca_n_components": None,
                "pca_gamma": None,
                "pca_M": None,
                "pca_mode": None,
            }
        )
        meta: dict[str, Any] = {
            "method": self.method,
            "space": self.space,
            "dtype": str(self._dtype),
            "device": self._device,
            **pca_meta,
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
        self._index_params = meta.get("index_params", {})
        paths: dict[str, str] = meta.get("paths", {})

        if "ref" in paths and Path(paths["ref"]).exists():
            self._ref_embeddings = torch.load(paths["ref"], weights_only=True)

        if "cal" in paths and Path(paths["cal"]).exists():
            self._cal_embeddings = torch.load(paths["cal"], weights_only=True)

        if "pca" in paths and Path(paths["pca"]).exists():
            pca_n_components = meta.get("pca_n_components")
            if pca_n_components is not None:
                pca = TensorPCA(
                    n_components=pca_n_components,
                    gamma=meta.get("pca_gamma"),
                    M=meta.get("pca_M"),
                    mode=meta.get("pca_mode"),
                )
                sd = torch.load(paths["pca"], weights_only=True)
                pca.load_state_dict(sd)
                self.pca = pca

        if self.method == "hnsw" and "index" in paths:
            try:
                import nmslib
            except ImportError:
                raise ImportError(
                    "nmslib is required to load an HNSW index. "
                    "Install it with `pip install nmslib`."
                )
            index = nmslib.init(method="hnsw", space=self.space)
            index.loadIndex(paths["index"], load_data=True)
            self._index = index
            self._index_built = True

        elif self.method == "brute" and "brute" in paths:
            if Path(paths["brute"]).exists():
                self._index = torch.load(paths["brute"], weights_only=True)
                self._index_built = True

        self._logger.info("load: restored from prefix '%s'", path)

    def reset(self) -> None:
        """Reset all internal state.

        Clears stored embeddings, the index, PCA accumulators, batch tracking
        state, and index parameters.  Any temporary on-disk files created
        during batch-wise population are deleted.  The construction-time
        hyper-parameters (*method*, *space*, *dtype*, *device*, *pca*) are
        preserved; if a :class:`TensorPCA` instance was supplied, its
        partial-fit accumulators are reset so it can be re-fitted on new data.
        """
        self._ref_embeddings = None
        self._cal_embeddings = None
        self._index = None
        self._index_built = False
        self._index_params = {}
        self._pca_partial_batches = False
        self._embedding_dim = None
        self._n_total_vectors = 0
        self._cleanup_batch_tmp_dir()
        if self.pca is not None:
            self.pca.reset_partial()
        self._logger.debug("reset: all state cleared")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_batch_tmp_dir(self) -> None:
        """Create a temporary directory for batch files if not already done."""
        if self._batch_tmp_dir is None:
            self._batch_tmp_dir = tempfile.mkdtemp(prefix="seapig_index_")

    def _cleanup_batch_tmp_dir(self) -> None:
        """Delete the temporary batch directory and reset tracking lists."""
        if self._batch_tmp_dir is not None:
            try:
                shutil.rmtree(self._batch_tmp_dir)
            except OSError as exc:
                self._logger.warning(
                    "Failed to remove temporary batch directory %r: %s",
                    self._batch_tmp_dir,
                    exc,
                )
            self._batch_tmp_dir = None
        self._batch_file_paths = []

    def _prepare_embs(self, embs: torch.Tensor) -> torch.Tensor:
        """Optionally apply the fitted PCA transform to *embs*.

        When :attr:`pca` is set the embeddings are projected onto the
        retained principal components and cast to :attr:`_dtype`.
        """
        if self.pca is not None:
            return self.pca.transform(embs).to(dtype=self._dtype)
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
        return IndexManager._suggest_hnsw_params_from_dims(N=N, D=D, k=k)

    @staticmethod
    def _suggest_hnsw_params_from_dims(
        N: int, D: int, k: int = 10
    ) -> dict[str, Any]:
        """Suggest conservative HNSW parameters from dataset dimensions.

        This variant accepts integer ``(N, D)`` directly and is used in
        batch-wise mode where a single tensor containing all embeddings is
        not available.

        Parameters
        ----------
        N : int
            Total number of reference vectors.
        D : int
            Embedding dimension (after PCA, if applicable).
        k : int, optional
            Number of neighbours used to derive ``efSearch``.

        Returns
        -------
        dict[str, Any]
            Dict with keys ``"build_defaults"`` and ``"query_defaults"``.
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
