"""Abstract Base Method for embeddings based confidence scores."""

import inspect
import warnings
from abc import ABC
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader
from typing_extensions import override

from seapig.scores.base import ConfidenceScore
from seapig.scores.utils import TensorPCA
from seapig.utils import get_logger
from seapig.utils.progress import track

logger = get_logger(__name__)


class EmbeddingScore(ConfidenceScore, ABC):
    """Base class for embedding-based confidence scores.

    Embedding-based scores quantify deviation from the training distribution using
    latent-space embeddings. Low scores indicate samples similar to the training
    distribution (likely inliers), while high scores indicate samples deviating
    from the training distribution (likely outliers).

    Parameters
    ----------
    pca:
        A `TensorPCA` instance or `None`. If provided, this `TensorPCA` object will
        be used to perform dimensionality reduction on embeddings prior to
        scoring (for example, to retain a specified explained variance).
        Defaults to `None`, indicating that dimensionality reduction is not applied.

    Attributes
    ----------
    ref_embeddings:
        A `torch.Tensor` with the embeddings of trainings samples. Defaults to `None`.
    cal_embeddings:
        A `torch.Tensor` with the embeddings of validation samples. Defaults to `None`.
    scores:
        A `torch.Tensor` with the confidence scores of the validation samples.
        Low scores indicate likely inliers, high scores indicate likely outliers.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Samples with scores higher
        than this threshold are excluded from prediction. Defaults to `None`.
    """

    ref_embeddings: torch.Tensor | None
    cal_embeddings: torch.Tensor | None
    train_required: bool = True
    pca: TensorPCA | None

    def __init__(self, pca: TensorPCA | None = None) -> None:
        super().__init__()
        self.pca = pca
        self.register_buffer("ref_embeddings", None)
        self.register_buffer("cal_embeddings", None, persistent=False)
        # Incremental fitting state
        self._partial_active: bool = False
        self._ref_embs_batches: list[torch.Tensor] = []
        self._cal_embs_batches: list[torch.Tensor] = []
        self._batch_paths: list[Path] = []
        self._cal_batch_paths: list[Path] = []
        self._n_ref_samples: int = 0
        self._batch_count: int = 0

    @staticmethod
    def _setup_path(
        outdir: Path | None = None, prefix: str | None = None
    ) -> Path | None:
        """Construct the output path for a parquet file."""
        if outdir is None or prefix is None:
            return None
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)
        return outdir / f"{prefix}.pt"

    @staticmethod
    def _check_model(model: torch.nn.Module) -> None:
        """Check a model for compatibility with embeddings-based confidence scores."""
        assert isinstance(model, torch.nn.Module)
        if not callable(model.embed):
            raise Exception("model is required to have a `.embed()` method.")
        sig = inspect.signature(obj=model.embed)
        if "x" not in sig.parameters.keys():
            raise Exception(
                "`.embed()` method is required to except `x` as argument."
            )

    @staticmethod
    def _write_pt(x: torch.Tensor, path: Path) -> None:
        """Write a `torch.Tensor` to disk."""
        torch.save(x.cpu(), path)

    @staticmethod
    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _load_pt(path: Path) -> torch.Tensor:
        """Read a file from disk to a `torch.Tensor`."""
        v: torch.Tensor = torch.load(path)
        return v

    @classmethod
    def _loadorembed(
        self,
        path: Path | None,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Load from file or iterate over dataloader to extract embeddings."""
        if path is not None and path.is_file():
            warnings.warn(
                f"Loading pre-existing embeddings from {path}.", UserWarning
            )
            v = self._load_pt(path)
            device = next(model.parameters()).device
            v = v.to(device)
        else:
            v = self._embed_dl(model=model, loader=loader)
            if path is not None:
                self._write_pt(v, path)
        return v

    @classmethod
    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _embed(
        self, X: torch.Tensor | dict[str, torch.Tensor], model: torch.nn.Module
    ) -> torch.Tensor:
        """Embed a batch based on a models embed method."""
        assert callable(model.embed)
        if isinstance(X, dict):
            if "image" not in X.keys():
                raise KeyError(
                    'A batch dictionary is required to contain the "image" key.'
                )
            z = model.embed(X["image"])
        elif isinstance(X, (list, tuple)):
            z = model.embed(X[0])
        else:
            z = model.embed(X)
        assert isinstance(z, torch.Tensor)
        if len(z.shape) > 2:  # we expect (B,D)
            raise ValueError(
                f"Expected embed method to return tensor of shape (B,D) but got {z.shape}"
            )
        return z

    @classmethod
    def _embed_dl(
        self,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Extract embeddings by iterating over a DataLoader.

        This method ensures the model is in eval mode during embedding extraction
        to ensure consistent behavior regardless of the model's initial state.
        The model's original training state is restored after embedding extraction.
        """
        assert callable(model.embed)
        # Save the current training state and set model to eval mode
        was_training = model.training
        model.eval()

        pbar_desc = f"Embedding {len(loader)} batches"
        embs_ls = list()
        for batch in track(
            loader, total=len(loader), desc=pbar_desc, unit="batches"
        ):
            z = self._embed(X=batch, model=model)
            embs_ls.append(z)
        embs = torch.cat(embs_ls, dim=0)

        # Restore the original training state
        if was_training:
            model.train()

        return embs

    @classmethod
    def _embed_from_dict(
        self,
        model: torch.nn.Module,
        loaders: dict[str, DataLoader[torch.Tensor | dict[str, torch.Tensor]]],
        key: Literal["train", "val"],
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> torch.Tensor:
        """Embed a loader from a specified key in a dictionary."""
        path = None
        assert isinstance(loaders, dict)
        assert isinstance(model, torch.nn.Module)
        if outdir is not None and prefix is None:
            warnings.warn(
                "'outdir' has been specified but 'prefix' is None.\n"
                "Consider specifying 'prefix' as well to enable saving embeddings.",
                UserWarning,
            )
        self._check_model(model)
        if key not in loaders.keys():
            raise KeyError(f"Missing key `{key}` in loaders dictionary.")
        loader = loaders[key]
        assert isinstance(loader, DataLoader)
        if prefix is not None:
            path = self._setup_path(outdir, prefix + f"-embeddings-{key}")
        embs = self._loadorembed(path, model, loader)
        return embs

    def _fit_pca(self) -> None:
        assert self.ref_embeddings is not None
        assert isinstance(self.pca, TensorPCA)
        self.pca.fit(self.ref_embeddings)

    # ------------------------------------------------------------------
    # Incremental (batch-wise) fitting helpers and public API
    # ------------------------------------------------------------------

    def reset_partial(self) -> None:
        """Reset all incremental accumulators and counters.

        This should be called before starting a new incremental fitting
        session. It is called automatically by ``partial_fit`` on its
        first invocation when no session is active.
        """
        self._partial_active = False
        self._ref_embs_batches = []
        self._cal_embs_batches = []
        self._batch_paths = []
        self._cal_batch_paths = []
        self._n_ref_samples = 0
        self._batch_count = 0
        if self.pca is not None:
            self.pca.reset_partial()

    def _accumulate_ref_embeddings(
        self,
        embs: torch.Tensor,
        write_to_disk: bool = False,
        outdir: Path | None = None,
        prefix: str | None = None,
        key: str = "train",
    ) -> None:
        """Accumulate a batch of embeddings in memory or write to disk.

        Parameters
        ----------
        embs:
            Embeddings tensor for the current batch.
        write_to_disk:
            If ``True`` and ``outdir`` is provided, write the batch to a
            ``.pt`` file instead of keeping it in memory.
        outdir:
            Directory for batch files. Only used when ``write_to_disk=True``.
        prefix:
            Filename prefix for batch files. Defaults to ``"emb"`` when
            ``outdir`` is given but ``prefix`` is ``None``.
        key:
            Either ``"train"`` or ``"val"``, selects the accumulator.
        """
        if write_to_disk and outdir is not None:
            eff_prefix = prefix
            if prefix is None:
                warnings.warn(
                    "'outdir' has been specified but 'prefix' is None."
                    " Using default prefix 'emb'.",
                    UserWarning,
                )
                eff_prefix = "emb"
            batch_idx = (
                self._batch_count
                if key == "train"
                else len(self._cal_batch_paths)
            )
            file_prefix = f"{eff_prefix}-embeddings-{key}-batch-{batch_idx:03d}"
            path = self._setup_path(outdir, file_prefix)
            assert path is not None
            self._write_pt(embs, path)
            if key == "train":
                self._batch_paths.append(path)
            else:
                self._cal_batch_paths.append(path)
        else:
            if key == "train":
                self._ref_embs_batches.append(embs)
            else:
                self._cal_embs_batches.append(embs)

    def _finalize_ref_embeddings(self, keep_batch_files: bool = False) -> None:
        """Concatenate accumulated batches into ``ref_embeddings`` / ``cal_embeddings``.

        Parameters
        ----------
        keep_batch_files:
            If ``False`` (default), per-batch files written during
            batch-write mode are deleted after loading.
        """
        if self._batch_paths:
            emb_list = [self._load_pt(p) for p in self._batch_paths]
            self.ref_embeddings = torch.cat(emb_list, dim=0)
            if not keep_batch_files:
                for p in self._batch_paths:
                    p.unlink(missing_ok=True)
        elif self._ref_embs_batches:
            self.ref_embeddings = torch.cat(self._ref_embs_batches, dim=0)

        if self._cal_batch_paths:
            emb_list = [self._load_pt(p) for p in self._cal_batch_paths]
            self.cal_embeddings = torch.cat(emb_list, dim=0)
            if not keep_batch_files:
                for p in self._cal_batch_paths:
                    p.unlink(missing_ok=True)
        elif self._cal_embs_batches:
            self.cal_embeddings = torch.cat(self._cal_embs_batches, dim=0)

        # clear accumulators
        self._ref_embs_batches = []
        self._cal_embs_batches = []
        self._batch_paths = []
        self._cal_batch_paths = []

    def partial_fit(
        self,
        X: torch.Tensor | None = None,
        *,
        model: torch.nn.Module | None = None,
        batch: torch.Tensor | dict[str, torch.Tensor] | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
        write_batch: bool | None = None,
    ) -> None:
        """Process a single training batch for incremental fitting.

        This method accumulates embeddings batch by batch. Call
        ``finalize()`` once all batches have been processed to set
        ``ref_embeddings`` and complete the incremental fitting session.

        If no session is active (i.e. ``reset_partial`` was not called
        beforehand), one is started automatically on the first call.

        Parameters
        ----------
        X:
            Pre-computed embeddings for this batch. Mutually exclusive
            with ``model`` + ``batch``.
        model:
            A ``torch.nn.Module`` with an ``.embed()`` method. Required
            when ``X`` is not provided.
        batch:
            A raw input batch (tensor or dict with ``"image"`` key).
            Required when ``X`` is not provided.
        outdir:
            Output directory for per-batch embedding files.
            Only used when ``write_batch=True``.
        prefix:
            Filename prefix for per-batch embedding files.
            Only used when ``write_batch=True``.
        write_batch:
            Whether to write this batch to disk instead of accumulating
            in memory. Defaults to ``False``.
        """
        if not self._partial_active:
            self.reset_partial()
            self._partial_active = True

        if X is not None:
            embs = X
        else:
            if model is None or batch is None:
                raise ValueError(
                    "Provide X (embeddings) or both model and batch "
                    "when using partial_fit."
                )
            embs = self._embed(X=batch, model=model)

        write_batch_flag = (
            bool(write_batch) if write_batch is not None else False
        )

        if self.pca is not None:
            self.pca.partial_fit(embs)

        self._accumulate_ref_embeddings(
            embs,
            write_to_disk=write_batch_flag,
            outdir=outdir,
            prefix=prefix,
            key="train",
        )
        self._n_ref_samples += embs.shape[0]
        self._batch_count += 1

    def finalize(self, keep_batch_files: bool = False) -> None:
        """Finalize incremental fitting and populate ``ref_embeddings``.

        Concatenates all accumulated batch embeddings (or loads them from
        disk) into ``ref_embeddings`` and, if validation batches were
        accumulated, into ``cal_embeddings``. If a ``TensorPCA`` is
        attached, ``pca.finalize()`` is also called.

        Parameters
        ----------
        keep_batch_files:
            Whether to keep per-batch ``.pt`` files written during
            batch-write mode. Defaults to ``False`` (files are removed
            after loading).

        Raises
        ------
        RuntimeError
            If called without any prior ``partial_fit`` calls.
        """
        if (
            not self._partial_active
            and not self._ref_embs_batches
            and not self._batch_paths
        ):
            raise RuntimeError(
                "No data provided to partial_fit before calling finalize()."
            )

        if self.pca is not None and self._n_ref_samples > 0:
            self.pca.finalize()

        self._finalize_ref_embeddings(keep_batch_files=keep_batch_files)

        self._partial_active = False
        self._n_ref_samples = 0
        self._batch_count = 0

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
        incremental: Literal["auto", "full", "batch"] = "auto",
        batch_write: bool = False,
        chunk_size: int | None = None,
        incremental_val: bool | None = None,
        keep_batch_files: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Train a confidence score based on sample embeddings.

        This method supports two usage modes:

        1. **Precomputed embeddings**: Supply training embeddings via `X` and
           optional calibration embeddings via `Y`.
        2. **On-the-fly extraction**: Supply a `model` with an `.embed()` method
           and a dictionary of `DataLoaders` to extract embeddings automatically.

        You must use either embeddings (X/Y) OR model+loaders, but not both.

        ```python
        # Mode 1: Precomputed embeddings (full)
        my_score = EmbeddingScore(k=2)
        my_score.fit(X=train_embs, Y=val_embs)

        # Mode 2: On-the-fly extraction (auto → batch when >1 train batch)
        my_score = EmbeddingScore(k=2)
        my_score.fit(model=model, loaders={"train": train_loader, "val": val_loader})

        # Mode 3: Chunked precomputed embeddings (batch)
        my_score.fit(X=big_X, incremental="batch", chunk_size=1024)
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
        incremental:
            Fitting mode. One of ``"auto"`` (default), ``"full"``, or
            ``"batch"``.

            - ``"full"``: collect all embeddings first, then fit (existing
              behaviour, unchanged).
            - ``"batch"``: process training data incrementally using
              ``partial_fit`` per batch, then call ``finalize()``.
            - ``"auto"``: choose ``"batch"`` when using ``model`` + ``loaders``
              with more than one training batch, or when ``chunk_size`` is set
              for precomputed ``X``; otherwise choose ``"full"``.
        batch_write:
            If ``True``, write each batch of embeddings to a separate ``.pt``
            file under ``outdir`` instead of accumulating in memory. Only used
            when ``incremental`` resolves to ``"batch"``. Defaults to ``False``.
        chunk_size:
            Split precomputed ``X`` into chunks of this size for batch-mode
            processing. Only used when ``incremental`` resolves to ``"batch"``
            and ``X`` is provided. Defaults to ``None`` (no chunking).
        incremental_val:
            Whether to process the validation loader incrementally in batch
            mode. Defaults to the same as the resolved ``incremental`` mode
            (i.e. ``True`` when batch, ``False`` when full).
        keep_batch_files:
            If ``True``, per-batch ``.pt`` files written during batch-write
            mode are kept on disk after ``finalize()``. Defaults to ``False``.
        """
        # Validate parameter combinations
        using_embeddings = X is not None
        using_model = model is not None or loaders is not None

        if using_embeddings and using_model:
            raise ValueError(
                "Cannot specify both embeddings (X/Y) and model+loaders. "
                "Use either precomputed embeddings OR on-the-fly extraction."
            )

        if not using_embeddings and not using_model:
            raise ValueError(
                "Must specify either embeddings (X) or model+loaders for fitting."
            )

        # Resolve incremental mode
        mode: Literal["full", "batch"] = "full"
        if incremental == "full":
            mode = "full"
        elif incremental == "batch":
            mode = "batch"
        else:  # "auto"
            if using_embeddings and chunk_size is not None:
                mode = "batch"
            elif using_model and loaders is not None and "train" in loaders:
                try:
                    train_len = len(loaders["train"])
                    mode = "batch" if train_len > 1 else "full"
                except TypeError:
                    mode = "batch"
            else:
                mode = "full"

        if mode == "full":
            # ---- existing full-mode behaviour ----
            if using_embeddings:
                self.ref_embeddings = X
                self.cal_embeddings = Y
            else:
                if model is None:
                    raise ValueError(
                        "model is required when not using precomputed embeddings."
                    )
                if loaders is None:
                    raise ValueError("loaders is required when using a model.")

                assert isinstance(loaders, dict)
                assert isinstance(model, torch.nn.Module)
                self._check_model(model)
                self.ref_embeddings = self._embed_from_dict(
                    loaders=loaders,
                    model=model,
                    key="train",
                    outdir=outdir,
                    prefix=prefix,
                )
                if "val" in loaders.keys():
                    self.cal_embeddings = self._embed_from_dict(
                        loaders=loaders,
                        model=model,
                        key="val",
                        outdir=outdir,
                        prefix=prefix,
                    )
        else:
            # ---- batch mode ----
            if using_embeddings:
                assert X is not None
                self.reset_partial()
                self._partial_active = True
                if chunk_size is not None:
                    n = X.shape[0]
                    for start in range(0, n, chunk_size):
                        chunk = X[start : start + chunk_size]
                        if self.pca is not None:
                            self.pca.partial_fit(chunk)
                        self._accumulate_ref_embeddings(
                            chunk,
                            write_to_disk=batch_write,
                            outdir=outdir,
                            prefix=prefix,
                            key="train",
                        )
                        self._n_ref_samples += chunk.shape[0]
                        self._batch_count += 1
                else:
                    if self.pca is not None:
                        self.pca.partial_fit(X)
                    self._accumulate_ref_embeddings(
                        X,
                        write_to_disk=batch_write,
                        outdir=outdir,
                        prefix=prefix,
                        key="train",
                    )
                    self._n_ref_samples = X.shape[0]
                    self._batch_count = 1
                if Y is not None:
                    val_chunk_size = chunk_size
                    if val_chunk_size is not None:
                        for start in range(0, Y.shape[0], val_chunk_size):
                            chunk = Y[start : start + val_chunk_size]
                            self._accumulate_ref_embeddings(
                                chunk,
                                write_to_disk=batch_write,
                                outdir=outdir,
                                prefix=prefix,
                                key="val",
                            )
                    else:
                        self._accumulate_ref_embeddings(
                            Y,
                            write_to_disk=batch_write,
                            outdir=outdir,
                            prefix=prefix,
                            key="val",
                        )
                self.finalize(keep_batch_files=keep_batch_files)
            else:
                # model + loaders in batch mode
                if model is None:
                    raise ValueError(
                        "model is required when not using precomputed embeddings."
                    )
                if loaders is None:
                    raise ValueError("loaders is required when using a model.")

                assert isinstance(loaders, dict)
                assert isinstance(model, torch.nn.Module)
                self._check_model(model)

                do_batch_val = (
                    incremental_val if incremental_val is not None else True
                )

                self.reset_partial()
                self._partial_active = True

                # ---- train batches ----
                was_training = model.training
                model.eval()
                train_loader = loaders["train"]
                try:
                    n_train: int | None = len(train_loader)
                except TypeError:
                    n_train = None
                pbar_desc = f"Embedding {n_train if n_train is not None else '?'} train batches"
                for train_batch in track(
                    train_loader, total=n_train, desc=pbar_desc, unit="batches"
                ):
                    embs = self._embed(X=train_batch, model=model)
                    if self.pca is not None:
                        self.pca.partial_fit(embs)
                    self._accumulate_ref_embeddings(
                        embs,
                        write_to_disk=batch_write,
                        outdir=outdir,
                        prefix=prefix,
                        key="train",
                    )
                    self._n_ref_samples += embs.shape[0]
                    self._batch_count += 1
                if was_training:
                    model.train()

                # ---- val batches ----
                if "val" in loaders:
                    if do_batch_val:
                        was_training_val = model.training
                        model.eval()
                        val_loader = loaders["val"]
                        try:
                            n_val: int | None = len(val_loader)
                        except TypeError:
                            n_val = None
                        pbar_desc_val = f"Embedding {n_val if n_val is not None else '?'} val batches"
                        for val_batch in track(
                            val_loader,
                            total=n_val,
                            desc=pbar_desc_val,
                            unit="batches",
                        ):
                            embs = self._embed(X=val_batch, model=model)
                            self._accumulate_ref_embeddings(
                                embs,
                                write_to_disk=batch_write,
                                outdir=outdir,
                                prefix=prefix,
                                key="val",
                            )
                        if was_training_val:
                            model.train()
                    else:
                        self.cal_embeddings = self._embed_from_dict(
                            loaders=loaders,
                            model=model,
                            key="val",
                            outdir=outdir,
                            prefix=prefix,
                        )

                self.finalize(keep_batch_files=keep_batch_files)

    @override
    def set_threshold(self, q: float = 0.99) -> None:
        """Set a threshold based on quantiles on the reference confidence scores.

        This method sets the selection threshold based on the quantile on
        the values found in the `scores` attribute. Samples with scores higher
        than this threshold are excluded from prediction. If the confidence score
        is trained, but uncalibrated, this will be based on the k-nearest-neighbor
        distances of the training samples, excluding the distance to the
        point itself. If calibrated, the distance of the calibration samples to
        the k-closest training samples are used.

        Parameters
        ----------
        q:
            A `float` indicating the quantile of confidence scores of the
            samples to set the rejection threshold to.
        """
        if self.train_required:
            assert self.is_trained()
        if self.cal_required:
            assert self.is_calibrated()
        assert self.scores is not None
        self.threshold = self.scores.float().quantile(q=q)

    @override
    def score(
        self,
        X: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]]
        | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> torch.Tensor:
        """Compute confidence scores for query samples.

        This method supports two usage modes:

        1. **Precomputed embeddings**: Supply query embeddings via `X`.
        2. **On-the-fly extraction**: Supply a `model` with an `.embed()` method
           and a `DataLoader` to extract embeddings automatically.

        You must use either embeddings (X) OR model+loader, but not both.

        Iterates over a dataloader (if provided), embeds samples on-the-fly using
        the supplied model's `.embed()` method and returns their confidence scores.

        ```python
        # Mode 1: Precomputed embeddings
        my_score = KNNScore()
        scores = my_score.score(X=test_embeddings)

        # Mode 2: On-the-fly extraction
        my_score = KNNScore()
        scores = my_score.score(model=model, loader=test_dl)
        ```

        Parameters
        ----------
        X:
            A `torch.Tensor` with query embeddings of shape (N, D).
            Required when not using `model` and `loader`.
        model:
            A torch.nn.Module representing a trained model instance. It is
            required to have an `.embed()` method.
            Required when not using `X`.
        loader:
            A `torch.utils.data.DataLoader` object returning `torch.Tensor`s or
            a `dict` of `torch.Tensor`s with the `"image"` key.
            Required when using `model`.
        outdir:
            A `pathlib.Path` object pointing towards a directory, by default `None`.
            If specified, embeddings are read to disk, if previously written. Otherwise,
            embeddings will be written to disk. Only used with `model` and `loader`.
        prefix:
            A `str`ing used as filename prefix to save embeddings, by default
            `None`. Only used with `model` and `loader`.
        """
        # Validate parameter combinations
        using_embeddings = X is not None
        using_model = model is not None or loader is not None

        if using_embeddings and using_model:
            raise ValueError(
                "Cannot specify both embeddings (X) and model+loader. "
                "Use either precomputed embeddings OR on-the-fly extraction."
            )

        if not using_embeddings and not using_model:
            raise ValueError(
                "Must specify either embeddings (X) or model+loader for scoring."
            )

        if using_embeddings:
            # Mode 1: Use precomputed embeddings - call subclass implementation
            assert X is not None, (
                "X is required when using precomputed embeddings."
            )
            return self._score_embeddings(X)
        else:
            # Mode 2: Extract embeddings on-the-fly
            if model is None:
                raise ValueError(
                    "model is required when not using precomputed embeddings."
                )
            if loader is None:
                raise ValueError("loader is required when using a model.")

            path = None
            if prefix is not None:
                path = self._setup_path(outdir, prefix)
            embeddings = self._loadorembed(path, model, loader)
            return self._score_embeddings(embeddings)

    def _score_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        """Compute confidence scores based on query embeddings.

        This method should be implemented by subclasses to compute confidence
        scores based on the query embeddings `X`. The base class does not
        implement any specific scoring logic, as this will depend on the
        particular method (e.g., k-nearest neighbors, PyOD scores, etc.).

        Parameters
        ----------
        X:
            A `torch.Tensor` with query embeddings of shape (N, D).

        Returns
        -------
        torch.Tensor
            A `torch.Tensor` with confidence scores for each query sample.
            Low scores indicate likely inliers, high scores indicate likely outliers.
        """
        raise NotImplementedError(
            "Subclasses must implement the `_score_embeddings` method."
        )

    @override
    def select(
        self,
        X: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]]
        | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Select samples for prediction based on their confidence score.

        This method supports two usage modes:

        1. **Precomputed embeddings**: Supply query embeddings via `X`.
        2. **On-the-fly extraction**: Supply a `model` with an `.embed()` method
           and a `DataLoader` to extract embeddings automatically.

        You must use either embeddings (X) OR model+loader, but not both.

        Samples are selected for prediction based on their confidence score compared
        to a threshold. Samples with scores lower than the threshold are selected,
        while samples with scores higher than the threshold are excluded. It is
        expected that the threshold was previously calibrated on, e.g. validation samples.

        ```python
        # Mode 1: Precomputed embeddings
        my_score = ConfidenceScore()
        my_score = my_score.fit(X=train_data, Y=val_data)
        scores = my_score.select(X=test_data)

        # Mode 2: On-the-fly extraction
        my_score = ConfidenceScore()
        my_score = my_score.fit(X=train_data, Y=val_data)
        scores = my_score.select(model=model, loader=test_loader)
        ```

        Parameters
        ----------
        X:
            A `torch.tensor` with samples representing testing
            embeddings to select based on a pre-calibrated threshold.
            Required when not using `model` and `loader`.
        model:
            A torch.nn.Module representing a trained model instance. It is
            required to have an `.embed()` method.
            Required when not using `X`.
        loader:
            A `torch.utils.data.DataLoader` object returning `torch.Tensor`s or
            a `dict` of `torch.Tensor`s with the `"image"` key available.
            Required when using `model`.
        outdir:
            A `pathlib.Path` object pointing towards a directory, by default `None`.
            If specified, embeddings are read to disk, if previously written. Otherwise,
            embeddings will be written to disk. Only used with `model` and `loader`.
        prefix:
            A `str`ing used as filename prefix to save embeddings, by default
            `None`. Only used with `model` and `loader`.
        """
        if self.train_required:
            assert self.is_trained()
        if self.cal_required:
            assert self.is_calibrated()
        if self.get_threshold() is None:
            logger.warning(
                "Threshold has not been set. Trying to set it via `set_threshold()`."
            )
            self.set_threshold()
        assert self.threshold is not None

        score = self.score(
            X=X, model=model, loader=loader, outdir=outdir, prefix=prefix
        )
        return {"score": score, "selected": score < self.threshold}

    def plot_embs(
        self,
        query_embeddings: torch.Tensor | None,
        method: Literal["tsne", "umap"] = "tsne",
        method_args: dict[str, Any] | None = None,
    ) -> None:
        """Visualize training, validation, and query embeddings in 2D.

        Parameters
        ----------
        query_embeddings : torch.Tensor | None, optional
            Embeddings of query samples to visualize.
        method : {"tsne", "umap"}, optional
            Dimensionality reduction method, by default "tsne".
        method_args : dict[str, Any] | None, optional
            A dictionary of arguments to pass to the dimensionality
            reduction method, by default None.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is not installed. Please install it with `pip install matplotlib`."
            )
        assert self.ref_embeddings is not None, (
            "Training embeddings are not set."
        )
        # Combine embeddings
        embeddings = [self.ref_embeddings]
        labels = ["train"] * len(self.ref_embeddings)

        if self.cal_embeddings is not None:
            embeddings.append(self.cal_embeddings)
            labels.extend(["cal"] * len(self.cal_embeddings))

        if query_embeddings is not None:
            embeddings.append(query_embeddings)
            labels.extend(["query"] * len(query_embeddings))

        all_embeddings: torch.Tensor = torch.cat(embeddings, dim=0)

        method_args = method_args or {}
        if method == "tsne":
            try:
                from sklearn.manifold import TSNE
            except ImportError:
                raise ImportError(
                    "t-SNE is not installed. Please install it with `pip install scikit-learn`."
                )
            reducer = TSNE(n_components=2, **method_args)
        elif method == "umap":  # pragma: no cover
            try:
                from umap import UMAP
            except ImportError:
                raise ImportError(
                    "UMAP is not installed. Please install it with `pip install umap-learn`."
                )
            reducer = UMAP(n_components=2, **method_args)
        else:
            raise ValueError("Invalid method. Choose 'tsne' or 'umap'.")

        reduced_embeddings = reducer.fit_transform(all_embeddings.cpu())

        label2col = {"train": "#1d7990", "cal": "#25901D", "query": "#f18e26"}

        plt.figure(figsize=(10, 8))
        for label in set(labels):
            idx = [i for i, la in enumerate(labels) if la == label]
            plt.scatter(
                reduced_embeddings[idx, 0],
                reduced_embeddings[idx, 1],
                label=label,
                color=label2col[label],
                alpha=0.1,
            )
        plt.legend()
        plt.title(f"Embedding Visualization ({method})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()
