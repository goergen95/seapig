"""Abstract Base Method for embeddings based confidence scores."""

import inspect
import warnings
from abc import ABC
from collections.abc import Generator
from contextlib import contextmanager
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

    @staticmethod
    def _validate_inputs(
        using_embeddings: bool, using_model: bool, context: str
    ) -> None:
        """Validate that exactly one input mode (embeddings or model) is specified.

        Parameters
        ----------
        using_embeddings:
            ``True`` when precomputed embeddings were supplied.
        using_model:
            ``True`` when a model (and loader/loaders) were supplied.
        context:
            Name of the calling method, used in error messages.
        """
        if using_embeddings and using_model:
            raise ValueError(
                f"Cannot specify both embeddings (X) and model+loader(s) in "
                f"{context}. Use either precomputed embeddings OR on-the-fly "
                f"extraction."
            )
        if not using_embeddings and not using_model:
            raise ValueError(
                f"Must specify either embeddings (X) or model+loader(s) in "
                f"{context}."
            )

    @staticmethod
    def _resolve_incremental_mode(
        incremental: Literal["auto", "full", "batch"],
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
    ) -> Literal["full", "batch"]:
        """Resolve ``"auto"`` to ``"full"`` or ``"batch"`` based on loader length.

        Parameters
        ----------
        incremental:
            The requested incremental mode.
        loader:
            The training DataLoader; its length is used when mode is ``"auto"``.
        """
        if incremental == "batch":
            return "batch"
        if incremental == "full":
            return "full"
        # "auto": choose batch when there is more than one training batch
        try:
            return "batch" if len(loader) > 1 else "full"
        except TypeError:
            return "batch"

    @staticmethod
    @contextmanager
    def _model_eval_context(
        model: torch.nn.Module,
    ) -> Generator[None, None, None]:
        """Context manager that temporarily sets a model to eval mode.

        Restores the original training state on exit, even if an exception
        is raised.
        """
        was_training = model.training
        model.eval()
        try:
            yield
        finally:
            model.train(was_training)

    @classmethod
    def _extract_embeddings(
        cls,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
        pca: TensorPCA | None = None,
        path: Path | None = None,
        pca_chunk_size: int = 1000,
    ) -> torch.Tensor:
        """Extract or load embeddings, fitting PCA incrementally when provided.

        If ``path`` points to an existing file the embeddings are loaded from
        disk and, when ``pca`` is provided, it is fitted from the cached tensor
        in chunks of ``pca_chunk_size``.  Otherwise the model is iterated
        over the loader, PCA (if given) is fitted incrementally per batch, and
        the full embedding tensor is written to ``path`` when provided.

        Parameters
        ----------
        model:
            Model with an ``.embed()`` method.  Also used for device resolution
            when loading from disk.
        loader:
            DataLoader over input batches.
        pca:
            Optional ``TensorPCA`` to fit.  When ``None`` no PCA fitting is
            performed.
        path:
            Optional path to save (or load) the full embedding tensor.
        pca_chunk_size:
            Chunk size used when iterating over a cached file to fit PCA.
            Defaults to ``1000``.
        """
        if path is not None and path.is_file():
            warnings.warn(
                f"Loading pre-existing embeddings from {path}.", UserWarning
            )
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            all_embs = cls._load_pt(path).to(device)
            if pca is not None:
                pca.reset_partial()
                n = all_embs.shape[0]
                for start in range(0, n, pca_chunk_size):
                    pca.partial_fit(all_embs[start : start + pca_chunk_size])
                pca.finalize()
            return all_embs

        # Extract embeddings from the model
        if pca is not None:
            pca.reset_partial()
        try:
            n_batches: int | None = len(loader)
        except TypeError:
            n_batches = None
        pbar_desc = (
            f"Embedding {n_batches if n_batches is not None else '?'} batches"
        )
        embs_list: list[torch.Tensor] = []
        with cls._model_eval_context(model):
            for batch in track(
                loader, total=n_batches, desc=pbar_desc, unit="batches"
            ):
                embs = cls._embed(X=batch, model=model)
                embs_list.append(embs)
                if pca is not None:
                    pca.partial_fit(embs)
        if pca is not None:
            pca.finalize()
        all_embs = torch.cat(embs_list, dim=0)
        if path is not None:
            cls._write_pt(all_embs, path)
        return all_embs

    @classmethod
    def _loadorembed(
        cls,
        path: Path | None,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Load embeddings from file or extract them from the DataLoader."""
        return cls._extract_embeddings(model=model, loader=loader, path=path)

    @classmethod
    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _embed(
        cls, X: torch.Tensor | dict[str, torch.Tensor], model: torch.nn.Module
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
        cls,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Extract embeddings by iterating over a DataLoader.

        This method ensures the model is in eval mode during embedding extraction
        to ensure consistent behavior regardless of the model's initial state.
        The model's original training state is restored after embedding extraction.
        """
        assert callable(model.embed)
        return cls._extract_embeddings(model=model, loader=loader)

    @classmethod
    def _embed_from_dict(
        cls,
        model: torch.nn.Module,
        loaders: dict[str, DataLoader[torch.Tensor | dict[str, torch.Tensor]]],
        key: Literal["train", "val"],
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> torch.Tensor:
        """Embed a loader from a specified key in a dictionary."""
        assert isinstance(loaders, dict)
        assert isinstance(model, torch.nn.Module)
        if outdir is not None and prefix is None:
            warnings.warn(
                "'outdir' has been specified but 'prefix' is None.\n"
                "Consider specifying 'prefix' as well to enable saving embeddings.",
                UserWarning,
            )
        cls._check_model(model)
        if key not in loaders.keys():
            raise KeyError(f"Missing key `{key}` in loaders dictionary.")
        loader = loaders[key]
        assert isinstance(loader, DataLoader)
        path = (
            cls._setup_path(outdir, prefix + f"-embeddings-{key}")
            if prefix is not None
            else None
        )
        return cls._extract_embeddings(model=model, loader=loader, path=path)

    def _fit_pca(self) -> None:
        assert self.ref_embeddings is not None
        assert isinstance(self.pca, TensorPCA)
        self.pca.fit(self.ref_embeddings)

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
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Train a confidence score based on sample embeddings.

        This method supports two usage modes:

        1. **Precomputed embeddings**: Supply training embeddings via `X` and
           optional calibration embeddings via `Y`. Embeddings are assigned
           directly, and any attached ``TensorPCA`` is fitted on the training
           embeddings.
        2. **On-the-fly extraction**: Supply a `model` with an `.embed()` method
           and a dictionary of `DataLoaders` to extract embeddings automatically.

        You must use either embeddings (X/Y) OR model+loaders, but not both.

        ```python
        # Mode 1: Precomputed embeddings
        my_score = EmbeddingScore(k=2)
        my_score.fit(X=train_embs, Y=val_embs)

        # Mode 2: On-the-fly extraction (auto → batch when >1 train batch)
        my_score = EmbeddingScore(k=2)
        my_score.fit(model=model, loaders={"train": train_loader, "val": val_loader})

        # Mode 3: Force batch extraction with disk persistence
        my_score.fit(
            model=model,
            loaders={"train": train_loader},
            incremental="batch",
            outdir=Path("embs"),
            prefix="run1",
        )
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
            Fitting mode for model+loaders extraction. One of ``"auto"``
            (default), ``"full"``, or ``"batch"``.

            - ``"full"``: extract all embeddings at once, then fit PCA.
            - ``"batch"``: extract embeddings batch-by-batch, fitting PCA
              incrementally during extraction. The full tensor is saved to a
              single file under ``outdir / prefix`` when both are provided.
              If that file already exists, re-embedding is skipped and PCA is
              fitted from the cached file in chunks.
            - ``"auto"``: choose ``"batch"`` when the train loader contains
              more than one batch, otherwise ``"full"``.

            This parameter is ignored when precomputed embeddings (``X``) are
            used; those are always assigned directly.
        """
        using_embeddings = X is not None
        using_model = model is not None or loaders is not None
        self._validate_inputs(using_embeddings, using_model, "fit()")

        if using_embeddings:
            # Mode 1: Pre-computed embeddings — assign directly
            self.ref_embeddings = X
            self.cal_embeddings = Y
            if self.pca is not None:
                self._fit_pca()
            return

        # Mode 2/3: Extract embeddings on-the-fly
        if model is None:
            raise ValueError(
                "model is required when not using precomputed embeddings."
            )
        if loaders is None:
            raise ValueError("loaders is required when using a model.")

        assert isinstance(loaders, dict)
        assert isinstance(model, torch.nn.Module)
        self._check_model(model)

        if outdir is not None and prefix is None:
            warnings.warn(
                "'outdir' has been specified but 'prefix' is None.\n"
                "Consider specifying 'prefix' as well to enable saving embeddings.",
                UserWarning,
            )

        mode = self._resolve_incremental_mode(incremental, loaders["train"])

        train_path = (
            self._setup_path(outdir, prefix + "-embeddings-train")
            if prefix is not None
            else None
        )

        if mode == "full":
            # Full mode: extract all train embeddings, then fit PCA post-hoc
            self.ref_embeddings = self._extract_embeddings(
                model=model, loader=loaders["train"], path=train_path
            )
            if self.pca is not None:
                self._fit_pca()
        else:
            # Batch mode: extract with incremental PCA during extraction
            self.ref_embeddings = self._extract_embeddings(
                model=model,
                loader=loaders["train"],
                pca=self.pca,
                path=train_path,
            )

        if "val" in loaders:
            val_path = (
                self._setup_path(outdir, prefix + "-embeddings-val")
                if prefix is not None
                else None
            )
            self.cal_embeddings = self._extract_embeddings(
                model=model, loader=loaders["val"], path=val_path
            )

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
        using_embeddings = X is not None
        using_model = model is not None or loader is not None
        self._validate_inputs(using_embeddings, using_model, "score()")

        if using_embeddings:
            assert X is not None
            return self._score_embeddings(X)

        assert model is not None
        if loader is None:
            raise ValueError("loader is required when using a model.")
        path = self._setup_path(outdir, prefix) if prefix is not None else None
        embeddings = self._extract_embeddings(
            model=model, loader=loader, path=path
        )
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
