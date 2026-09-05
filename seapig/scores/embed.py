"""Abstract Base Method for embeddings based uncertainty scores."""

from abc import ABC
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader
from typing_extensions import override

from seapig.scores.base import UncertaintyScore
from seapig.scores.utils import TensorPCA
from seapig.utils import get_logger

logger = get_logger(__name__)


from seapig.scores.mixins import ModelExtractorMixin


class EmbeddingScore(UncertaintyScore, ModelExtractorMixin, ABC):
    """Base class for embedding-based uncertainty scores.

    Embedding-based scores quantify deviation from the training distribution using
    latent-space embeddings. Low scores indicate samples similar to the training
    distribution (likely inliers), while high scores indicate samples deviating
    from the training distribution (likely outliers).

    Parameters
    ----------
    pca : `TensorPCA` or None, default None
        Optional `TensorPCA` object for dimensionality reduction prior to scoring. When
        provided, embeddings are projected onto the principal components
        before the score is computed.

    Attributes
    ----------
    ref_embeddings : torch.Tensor or None
        Embeddings of training samples used to fit the score.
    cal_embeddings : torch.Tensor or None
        Embeddings of validation/calibration samples. Optional.
    scores : torch.Tensor or None
        Uncertainty scores of the calibration (or training) samples.
    threshold : torch.Tensor or None
        Rejection threshold. Samples with scores above this value are excluded.

    See Also
    --------
    `scores.UncertaintyScore`
    `scores.KNNScore`
    `scores.PCAScore`
    `scores.PyODScore`
    `scores.EuclideanScore`
    `scores.CosineScore`
    `scores.MahalanobisScore`
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

    def _fit_pca(self) -> None:
        assert self.ref_embeddings is not None
        assert isinstance(self.pca, TensorPCA)
        self.pca.fit(self.ref_embeddings)

    def _apply_pca(self) -> None:
        assert self.ref_embeddings is not None
        if self.pca is None:
            return
        self._fit_pca()
        self.ref_embeddings = self.pca.transform(self.ref_embeddings)
        if self.cal_embeddings is not None:
            self.cal_embeddings = self.pca.transform(self.cal_embeddings)

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
        """Train a uncertainty score based on sample embeddings.

        This method supports two usage modes:

        1. **Precomputed embeddings**: Supply training embeddings via `X` and
           optional calibration embeddings via `Y`.
        2. **On-the-fly extraction**: Supply a `model` with an `.embed()` method
           and a dictionary of `DataLoaders` to extract embeddings automatically.

        You must use either embeddings (X/Y) OR model+loaders, but not both.

        ```python
        # Mode 1: Precomputed embeddings
        from seapig.scores import EuclideanScore
        my_score = EuclideanScore(k=2)
        my_score.fit(X=train_embs, Y=val_embs)

        # Mode 2: On-the-fly extraction
        my_score = EuclideanScore(k=2)
        my_score.fit(model=model, loaders={"train": train_loader, "val": val_loader})
        ```

        Parameters
        ----------
        X:
            A `torch.Tensor` with training sample embeddings. Required when not
            using `model` and `loaders`.
        Y:
            A `torch.Tensor` with calibration sample embeddings. Optional.
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
            A `float` or `bool` indicating if outliers from the training
            distribution should be filtered before fitting. Defaults to `False`.
        """
        tensor_mode = X is not None
        model_mode = model is not None or loaders is not None
        if tensor_mode == model_mode:
            raise ValueError(
                "Specify either pre-computed tensors (X and Y) or a model with a loader, but not both."
            )
        if model_mode:
            assert model is not None
            assert loaders is not None
            assert "train" in loaders
            out = self._extract_dict(
                model=model,
                loaders=loaders,
                input_keys=["image"],
                output_key="embedding",
                key="train",
                outdir=outdir,
                prefix=prefix,
            )
            X = out.get("embedding")
            if "val" in loaders:
                out = self._extract_dict(
                    model=model,
                    loaders=loaders,
                    input_keys=["image"],
                    output_key="embedding",
                    key="val",
                    outdir=outdir,
                    prefix=prefix,
                )
                Y = out.get("embedding")
        self.ref_embeddings = X
        self.cal_embeddings = Y
        self._fit(q=q)

    def _fit(self, q: bool | float = False):
        raise NotImplementedError(
            "Subclasses must implement the `_fit` method."
        )

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
        """Compute uncertainty scores for query samples.

        This method supports two usage modes:

        1. **Precomputed embeddings**: Supply query embeddings via `X`.
        2. **On-the-fly extraction**: Supply a `model` with an `.embed()` method
           and a `DataLoader` to extract embeddings automatically.

        You must use either embeddings (X) OR model+loader, but not both.

        ```python
        # Mode 1: Precomputed embeddings
        from seapig.scores import EuclideanScore
        my_score = EuclideanScore()
        scores = my_score.score(X=test_embeddings)

        # Mode 2: On-the-fly extraction
        my_score = EuclideanScore()
        scores = my_score.score(model=model, loader=test_dl)
        ```

        Parameters
        ----------
        X:
            A `torch.Tensor` with query embeddings of shape `(N, D)`.
            Required when not using `model` and `loader`.
        model:
            A `torch.nn.Module` with an `.embed()` method.
            Required when not using `X`.
        loader:
            A `torch.utils.data.DataLoader` returning `torch.Tensor`s or
            dicts with the `"image"` key. Required when using `model`.
        outdir:
            A `pathlib.Path` pointing to a directory for saving/loading embeddings.
            Only used with `model` and `loader`.
        prefix:
            A `str` used as filename prefix for saved embeddings.
            Only used with `model` and `loader`.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape `(N,)` with uncertainty scores.
            Low values indicate likely inliers, high values indicate likely outliers.
        """
        tensors_mode = X is not None
        model_mode = model is not None and loader is not None

        if tensors_mode == model_mode:
            raise ValueError(
                "Specify either pre-computed tensors (X and Y) or a model with a loader, but not both."
            )
        if model_mode:
            out = self._extract_loader(
                model=model,
                loader=loader,
                input_keys=["image"],
                output_key="embedding",
                outdir=outdir,
                prefix=prefix,
            )
            X = out.get("embedding")
        assert isinstance(X, torch.Tensor)
        return self._score(X)

    def _score(self, X: torch.Tensor):
        raise NotImplementedError(
            "Subclasses must implement the `_score` method."
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
        """Select samples for prediction based on their uncertainty score.

        This method supports two usage modes:

        1. **Precomputed embeddings**: Supply query embeddings via `X`.
        2. **On-the-fly extraction**: Supply a `model` with an `.embed()` method
           and a `DataLoader` to extract embeddings automatically.

        You must use either embeddings (X) OR model+loader, but not both.

        Samples are selected based on their uncertainty score relative to a
        threshold. Samples with scores lower than the threshold are selected,
        while samples with scores higher than the threshold are excluded. The
        threshold should be calibrated beforehand (e.g., on validation samples).

        ```python
        # Mode 1: Precomputed embeddings
        from seapig.scores import EuclideanScore
        my_score = EuclideanScore()
        my_score.fit(X=train_data, Y=val_data)
        result = my_score.select(X=test_data)

        # Mode 2: On-the-fly extraction
        my_score = EuclideanScore()
        my_score.fit(X=train_data, Y=val_data)
        result = my_score.select(model=model, loader=test_loader)
        ```

        Parameters
        ----------
        X:
            A `torch.Tensor` with query sample embeddings of shape `(N, D)`.
            Required when not using `model` and `loader`.
        model:
            A `torch.nn.Module` with an `.embed()` method.
            Required when not using `X`.
        loader:
            A `torch.utils.data.DataLoader` returning `torch.Tensor`s or
            dicts with the `"image"` key. Required when using `model`.
        outdir:
            A `pathlib.Path` pointing to a directory for saving/loading embeddings.
            Only used with `model` and `loader`.
        prefix:
            A `str` used as filename prefix for saved embeddings.
            Only used with `model` and `loader`.

        Returns
        -------
        dict[str, torch.Tensor]
            A dict with keys `'score'` (uncertainty scores) and `'selected'`
            (boolean mask where `True` means the sample is selected).
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

    @override
    def set_threshold(self, q: float = 0.99) -> None:
        """Set a threshold based on a quantile of the available uncertainty scores.

        Samples with scores higher than the threshold are excluded from prediction.
        If calibration embeddings were provided during `fit`, the threshold is
        computed from their scores; otherwise the training sample scores are used.

        Parameters
        ----------
        q : float
            Quantile in `(0, 1)` used to determine the threshold. Defaults to
            `0.99` (i.e., 1% of samples are rejected as outliers).
        """
        if self.train_required:
            assert self.is_trained()
        if self.cal_required:
            assert self.is_calibrated()
        assert self.scores is not None
        self.threshold = self.scores.float().quantile(q=q)

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
        elif method == "umap":
            try:
                from umap import UMAP
            except ImportError:
                raise ImportError(
                    "UMAP is not installed. Please install it with `pip install umap-learn`."
                )
            reducer = UMAP(n_components=2, **method_args)  # pragma: no cover
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
