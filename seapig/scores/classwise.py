"""Generic class-wise wrapper for any per-sample uncertainty score.

The implementation mirrors :class:`KNNClassWiseScore` but is agnostic to the
underlying score type. It builds a separate scorer instance for each class
label and aggregates the results into a `(N, C)`matrix where `C`is the
number of discovered classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from typing_extensions import override

from seapig import scores
from seapig.scores.extractor import ModelExtractor
from seapig.utils import get_logger

logger = get_logger(__name__)


class ClassWiseScore(scores.UncertaintyScore):
    """Base class-wise wrapper.

    Parameters
    ----------
    base_score_cls:
        The concrete `UncertaintyScore` subclass to instantiate for each class.
    base_kwargs:
        Keyword arguments passed to each `base_score_cls` instance.
    """

    def __init__(
        self, base_score_cls: type[scores.UncertaintyScore], **base_kwargs: Any
    ) -> None:
        super().__init__()
        self.base_score_cls = base_score_cls
        self._base_kwargs: dict[str, Any] = base_kwargs
        self._class_labels: torch.Tensor | None = None
        self._scorers: dict[int, scores.UncertaintyScore] = {}
        self._thresholds: dict[int, torch.Tensor] = {}

    def _make_extractor(self, want_labels: bool) -> ModelExtractor:
        """Return a `ModelExtractor` configured for the wrapped scorer.

        * For KNN based scores we request the `embed` method.
        * For Logit based scores we request the `logits` method.
        `want_labels` determines whether the `label` key is part of the
        `input_keys` (required during calibration / training but not during
        scoring).
        """
        if issubclass(self.base_score_cls, scores.KNNScore):
            method = "embed"
            out_key = "embedding"
            keys = ("image", "label")
        elif issubclass(self.base_score_cls, scores.LogitScore):
            method = "logits"
            out_key = "logit"
            keys = ("image", "label") if want_labels else ("image",)
        else:  # pragma: no cover
            raise TypeError(
                "ClassWiseScore only supports KNNScore or LogitScore subclasses"
            )
        return ModelExtractor(
            method_name=method, output_key=out_key, input_keys=keys
        )

    def _extract_class_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        target_label: int,
        multi_label: bool,
    ) -> torch.Tensor:
        if multi_label:
            mask = y[:, target_label] == 1
        else:
            mask = y == target_label
        return X[mask]

    @override
    def fit(
        self,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loaders: dict[str, DataLoader[torch.Tensor | dict[str, torch.Tensor]]]
        | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit a separate scorer for each class.

        The method supports two mutually exclusive modes:

        * **Tensor mode** - pre-computed feature tensors `X`and label `y` are
          provided directly.
        * **Model mode** - a `torch.nn.Module` together with a `DataLoader`
          is supplied; the underlying `ModelExtractor` extracts the required
          embeddings or logits.

        Exactly one of these modes must be selected.  If both or neither are
        provided a `ValueError`is raised.

        Parameters
        ----------
        X, y:
            Training tensors. `X` holds the feature representation required by
            the wrapped `base_score_cls` (embeddings for KNN-based scores or
            logits for logit-based scores). `y` contains class labels; a 1-D
            tensor for single-label classification or a 2-D binary matrix for
            multi-label tasks.
        X_val, y_val:
            Optional validation tensors used for calibration of per-class
            scorers.
        model:
            A `torch.nn.Module` whose forward method yields the representation
            needed by the scorer.
        loaders:
            Mapping of split names to `DataLoader` objects. At minimum a
            `"train"` loader is required; a `"val"` loader is used if
            validation tensors are not supplied directly.
        outdir, prefix:
            Destination directory and filename prefix for any intermediate files
            produced by the extractor.
        **kwargs:
            Additional keyword arguments forwarded to the concrete scorer's
            `fit` method.

        Raises
        ------
        ValueError
            If both tensor and model modes are specified or neither is.
        RuntimeError
            Propagated from the underlying scorer when training data are
            missing for a particular class.

        """
        tensor_mode = X is not None
        model_mode = model is not None
        if tensor_mode == model_mode:
            raise ValueError(
                "Specify either pre-computed tensors (X and Y) or a model with a loader, but not both."
            )

        if model_mode:
            assert (
                model is not None and loaders is not None and "train" in loaders
            )
            # Choose the correct output key based on the underlying scorer type
            out_key = (
                "embedding"
                if issubclass(self.base_score_cls, scores.KNNScore)
                else "logit"
            )
            extractor = self._make_extractor(want_labels=True)
            data = extractor.extract(
                model=model,
                loader=loaders["train"],
                outdir=outdir,
                prefix=prefix,
                overwrite=False,
            )
            X = data.get(out_key)
            y = data.get("label")
            if "val" in loaders:
                data = extractor.extract(
                    model=model,
                    loader=loaders["val"],
                    outdir=outdir,
                    prefix=prefix,
                    overwrite=False,
                )
                X_val = data.get(out_key)
                y_val = data.get("label")

        assert X is not None and y is not None, "Training data must be provided"
        assert X.shape[0] == y.shape[0], (
            "X and y must have the same first dimension"
        )
        multi_label = y.dim() == 2
        if multi_label:
            class_indices = torch.arange(y.shape[1], device=y.device)
        else:
            class_indices = torch.unique(y, sorted=True)
        self._class_labels = class_indices

        for label in class_indices:
            lbl = int(label.item())
            X_c = self._extract_class_data(X, y, lbl, multi_label)
            if X_c.shape[0] == 0:
                raise ValueError(f"No training samples found for class {lbl}")
            Y_c: torch.Tensor | None = None
            if X_val is not None and y_val is not None:
                assert X_val.shape[0] == y_val.shape[0]
                Y_c = self._extract_class_data(X_val, y_val, lbl, multi_label)
            scorer = self.base_score_cls(**self._base_kwargs)
            scorer.fit(X=X_c, Y=Y_c, **kwargs)  # type: ignore[arg-type]
            self._scorers[lbl] = scorer

        self.set_trained()

    @override
    def set_threshold(self, q: float = 0.99) -> None:
        """Set per-class thresholds based on the calibrated scores.

        The underlying scorer for each class provides its own `set_threshold`
        implementation (typically based on a quantile of the validation scores).
        This wrapper forwards the requested quantile `q` to each scorer, stores
        the resulting scalar threshold in `self._thresholds` and marks the
        wrapper as calibrated.

        Parameters
        ----------
        q:
            Quantile to use for threshold determination. `0.99` (default)
            selects the 99th percentile of the validation score distribution.

        Raises
        ------
        RuntimeError
            If `fit` has not been called yet.
        AssertionError
            If a scorer fails to provide a threshold.
        """
        if not self.is_trained():
            raise RuntimeError("Fit must be called before setting thresholds.")
        for lbl, scorer in self._scorers.items():
            scorer.set_threshold(q)
            thres = scorer.get_threshold()
            assert thres is not None
            self._thresholds[lbl] = thres
        self.set_calibrated()

    @override
    def get_threshold(self, id: int | None = None) -> torch.Tensor | None:
        """Return the threshold for a specific class.

        After calibration `self._thresholds` maps each class label to its scalar
        threshold tensor. If the wrapper has not been calibrated or `id` is
        `None`, `None` is returned; otherwise the threshold tensor for the
        requested class identifier is returned.
        """
        assert isinstance(self._class_labels, torch.Tensor)
        if (
            not self.is_calibrated()
            or id is None
            or id > len(self._class_labels)
        ):
            return None

        return self._thresholds[id]

    def _score(self, X: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "Direct _score is not used; call .score() instead."
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
        """Compute per-class uncertainty scores.

        Mirrors the `fit` method in accepting either pre-computed tensors or a
        `torch.nn.Module` with a `DataLoader`. The appropriate representation
        (embeddings for KNN-based scorers or logits for logit-based scorers) is
        extracted via `_make_extractor` when a model is supplied.

        Parameters
        ----------
        X:
            Tensor of shape `(N, D)` containing the features for which scores
            should be computed. Required in tensor mode.
        model:
            `torch.nn.Module` that produces the necessary representation.
        loader:
            DataLoader yielding the input data for `model`when `model` is
            provided.
        outdir, prefix:
            Forwarded to the extractor for any intermediate files.

        Returns
        -------
        torch.Tensor
            A `(N, C)` tensor where `C` is the number of discovered classes.
            Each column contains the scores for a particular class.

        Raises
        ------
        ValueError
            If both tensor and model modes are specified or neither is.
        RuntimeError
            If `fit` has not been called before scoring.
        """
        tensors_mode = X is not None
        model_mode = model is not None
        if tensors_mode == model_mode:
            raise ValueError(
                "Specify either pre-computed tensors (X and Y) or a model with a loader, but not both."
            )

        if model_mode:
            assert loader is not None
            extractor = self._make_extractor(want_labels=False)
            data = extractor.extract(
                model=model, loader=loader, outdir=outdir, prefix=prefix
            )
            out_key = (
                "embedding"
                if issubclass(self.base_score_cls, scores.KNNScore)
                else "logit"
            )
            X = data.get(out_key)

        assert isinstance(X, torch.Tensor)
        if self._class_labels is None:
            raise RuntimeError("fit must be called before scoring.")
        N = X.shape[0]
        C = len(self._class_labels)
        _scores = torch.empty((N, C), device=X.device, dtype=X.dtype)
        for col_idx, label in enumerate(self._class_labels):
            scorer = self._scorers[int(label.item())]
            _scores[:, col_idx] = scorer.score(X)  # type: ignore[arg-type]
        return _scores

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
        """Select samples below per-class thresholds.

        The method first ensures that thresholds have been calibrated (by
        invoking `set_threshold` if necessary) and then computes the
        class-wise scores via `score`. A boolean mask of shape `(N, C)`
        is returned where `True` indicates that the score for a given sample
        and class falls below the corresponding threshold.

        Parameters
        ----------
        X, model, loader, outdir, prefix:
            Same semantics as `score`; either pre-computed tensors or a
            model with a DataLoader must be supplied.

        Returns
        -------
        dict[str, torch.Tensor]
            `{"score": scores, "selected": mask}` where `scores` is the
            `(N, C)` tensor of raw scores and `mask` is the boolean selection
            mask.
        """
        if self.get_threshold() is None:
            logger.warning(
                "Threshold has not been set. Trying to set it via `set_threshold()`."
            )
            self.set_threshold()
        scores = self.score(
            X=X, model=model, loader=loader, outdir=outdir, prefix=prefix
        )
        mask = torch.empty_like(scores, dtype=torch.bool)
        assert self._class_labels is not None
        for col_idx, label in enumerate(self._class_labels):
            thr = self.get_threshold(id=int(label.item()))
            assert thr is not None
            mask[:, col_idx] = scores[:, col_idx] < thr
        return {"score": scores, "selected": mask}

    def plot(
        self, query_scores: torch.Tensor | None = None, bins: int = 100
    ) -> None:
        """Forward the plotting method to the score implementation."""
        for lbl, scorer in self._scorers.items():
            try:
                scorer.plot(query_scores, bins=bins)
            except Exception as exc:  # pragma: no cover – defensive
                raise RuntimeError(
                    f"Plot failed for class {lbl}: {exc}"
                ) from exc
