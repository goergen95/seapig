"""Generic class-wise wrapper for any per-sample uncertainty score.

The implementation mirrors :class:`KNNClassWiseScore` but is agnostic to the
underlying score type. It builds a separate scorer instance for each class
label and aggregates the results into a `(N, C)`matrix where `C`is the
number of discovered classes.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from typing_extensions import override

from seapig import scores as sp
from seapig.scores.extractor import ModelExtractor
from seapig.scores.utils import TensorPCA
from seapig.utils import get_logger


class ClassWiseMode(Enum):
    """Mode of class-wise scoring."""

    SINGLE_LABEL = "single_label"
    MULTI_LABEL = "multi_label"


logger = get_logger(__name__)


class ClassWiseScore(sp.UncertaintyScore):
    """Base class-wise wrapper for per-class uncertainty scoring.

    The wrapper supports three usage modes:

    * **Tensor mode** - Directly provide pre-computed feature tensors `X` and label
      tensor `y` to :meth:`fit` and optionally to :meth:`score`.
    * **Model mode** - Supply a `torch.nn.Module` and a `DataLoader`; the
      :class:`ModelExtractor` extracts the required embeddings.
    * **Full-matrix mode** - When calling :meth:`score`, setting `full_matrix=True`
      returns the raw `(N, C)` score matrix without aggregating over classes.
      This mode is useful when downstream code needs per-class scores.

    Parameters
    ----------
    base_score_cls:
        The concrete `UncertaintyScore` subclass to instantiate for each class.
    base_kwargs:
        Keyword arguments passed to each `base_score_cls` instance.
    global_pca:
        Optional :class:`TensorPCA` applied globally to inputs before scoring.
    aggregation:
        Aggregation function or name (`"mean"`, `"max"`, `"min"`) used to
        reduce per-class scores to a single scalar per sample in multi-label
        scenarios. Ignored in single-label mode.
    """

    def __init__(
        self,
        base_score_cls: type[sp.UncertaintyScore],
        global_pca: TensorPCA | None = None,
        aggregation: str
        | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "mean",
        **base_kwargs: Any,
    ) -> None:
        super().__init__()
        if not issubclass(base_score_cls, sp.KNNScore):
            raise TypeError(
                "Class-wise scores are currently only supported for KNNScore."
            )
        self.base_score_cls = base_score_cls
        self.pca = global_pca
        self.aggregation = self._resolve_aggregation(aggregation)
        self._class_labels: list[int] | None = None
        self._scorers: dict[int, sp.UncertaintyScore] = {}
        self._thresholds: dict[int, torch.Tensor] = {}
        self._threshold: torch.Tensor | None = None
        self._mode: ClassWiseMode | None = None
        self._base_kwargs: dict[str, Any] = base_kwargs

    @property
    def mode(self) -> ClassWiseMode:
        """Return the mode inferred during fitting.

        Raises
        ------
        RuntimeError
            If `fit` has not been called yet.
        """
        if self._mode is None:
            raise RuntimeError("fit() must be called before mode is available.")
        return self._mode

    @staticmethod
    def _infer_mode(y: torch.Tensor) -> ClassWiseMode:
        """Infer the scoring mode from the shape of `y`."""
        if y.dim() == 1:
            return ClassWiseMode.SINGLE_LABEL
        if y.dim() == 2:
            return ClassWiseMode.MULTI_LABEL
        raise ValueError(
            f"Expected y to either be of size (N,) or (N,C) but found {y.shape}"
        )

    def _resolve_aggregation(
        self, agg: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Resolve aggregation specification to a callable."""
        if callable(agg):
            assert not isinstance(agg, str)
            return agg
        if agg == "mean":
            return lambda s, m: torch.nanmean(
                s.masked_fill(~m, float("nan")), dim=1
            )
        if agg == "max":
            return lambda s, m: s.masked_fill(~m, float("-inf")).amax(dim=1)
        if agg == "min":
            return lambda s, m: s.masked_fill(~m, float("inf")).amin(dim=1)
        raise ValueError(f"Unknown aggregation '{agg}'")

    def _make_extractor(self, labels_from: str | None = None) -> ModelExtractor:
        """Return a `ModelExtractor` configured for the wrapped scorer.

        `labels_from` determines whether the `label` key is part of the
        `input_keys` (required during fit) or from the
        model outputs (required during scoring/selection).
        """
        in_keys = ("image",)
        out_keys = ("embedding",)
        if labels_from == "input":
            in_keys += ("label",)
        if labels_from == "output":
            out_keys += ("prediction",)
        return ModelExtractor(output_keys=out_keys, input_keys=in_keys)

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
          embeddings.

        Exactly one of these modes must be selected.  If both or neither are
        provided a `ValueError`is raised.

        Parameters
        ----------
        X, y:
            Training tensors. `X` holds the feature representation required by
            the wrapped `base_score_cls`. `y` contains class labels; a 1-D
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
            extractor = self._make_extractor(labels_from="input")
            data = extractor.extract(
                model=model,
                loader=loaders["train"],
                outdir=outdir,
                prefix=None if prefix is None else prefix + "-train",
                overwrite=False,
            )
            X = data.get("embedding")
            y = data.get("label")

            if "val" in loaders:
                data = extractor.extract(
                    model=model,
                    loader=loaders["val"],
                    outdir=outdir,
                    prefix=None if prefix is None else prefix + "-val",
                    overwrite=False,
                )
                X_val = data.get("embedding")
                y_val = data.get("label")

        assert X is not None and y is not None, "Training data must be provided"
        # Infer and store the scoring mode based on label tensor shape
        self._mode = self._infer_mode(y)
        assert X.shape[0] == y.shape[0], (
            "X and y must have the same first dimension"
        )

        if self.pca is not None:
            X = self.pca.fit_transform(X)
            if X_val is not None:
                X_val = self.pca.transform(X_val)

        multi_label = y.dim() == 2
        if multi_label:
            class_indices = torch.arange(y.shape[1], device=y.device)
        else:
            class_indices = torch.unique(y, sorted=True)
        self._class_labels = class_indices.tolist()

        for lbl in self._class_labels:
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

        if self._mode is ClassWiseMode.MULTI_LABEL:
            if X_val is not None and y_val is not None:
                self.scores = self._score_multi_label(X_val, y_val)
            else:
                self.scores = self._score_multi_label(X, y)

        # Ensure mode was inferred
        assert self._mode is not None, "Mode inference failed during fit"
        self.set_trained()

    @override
    def set_threshold(self, q: float = 0.99) -> None:
        if not self.is_trained():
            raise RuntimeError(
                "fit() must be called before setting thresholds."
            )

        # Always populate per-class thresholds (used directly in single-label
        # mode, and available for full_matrix consumers in any mode).
        for lbl, scorer in self._scorers.items():
            scorer.set_threshold(q)
            thr = scorer.get_threshold()
            assert isinstance(thr, torch.Tensor)
            self._thresholds[lbl] = thr

        if self._mode is ClassWiseMode.MULTI_LABEL:
            assert isinstance(self.scores, torch.Tensor)
            self._threshold = torch.quantile(self.scores, q=q)

        self.set_calibrated()

    @override
    def get_threshold(
        self, full_matrix: bool = False
    ) -> dict[int, torch.Tensor] | torch.Tensor | None:
        """Return the threshold for a specific class.

        After calibration `self._thresholds` maps each class label to its scalar
        threshold tensor. If the wrapper has not been calibrated, `None` is returned;
        otherwise the threshold dictionary with keys identifying the class label is
        returned.
        """
        if not self.is_calibrated():
            return None
        if full_matrix or self._mode is ClassWiseMode.SINGLE_LABEL:
            return self._thresholds
        return self._threshold

    def _score_full_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Score all classes and return the full `(N, C)` matrix.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor embeddings with shape `(N, D)`.

        Returns
        -------
        torch.Tensor
            Matrix of shape `(N,C)` where each entry is the score produced by
            the specific scorer for class c.
        """
        if self._class_labels is None:
            raise RuntimeError("fit() must be called before scoring.")
        N = X.shape[0]
        C = len(self._class_labels)
        scores = torch.empty((N, C), device=X.device, dtype=X.dtype)
        for col_idx, lbl in enumerate(self._class_labels):
            scorer = self._scorers[lbl]
            scores[:, col_idx] = scorer.score(X)
        return scores

    def _score_single_label(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Score samples according to their single-label class.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor (embeddings) with shape `(N, D)`.
        y : torch.Tensor
            1-D tensor of class labels with length `N`.

        Returns
        -------
        torch.Tensor
            Vector of shape `(N,)` where each entry is the score produced by
            the scorer corresponding to the sample's class.
        """
        if self._class_labels is None:
            raise RuntimeError("fit must be called before scoring.")
        if y.dim() != 1:
            raise ValueError("y must be a 1-D tensor for single-label scoring.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        # Validate that all labels are known
        unknown = set(y.tolist()) - set(self._class_labels)
        if unknown:
            raise ValueError(f"Unknown class labels in y: {sorted(unknown)}")
        N = X.shape[0]
        scores = torch.empty(N, device=X.device, dtype=X.dtype)
        for lbl in self._class_labels:
            mask = y == lbl
            if not mask.any():
                continue
            scorer = self._scorers[lbl]
            col_scores = scorer.score(X)
            scores[mask] = col_scores[mask]
        return scores

    def _score_multi_label(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Score multi-label samples by aggregating per-class scores.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor features of shape `(N, D)` .
        y : torch.Tensor
            Binary label matrix of shape `(N, C)` where `C` matches the
            number of classes available during `fit()`. Each row must contain at
            least one positive entry.

        Returns
        -------
        torch.Tensor
            Aggregated per-sample scores of shape `(N,)`.

        Raises
        ------
        RuntimeError
            If `fit` has not been called before scoring.
        ValueError
            If `y` does not have the expected shape, contains an unexpected
            number of classes, or any sample has no positive label.
        """
        if self._class_labels is None:
            raise RuntimeError("fit must be called before scoring.")
        if y.dim() != 2:
            raise ValueError(
                "y must be a 2-D binary matrix for multi-label scoring."
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if y.shape[1] != len(self._class_labels):
            raise ValueError(
                f"Number of label columns ({y.shape[1]}) does not match number of classes ({len(self._class_labels)})."
            )
        if (y.sum(dim=1) == 0).any():
            logger.warning(
                "Samples with no positive labels encountered; they will be filled with average scores."
            )
        _scores = self._score_full_matrix(X)
        mask = y.to(dtype=torch.bool)
        aggregated = self.aggregation(_scores, mask)
        # fill in scores for rows with no positive label and emit a warnings
        nan_mask = torch.isnan(aggregated)
        if nan_mask.any():
            logger.warning(
                "Encountered missing values after aggregation; using average score for these rows."
            )
            avg_scores = torch.nanmean(_scores, dim=1)
            aggregated[nan_mask] = avg_scores[nan_mask]
        return aggregated

    @override
    def score(
        self,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]]
        | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
        full_matrix: bool = False,
        return_label: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute per-class uncertainty scores.

        Mirrors the `fit` method in accepting either pre-computed tensors or a
        `torch.nn.Module` with a `DataLoader`. The appropriate embeddings for KNN-based
        scorers are extracted via `_make_extractor` when a model is supplied.

        Parameters
        ----------
        X:
            Tensor of shape `(N, D)` containing the features for which scores
            should be computed. Required in tensor mode.
        model:
            `torch.nn.Module` that produces the necessary representation.
        loader:
            DataLoader yielding the input data for `model` when `model` is
            provided.
        outdir, prefix:
            Forwarded to the extractor for any intermediate files.
        full_matrix:
            If `True`, returns the full `(N, C)` score matrix without
            requiring label information. `False` by default.
        return_label:
            Boolean. Besides the scores also returns the labels. Useful
            for the `select()` method. `False` by default.

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
            # If full_matrix we don't need labels, otherwise we extract from output
            extractor = self._make_extractor(
                labels_from=None if full_matrix else "output"
            )
            data = extractor.extract(
                model=model, loader=loader, outdir=outdir, prefix=prefix
            )
            X = data.get("embedding")
            if not full_matrix:
                y = data.get("prediction")

        assert isinstance(X, torch.Tensor)

        if self.pca is not None:
            X = self.pca.transform(X)

        if self._mode is None:
            raise RuntimeError("fit() must be called before scoring.")

        if full_matrix:
            return self._score_full_matrix(X)

        if y is None:
            raise ValueError(
                f"Mode '{self._mode.value}' requires labels; pass `y=` or use "
                "`full_matrix=True` for the label-free matrix mode."
            )

        inferred = self._infer_mode(y)
        if inferred is not self._mode:
            raise ValueError(
                f"Model was fit in '{self._mode.value}' mode but received labels "
                f"of shape {tuple(y.shape)} (inferred '{inferred.value}')."
            )

        if self._mode is ClassWiseMode.SINGLE_LABEL:
            _scores = self._score_single_label(X, y)
        else:
            _scores = self._score_multi_label(X, y)

        if return_label:
            return _scores, y

        return _scores

    @override
    def select(
        self,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]]
        | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
        full_matrix: bool = False,
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
        if self.get_threshold(full_matrix=full_matrix) is None:
            logger.warning(
                "Threshold has not been set; calling set_threshold() with defaults."
            )
            self.set_threshold()

        _scores = self.score(
            X=X,
            y=y,
            model=model,
            loader=loader,
            outdir=outdir,
            prefix=prefix,
            full_matrix=full_matrix,
            return_label=self._mode == ClassWiseMode.SINGLE_LABEL,
        )

        if full_matrix:
            thr = self.get_threshold(full_matrix=True)
            assert isinstance(_scores, torch.Tensor)
            assert isinstance(self._class_labels, list)
            assert isinstance(thr, dict)
            mask = torch.stack(
                [
                    _scores[:, i] < thr[lbl]
                    for i, lbl in enumerate(self._class_labels)
                ],
                dim=1,
            )
        elif self._mode is ClassWiseMode.SINGLE_LABEL:
            _scores, labels = _scores
            thr = self.get_threshold()
            assert labels is not None
            assert isinstance(thr, dict)
            thr_vec = torch.stack([thr[int(lbl)] for lbl in labels.tolist()])
            mask = _scores < thr_vec
        else:  # multi-label
            thres = self.get_threshold()
            assert isinstance(_scores, torch.Tensor)
            assert isinstance(thres, torch.Tensor)
            mask = _scores < thres

        return {"score": _scores, "selected": mask}

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
