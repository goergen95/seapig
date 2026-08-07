"""KNN-based uncertainty scores."""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing_extensions import override

from seapig.scores.embed import EmbeddingScore
from seapig.scores.utils import TensorPCA
from seapig.utils import get_logger
from seapig.utils.progress import track

__all__ = ["CosineScore", "EuclideanScore", "KNNScore", "MahalanobisScore"]

logger = get_logger(__name__)

LabelMode = Literal["single", "multi"]
ClassAgg = Literal["min", "mean", "per_class", "target"]
ThresholdMode = Literal["global", "per_class"]
TargetReduce = Literal["mean", "min", "max"]


class KNNScore(EmbeddingScore, ABC):
    """Abstract base class for KNN distance-based uncertainty scores.

    Computes distance-based uncertainty scores where low scores indicate samples
    similar to the training distribution (low uncertainty) and high scores indicate
    samples deviating from the training distribution (high uncertainty).

    Parameters
    ----------
    k : int, default 1
        Number of nearest neighbors used to compute the distance score.
    stat : {'max', 'mean', 'median', 'min'}, default 'max'
        Statistic applied to aggregate distances across the k neighbors.
    pca : `TensorPCA` or None, default None
        Optional `TensorPCA` object for dimensionality reduction prior to scoring.
        In class-wise mode, PCA is always fit **globally** on all reference embeddings.
    save_index : bool or Path, default False
        In class-agnostic mode, `True` saves the HNSW index to a default `.bin` file,
        or a `Path` (ending in `.bin`) specifies the file. In class-wise mode, this
        is treated as a **directory**; per-class indices are stored under
        `class{c}.bin` (Mahalanobis additionally saves `vi_zero.pt`).

    Class-wise mode
    ---------------
    When `class_wise=True`, the score builds a **separate index per class**
    (supports single- and multi-label). Labels are required for `fit`.

    - `class_aggregation`:
        * `"min"` (default): distance to nearest class,
        * `"mean"`: mean over classes (ignoring empty classes),
        * `"per_class"`: return raw `(N, C)` distance matrix,
        * `"target"`: distance to *predicted* target class(es);
          multi-label targets are reduced via `target_reduce`.
    - `threshold_mode`:
        * `"global"`: single threshold from concatenated per-class calibration scores,
        * `"per_class"`: one threshold per class.
    - `num_classes`: inferred from labels at fit-time if not provided.
    - `min_samples_per_class` (default 5): for Mahalanobis, classes below this
      threshold fall back to a shared (pooled) covariance.

    Label auto-detection
    --------------------
    Labels are normalized internally to a multi-hot `(N, C) bool` tensor.
    - 1-D integer tensor → treated as **single-label** (one-hot expanded).
    - 2-D bool or {0,1}-valued integer tensor → treated as **multi-label**.

    See Also
    --------
    `scores.EuclideanScore`
    `scores.CosineScore`
    `scores.MahalanobisScore`
    """

    k: int = 1
    cal_embeddings: torch.Tensor | None
    index: Any | None = None
    index_path: Path | None = None

    def __init__(
        self,
        k: int = 1,
        stat: str = "max",
        pca: TensorPCA | None = None,
        save_index: bool | Path = False,
        *,
        class_wise: bool = False,
        class_aggregation: ClassAgg = "min",
        threshold_mode: ThresholdMode = "global",
        target_reduce: TargetReduce = "mean",
        num_classes: int | None = None,
        min_samples_per_class: int = 5,
        label_key: str = "label",
    ) -> None:
        super().__init__(pca=pca)
        assert stat in ["max", "mean", "median", "min"]
        self.stat: str = stat
        self.k = k

        # class-wise config
        self.class_wise: bool = class_wise
        self.class_aggregation: ClassAgg = class_aggregation
        self.threshold_mode: ThresholdMode = threshold_mode
        self.target_reduce: TargetReduce = target_reduce
        self.num_classes: int | None = num_classes
        self.min_samples_per_class: int = min_samples_per_class
        self.label_key: str = label_key
        self.label_mode: LabelMode | None = None

        # per-class state
        self.indices_by_class: dict[int, Any] = {}
        self.class_counts: dict[int, int] = {}
        self.fallback_classes: set[int] = set()
        self.empty_classes: set[int] = set()
        self.scores_by_class: dict[int, torch.Tensor] | None = None
        self.threshold_by_class: dict[int, torch.Tensor] | None = None

        # labels (plain attrs, not buffers)
        self.ref_labels: torch.Tensor | None = None
        self.cal_labels: torch.Tensor | None = None

        # config-time validation
        if (
            class_wise
            and class_aggregation == "per_class"
            and threshold_mode == "global"
        ):
            raise ValueError(
                "class_aggregation='per_class' with threshold_mode='global' is "
                "not supported (score is (N, C)); use threshold_mode='per_class'."
            )
        if not class_wise:
            _cw_defaults = {
                "class_aggregation": "min",
                "threshold_mode": "global",
                "target_reduce": "mean",
                "num_classes": None,
                "min_samples_per_class": 5,
                "label_key": "label",
            }
            _current = {
                "class_aggregation": class_aggregation,
                "threshold_mode": threshold_mode,
                "target_reduce": target_reduce,
                "num_classes": num_classes,
                "min_samples_per_class": min_samples_per_class,
                "label_key": label_key,
            }
            _diff = {
                k_: v for k_, v in _current.items() if v != _cw_defaults[k_]
            }
            if _diff:
                warnings.warn(
                    f"class_wise=False but class-wise kwargs provided: {_diff}. "
                    "These will be ignored.",
                    UserWarning,
                )

        cw_tag = "-cw" if class_wise else ""
        self.ident: str = (
            f"{self.ident}-k{self.k}-{'full' if pca is None else 'pca'}{cw_tag}"
        )

        if save_index:
            if isinstance(save_index, bool):
                if class_wise:
                    self.index_path = Path(f"{self.ident}_indices")
                    self.index_path.mkdir(parents=True, exist_ok=True)
                else:
                    self.index_path = Path(f"{self.ident}_index.bin")
            else:
                assert isinstance(save_index, Path)
                if class_wise:
                    self.index_path = save_index
                    self.index_path.mkdir(parents=True, exist_ok=True)
                else:
                    assert save_index.suffix == ".bin", (
                        "Index file must have a .bin extension"
                    )
                    save_index.parent.mkdir(parents=True, exist_ok=True)
                    self.index_path = save_index

    @staticmethod
    def _normalize_labels(
        labels: torch.Tensor, num_classes: int | None = None
    ) -> tuple[torch.Tensor, int, LabelMode]:
        """Convert labels to `(N, C)` bool multi-hot.

        Detection rules:
          * 1-D integer tensor       → single-label (one-hot expanded)
          * 2-D bool / {0,1} tensor  → multi-label
        """
        if labels.ndim == 1:
            if labels.dtype.is_floating_point:
                raise ValueError(
                    "1-D labels must be of integer dtype (single-label); "
                    f"got {labels.dtype}."
                )
            n = int(labels.shape[0])
            C = (
                int(num_classes)
                if num_classes is not None
                else int(labels.max().item()) + 1
            )
            mh = torch.zeros((n, C), dtype=torch.bool, device=labels.device)
            mh[torch.arange(n, device=labels.device), labels.long()] = True
            return mh, C, "single"
        elif labels.ndim == 2:
            if labels.dtype == torch.bool:
                mh = labels
            else:
                uniq = torch.unique(labels)
                if not bool(torch.all((uniq == 0) | (uniq == 1)).item()):
                    raise ValueError(
                        "2-D labels must be bool- or {0,1}-valued for multi-label."
                    )
                mh = labels.bool()
            C = int(mh.shape[1])
            if num_classes is not None and num_classes != C:
                raise ValueError(
                    f"num_classes={num_classes} does not match label C={C}."
                )
            return mh, C, "multi"
        else:
            raise ValueError(
                f"labels must be 1-D or 2-D, got ndim={labels.ndim}."
            )

    @classmethod
    @torch.inference_mode()
    def _embed_and_label_dl(
        cls,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
        *,
        label_key: str = "label",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Iterate loader; return `(embeddings, labels)` tensors.

        Label resolution per batch:
          * dict → `batch[label_key]`
          * tuple/list → `batch[1]`
        """
        assert callable(model.embed)
        was_training = model.training
        model.eval()

        pbar_desc = f"Embedding+labelling {len(loader)} batches"
        embs_ls, lbls_ls = [], []
        for batch in track(
            loader, total=len(loader), desc=pbar_desc, unit="batches"
        ):
            z = cls._embed(X=batch, model=model)
            if isinstance(batch, dict):
                if label_key not in batch:
                    raise KeyError(
                        f"Batch dict does not contain label key '{label_key}'. "
                        f"Available keys: {list(batch.keys())}."
                    )
                y = batch[label_key]
            elif isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    raise ValueError(
                        "Tuple/list batch must contain (x, y) but has length "
                        f"{len(batch)}."
                    )
                y = batch[1]
            else:
                raise TypeError(
                    f"Cannot extract labels from batch of type {type(batch)}."
                )
            embs_ls.append(z)
            lbls_ls.append(y)
        embs = torch.cat(embs_ls, dim=0)
        lbls = torch.cat(lbls_ls, dim=0)

        if was_training:
            model.train()
        return embs, lbls

    @classmethod
    def _embed_and_label_from_dict(
        cls,
        model: torch.nn.Module,
        loaders: dict[str, DataLoader[torch.Tensor | dict[str, torch.Tensor]]],
        key: Literal["train", "val"],
        outdir: Path | None = None,
        prefix: str | None = None,
        *,
        label_key: str = "label",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings and labels for a loader key, with optional caching."""
        assert isinstance(loaders, dict)
        assert isinstance(model, torch.nn.Module)
        if outdir is not None and prefix is None:
            warnings.warn(
                "'outdir' specified but 'prefix' is None; caching disabled.",
                UserWarning,
            )
        cls._check_model(model)
        if key not in loaders:
            raise KeyError(f"Missing key `{key}` in loaders dictionary.")
        loader = loaders[key]
        assert isinstance(loader, DataLoader)

        embs_path = None
        lbls_path = None
        if prefix is not None:
            embs_path = cls._setup_path(outdir, prefix + f"-embeddings-{key}")
            lbls_path = cls._setup_path(outdir, prefix + f"-labels-{key}")

        if (
            embs_path is not None
            and embs_path.is_file()
            and lbls_path is not None
            and lbls_path.is_file()
        ):
            warnings.warn(
                f"Loading pre-existing embeddings+labels for '{key}' from disk.",
                UserWarning,
            )
            embs = cls._load_pt(embs_path)
            lbls = cls._load_pt(lbls_path)
            device = next(model.parameters()).device
            embs = embs.to(device)
            lbls = lbls.to(device)
        else:
            embs, lbls = cls._embed_and_label_dl(
                model=model, loader=loader, label_key=label_key
            )
            if embs_path is not None:
                cls._write_pt(embs, embs_path)
            if lbls_path is not None:
                cls._write_pt(lbls, lbls_path)
        return embs, lbls

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
        labels: torch.Tensor | None = None,
        cal_labels: torch.Tensor | None = None,
        q: bool | float = False,
    ) -> None:
        """Train an uncertainty score based on sample embeddings.

        Two usage modes:
          1. **Precomputed embeddings** via `X` (and optional `Y`).
          2. **On-the-fly extraction** via `model` + `loaders`.

        When `class_wise=True`, `labels` (and `cal_labels`) are required for
        precomputed mode; in loader mode, labels are extracted from batches
        using `label_key` (dict batches) or index `1` (tuple/list batches).

        See class docstring for label auto-detection rules and `num_classes`
        inference behavior.
        """
        # class-agnostic path: reuse base implementation
        if not self.class_wise:
            if labels is not None or cal_labels is not None:
                warnings.warn(
                    "labels/cal_labels provided but class_wise=False — ignoring.",
                    UserWarning,
                )
            super().fit(
                X=X,
                Y=Y,
                model=model,
                loaders=loaders,
                outdir=outdir,
                prefix=prefix,
            )
            self._fit_impl(q=q)
            return

        # class-wise path: also collect labels
        using_embeddings = X is not None
        using_model = model is not None or loaders is not None
        if using_embeddings and using_model:
            raise ValueError(
                "Cannot specify both embeddings (X/Y) and model+loaders."
            )
        if not using_embeddings and not using_model:
            raise ValueError(
                "Must specify either embeddings (X) or model+loaders."
            )

        if using_embeddings:
            if labels is None:
                raise ValueError(
                    "class_wise=True requires `labels` when using precomputed X."
                )
            self.ref_embeddings = X
            self.cal_embeddings = Y
            self.ref_labels = labels
            self.cal_labels = cal_labels
            if Y is not None and cal_labels is None:
                raise ValueError(
                    "class_wise=True with cal embeddings requires `cal_labels`."
                )
        else:
            assert model is not None
            assert loaders is not None
            self._check_model(model)
            embs, labels = self._embed_and_label_from_dict(
                model=model,
                loaders=loaders,
                key="train",
                outdir=outdir,
                prefix=prefix,
                label_key=self.label_key,
            )
            self.ref_embeddings = embs
            self.ref_labels = labels
            if "val" in loaders:
                embs_v, labels_v = self._embed_and_label_from_dict(
                    model=model,
                    loaders=loaders,
                    key="val",
                    outdir=outdir,
                    prefix=prefix,
                    label_key=self.label_key,
                )
                self.cal_embeddings = embs_v
                self.cal_labels = labels_v

        self._fit_impl(q=q)

    def _fit_impl(self, q: float | None = None) -> None:
        if not self.class_wise:
            return self._fit_impl_global(q)
        return self._fit_impl_classwise(q)

    def _fit_impl_global(self, q: float | None = None) -> None:
        """Class-agnostic fit."""
        assert self.ref_embeddings is not None
        if self.cal_required:
            assert self.cal_embeddings is not None

        if self.pca is not None:
            self._fit_pca()
            self.ref_embeddings = self.pca.transform(self.ref_embeddings)
            if self.cal_embeddings is not None:
                self.cal_embeddings = self.pca.transform(self.cal_embeddings)

        if q:
            assert (q >= 0.0) & (q <= 1.0)
            if self.index is None:
                self._setup_index()
            scores, _ = self._distance(self.ref_embeddings, offset=1)
            scores = self._stat(scores)
            threshold = torch.quantile(scores.float(), q=q)
            index = scores < threshold
            self.ref_embeddings = self.ref_embeddings[index, :]

        self._setup_index()
        self.set_trained()

        if self.cal_embeddings is None:
            scores, _ = self._distance(self.ref_embeddings, offset=1)
            self.scores = self._stat(scores)
        else:
            scores, _ = self._distance(self.cal_embeddings, offset=0)
            self.scores = self._stat(scores)
            self.set_calibrated()

    def _fit_impl_classwise(self, q: float | None = None) -> None:
        """Class-wise fit: per-class indices + per-class calibration scores."""
        assert self.ref_embeddings is not None
        assert self.ref_labels is not None
        if self.cal_required:
            assert self.cal_embeddings is not None

        # Normalize labels
        ref_mh, C, mode = self._normalize_labels(
            self.ref_labels, self.num_classes
        )
        if self.num_classes is None:
            logger.info(
                f"Inferred num_classes={C} from training labels (mode={mode})."
            )
        self.num_classes = C
        self.label_mode = mode
        self.ref_labels = ref_mh
        if self.cal_labels is not None:
            cal_mh, _, _ = self._normalize_labels(self.cal_labels, C)
            self.cal_labels = cal_mh

        # Per-class counts
        counts = ref_mh.sum(dim=0).long().tolist()
        self.class_counts = {c: int(counts[c]) for c in range(C)}
        self.empty_classes = {c for c in range(C) if counts[c] == 0}

        self.fallback_classes = {
            c for c in range(C) if 0 < counts[c] < self.min_samples_per_class
        }
        logger.info(
            f"Class-wise fit: mode={mode}, C={C}, counts={self.class_counts}"
        )
        for c in sorted(self.empty_classes):
            warnings.warn(
                f"Class {c} has 0 training samples — its distances default to +inf.",
                UserWarning,
            )
        for c in sorted(self.fallback_classes):
            warnings.warn(
                f"Class {c} has {counts[c]} < min_samples_per_class="
                f"{self.min_samples_per_class} samples — may use shared "
                f"covariance fallback (Mahalanobis) or a small index.",
                UserWarning,
            )

        # Global PCA
        if self.pca is not None:
            self._fit_pca()
            self.ref_embeddings = self.pca.transform(self.ref_embeddings)
            if self.cal_embeddings is not None:
                self.cal_embeddings = self.pca.transform(self.cal_embeddings)

        # Optional shared setup (e.g. Mahalanobis pooled covariance)
        self._setup_shared()

        # Per-class indices (with optional outlier filter)
        for c in range(C):
            if c in self.empty_classes:
                continue
            mask = ref_mh[:, c]
            embs_c = self.ref_embeddings[mask]
            self._setup_index_for_class(c, embs_c)
            if q:
                assert (q >= 0.0) & (q <= 1.0)
                d, _ = self._distance_for_class(c, embs_c, offset=1)
                d = self._stat(d)
                thr = torch.quantile(d.float(), q=q)
                keep = d < thr
                embs_c = embs_c[keep, :]
                self._setup_index_for_class(c, embs_c)

        self.set_trained()

        # Per-class calibration scores
        self.scores_by_class = {}
        for c in range(C):
            if c in self.empty_classes:
                continue
            if self.cal_embeddings is None:
                mask = ref_mh[:, c]
                q_c = self.ref_embeddings[mask]
                if q_c.shape[0] < 2:
                    continue
                d, _ = self._distance_for_class(c, q_c, offset=1)
            else:
                assert self.cal_labels is not None
                mask = self.cal_labels[:, c]
                if int(mask.sum().item()) == 0:
                    warnings.warn(
                        f"Class {c} has no calibration samples; skipping.",
                        UserWarning,
                    )
                    continue
                q_c = self.cal_embeddings[mask]
                d, _ = self._distance_for_class(c, q_c, offset=0)
            self.scores_by_class[c] = self._stat(d)

        if self.threshold_mode == "global" and self.scores_by_class:
            self.scores = torch.cat(list(self.scores_by_class.values()))

        if self.cal_embeddings is not None:
            self.set_calibrated()

    @override
    def _score_embeddings(
        self, X: torch.Tensor, *, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute uncertainty scores for query embeddings.

        In class-wise mode with `class_aggregation='target'`, the `target`
        argument must be a label tensor indicating the predicted class(es)
        for each query sample.
        """
        if self.pca is not None:
            X = self.pca.transform(X)

        if not self.class_wise:
            assert self.index is not None, "Index must be built before scoring"
            score, _ = self._distance(query=X)
            score = self._stat(score)
            return score.to(device=X.device)

        # class-wise
        assert self.num_classes is not None
        assert self.indices_by_class, "No per-class indices built."
        C = self.num_classes
        per_class = torch.full(
            (X.shape[0], C), float("inf"), device=X.device, dtype=torch.float32
        )
        for c in range(C):
            if c in self.empty_classes or c not in self.indices_by_class:
                continue
            d, _ = self._distance_for_class(c, X)
            per_class[:, c] = self._stat(d).to(X.device).float()

        agg = self.class_aggregation
        if agg == "per_class":
            return per_class
        if agg == "min":
            return per_class.min(dim=1).values
        if agg == "mean":
            finite = per_class.isfinite()
            safe = per_class.masked_fill(~finite, 0.0)
            return safe.sum(dim=1) / finite.sum(dim=1).clamp_min(1)
        if agg == "target":
            if target is None:
                raise ValueError(
                    "class_aggregation='target' requires `target` labels."
                )
            target_mh, _, _ = self._normalize_labels(target, C)
            target_mh = target_mh.to(X.device)
            return self._reduce_target(per_class, target_mh)
        raise ValueError(f"Unknown class_aggregation={agg}")

    def _reduce_target(
        self, per_class: torch.Tensor, target_mh: torch.Tensor
    ) -> torch.Tensor:
        """Reduce per-class distances over positive target classes."""
        valid = target_mh & per_class.isfinite()
        if self.target_reduce == "mean":
            safe = per_class.masked_fill(~valid, 0.0)
            return safe.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        if self.target_reduce == "min":
            return per_class.masked_fill(~valid, float("inf")).min(dim=1).values
        if self.target_reduce == "max":
            return (
                per_class.masked_fill(~valid, float("-inf")).max(dim=1).values
            )
        raise ValueError(f"Unknown target_reduce={self.target_reduce}")

    @override  # override to accept target argument
    def score(
        self,
        X: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]]
        | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
        *,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute uncertainty scores for query samples.

        When `class_wise=True` and `class_aggregation='target'`:
          * With precomputed `X`: `target` **must** be provided.
          * With `loader`: target labels are extracted from the loader
            (via `self.label_key`); `target` must be `None`.
        """
        if not (self.class_wise and self.class_aggregation == "target"):
            if target is not None:
                warnings.warn(
                    "`target` argument is only used when class_wise=True and "
                    "class_aggregation='target'; ignoring.",
                    UserWarning,
                )
            return super().score(
                X=X, model=model, loader=loader, outdir=outdir, prefix=prefix
            )

        # class_wise & target aggregation
        using_embeddings = X is not None
        using_model = model is not None or loader is not None
        if using_embeddings and using_model:
            raise ValueError("Cannot specify both X and model+loader.")
        if not using_embeddings and not using_model:
            raise ValueError("Must specify either X or model+loader.")

        if using_embeddings:
            if target is None:
                raise ValueError(
                    "target labels are required with precomputed X in "
                    "class_aggregation='target' mode."
                )
            return self._score_embeddings(X, target=target)

        if target is not None:
            raise ValueError(
                "target must be None when using loader mode; labels are "
                "extracted from the loader via `label_key`."
            )
        assert model is not None and loader is not None
        embs, lbls = self._embed_and_label_dl(
            model=model, loader=loader, label_key=self.label_key
        )
        return self._score_embeddings(embs, target=lbls)

    @override  # override to accept target argument
    def select(
        self,
        X: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]]
        | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
        *,
        target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Select samples based on their uncertainty score.

        See `score` for `target`-mode semantics. Per-class thresholds are
        applied when `threshold_mode='per_class'`.
        """
        if self.train_required:
            assert self.is_trained()
        if self.cal_required:
            assert self.is_calibrated()

        # ensure a threshold exists
        if not self.class_wise or self.threshold_mode == "global":
            if self.get_threshold() is None:
                logger.warning(
                    "Threshold has not been set. Calling set_threshold()."
                )
                self.set_threshold()
        else:
            if self.threshold_by_class is None:
                logger.warning(
                    "Per-class thresholds not set. Calling set_threshold()."
                )
                self.set_threshold()

        # class-agnostic path
        if not self.class_wise:
            score = self.score(
                X=X, model=model, loader=loader, outdir=outdir, prefix=prefix
            )
            assert self.threshold is not None
            return {"score": score, "selected": score < self.threshold}

        # class-wise: compute score, keeping targets when needed
        target_mh: torch.Tensor | None = None
        if self.class_aggregation == "target":
            using_embeddings = X is not None
            using_model = model is not None or loader is not None
            if using_embeddings and using_model:
                raise ValueError("Cannot specify both X and model+loader.")
            if using_embeddings:
                if target is None:
                    raise ValueError(
                        "target labels required with precomputed X in "
                        "class_aggregation='target' mode."
                    )
                score = self._score_embeddings(X, target=target)
                target_mh, _, _ = self._normalize_labels(
                    target, self.num_classes
                )
            else:
                if target is not None:
                    raise ValueError(
                        "target must be None with loader mode; extracted from loader."
                    )
                assert model is not None and loader is not None
                embs, lbls = self._embed_and_label_dl(
                    model=model, loader=loader, label_key=self.label_key
                )
                score = self._score_embeddings(embs, target=lbls)
                target_mh, _, _ = self._normalize_labels(lbls, self.num_classes)
        else:
            score = self.score(
                X=X, model=model, loader=loader, outdir=outdir, prefix=prefix
            )

        # apply thresholds
        if self.threshold_mode == "global":
            assert self.threshold is not None
            selected = score < self.threshold
            return {"score": score, "selected": selected}

        # per-class thresholds
        assert self.threshold_by_class is not None
        assert self.num_classes is not None
        thr_vec = torch.full(
            (self.num_classes,),
            float("inf"),
            device=score.device,
            dtype=torch.float32,
        )
        for c, t in self.threshold_by_class.items():
            thr_vec[c] = t.to(score.device).float()

        agg = self.class_aggregation
        if agg == "per_class":
            selected = score < thr_vec
        elif agg == "min":
            finite = thr_vec.isfinite()
            if not finite.any():
                raise RuntimeError("No finite per-class thresholds available.")
            selected = score < thr_vec[finite].min()
        elif agg == "mean":
            finite = thr_vec.isfinite()
            if not finite.any():
                raise RuntimeError("No finite per-class thresholds available.")
            selected = score < thr_vec[finite].mean()
        elif agg == "target":
            assert target_mh is not None
            per_sample_thr = self._reduce_target(
                thr_vec.unsqueeze(0).expand(score.shape[0], -1),
                target_mh.to(score.device),
            )
            selected = score < per_sample_thr
        else:
            raise ValueError(f"Unknown class_aggregation={agg}")

        return {"score": score, "selected": selected}

    @override
    def set_threshold(self, q: float = 0.99) -> None:
        """Set threshold(s) from calibration scores.

        In `threshold_mode='global'` a single quantile is computed from the
        concatenated per-class scores (class-wise mode) or from the flat
        `self.scores` (class-agnostic). In `threshold_mode='per_class'` one
        quantile per class is stored in `self.threshold_by_class`.
        """
        if self.train_required:
            assert self.is_trained()
        if self.cal_required:
            assert self.is_calibrated()

        if not self.class_wise or self.threshold_mode == "global":
            assert self.scores is not None
            self.threshold = self.scores.float().quantile(q=q)
            return

        assert (
            self.scores_by_class is not None and len(self.scores_by_class) > 0
        )
        self.threshold_by_class = {
            c: s.float().quantile(q=q) for c, s in self.scores_by_class.items()
        }
        self.threshold = torch.stack(
            list(self.threshold_by_class.values())
        ).max()

    def _setup_shared(self) -> None:
        """Compute shared state used across classes."""
        return

    @abstractmethod
    def _setup_index(self) -> None:
        """Prepare an index for KNN search (class-agnostic path)."""

    @abstractmethod
    def _distance(
        self, query: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """KNN distances/indices against the class-agnostic index."""

    @abstractmethod
    def _setup_index_for_class(self, c: int, embs: torch.Tensor) -> None:
        """Prepare a per-class KNN index (class-wise path)."""

    @abstractmethod
    def _distance_for_class(
        self, c: int, query: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """KNN distances/indices against the class `c` index."""

    def knn_search(
        self,
        query: torch.Tensor,
        offset: int = 0,
        *,
        class_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(distances, indices)` of KNN search.

        If `class_wise=True` and `class_id` is given, searches the per-class
        index for `class_id`; otherwise searches the global index.
        """
        if self.class_wise and class_id is not None:
            return self._distance_for_class(class_id, query, offset=offset)
        return self._distance(query=query, offset=offset)

    @staticmethod
    def _suggest_build_params(embs: torch.Tensor, k: int = 1) -> dict[str, Any]:
        if embs.dim() != 2:
            raise ValueError("embeddings must be 2D (N, D)")
        n, d = map(int, embs.shape)

        if d <= 64:
            M = 16
        elif d <= 128:  # pragma: no cover
            M = 24
        elif d <= 256:  # pragma: no cover
            M = 32
        elif d <= 512 or d <= 1024:  # pragma: no cover
            M = 48
        else:  # pragma: no cover
            M = 64

        # adjust upward for large n
        if n > 5_000_000:  # pragma: no cover
            M = max(M, 32)
        if n > 50_000_000:  # pragma: no cover
            M = max(M, 48)

        # adjust downward for very small n
        if n < 10_000:
            M = min(M, 16)

        # ef_construction based on M
        C = max(4 * M, 128)
        # cap at a high maximum value
        C = min(C, 1024)

        return {"M": M, "efConstruction": C}

    @staticmethod
    def _suggest_query_params(embs: torch.Tensor, k: int = 1) -> dict[str, Any]:
        """Suggest query parameters for HNSW index."""
        params = KNNScore._suggest_build_params(embs, k)

        S = max(k * 8, 512)
        S = min(S, params["efConstruction"])
        S = max(S, k)

        return {"efSearch": S}

    def _make_faiss_index(self, embs: torch.Tensor) -> Any:
        params = self._suggest_build_params(embs=embs, k=self.k)
        d = int(embs.shape[1])
        N = int(embs.shape[0])
        if N <= 10_000:
            index = faiss.IndexFlatL2(d)  # type: ignore[possibly-missing-attribute]
        else:
            M = params["M"]
            index = faiss.IndexHNSWFlat(  # type: ignore[possibly-missing-attribute]
                d, M, faiss.METRIC_L2
            )
            index.hnsw.efConstruction = params["efConstruction"]
        return index

    def _build_index(self, embs: torch.Tensor) -> None:
        """Build the class-agnostic index from `embs`."""
        assert isinstance(embs, torch.Tensor)
        index_path = self.index_path if not self.class_wise else None
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

    def _build_index_for_class(self, c: int, embs: torch.Tensor) -> None:
        """Build the per-class index for class `c` from `embs`."""
        assert isinstance(embs, torch.Tensor)
        index = self._make_faiss_index(embs)
        idx_path: Path | None = None
        if self.index_path is not None:
            idx_path = self.index_path / f"class{c}.bin"

        if idx_path is not None and idx_path.exists():
            warnings.warn(
                f"Class index {idx_path} already exists. Loading from disk.",
                UserWarning,
            )
            index = faiss.read_index(str(idx_path))  # type: ignore[possibly-missing-attribute]
        else:
            embs_np = embs.cpu().numpy().astype(np.float32)
            index.add(embs_np)
            if idx_path is not None:
                faiss.write_index(index, str(idx_path))  # type: ignore[possibly-missing-attribute]

        self.indices_by_class[c] = index

    def _query_index(
        self, query: torch.Tensor, offset: int = 0, *, index: Any | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query a FAISS index for KNN distances.

        If `index` is None, uses `self.index`.
        """
        if index is None:
            index = self.index
        assert index is not None, "Index must be built before querying."
        if isinstance(index, faiss.IndexHNSW):  # type: ignore[possibly-missing-attribute]
            params = KNNScore._suggest_query_params(query, self.k + offset)
            ef_search = params.get("efSearch", index.hnsw.efSearch)
            index.hnsw.efSearch = ef_search
        if index.d != query.shape[1]:
            raise ValueError(
                f"Query dimension {query.shape[1]} does not match "
                f"index dimension {index.d}"
            )
        query_np = query.cpu().numpy().astype(np.float32)
        search_results = index.search(query_np, self.k + offset)
        search_results = tuple(
            torch.from_numpy(x).to(query.device) for x in search_results
        )
        distances, indices = search_results
        if offset > 0:
            distances = distances[:, offset:]
            indices = indices[:, offset:]
        return (distances, indices)

    def _stat(self, x: torch.Tensor) -> torch.Tensor:
        assert self.stat in ["max", "mean", "median", "min"]
        if self.stat == "max":
            x = x.amax(1)
        if self.stat == "mean":
            x = x.mean(1)
        if self.stat == "median":
            x = x.median(1).values
        if self.stat == "min":
            x = x.amin(1)
        return x


class EuclideanScore(KNNScore):
    """KNN distance based on Euclidean distance.

    Computes Euclidean distance-based uncertainty scores where low scores indicate
    samples similar to the training distribution (low uncertainty) and high scores
    indicate samples deviating from the training distribution (high uncertainty).

    Parameters
    ----------
    k : int, default 1
        Number of nearest neighbors to use.
    stat : {'max', 'mean', 'median', 'min'}, default 'max'
        Statistic to aggregate distances across the k neighbors.
    pca : `TensorPCA` or None, default None
        Optional `TensorPCA` object for dimensionality reduction prior to scoring.
    save_index : bool or Path, default False
        Whether (and where) to save the HNSW index to disk.

    Examples
    --------
    ```python
    # Class-agnostic
    from seapig.scores import EuclideanScore
    s = EuclideanScore(k=5)
    s.fit(X=train_embs, Y=val_embs)
    s.set_threshold(q=0.95)
    out = s.select(X=test_embs)

    # Class-wise (single-label)
    s = EuclideanScore(k=5, class_wise=True, class_aggregation="min")
    s.fit(X=train_embs, labels=train_labels)
    s.set_threshold(q=0.95)
    out = s.select(X=test_embs)
    ```
    """

    k: int
    ident: str = "euclidean"

    def __init__(
        self,
        k: int = 1,
        stat: str = "max",
        pca: TensorPCA | None = None,
        save_index: bool | Path = False,
        *,
        class_wise: bool = False,
        class_aggregation: ClassAgg = "min",
        threshold_mode: ThresholdMode = "global",
        target_reduce: TargetReduce = "mean",
        num_classes: int | None = None,
        min_samples_per_class: int = 5,
        label_key: str = "label",
    ) -> None:
        super().__init__(
            k=k,
            stat=stat,
            pca=pca,
            save_index=save_index,
            class_wise=class_wise,
            class_aggregation=class_aggregation,
            threshold_mode=threshold_mode,
            target_reduce=target_reduce,
            num_classes=num_classes,
            min_samples_per_class=min_samples_per_class,
            label_key=label_key,
        )

    @override
    def _setup_index(self) -> None:
        assert isinstance(self.ref_embeddings, torch.Tensor)
        self._build_index(self.ref_embeddings)

    @override
    @torch.inference_mode()
    def _distance(
        self, query: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        squared, indices = self._query_index(query, offset)
        return (torch.sqrt(squared), indices)

    @override
    def _setup_index_for_class(self, c: int, embs: torch.Tensor) -> None:
        self._build_index_for_class(c, embs)

    @override
    @torch.inference_mode()
    def _distance_for_class(
        self, c: int, query: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        squared, indices = self._query_index(
            query, offset, index=self.indices_by_class[c]
        )
        return (torch.sqrt(squared), indices)


class CosineScore(KNNScore):
    """KNN distance based on cosine distance (`1 - cosine_similarity`).

    Computes cosine distance-based uncertainty scores where low scores indicate
    samples similar to the training distribution (low uncertainty) and high scores
    indicate samples deviating from the training distribution (high uncertainty).

    The cosine distance is computed as `(1 - cosine_similarity)`, with a range
    of `[0, 2]` where `0` indicates identical vectors, `1` indicates orthogonal
    vectors, and `2` indicates opposite vectors.

    Parameters
    ----------
    k : int, default 1
        Number of nearest neighbors to use.
    stat : {'max', 'mean', 'median', 'min'}, default 'max'
        Statistic to aggregate distances across the k neighbors.
    pca : `TensorPCA` or None, default None
        Optional `TensorPCA` object for dimensionality reduction prior to scoring.
    save_index : bool or Path, default False
        Whether (and where) to save the HNSW index to disk.

    See Also
    --------
    `scores.KNNScore`
    `scores.EuclideanScore`
    `scores.MahalanobisScore`
    """

    k: int = 1
    ident: str = "cosine"

    def __init__(
        self,
        k: int = 1,
        stat: str = "max",
        pca: TensorPCA | None = None,
        save_index: bool | Path = False,
        *,
        class_wise: bool = False,
        class_aggregation: ClassAgg = "min",
        threshold_mode: ThresholdMode = "global",
        target_reduce: TargetReduce = "mean",
        num_classes: int | None = None,
        min_samples_per_class: int = 5,
        label_key: str = "label",
    ) -> None:
        super().__init__(
            k=k,
            stat=stat,
            pca=pca,
            save_index=save_index,
            class_wise=class_wise,
            class_aggregation=class_aggregation,
            threshold_mode=threshold_mode,
            target_reduce=target_reduce,
            num_classes=num_classes,
            min_samples_per_class=min_samples_per_class,
            label_key=label_key,
        )

    @override
    def _setup_index(self) -> None:
        assert isinstance(self.ref_embeddings, torch.Tensor)
        normalized = torch.nn.functional.normalize(self.ref_embeddings)
        self._build_index(normalized)

    @override
    @torch.inference_mode()
    def _distance(
        self, query: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.index is not None
        normalized = torch.nn.functional.normalize(query)
        distances, indices = self._query_index(normalized, offset)
        return (0.5 * distances, indices)

    @override
    def _setup_index_for_class(self, c: int, embs: torch.Tensor) -> None:
        normalized = torch.nn.functional.normalize(embs)
        self._build_index_for_class(c, normalized)

    @override
    @torch.inference_mode()
    def _distance_for_class(
        self, c: int, query: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = torch.nn.functional.normalize(query)
        distances, indices = self._query_index(
            normalized, offset, index=self.indices_by_class[c]
        )
        return (0.5 * distances, indices)


class MahalanobisScore(KNNScore):
    """Mahalanobis distance to training samples.

    Computes Mahalanobis distance-based uncertainty scores where low scores indicate
    samples similar to the training distribution (low uncertainty) and high scores
    indicate samples deviating from the training distribution (high uncertainty).

    The Mahalanobis distance accounts for correlations in the training data by
    whitening the embeddings with the Cholesky factor of the training covariance
    matrix prior to computing Euclidean nearest-neighbour distances.

    In class-wise mode, each class has its own covariance (with regularization).
    Classes with fewer than `min_samples_per_class` samples fall back to a
    **shared pooled covariance** computed once from the entire reference set.

    Parameters
    ----------
    k : int, default 1
        Number of nearest neighbors to use.
    stat : {'max', 'mean', 'median', 'min'}, default 'max'
        Statistic to aggregate distances across the k neighbors.
    pca : `TensorPCA` or None, default None
        Optional `TensorPCA` object for dimensionality reduction prior to scoring.
    save_index : bool or Path, default False
        Whether (and where) to save the HNSW index to disk.

    See Also
    --------
    `scores.KNNScore`
    `scores.EuclideanScore`
    `scores.CosineScore`
    """

    k: int
    vi_zero: torch.Tensor
    ident: str = "mahalanobis"

    def __init__(
        self,
        k: int = 1,
        stat: str = "max",
        pca: TensorPCA | None = None,
        save_index: bool | Path = False,
        *,
        class_wise: bool = False,
        class_aggregation: ClassAgg = "min",
        threshold_mode: ThresholdMode = "global",
        target_reduce: TargetReduce = "mean",
        num_classes: int | None = None,
        min_samples_per_class: int = 5,
        label_key: str = "label",
    ) -> None:
        super().__init__(
            k=k,
            stat=stat,
            pca=pca,
            save_index=save_index,
            class_wise=class_wise,
            class_aggregation=class_aggregation,
            threshold_mode=threshold_mode,
            target_reduce=target_reduce,
            num_classes=num_classes,
            min_samples_per_class=min_samples_per_class,
            label_key=label_key,
        )
        self.register_buffer("vi_zero", None)
        self.vi_zero_shared: torch.Tensor | None = None
        self.vi_zero_by_class: dict[int, torch.Tensor] = {}

    @staticmethod
    def _safe_inv_cholesky(
        cov: torch.Tensor, eps: float = 1e-4
    ) -> torch.Tensor:
        """Regularized inverse Cholesky factor: `L^{-1}` for `Σ + εI`."""
        d = cov.shape[0]
        diag_mean = cov.diagonal().mean().clamp_min(1e-12)
        eye = torch.eye(d, device=cov.device, dtype=cov.dtype)
        reg = cov + eps * diag_mean * eye
        L = torch.linalg.cholesky(reg)
        return torch.linalg.inv(L)

    @override
    def _setup_index(self) -> None:
        assert isinstance(self.ref_embeddings, torch.Tensor)
        cov = self.ref_embeddings.T.cov()
        self.vi_zero = self._safe_inv_cholesky(cov)
        transformed = self.ref_embeddings @ self.vi_zero.T
        self._build_index(transformed)

    @override
    @torch.inference_mode()
    def _distance(
        self, query: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.index is not None
        transformed = query.float() @ self.vi_zero.T
        distances, indices = self._query_index(transformed, offset)
        return torch.sqrt(distances), indices

    @override
    def _setup_shared(self) -> None:
        assert isinstance(self.ref_embeddings, torch.Tensor)
        cov_shared = self.ref_embeddings.T.cov()
        self.vi_zero_shared = self._safe_inv_cholesky(cov_shared)

    @override
    def _setup_index_for_class(self, c: int, embs: torch.Tensor) -> None:
        assert self.vi_zero_shared is not None, "_setup_shared must run first"
        if embs.shape[0] < self.min_samples_per_class:
            self.fallback_classes.add(c)
            vi = self.vi_zero_shared
        else:
            try:
                cov_c = embs.T.cov()
                vi = self._safe_inv_cholesky(cov_c)
            except RuntimeError as e:  # pragma: no cover
                logger.warning(
                    f"Cholesky failed for class {c}: {e}. Using shared covariance."
                )
                self.fallback_classes.add(c)
                vi = self.vi_zero_shared
        self.vi_zero_by_class[c] = vi
        transformed = embs @ vi.T
        self._build_index_for_class(c, transformed)

    @override
    @torch.inference_mode()
    def _distance_for_class(
        self, c: int, query: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vi = self.vi_zero_by_class[c]
        transformed = query.float() @ vi.T
        distances, indices = self._query_index(
            transformed, offset, index=self.indices_by_class[c]
        )
        return torch.sqrt(distances), indices
