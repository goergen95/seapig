"""Label-diversity scores."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from seapig.scores.knn import (
    CosineScore,
    EuclideanScore,
    KNNScore,
    LabelMode,
    MahalanobisScore,
)
from seapig.scores.utils import TensorPCA
from seapig.utils import get_logger

logger = get_logger(__name__)
__all__ = [
    "LabelDiversityCosineScore",
    "LabelDiversityEuclideanScore",
    "LabelDiversityMahalanobisScore",
]


class _LabelDiversityMixin(KNNScore):
    """Mixin turning any `KNNScore` into a label-diversity score.

    The score for a query is a measure of **how diverse the labels of its
    ``k`` nearest reference neighbors are** — no query label is needed.

    Two diversity measures are supported (``diversity``):

    * ``"entropy"`` (default): Shannon entropy of the neighbor label
      distribution, normalized to ``[0, 1]`` by ``log(num_classes)``.
      For multi-label references we treat the per-class frequency vector
      (mean of the neighbors' multi-hot rows) as the distribution after
      renormalization; a neighborhood dominated by a single class scores
      near 0, a maximally spread neighborhood scores near 1.
    * ``"unique"``: fraction of *distinct* classes appearing among the
      neighbors, computed as ``(# unique classes present) / min(k, C)``.
      In the multi-label case, a class is "present" if any neighbor has
      that class bit set.

    Requires
    --------
    * Reference ``labels`` at fit time (precomputed) *or* labels embedded
      in batches (loader mode, via ``self.label_key``).

    Not compatible with ``class_wise=True`` — this score keeps a single
    global index and reasons about labels itself.
    """

    ident: str = "label-diversity"

    def __init__(
        self,
        k: int = 5,
        pca: TensorPCA | None = None,
        save_index: bool | Path = False,
        *,
        num_classes: int | None = None,
        label_key: str = "label",
        diversity: str = "entropy",
        **kwargs,
    ) -> None:
        if kwargs.get("class_wise", False):
            raise ValueError(
                f"{type(self).__name__} does not support class_wise=True."
            )
        if diversity not in ("entropy", "unique"):
            raise ValueError(
                f"Unknown diversity='{diversity}'. "
                "Expected 'entropy' or 'unique'."
            )
        # we pass "mean" purely to satisfy the base-class assertion.
        super().__init__(
            k=k, stat="mean", pca=pca, save_index=save_index, class_wise=False
        )
        self.num_classes: int | None = num_classes
        self.label_key: str = label_key
        self.diversity: str = diversity
        self.label_mode: LabelMode | None = None
        self.ref_labels: torch.Tensor | None = None
        self.cal_labels: torch.Tensor | None = None

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
        """Fit reference embeddings + labels and (optionally) calibrate.

        Note: ``cal_labels`` is not used by the diversity computation
        itself (no query label is needed), but is still accepted so that
        the same ``KNNScore._normalize_labels`` machinery can validate
        the calibration labels' schema against the reference schema.
        """
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
                    f"{type(self).__name__} requires `labels` for the "
                    "reference set."
                )
            self.ref_embeddings = X
            self.cal_embeddings = Y
            self.ref_labels = labels
            self.cal_labels = cal_labels
        else:
            assert model is not None and loaders is not None
            self._check_model(model)
            embs, lbls = self._embed_and_label_from_dict(
                model=model,
                loaders=loaders,
                key="train",
                outdir=outdir,
                prefix=prefix,
                label_key=self.label_key,
            )
            self.ref_embeddings = embs
            self.ref_labels = lbls
            if "val" in loaders:
                embs_v, lbls_v = self._embed_and_label_from_dict(
                    model=model,
                    loaders=loaders,
                    key="val",
                    outdir=outdir,
                    prefix=prefix,
                    label_key=self.label_key,
                )
                self.cal_embeddings = embs_v
                self.cal_labels = lbls_v

        assert self.ref_labels is not None
        ref_mh, C, mode = KNNScore._normalize_labels(
            self.ref_labels, self.num_classes
        )
        if self.num_classes is None:
            logger.info(
                f"Inferred num_classes={C} from reference labels (mode={mode})."
            )
        self.num_classes = C
        self.label_mode = mode
        self.ref_labels = ref_mh
        if self.cal_labels is not None:
            cal_mh, _, _ = KNNScore._normalize_labels(self.cal_labels, C)
            self.cal_labels = cal_mh

        self._fit_impl_global(q=q)

        if self.cal_embeddings is None:
            self.scores = self._compute_diversity(self.ref_embeddings, offset=1)
        else:
            self.scores = self._compute_diversity(self.cal_embeddings, offset=0)

    def _compute_diversity(
        self, query: torch.Tensor, offset: int = 0
    ) -> torch.Tensor:
        """Diversity of KNN reference labels around each query.

        `query` is assumed to already live in the index space (i.e. PCA
        has already been applied if configured). Returns a ``(N,)`` tensor.
        """
        _, indices = self._distance(query, offset=offset)  # (N, k)
        assert self.ref_labels is not None
        assert self.num_classes is not None
        ref_lbl = self.ref_labels.to(indices.device)
        neighbor_lbl = ref_lbl[indices].float()  # (N, k, C)

        if self.diversity == "unique":
            present = (neighbor_lbl > 0).any(dim=1).float()  # (N, C)
            n_unique = present.sum(dim=-1)  # (N,)
            denom = float(min(self.k, self.num_classes))
            score = n_unique / denom if denom > 0 else n_unique
            return score.to(query.device)

        freq = neighbor_lbl.sum(dim=1)  # (N, C)
        totals = freq.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # (N, 1)
        p = freq / totals  # (N, C)
        # 0 * log(0) := 0
        log_p = torch.where(
            p > 0, torch.log(p.clamp_min(1e-12)), torch.zeros_like(p)
        )
        entropy = -(p * log_p).sum(dim=-1)  # (N,)
        # Normalize to [0, 1] by log(C) so the score is scale-free.
        if self.num_classes > 1:
            import math

            entropy = entropy / math.log(self.num_classes)
        return entropy.to(query.device)

    def _score_embeddings(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.pca is not None:
            X = self.pca.transform(X)
        return self._compute_diversity(X, offset=0)

    def score(
        self,
        X: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]]
        | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute label-diversity scores.

        No query labels are required (neither precomputed nor via
        batches): diversity is a property of the reference labels in
        the neighborhood of each query embedding.
        """
        using_embeddings = X is not None
        using_model = model is not None or loader is not None
        if using_embeddings and using_model:
            raise ValueError("Cannot specify both X and model+loader.")
        if not using_embeddings and not using_model:
            raise ValueError("Must specify either X or model+loader.")

        if using_embeddings:
            return self._score_embeddings(X)

        assert model is not None and loader is not None
        path = None
        if prefix is not None:
            path = KNNScore._setup_path(outdir, prefix)
        embs = KNNScore._loadorembed(path, model, loader)
        return self._score_embeddings(embs)

    def select(
        self,
        X: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]]
        | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Select samples whose diversity score is below the threshold."""
        if self.train_required:
            assert self.is_trained()
        if self.cal_required:
            assert self.is_calibrated()
        if self.get_threshold() is None:
            logger.warning(
                "Threshold has not been set. Calling set_threshold()."
            )
            self.set_threshold()
        score = self.score(
            X=X, model=model, loader=loader, outdir=outdir, prefix=prefix
        )
        assert self.threshold is not None
        return {"score": score, "selected": score < self.threshold}


class LabelDiversityEuclideanScore(_LabelDiversityMixin, EuclideanScore):
    """Label-diversity score using Euclidean KNN search."""

    ident: str = "label-diversity-euclidean"


class LabelDiversityCosineScore(_LabelDiversityMixin, CosineScore):
    """Label-diversity score using cosine KNN search."""

    ident: str = "label-diversity-cosine"


class LabelDiversityMahalanobisScore(_LabelDiversityMixin, MahalanobisScore):
    """Label-diversity score using Mahalanobis KNN search."""

    ident: str = "label-diversity-mahalanobis"
