"""Tests for the label-diversity KNN mixin."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from seapig.scores.diversity import (
    LabelDiversityCosineScore,
    LabelDiversityEuclideanScore,
    LabelDiversityMahalanobisScore,
    _LabelDiversityMixin,
)
from seapig.scores.knn import (
    CosineScore,
    EuclideanScore,
    KNNScore,
    MahalanobisScore,
)
from seapig.scores.utils import TensorPCA

DIVERSITY_CLASSES = [
    LabelDiversityEuclideanScore,
    LabelDiversityCosineScore,
    LabelDiversityMahalanobisScore,
]


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.fixture
def clustered_data() -> dict[str, torch.Tensor | int]:
    """Three well-separated Gaussian blobs in R^8, 60 samples per class."""
    C, D, N = 3, 8, 60
    centers = torch.tensor(
        [
            [5.0] + [0.0] * (D - 1),
            [-5.0] + [0.0] * (D - 1),
            [0.0, 5.0] + [0.0] * (D - 2),
        ]
    )
    xs, ys = [], []
    for c in range(C):
        xs.append(centers[c] + 0.3 * torch.randn(N, D))
        ys.append(torch.full((N,), c, dtype=torch.long))
    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    perm = torch.randperm(X.shape[0])
    return {"X": X[perm], "y": y[perm], "C": C, "D": D}


@pytest.fixture
def cal_data(
    clustered_data: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """A held-out calibration set drawn from the same distribution."""
    C, D, N = clustered_data["C"], clustered_data["D"], 20
    centers = torch.tensor(
        [
            [5.0] + [0.0] * (D - 1),
            [-5.0] + [0.0] * (D - 1),
            [0.0, 5.0] + [0.0] * (D - 2),
        ]
    )
    xs, ys = [], []
    for c in range(int(C)):
        xs.append(centers[c] + 0.3 * torch.randn(int(N), int(D)))
        ys.append(torch.full((int(N),), c, dtype=torch.long))
    return {"X": torch.cat(xs), "y": torch.cat(ys)}


@pytest.fixture
def mixed_data() -> dict[str, torch.Tensor | int]:
    """Overlapping classes — same center, different labels — so KNN
    neighborhoods are label-heterogeneous by construction."""
    C, D, N = 4, 6, 40
    X = torch.randn(N * C, D) * 0.3  # all clustered near origin
    y = torch.arange(N * C) % C  # round-robin labels
    perm = torch.randperm(X.shape[0])
    return {"X": X[perm], "y": y[perm].long(), "C": C, "D": D}


class _TinyEmbedModel(torch.nn.Module):
    """Identity embedder — returns the input untouched."""

    def __init__(self) -> None:
        super().__init__()
        # Need one param so `next(model.parameters()).device` works.
        self._p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def embed(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, dict):
            x = x["x"]
        elif isinstance(x, (list, tuple)):
            x = x[0]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.embed(x)


@pytest.fixture
def loader_pair(
    clustered_data: dict[str, torch.Tensor], cal_data: dict[str, torch.Tensor]
) -> dict[str, DataLoader[Any]]:
    train_ds = TensorDataset(clustered_data["X"], clustered_data["y"])
    val_ds = TensorDataset(cal_data["X"], cal_data["y"])
    return {
        "train": DataLoader(train_ds, batch_size=32, shuffle=False),
        "val": DataLoader(val_ds, batch_size=32, shuffle=False),
    }


class TestTaxonomy:
    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_is_knn_score(self, cls: type) -> None:
        assert issubclass(cls, KNNScore)

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_uses_mixin(self, cls: type) -> None:
        assert issubclass(cls, _LabelDiversityMixin)

    def test_bindings(self) -> None:
        assert issubclass(LabelDiversityEuclideanScore, EuclideanScore)
        assert issubclass(LabelDiversityCosineScore, CosineScore)
        assert issubclass(LabelDiversityMahalanobisScore, MahalanobisScore)

    @pytest.mark.parametrize(
        "cls,expected",
        [
            (LabelDiversityEuclideanScore, "label-diversity-euclidean"),
            (LabelDiversityCosineScore, "label-diversity-cosine"),
            (LabelDiversityMahalanobisScore, "label-diversity-mahalanobis"),
        ],
    )
    def test_ident_prefix(self, cls: type, expected: str) -> None:
        s = cls(k=3)
        assert s.ident.startswith(expected)
        assert "-k3-" in s.ident


class TestConstruction:
    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_defaults(self, cls: type) -> None:
        s = cls()
        assert s.k == 5
        assert s.label_key == "label"
        assert s.num_classes is None
        assert s.ref_labels is None
        assert s.cal_labels is None
        assert s.pca is None
        assert s.diversity == "entropy"

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_rejects_class_wise(self, cls: type) -> None:
        with pytest.raises(ValueError, match="class_wise"):
            cls(k=3, class_wise=True)

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_rejects_unknown_diversity(self, cls: type) -> None:
        with pytest.raises(ValueError, match="diversity"):
            cls(k=3, diversity="gini")

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    @pytest.mark.parametrize("mode", ["entropy", "unique"])
    def test_accepts_supported_diversity(self, cls: type, mode: str) -> None:
        s = cls(k=3, diversity=mode)
        assert s.diversity == mode

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_custom_k_and_label_key(self, cls: type) -> None:
        s = cls(k=7, label_key="y")
        assert s.k == 7
        assert s.label_key == "y"

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_pca_accepted(self, cls: type) -> None:
        pca = TensorPCA(n_components=4)
        s = cls(k=3, pca=pca)
        assert s.pca is pca


class TestFitEmbeddings:
    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_fit_requires_labels(
        self, cls: type, clustered_data: dict[str, torch.Tensor]
    ) -> None:
        s = cls(k=3)
        with pytest.raises(ValueError, match="labels"):
            s.fit(X=clustered_data["X"])

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_fit_rejects_X_and_loaders(
        self,
        cls: type,
        clustered_data: dict[str, torch.Tensor],
        loader_pair: dict[str, DataLoader[Any]],
    ) -> None:
        s = cls(k=3)
        with pytest.raises(ValueError, match="Cannot specify both"):
            s.fit(
                X=clustered_data["X"],
                labels=clustered_data["y"],
                loaders=loader_pair,
                model=_TinyEmbedModel(),
            )

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_fit_requires_something(self, cls: type) -> None:
        s = cls(k=3)
        with pytest.raises(ValueError, match="Must specify"):
            s.fit()

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_fit_Y_without_cal_labels_allowed(
        self,
        cls: type,
        clustered_data: dict[str, torch.Tensor],
        cal_data: dict[str, torch.Tensor],
    ) -> None:
        # Diversity does not need cal labels — only reference labels.
        s = cls(k=3)
        s.fit(
            X=clustered_data["X"], labels=clustered_data["y"], Y=cal_data["X"]
        )
        assert s.is_trained() and s.is_calibrated()
        assert s.scores is not None
        assert s.scores.shape == (cal_data["X"].shape[0],)

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_fit_leaveoneout(
        self, cls: type, clustered_data: dict[str, torch.Tensor]
    ) -> None:
        s = cls(k=5)
        s.fit(X=clustered_data["X"], labels=clustered_data["y"])
        assert s.is_trained()
        # No cal set → not calibrated, but scores exist from LOO.
        assert not s.is_calibrated()
        assert s.scores is not None
        assert s.scores.shape == (clustered_data["X"].shape[0],)
        assert (s.scores >= 0.0).all() and (s.scores <= 1.0).all()
        assert s.num_classes == clustered_data["C"]
        assert s.label_mode == "single"
        # Labels normalized to multi-hot.
        assert s.ref_labels.dtype == torch.bool
        assert s.ref_labels.shape == (
            clustered_data["X"].shape[0],
            clustered_data["C"],
        )

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_fit_with_cal(
        self,
        cls: type,
        clustered_data: dict[str, torch.Tensor],
        cal_data: dict[str, torch.Tensor],
    ) -> None:
        s = cls(k=5)
        s.fit(
            X=clustered_data["X"],
            labels=clustered_data["y"],
            Y=cal_data["X"],
            cal_labels=cal_data["y"],
        )
        assert s.is_trained() and s.is_calibrated()
        assert s.scores is not None
        assert s.scores.shape == (cal_data["X"].shape[0],)
        assert (s.scores >= 0.0).all() and (s.scores <= 1.0).all()

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_num_classes_inference(
        self, cls: type, clustered_data: dict[str, torch.Tensor]
    ) -> None:
        s = cls(k=3)
        s.fit(X=clustered_data["X"], labels=clustered_data["y"])
        assert s.num_classes == clustered_data["C"]

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_num_classes_override(
        self, cls: type, clustered_data: dict[str, torch.Tensor]
    ) -> None:
        s = cls(k=3, num_classes=5)
        s.fit(X=clustered_data["X"], labels=clustered_data["y"])
        assert s.num_classes == 5
        assert s.ref_labels.shape[1] == 5


class TestFitLoader:
    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_fit_from_loaders(
        self, cls: type, loader_pair: dict[str, DataLoader[Any]]
    ) -> None:
        s = cls(k=3)
        s.fit(model=_TinyEmbedModel(), loaders=loader_pair)
        assert s.is_trained() and s.is_calibrated()
        assert s.ref_labels is not None and s.cal_labels is not None
        assert s.ref_labels.dtype == torch.bool

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_fit_train_only_loader(
        self, cls: type, loader_pair: dict[str, DataLoader[Any]]
    ) -> None:
        s = cls(k=3)
        s.fit(model=_TinyEmbedModel(), loaders={"train": loader_pair["train"]})
        assert s.is_trained()
        assert not s.is_calibrated()
        assert s.cal_labels is None


class TestScoreSemantics:
    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_score_no_target_needed(
        self, cls: type, clustered_data: dict[str, torch.Tensor]
    ) -> None:
        s = cls(k=3)
        s.fit(X=clustered_data["X"], labels=clustered_data["y"])
        # No target required, no error.
        scores = s.score(X=clustered_data["X"][:5])
        assert scores.shape == (5,)

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_score_range(
        self, cls: type, clustered_data: dict[str, torch.Tensor]
    ) -> None:
        s = cls(k=5)
        s.fit(X=clustered_data["X"], labels=clustered_data["y"])
        Q = torch.randn(32, int(clustered_data["D"]))
        scores = s.score(X=Q)
        assert scores.shape == (32,)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_pure_neighborhood_gives_zero_entropy(
        self, cls: type, clustered_data: dict[str, torch.Tensor]
    ) -> None:
        """Well-separated clusters → all k neighbors share a class →
        entropy = 0."""
        s = cls(k=5, diversity="entropy")
        s.fit(X=clustered_data["X"], labels=clustered_data["y"])
        idx = torch.arange(0, clustered_data["X"].shape[0], step=10)
        Q = clustered_data["X"][idx]
        scores = s.score(X=Q)
        # Entropy of a delta distribution is 0.
        torch.testing.assert_close(
            scores, torch.zeros_like(scores), atol=1e-6, rtol=0.0
        )

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_pure_neighborhood_gives_low_unique(
        self, cls: type, clustered_data: dict[str, torch.Tensor]
    ) -> None:
        """Pure single-class neighborhood → unique fraction = 1/min(k,C)."""
        s = cls(k=5, diversity="unique")
        s.fit(X=clustered_data["X"], labels=clustered_data["y"])
        idx = torch.arange(0, clustered_data["X"].shape[0], step=10)
        Q = clustered_data["X"][idx]
        scores = s.score(X=Q)
        denom = min(5, int(clustered_data["C"]))
        expected = 1.0 / denom
        torch.testing.assert_close(
            scores, torch.full_like(scores, expected), atol=1e-6, rtol=0.0
        )

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_mixed_neighborhood_high_entropy(
        self, cls: type, mixed_data: dict[str, torch.Tensor]
    ) -> None:
        """Overlapping classes → neighborhood covers many labels → high
        entropy."""
        s = cls(k=8, diversity="entropy")
        s.fit(X=mixed_data["X"], labels=mixed_data["y"])
        Q = torch.randn(20, int(mixed_data["D"])) * 0.3
        scores = s.score(X=Q)
        # On average, the normalized entropy should be well above 0.
        assert scores.mean() >= 0.5

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_mixed_neighborhood_high_unique(
        self, cls: type, mixed_data: dict[str, torch.Tensor]
    ) -> None:
        s = cls(k=8, diversity="unique")
        s.fit(X=mixed_data["X"], labels=mixed_data["y"])
        Q = torch.randn(20, int(mixed_data["D"])) * 0.3
        scores = s.score(X=Q)
        # With C=4 classes and k=8 well-mixed neighbors, most queries
        # should see all classes present.
        assert scores.mean() >= 0.75

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_entropy_upper_bound_is_uniform(
        self, cls: type, mixed_data: dict[str, torch.Tensor]
    ) -> None:
        """Normalized entropy is bounded above by 1.0."""
        s = cls(k=8, diversity="entropy")
        s.fit(X=mixed_data["X"], labels=mixed_data["y"])
        Q = torch.randn(50, int(mixed_data["D"])) * 0.3
        scores = s.score(X=Q)
        assert (scores <= 1.0 + 1e-6).all()

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_pure_vs_mixed_ordering(
        self,
        cls: type,
        clustered_data: dict[str, torch.Tensor],
        mixed_data: dict[str, torch.Tensor],
    ) -> None:
        """Diversity on well-separated clusters < diversity on overlapping
        labels, on average."""
        pure = cls(k=5, diversity="entropy")
        pure.fit(X=clustered_data["X"], labels=clustered_data["y"])
        Qp = clustered_data["X"][:20]
        sp = pure.score(X=Qp)

        mixed = cls(k=5, diversity="entropy")
        mixed.fit(X=mixed_data["X"], labels=mixed_data["y"])
        Qm = torch.randn(20, int(mixed_data["D"])) * 0.3
        sm = mixed.score(X=Qm)

        assert sp.mean() < sm.mean()

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_score_from_loader(
        self,
        cls: type,
        clustered_data: dict[str, torch.Tensor],
        loader_pair: dict[str, DataLoader[Any]],
    ) -> None:
        s = cls(k=3)
        s.fit(X=clustered_data["X"], labels=clustered_data["y"])
        scores = s.score(model=_TinyEmbedModel(), loader=loader_pair["val"])
        n_val = sum(b[0].shape[0] for b in loader_pair["val"])
        assert scores.shape == (n_val,)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_score_rejects_both_X_and_loader(
        self,
        cls: type,
        clustered_data: dict[str, torch.Tensor],
        loader_pair: dict[str, DataLoader[Any]],
    ) -> None:
        s = cls(k=3)
        s.fit(X=clustered_data["X"], labels=clustered_data["y"])
        with pytest.raises(ValueError, match="Cannot specify both"):
            s.score(
                X=clustered_data["X"][:4],
                model=_TinyEmbedModel(),
                loader=loader_pair["val"],
            )

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_score_requires_something(
        self, cls: type, clustered_data: dict[str, torch.Tensor]
    ) -> None:
        s = cls(k=3)
        s.fit(X=clustered_data["X"], labels=clustered_data["y"])
        with pytest.raises(ValueError, match="Must specify"):
            s.score()


class TestEntropyExactValues:
    """Sanity checks on the entropy computation using a hand-built case."""

    def test_two_class_uniform_neighborhood(self) -> None:
        """A neighborhood equally split between 2 classes has
        normalized entropy = 1."""
        # 2 classes, 6 points; queries far from cluster centers so KNN
        # is forced to mix.
        X = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [10.0, 10.0],
                [10.1, 10.0],
                [10.0, 10.1],
            ]
        )
        y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        s = LabelDiversityEuclideanScore(k=6, diversity="entropy")
        s.fit(X=X, labels=y)
        # Any query point will pull in all 6 refs when k=6 → uniform split.
        Q = torch.tensor([[5.0, 5.0]])
        score = s.score(X=Q)
        torch.testing.assert_close(
            score, torch.tensor([1.0]), atol=1e-5, rtol=0.0
        )

    def test_unique_two_of_two_classes(self) -> None:
        X = torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
        y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        s = LabelDiversityEuclideanScore(k=4, diversity="unique")
        s.fit(X=X, labels=y)
        Q = torch.tensor([[5.0, 5.0]])
        score = s.score(X=Q)
        # 2 classes present out of min(k=4, C=2) = 2 → 1.0.
        torch.testing.assert_close(
            score, torch.tensor([1.0]), atol=1e-6, rtol=0.0
        )


class TestMultiLabel:
    @pytest.fixture
    def multi_data(self) -> dict[str, torch.Tensor | int]:
        D, C, N = 6, 4, 40
        X = torch.randn(N, D)
        y = (torch.rand(N, C) > 0.6).bool()
        # Guarantee at least one True per row.
        y[torch.arange(N), torch.randint(0, int(C), (N,))] = True
        return {"X": X, "y": y, "C": int(C), "D": int(D)}

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_fit_multi_label(
        self, cls: type, multi_data: dict[str, torch.Tensor]
    ) -> None:
        s = cls(k=3)
        s.fit(X=multi_data["X"], labels=multi_data["y"])
        assert s.label_mode == "multi"
        assert s.num_classes == multi_data["C"]
        assert s.ref_labels.dtype == torch.bool
        assert s.ref_labels.shape == (multi_data["X"].shape[0], multi_data["C"])

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_score_multi_label(
        self, cls: type, multi_data: dict[str, torch.Tensor]
    ) -> None:
        s = cls(k=3)
        s.fit(X=multi_data["X"], labels=multi_data["y"])
        Q = torch.randn(10, int(multi_data["D"]))
        scores = s.score(X=Q)
        assert scores.shape == (10,)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_multi_label_int_form_accepted(
        self, cls: type, multi_data: dict[str, torch.Tensor]
    ) -> None:
        s = cls(k=3)
        # {0,1} int tensor should be treated as multi-label.
        s.fit(X=multi_data["X"], labels=multi_data["y"].long())
        assert s.label_mode == "multi"


class TestSelect:
    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_set_threshold(
        self,
        cls: type,
        clustered_data: dict[str, torch.Tensor],
        cal_data: dict[str, torch.Tensor],
    ) -> None:
        s = cls(k=5)
        s.fit(
            X=clustered_data["X"],
            labels=clustered_data["y"],
            Y=cal_data["X"],
            cal_labels=cal_data["y"],
        )
        s.set_threshold(q=0.95)
        assert s.threshold is not None
        assert 0.0 <= float(s.threshold) <= 1.0

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_select_shapes(
        self,
        cls: type,
        clustered_data: dict[str, torch.Tensor],
        cal_data: dict[str, torch.Tensor],
    ) -> None:
        s = cls(k=5)
        s.fit(
            X=clustered_data["X"],
            labels=clustered_data["y"],
            Y=cal_data["X"],
            cal_labels=cal_data["y"],
        )
        s.set_threshold(q=0.95)
        Q = torch.randn(16, int(clustered_data["D"]))
        out = s.select(X=Q)
        assert set(out.keys()) == {"score", "selected"}
        assert out["score"].shape == (16,)
        assert out["selected"].shape == (16,)
        assert out["selected"].dtype == torch.bool

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_select_auto_threshold(
        self, cls: type, clustered_data: dict[str, torch.Tensor]
    ) -> None:
        s = cls(k=5)
        s.fit(X=clustered_data["X"], labels=clustered_data["y"])
        # Threshold not set — should be auto-computed with a warning.
        Q = clustered_data["X"][:8]
        out = s.select(X=Q)
        assert s.threshold is not None
        assert out["selected"].dtype == torch.bool

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_select_no_target_accepted(
        self,
        cls: type,
        clustered_data: dict[str, torch.Tensor],
        cal_data: dict[str, torch.Tensor],
    ) -> None:
        """Select must not require a `target` kwarg — diversity is
        label-free at query time."""
        s = cls(k=5)
        s.fit(
            X=clustered_data["X"],
            labels=clustered_data["y"],
            Y=cal_data["X"],
            cal_labels=cal_data["y"],
        )
        s.set_threshold(q=0.95)
        out = s.select(X=torch.randn(4, int(clustered_data["D"])))
        assert out["selected"].shape == (4,)


class TestPCA:
    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_pca_reduces_dim(
        self, cls: type, clustered_data: dict[str, torch.Tensor]
    ) -> None:
        pca = TensorPCA(n_components=4)
        s = cls(k=3, pca=pca)
        s.fit(X=clustered_data["X"], labels=clustered_data["y"])
        # After fitting, ref_embeddings should live in PCA space.
        assert s.ref_embeddings.shape[1] == 4

    @pytest.mark.parametrize("cls", DIVERSITY_CLASSES)
    def test_pca_score_consistency(
        self, cls: type, clustered_data: dict[str, torch.Tensor]
    ) -> None:
        pca = TensorPCA(n_components=4)
        s = cls(k=5, pca=pca, diversity="entropy")
        s.fit(X=clustered_data["X"], labels=clustered_data["y"])
        # Pure-cluster query should still give ~0 entropy even after PCA.
        idx = torch.arange(0, clustered_data["X"].shape[0], step=15)
        Q = clustered_data["X"][idx]  # raw, un-PCA'd — matches user API
        scores = s.score(X=Q)
        torch.testing.assert_close(
            scores, torch.zeros_like(scores), atol=1e-6, rtol=0.0
        )


class TestPersistence:
    def test_save_index_roundtrip(
        self, tmp_path: Path, clustered_data: dict[str, torch.Tensor]
    ) -> None:
        idx_path = tmp_path / "diversity.bin"
        s1 = LabelDiversityEuclideanScore(k=5, save_index=idx_path)
        s1.fit(X=clustered_data["X"], labels=clustered_data["y"])
        assert idx_path.exists()

        # A fresh score should load the same index from disk.
        s2 = LabelDiversityEuclideanScore(k=5, save_index=idx_path)
        s2.fit(X=clustered_data["X"], labels=clustered_data["y"])

        Q = clustered_data["X"][:10]
        torch.testing.assert_close(s1.score(X=Q), s2.score(X=Q))
