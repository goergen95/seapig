"""Tests for unified fit() method API."""

import re
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from seapig.scores.embed import EmbeddingScore
from seapig.scores.knn import EuclideanScore
from seapig.scores.logits import (
    EnergyScore,
    EntropyScore,
    MarginScore,
    MutualInformationScore,
    PredictiveVarianceScore,
    SoftmaxScore,
)
from seapig.scores.pca import PCAScore
from seapig.scores.utils import TensorPCA

_EmbedLoader = DataLoader[torch.Tensor | dict[str, torch.Tensor]]


class DummyModel(torch.nn.Module):
    """Dummy model for testing embedding extraction."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def embed(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, dict):
            x = x["image"]  # type: ignore[argument-type] # pragma: no cover
        return x


class MinimalEmbedding(EmbeddingScore):
    """Minimal concrete EmbeddingScore for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.train_required = False
        self.cal_required = False

    def _score(self, X: torch.Tensor) -> torch.Tensor:
        return X.sum(dim=1)  # pragma: no cover

    def _fit(self, q: bool | float = False):
        return


def test_fit_with_embeddings_only() -> None:
    """Test fit() with precomputed embeddings (mode 1)."""
    score = MinimalEmbedding()
    ref_embs = torch.randn(10, 5)
    cal_embs = torch.randn(5, 5)

    score.fit(X=ref_embs, Y=cal_embs)

    assert score.ref_embeddings is not None
    assert torch.equal(score.ref_embeddings, ref_embs)
    assert score.cal_embeddings is not None
    assert torch.equal(score.cal_embeddings, cal_embs)


def test_fit_with_model_loaders() -> None:
    """Test fit() with model and loaders (mode 2)."""
    model = DummyModel()
    train_data = torch.randn(10, 5)
    val_data = torch.randn(5, 5)
    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=2,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=2,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )

    score = MinimalEmbedding()
    score.fit(
        model=model,
        loaders={
            "train": cast(_EmbedLoader, train_loader),
            "val": cast(_EmbedLoader, val_loader),
        },
    )

    assert score.ref_embeddings is not None
    assert score.ref_embeddings.shape[0] == 10
    assert score.cal_embeddings is not None
    assert score.cal_embeddings.shape[0] == 5


def test_fit_with_model_train_only() -> None:
    """Test fit() with model and only train loader."""
    model = DummyModel()
    train_data = torch.randn(10, 5)
    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=2,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )

    score = MinimalEmbedding()
    score.fit(model=model, loaders={"train": cast(_EmbedLoader, train_loader)})

    assert score.ref_embeddings is not None
    assert score.ref_embeddings.shape[0] == 10
    assert score.cal_embeddings is None


def test_fit_rejects_both_embeddings_and_model() -> None:
    """Test that fit() raises error when both embeddings and model are provided."""
    model = DummyModel()
    ref_embs = torch.randn(10, 5)
    train_loader: _EmbedLoader = cast(
        _EmbedLoader,
        DataLoader([torch.randn(5)]),  # type: ignore[arg-type, ty:invalid-argument-type]
    )

    score = MinimalEmbedding()
    with pytest.raises(ValueError):
        score.fit(X=ref_embs, model=model, loaders={"train": train_loader})


def test_fit_rejects_neither_embeddings_nor_model() -> None:
    """Test that fit() raises error when neither embeddings nor model provided."""
    score = MinimalEmbedding()
    with pytest.raises(ValueError):
        score.fit()


def test_fit_rejects_model_without_loaders() -> None:
    """Test that fit() raises error when model provided without loaders."""
    model = DummyModel()
    score = MinimalEmbedding()
    with pytest.raises(AssertionError):
        score.fit(model=model)


def test_fit_rejects_loaders_without_model() -> None:
    """Test that fit() raises error when loaders provided without model."""
    train_loader: _EmbedLoader = cast(
        _EmbedLoader,
        DataLoader([torch.randn(5)]),  # type: ignore[arg-type, ty:invalid-argument-type]
    )
    score = MinimalEmbedding()
    with pytest.raises(AssertionError):
        score.fit(loaders={"train": train_loader})


def test_euclidean_score_fit_with_embeddings() -> None:
    """Test EuclideanScore fit() with precomputed embeddings."""
    score = EuclideanScore(k=2)
    ref_embs = torch.randn(20, 8)
    cal_embs = torch.randn(10, 8)

    score.fit(X=ref_embs, Y=cal_embs)

    assert score.ref_embeddings is not None
    assert score.is_trained()
    assert score.is_calibrated()


def test_euclidean_score_fit_with_model() -> None:
    """Test EuclideanScore fit() with model and loaders."""
    model = DummyModel()
    train_data = torch.randn(20, 8)
    val_data = torch.randn(10, 8)
    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=5,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=5,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )

    score = EuclideanScore(k=2)
    score.fit(
        model=model,
        loaders={
            "train": cast(_EmbedLoader, train_loader),
            "val": cast(_EmbedLoader, val_loader),
        },
    )

    assert score.ref_embeddings is not None
    assert score.is_trained()
    assert score.is_calibrated()


def test_pca_score_fit_with_embeddings() -> None:
    """Test PCAScore fit() with precomputed embeddings."""
    score = PCAScore(pca=TensorPCA(n_components=0.75))
    ref_embs = torch.randn(20, 8)
    cal_embs = torch.randn(10, 8)

    score.fit(X=ref_embs, Y=cal_embs)

    assert score.ref_embeddings is not None
    assert score.is_trained()
    assert score.is_calibrated()


def test_pca_score_fit_with_model() -> None:
    """Test PCAScore fit() with model and loaders."""
    model = DummyModel()
    train_data = torch.randn(20, 8)
    val_data = torch.randn(10, 8)
    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=5,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=5,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )

    score = PCAScore(pca=TensorPCA(n_components=0.75))
    score.fit(
        model=model,
        loaders={
            "train": cast(_EmbedLoader, train_loader),
            "val": cast(_EmbedLoader, val_loader),
        },
    )

    assert score.ref_embeddings is not None
    assert score.is_trained()
    assert score.is_calibrated()


def test_pyod_score_fit_with_embeddings() -> None:
    """Test PyODScore fit() with precomputed embeddings."""
    pytest.importorskip("pyod")
    from pyod.models.knn import KNN

    from seapig.scores.pyod import PyODScore

    score = PyODScore(detector=KNN(n_neighbors=2))
    ref_embs = torch.randn(20, 8)
    cal_embs = torch.randn(10, 8)

    score.fit(X=ref_embs, Y=cal_embs)

    assert score.ref_embeddings is not None
    assert score.is_trained()
    assert score.is_calibrated()


def test_pyod_score_fit_with_model() -> None:
    """Test PyODScore fit() with model and loaders."""
    pytest.importorskip("pyod")
    from pyod.models.knn import KNN

    from seapig.scores.pyod import PyODScore

    model = DummyModel()
    train_data = torch.randn(20, 8)
    val_data = torch.randn(10, 8)
    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=5,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=5,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )

    score = PyODScore(detector=KNN(n_neighbors=2))
    score.fit(
        model=model,
        loaders={
            "train": cast(_EmbedLoader, train_loader),
            "val": cast(_EmbedLoader, val_loader),
        },
    )

    assert score.ref_embeddings is not None
    assert score.is_trained()
    assert score.is_calibrated()


# Simple dummy model that returns logits of a specified shape
class DummyLogitModel(torch.nn.Module):
    def __init__(self, task: str, per_member: bool, K: int = 3, M: int = 5):
        super().__init__()
        self.task = task
        self.per_member = per_member
        self.K = K
        self.M = M

    def logits(self, x: torch.Tensor):
        N = x.shape[0]
        if self.task == "multiclass":
            if self.per_member:
                return torch.randn(N, self.K, self.M)
            return torch.randn(N, self.K)
        if self.task == "binary":
            if self.per_member:
                return torch.randn(N, self.M)
            return torch.randn(N)
        if self.task == "multilabel":
            if self.per_member:
                return torch.randn(N, self.K, self.M)
            return torch.randn(N, self.K)


@pytest.mark.parametrize(
    "ScoreClass,task,logits_shape,labels_fn,per_member",
    [
        # Pointwise categorical (default multiclass)
        (
            SoftmaxScore,
            "multiclass",
            (10, 3),
            lambda N, C: torch.randint(0, C, (N,)),
            False,
        ),
        (
            EntropyScore,
            "multiclass",
            (10, 3),
            lambda N, C: torch.randint(0, C, (N,)),
            False,
        ),
        (
            MarginScore,
            "multiclass",
            (10, 3),
            lambda N, C: torch.randint(0, C, (N,)),
            False,
        ),
        (
            EnergyScore,
            "multiclass",
            (10, 3),
            lambda N, C: torch.randint(0, C, (N,)),
            False,
        ),
        # Pointwise binary (single‑logit)
        (
            SoftmaxScore,
            "binary",
            (12,),
            lambda N, _: torch.randint(0, 2, (N,)).float(),
            False,
        ),
        (
            EntropyScore,
            "binary",
            (12,),
            lambda N, _: torch.randint(0, 2, (N,)).float(),
            False,
        ),
        (
            MarginScore,
            "binary",
            (12,),
            lambda N, _: torch.randint(0, 2, (N,)).float(),
            False,
        ),
        (
            EnergyScore,
            "binary",
            (12,),
            lambda N, _: torch.randint(0, 2, (N,)).float(),
            False,
        ),
        # Pointwise multilabel
        (
            SoftmaxScore,
            "multilabel",
            (8, 4),
            lambda N, C: torch.randint(0, 2, (N, C)).float(),
            False,
        ),
        (
            EntropyScore,
            "multilabel",
            (8, 4),
            lambda N, C: torch.randint(0, 2, (N, C)).float(),
            False,
        ),
        (
            MarginScore,
            "multilabel",
            (8, 4),
            lambda N, C: torch.randint(0, 2, (N, C)).float(),
            False,
        ),
        (
            EnergyScore,
            "multilabel",
            (8, 4),
            lambda N, C: torch.randint(0, 2, (N, C)).float(),
            False,
        ),
        # Ensemble categorical (multiple members)
        (
            MutualInformationScore,
            "multiclass",
            (7, 3, 5),
            lambda N, _: torch.randint(0, 3, (N,)),
            True,
        ),
        (
            PredictiveVarianceScore,
            "multiclass",
            (7, 3, 5),
            lambda N, _: torch.randint(0, 3, (N,)),
            True,
        ),
        # Ensemble binary (single‑logit with members)
        (
            MutualInformationScore,
            "binary",
            (9, 4),
            lambda N, _: torch.randint(0, 2, (N,)).float(),
            True,
        ),
        (
            PredictiveVarianceScore,
            "binary",
            (9, 4),
            lambda N, _: torch.randint(0, 2, (N,)).float(),
            True,
        ),
    ],
)
def test_fit_score_select_logits(
    ScoreClass, task, logits_shape, labels_fn, per_member
):
    """Unified ``fit``/``score``/``select`` works for every logit‑score variant.

    This test covers:
    * fitting with pre-computed logits (with and without temperature scaling)
    * fitting with a model+loader (the same behaviour as pre-computed)
    * all supported tasks (multiclass, binary, multilabel, ensemble)
    * ``score`` returns the stored scores and ``select`` returns a proper dict.
    """
    N = logits_shape[0]
    C = logits_shape[1] if len(logits_shape) > 1 else 1
    logits = torch.randn(*logits_shape)
    labels = labels_fn(N, C)

    # ---------- pre‑computed logits ----------
    scorer = ScoreClass(task=task)
    scorer.fit(logits, labels, temp_scale=False)
    assert scorer.logits is not None
    assert scorer.labels is not None
    assert scorer.temperature is None
    assert scorer.scores.shape == (N,)
    # score must match stored scores
    new_scores = scorer.score(logits)
    assert torch.allclose(new_scores, scorer.scores)
    # select returns dict with correct shapes/types
    sel = scorer.select(logits)
    assert isinstance(sel, dict)
    assert "score" in sel and "selected" in sel
    assert sel["score"].shape == (N,)
    assert sel["selected"].shape == (N,)
    assert sel["selected"].dtype == torch.bool

    # ---------- temperature scaling ----------
    scorer_ts = ScoreClass(task=task)
    scorer_ts.fit(logits, labels, temp_scale=True)
    assert (
        isinstance(scorer_ts.temperature, float) and scorer_ts.temperature > 0
    )

    # ---------- model + loader variant ----------
    dummy_data = torch.randn(N, 5)
    loader = DataLoader(
        TensorDataset(dummy_data),
        batch_size=2,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )
    # Determine K and M for the dummy model based on task and shape
    if per_member:
        if task == "binary":
            K = 1
            M = logits_shape[1]
        else:
            K = logits_shape[1]
            M = logits_shape[2]
    else:
        K = logits_shape[1] if len(logits_shape) > 1 else 1
        M = 1
    model = DummyLogitModel(task=task, per_member=per_member, K=K, M=M)
    scorer_model = ScoreClass(task=task)
    scorer_model.fit(model=model, loader=loader, temp_scale=False)
    assert scorer_model.logits is not None
    # Labels may be None when the loader does not provide them; attribute should exist
    assert hasattr(scorer_model, "labels")
    assert scorer_model.scores.shape == (N,)
    new_scores_m = scorer_model.score(scorer_model.logits)
    assert torch.allclose(new_scores_m, scorer_model.scores)
    sel_m = scorer_model.select(scorer_model.logits)
    assert isinstance(sel_m, dict)
    assert "score" in sel_m and "selected" in sel_m
    assert sel_m["score"].shape == (N,)
    assert sel_m["selected"].shape == (N,)
    assert sel_m["selected"].dtype == torch.bool


# ---------------------------------------------------------------------------
# Error handling tests – same for all logit score classes
# ---------------------------------------------------------------------------
match = re.escape(
    "Specify either pre-computed logits or a model with a loader, but not both."
)


@pytest.mark.parametrize(
    "ScoreClass",
    [
        SoftmaxScore,
        EntropyScore,
        MarginScore,
        EnergyScore,
        MutualInformationScore,
        PredictiveVarianceScore,
    ],
)
def test_logit_score_rejects_both_logits_and_model(ScoreClass):
    model = DummyLogitModel(task="multiclass", per_member=False)
    loader = DataLoader([torch.randn(5)], batch_size=1)  # type: ignore
    logits = torch.randn(10, 3)
    scorer = ScoreClass()
    with pytest.raises(ValueError, match=match):
        scorer.fit(X=logits, model=model, loader=loader)
    # also raises when one is specified, but missing the other
    with pytest.raises(
        ValueError, match="`model` and `loader` must be given together."
    ):
        scorer.fit(model=model, loader=None)


@pytest.mark.parametrize(
    "ScoreClass",
    [
        SoftmaxScore,
        EntropyScore,
        MarginScore,
        EnergyScore,
        MutualInformationScore,
        PredictiveVarianceScore,
    ],
)
def test_logit_score_rejects_neither_logits_nor_model(ScoreClass):
    scorer = ScoreClass()
    with pytest.raises(ValueError, match=match):
        scorer.fit()
