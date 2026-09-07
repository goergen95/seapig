"""Consolidated tests for ClassWiseScore wrappers.

Covers generic ClassWiseScore behavior as well as all KNN and Logit
class‑wise scores. Parameterized tests reduce duplication while ensuring
identical test logic across score types.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from seapig.scores import (
    ClassWiseScore,
    CosineClassWiseScore,
    EnergyClassWiseScore,
    EntropyClassWiseScore,
    EuclideanClassWiseScore,
    MahalanobisClassWiseScore,
    MarginClassWiseScore,
    MutualInformationClassWiseScore,
    PredictiveVarianceClassWiseScore,
    SoftmaxClassWiseScore,
)
from tests.fixtures import DummyModel, DummyScore


def test_set_threshold_without_fit():
    cw = ClassWiseScore(base_score_cls=DummyScore)
    with pytest.raises(
        RuntimeError, match="Fit must be called before setting thresholds"
    ):
        cw.set_threshold()


def test_score_without_fit():
    cw = ClassWiseScore(base_score_cls=DummyScore)
    X = torch.randn(4, 2)
    with pytest.raises(RuntimeError, match="fit must be called before scoring"):
        cw.score(X)


def test_fit_error_both_modes():
    X = torch.randn(4, 2)
    dummy_model = torch.nn.Linear(3, 3)
    cw = ClassWiseScore(base_score_cls=DummyScore)
    with pytest.raises(
        ValueError, match="Specify either pre-computed tensors.*or a model"
    ):
        cw.fit(X=X, model=dummy_model, loaders={"train": None})  # type: ignore


def test_fit_error_no_samples_multi_label():
    X = torch.randn(4, 2)
    y = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=torch.int64)
    cw = ClassWiseScore(base_score_cls=DummyScore)
    with pytest.raises(
        ValueError, match="No training samples found for class 1"
    ):
        cw.fit(X=X, y=y)


def test_score_error_both_modes():
    X = torch.randn(5, 3)
    y = torch.arange(5) % 2
    dummy_model = torch.nn.Linear(3, 3)
    cw = ClassWiseScore(base_score_cls=DummyScore)
    cw.fit(X=X, y=y)
    with pytest.raises(
        ValueError, match="Specify either pre-computed tensors.*or a model"
    ):
        cw.score(X=X, model=dummy_model, loader=None)


def test_plot_exception_handling():
    pytest.importorskip("matplotlib")
    from unittest.mock import patch

    import matplotlib.pyplot as plt

    class BadPlotScore(DummyScore):
        def plot(
            self, query_scores: torch.Tensor | None = None, bins: int = 100
        ) -> None:
            raise RuntimeError("plot failed deliberately")

    class BadPlotClassWiseScore(ClassWiseScore):
        def __init__(self, **kwargs):
            super().__init__(base_score_cls=BadPlotScore, **kwargs)

    X = torch.randn(6, 2)
    y = torch.tensor([0, 0, 1, 1, 0, 1])
    cw = BadPlotClassWiseScore()
    cw.fit(X=X, y=y)
    with (
        patch.object(plt, "show"),
        pytest.raises(
            RuntimeError,
            match="Plot failed for class 0: plot failed deliberately",
        ),
    ):
        cw.plot()


def make_loader(
    X: torch.Tensor, y: torch.Tensor, batch_size: int = 4
) -> DataLoader:
    data = [{"image": X[i], "label": y[i]} for i in range(len(y))]
    return DataLoader(
        data,  # ty: ignore[invalid-argument-type]
        batch_size=batch_size,
        collate_fn=lambda batch: {
            k: torch.stack([d[k] for d in batch]) for k in batch[0]
        },
    )


def _make_embeddings(num: int, dim: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(num, dim)


def _make_logits_multi(num: int, dim: int, members: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(num, members, dim)


# Parameter definitions
knn_cases = [
    (EuclideanClassWiseScore, {"k": 1}, True),
    (CosineClassWiseScore, {"k": 2}, True),
    (MahalanobisClassWiseScore, {"k": 1}, True),
]
logit_cases = [
    (SoftmaxClassWiseScore, {"per_member": True}, False),
    (EntropyClassWiseScore, {"per_member": True}, False),
    (MarginClassWiseScore, {"per_member": True}, False),
    (EnergyClassWiseScore, {"per_member": True}, False),
    (MutualInformationClassWiseScore, {}, False),
    (PredictiveVarianceClassWiseScore, {}, False),
]
all_cases = knn_cases + logit_cases


@pytest.mark.parametrize("score_cls, kwargs, is_knn", knn_cases)
def test_knn_single_label(score_cls, kwargs, is_knn):
    # generate single‑label embedding data
    X_train = _make_embeddings(30, 3)
    y_train = torch.tensor([0, 0, 1, 1, 2, 2, 2, 0, 1, 2] * 3)
    X_val = _make_embeddings(4, 3)
    y_val = torch.tensor([0, 1, 2, 2])
    X_test = _make_embeddings(3, 3)
    cw = score_cls(**kwargs)
    cw.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)
    cw.set_threshold(q=0.95)
    scores = cw.score(X_test)
    assert scores.shape == (X_test.shape[0], 3)
    for idx, label in enumerate(sorted(torch.unique(y_train).tolist())):
        class_scorer = cw._scorers[label]
        expected = class_scorer.score(X_test)
        torch.testing.assert_close(scores[:, idx], expected)
    result = cw.select(X_test)
    assert "score" in result and "selected" in result
    assert result["score"].shape == scores.shape
    assert result["selected"].shape == scores.shape
    for idx, label in enumerate(sorted(torch.unique(y_train).tolist())):
        thr = cw._thresholds[label]
        assert torch.equal(result["selected"][:, idx], scores[:, idx] < thr)


@pytest.mark.parametrize("score_cls, kwargs, is_knn", knn_cases)
def test_knn_multi_label(score_cls, kwargs, is_knn):
    X_train = _make_embeddings(24, 4)
    y_train = torch.tensor(
        [[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0]] * 3,
        dtype=torch.int64,
    )
    X_val = _make_embeddings(2, 4)
    y_val = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)
    X_test = _make_embeddings(5, 4)
    cw = score_cls(**kwargs)
    cw.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)
    cw.set_threshold(q=0.9)
    scores = cw.score(X_test)
    assert scores.shape == (X_test.shape[0], 2)
    for idx in range(2):
        class_scorer = cw._scorers[idx]
        expected = class_scorer.score(X_test)
        torch.testing.assert_close(scores[:, idx], expected)
    out = cw.select(X_test)
    assert out["selected"].shape == scores.shape
    for idx in range(2):
        thr = cw._thresholds[idx]
        assert torch.equal(out["selected"][:, idx], scores[:, idx] < thr)


@pytest.mark.parametrize("score_cls, kwargs, is_knn", knn_cases)
def test_knn_model_loader_single(score_cls, kwargs, is_knn):
    X_train = _make_embeddings(30, 3)
    y_train = torch.tensor([0, 0, 1, 1, 2, 2, 2, 0, 1, 2] * 3)
    X_val = _make_embeddings(4, 3)
    y_val = torch.tensor([0, 1, 2, 2])
    X_test = _make_embeddings(3, 3)
    train_loader = make_loader(X_train, y_train)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(
        X_test, torch.zeros(len(X_test), dtype=torch.int64)
    )
    cw = score_cls(**kwargs)
    cw.fit(
        model=DummyModel(), loaders={"train": train_loader, "val": val_loader}
    )
    cw.set_threshold(q=0.95)
    scores = cw.score(model=DummyModel(), loader=test_loader)
    assert scores.shape == (X_test.shape[0], 3)
    for lbl in sorted(torch.unique(y_train).tolist()):
        class_scorer = cw._scorers[lbl]
        expected = class_scorer.score(X_test)
        idx = sorted(torch.unique(y_train).tolist()).index(lbl)
        torch.testing.assert_close(scores[:, idx], expected)
    result = cw.select(model=DummyModel(), loader=test_loader)
    assert "score" in result and "selected" in result
    assert result["selected"].shape == scores.shape
    for lbl in sorted(torch.unique(y_train).tolist()):
        thr = cw._thresholds[lbl]
        idx = sorted(torch.unique(y_train).tolist()).index(lbl)
        assert torch.equal(result["selected"][:, idx], scores[:, idx] < thr)


@pytest.mark.parametrize("score_cls, kwargs, is_knn", knn_cases)
def test_knn_model_loader_multi(score_cls, kwargs, is_knn):
    X_train = _make_embeddings(30, 3)
    y_train = torch.tensor(
        [[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0]] * 3,
        dtype=torch.int64,
    )
    X_val = _make_embeddings(2, 3)
    y_val = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)
    X_test = _make_embeddings(5, 3)
    train_loader = make_loader(X_train, y_train)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(
        X_test, torch.zeros(len(X_test), 2, dtype=torch.int64)
    )
    cw = score_cls(**kwargs)
    cw.fit(
        model=DummyModel(), loaders={"train": train_loader, "val": val_loader}
    )
    cw.set_threshold(q=0.9)
    scores = cw.score(model=DummyModel(), loader=test_loader)
    assert scores.shape == (X_test.shape[0], 2)
    for lbl in range(2):
        class_scorer = cw._scorers[lbl]
        expected = class_scorer.score(X_test)
        torch.testing.assert_close(scores[:, lbl], expected)
    result = cw.select(model=DummyModel(), loader=test_loader)
    assert "score" in result and "selected" in result
    assert result["selected"].shape == scores.shape
    for lbl in range(2):
        thr = cw._thresholds[lbl]
        assert torch.equal(result["selected"][:, lbl], scores[:, lbl] < thr)


@pytest.mark.parametrize("score_cls, kwargs, is_knn", knn_cases)
def test_knn_pca_per_class(score_cls, kwargs, is_knn):
    from seapig.scores.utils import TensorPCA

    X_train = _make_embeddings(30, 12)
    y_train = torch.tensor([0, 0, 1, 1, 2, 2, 2, 0, 1, 2] * 3)
    X_val = _make_embeddings(4, 12)
    y_val = torch.tensor([0, 1, 2, 2])
    X_test = _make_embeddings(3, 12)
    orig_dim = X_train.shape[1]
    pca = TensorPCA(n_components=2)
    cw = score_cls(**kwargs, pca=pca)
    cw.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)
    for scorer in cw._scorers.values():
        assert isinstance(scorer.pca, TensorPCA)
        assert isinstance(scorer.pca.u, torch.Tensor)
        assert scorer.pca.u.numel() > 0
        assert isinstance(scorer.ref_embeddings, torch.Tensor)
        assert scorer.ref_embeddings.shape[1] < orig_dim
    scores = cw.score(X_test)
    assert scores.shape == (X_test.shape[0], len(cw._scorers))


# Logit scores – reuse similar structure without PCA
@pytest.mark.parametrize("score_cls, kwargs, is_knn", logit_cases)
def test_logit_single_label(score_cls, kwargs, is_knn):
    # Use multi‑member logits for all scores
    X_train = _make_logits_multi(10, 4, members=2)
    X_val = _make_logits_multi(4, 4, members=2)
    X_test = _make_logits_multi(3, 4, members=2)
    y_train = torch.randint(0, 4, (10,))
    y_val = torch.randint(0, 4, (4,))
    cw = score_cls(**kwargs)
    cw.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)
    cw.set_threshold(q=0.95)
    scores = cw.score(X_test)
    assert scores.shape == (X_test.shape[0], len(torch.unique(y_train)))
    for idx, label in enumerate(sorted(torch.unique(y_train).tolist())):
        class_scorer = cw._scorers[label]
        expected = class_scorer.score(X_test)
        torch.testing.assert_close(scores[:, idx], expected)
    result = cw.select(X_test)
    assert "score" in result and "selected" in result
    assert result["score"].shape == scores.shape
    assert result["selected"].shape == scores.shape
    for idx, label in enumerate(sorted(torch.unique(y_train).tolist())):
        thr = cw._thresholds[label]
        assert torch.equal(result["selected"][:, idx], scores[:, idx] < thr)


@pytest.mark.parametrize("score_cls, kwargs, is_knn", logit_cases)
def test_logit_multi_label(score_cls, kwargs, is_knn):
    # Use multi‑member logits for all scores
    X_train = _make_logits_multi(8, 3, members=2)
    X_val = _make_logits_multi(2, 3, members=2)
    X_test = _make_logits_multi(5, 3, members=2)
    y_train = torch.tensor(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=torch.int64,
    )
    y_val = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.int64)
    cw = score_cls(**kwargs)
    cw.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)
    cw.set_threshold(q=0.9)
    scores = cw.score(X_test)
    assert scores.shape == (X_test.shape[0], 3)
    for idx in range(3):
        class_scorer = cw._scorers[idx]
        expected = class_scorer.score(X_test)
        torch.testing.assert_close(scores[:, idx], expected)
    out = cw.select(X_test)
    assert out["selected"].shape == scores.shape
    for idx in range(3):
        thr = cw._thresholds[idx]
        assert torch.equal(out["selected"][:, idx], scores[:, idx] < thr)


@pytest.mark.parametrize("score_cls, kwargs, is_knn", logit_cases)
def test_logit_model_loader_single(score_cls, kwargs, is_knn):
    # Use multi‑member logits for all scores
    X_train = _make_logits_multi(10, 4, members=2)
    X_val = _make_logits_multi(4, 4, members=2)
    X_test = _make_logits_multi(3, 4, members=2)
    y_train = torch.randint(0, 4, (10,))
    y_val = torch.randint(0, 4, (4,))
    train_loader = make_loader(X_train, y_train)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(
        X_test, torch.zeros(len(X_test), dtype=torch.int64)
    )
    cw = score_cls(**kwargs)
    cw.fit(
        model=DummyModel(), loaders={"train": train_loader, "val": val_loader}
    )
    cw.set_threshold(q=0.95)
    scores = cw.score(model=DummyModel(), loader=test_loader)
    assert scores.shape == (X_test.shape[0], len(torch.unique(y_train)))
    for lbl in sorted(torch.unique(y_train).tolist()):
        class_scorer = cw._scorers[lbl]
        expected = class_scorer.score(X_test)
        idx = sorted(torch.unique(y_train).tolist()).index(lbl)
        torch.testing.assert_close(scores[:, idx], expected)
    result = cw.select(model=DummyModel(), loader=test_loader)
    assert "score" in result and "selected" in result
    assert result["selected"].shape == scores.shape
    for lbl in sorted(torch.unique(y_train).tolist()):
        thr = cw._thresholds[lbl]
        idx = sorted(torch.unique(y_train).tolist()).index(lbl)
        assert torch.equal(result["selected"][:, idx], scores[:, idx] < thr)


@pytest.mark.parametrize("score_cls, kwargs, is_knn", logit_cases)
def test_logit_model_loader_multi(score_cls, kwargs, is_knn):
    # Use multi‑member logits for all scores
    X_train = _make_logits_multi(8, 3, members=2)
    X_val = _make_logits_multi(2, 3, members=2)
    X_test = _make_logits_multi(5, 3, members=2)
    y_train = torch.tensor(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=torch.int64,
    )
    y_val = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.int64)
    train_loader = make_loader(X_train, y_train)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(
        X_test, torch.zeros(len(X_test), 3, dtype=torch.int64)
    )
    cw = score_cls(**kwargs)
    cw.fit(
        model=DummyModel(), loaders={"train": train_loader, "val": val_loader}
    )
    cw.set_threshold(q=0.9)
    scores = cw.score(model=DummyModel(), loader=test_loader)
    assert scores.shape == (X_test.shape[0], 3)
    for lbl in range(3):
        class_scorer = cw._scorers[lbl]
        expected = class_scorer.score(X_test)
        torch.testing.assert_close(scores[:, lbl], expected)
    result = cw.select(model=DummyModel(), loader=test_loader)
    assert "score" in result and "selected" in result
    assert result["selected"].shape == scores.shape
    for lbl in range(3):
        thr = cw._thresholds[lbl]
        assert torch.equal(result["selected"][:, lbl], scores[:, lbl] < thr)
