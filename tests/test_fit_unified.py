"""Tests for unified fit() method API."""

from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from seapig.scores.embed import EmbeddingScore
from seapig.scores.knn import EuclideanScore
from seapig.scores.logits import SoftmaxScore
from seapig.scores.pca import PCAScore
from seapig.scores.utils import TensorPCA

try:
    from pyod.models.knn import KNN

    from seapig.scores.pyod import PyODScore

except ImportError:  # pragma: no cover
    KNN = None
    print("PyOD is not installed; skipping PyODScore tests.")

_EmbedLoader = DataLoader[torch.Tensor | dict[str, torch.Tensor]]


class DummyModel(torch.nn.Module):
    """Dummy model for testing embedding extraction."""

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, dict):
            x = x["image"]  # pragma: no cover
        return x

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, dict):
            x = x["image"]  # pragma: no cover
        # Return simple logits based on input
        return torch.randn(x.shape[0], 3)


class MinimalEmbedding(EmbeddingScore):
    """Minimal concrete EmbeddingScore for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.train_required = False
        self.cal_required = False

    def _score_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        return X.sum(dim=1)  # pragma: no cover


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
        DataLoader([torch.randn(5)]),  # type: ignore[arg-type]
    )

    score = MinimalEmbedding()
    with pytest.raises(ValueError, match="Cannot specify both"):
        score.fit(X=ref_embs, model=model, loaders={"train": train_loader})


def test_fit_rejects_neither_embeddings_nor_model() -> None:
    """Test that fit() raises error when neither embeddings nor model provided."""
    score = MinimalEmbedding()
    with pytest.raises(ValueError, match="Must specify either"):
        score.fit()


def test_fit_rejects_model_without_loaders() -> None:
    """Test that fit() raises error when model provided without loaders."""
    model = DummyModel()
    score = MinimalEmbedding()
    with pytest.raises(ValueError, match="loaders is required"):
        score.fit(model=model)


def test_fit_rejects_loaders_without_model() -> None:
    """Test that fit() raises error when loaders provided without model."""
    train_loader: _EmbedLoader = cast(
        _EmbedLoader,
        DataLoader([torch.randn(5)]),  # type: ignore[arg-type]
    )
    score = MinimalEmbedding()
    with pytest.raises(ValueError, match="model is required"):
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


def test_logit_score_fit_with_logits() -> None:
    """Test LogitScore fit() with precomputed logits."""
    score = SoftmaxScore()
    logits = torch.randn(10, 3)
    labels = torch.randint(0, 3, (10,))

    # Use X and Y parameters (logits and labels as kwargs also supported)
    score.fit(X=logits, Y=labels)

    assert score.logits is not None
    assert score.labels is not None
    assert score.temperature is not None


def test_logit_score_fit_with_model() -> None:
    """Test LogitScore fit() with model and loader."""
    model = DummyModel()
    data = torch.randn(10, 8)
    loader = DataLoader(
        TensorDataset(data),
        batch_size=2,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )

    score = SoftmaxScore()
    score.fit(model=model, loader=loader)

    assert score.logits is not None
    assert score.logits.shape[0] == 10


def test_logit_score_rejects_both_logits_and_model() -> None:
    """Test that LogitScore fit() rejects both logits and model."""
    model = DummyModel()
    logits = torch.randn(10, 3)
    loader = cast(DataLoader[object], DataLoader([torch.randn(8)]))  # type: ignore[arg-type]

    score = SoftmaxScore()
    with pytest.raises(ValueError, match="Cannot specify both"):
        score.fit(X=logits, model=model, loader=loader)


def test_logit_score_rejects_neither_logits_nor_model() -> None:
    """Test that LogitScore fit() rejects neither logits nor model."""
    score = SoftmaxScore()
    with pytest.raises(ValueError, match="Must specify either"):
        score.fit()
