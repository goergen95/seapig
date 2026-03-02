"""Tests for ensuring SelectiveInferenceTask preserves model state.

This module tests that SelectiveInferenceTask and EmbeddingScore methods
properly preserve the model's training/evaluation state, which is critical
for models with BatchNorm or Dropout layers.
"""

import torch
from lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, MetricCollection

from seapig import SelectiveInferenceTask
from seapig.scores.knn import EuclideanScore


class DictDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset that returns dict with 'image' key for compatibility with EmbeddingScore."""

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.data[index]}


class ModelWithBatchNorm(LightningModule):
    """Model with BatchNorm to test training/eval mode sensitivity."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(16)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(16, 2)
        self.test_metrics = MetricCollection(
            Accuracy(task="multiclass", num_classes=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings before final classification layer."""
        x = self.conv(x)
        x = self.bn(x)  # BatchNorm will behave differently in train vs eval
        x = torch.relu(x)
        x = self.pool(x)
        return x.flatten(start_dim=1)


def test_model_state_preserved_in_forward() -> None:
    """Test that SelectiveInferenceTask.forward preserves model training state."""
    torch.manual_seed(42)
    model = ModelWithBatchNorm()
    score = EuclideanScore(k=2)

    # Fit the score with some dummy embeddings
    dummy_embs = torch.randn(10, 16)
    score.fit(dummy_embs, None)
    score.set_threshold(q=0.99)

    # Create SelectiveInferenceTask
    task = SelectiveInferenceTask(task=model, score=score)

    # Create test input
    x = torch.randn(2, 3, 8, 8)

    # Test 1: Model in eval mode should stay in eval mode
    model.eval()
    assert not model.training, "Model should be in eval mode"
    _ = task.forward(x)
    assert not model.training, (
        "Model should still be in eval mode after forward"
    )

    # Test 2: Model in training mode should stay in training mode
    model.train()
    assert model.training, "Model should be in training mode"
    _ = task.forward(x)
    assert model.training, (
        "Model should still be in training mode after forward"
    )


def test_fit_preserves_model_state() -> None:
    """Test that EmbeddingScore.fit preserves model training state."""
    torch.manual_seed(42)
    model = ModelWithBatchNorm()

    # Create dataloaders
    train_data = torch.randn(20, 3, 8, 8)
    train_dataset = DictDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=4)

    val_data = torch.randn(10, 3, 8, 8)
    val_dataset = DictDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=4)

    loaders = {"train": train_loader, "val": val_loader}

    score = EuclideanScore(k=2)

    # Test with model in eval mode
    model.eval()
    assert not model.training, "Model should be in eval mode"
    score.fit(model=model, loaders=loaders)
    assert not model.training, "Model should still be in eval mode after fit"

    # Reset score for next test
    score = EuclideanScore(k=2)

    # Test with model in training mode
    model.train()
    assert model.training, "Model should be in training mode"
    score.fit(model=model, loaders=loaders)
    assert model.training, "Model should still be in training mode after fit"


def test_score_preserves_model_state() -> None:
    """Test that EmbeddingScore.score preserves model training state."""
    torch.manual_seed(42)
    model = ModelWithBatchNorm()

    # Create dataloaders
    train_data = torch.randn(20, 3, 8, 8)
    train_dataset = DictDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=4)

    val_data = torch.randn(10, 3, 8, 8)
    val_dataset = DictDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=4)

    test_data = torch.randn(8, 3, 8, 8)
    test_dataset = DictDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=4)

    loaders = {"train": train_loader, "val": val_loader}

    score = EuclideanScore(k=2)

    # First fit the score
    model.eval()
    score.fit(model=model, loaders=loaders)
    score.set_threshold(q=0.99)

    # Test score with model in eval mode
    model.eval()
    assert not model.training, "Model should be in eval mode"
    _ = score.score(model=model, loader=test_loader, outdir=None, prefix=None)
    assert not model.training, "Model should still be in eval mode after score"

    # Test score with model in training mode
    model.train()
    assert model.training, "Model should be in training mode"
    _ = score.score(model=model, loader=test_loader, outdir=None, prefix=None)
    assert model.training, "Model should still be in training mode after score"


def test_loop_vs_standalone_scores_match() -> None:
    """Test that scores are consistent when model is used in a loop.

    This test reproduces the original issue: scores computed in a loop should
    match scores computed standalone, regardless of how many times the model
    has been called.
    """
    torch.manual_seed(42)
    model = ModelWithBatchNorm()

    # Create dataloaders
    train_data = torch.randn(20, 3, 8, 8)
    train_dataset = DictDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=4)

    val_data = torch.randn(10, 3, 8, 8)
    val_dataset = DictDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=4)

    test_data = torch.randn(8, 3, 8, 8)
    test_dataset = DictDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=4)

    loaders = {"train": train_loader, "val": val_loader}

    # Scenario 1: Fit score and compute test scores (standalone)
    model.eval()
    score1 = EuclideanScore(k=2)
    score1.fit(model=model, loaders=loaders)
    score1.set_threshold(q=0.99)
    scores_standalone = score1.score(
        model=model, loader=test_loader, outdir=None, prefix=None
    )

    # Scenario 2: Fit score in a "loop" (simulate repeated usage)
    model.eval()
    score2 = EuclideanScore(k=2)
    score2.fit(model=model, loaders=loaders)
    score2.set_threshold(q=0.99)

    # Simulate some intermediate model usage (like in a loop)
    for _ in range(3):
        _ = model(train_data[:4])

    scores_after_loop = score2.score(
        model=model, loader=test_loader, outdir=None, prefix=None
    )

    # Scores should be identical regardless of intermediate model calls
    assert torch.allclose(scores_standalone, scores_after_loop, atol=1e-5), (
        "Scores should be identical whether computed standalone or after loop usage"
    )


def test_embeddings_differ_in_train_vs_eval_mode() -> None:
    """Test that embeddings differ between training and eval modes (with BatchNorm).

    This demonstrates why it's critical to ensure the model is in the correct
    mode during embedding extraction.
    """
    torch.manual_seed(42)
    model = ModelWithBatchNorm()

    # Create test input
    x = torch.randn(4, 3, 8, 8)

    # Get embeddings in eval mode
    model.eval()
    with torch.inference_mode():
        emb_eval = model.embed(x)

    # Get embeddings in training mode (will use batch statistics for BatchNorm)
    model.train()
    with torch.inference_mode():
        emb_train = model.embed(x)

    # Embeddings should be DIFFERENT because BatchNorm behaves differently
    # in train vs eval mode
    assert not torch.allclose(emb_eval, emb_train, atol=1e-5), (
        "Embeddings should differ between training and eval modes due to BatchNorm"
    )


def test_fit_forces_eval_mode_during_embedding() -> None:
    """Test that fit forces eval mode during embedding extraction.

    This is the actual bug: if the model is in training mode when fit is called,
    the embeddings will be extracted using batch statistics (in training mode),
    which gives different results than when the model is in eval mode.

    The fix should ensure that embeddings are always extracted in eval mode,
    regardless of the model's initial state.
    """
    torch.manual_seed(42)
    model = ModelWithBatchNorm()

    # Create dataloaders
    train_data = torch.randn(20, 3, 8, 8)
    train_dataset = DictDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=4)

    val_data = torch.randn(10, 3, 8, 8)
    val_dataset = DictDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=4)

    loaders = {"train": train_loader, "val": val_loader}

    # Scenario 1: Fit with model in eval mode
    model.eval()
    score1 = EuclideanScore(k=2)
    score1.fit(model=model, loaders=loaders)
    assert score1.ref_embeddings is not None
    ref_emb_eval = score1.ref_embeddings.clone()
    assert score1.ref_embeddings is not None
    ref_emb_eval = score1.ref_embeddings.clone()

    # Scenario 2: Fit with model in training mode
    # The embeddings should be THE SAME as scenario 1, because fit should
    # force eval mode during embedding extraction
    model.train()
    score2 = EuclideanScore(k=2)
    score2.fit(model=model, loaders=loaders)
    assert score2.ref_embeddings is not None
    ref_emb_train = score2.ref_embeddings
    assert score2.ref_embeddings is not None
    ref_emb_train = score2.ref_embeddings

    # These should be IDENTICAL because fit should force eval mode
    # This will FAIL with the current implementation
    assert torch.allclose(ref_emb_eval, ref_emb_train, atol=1e-5), (
        "Embeddings should be identical regardless of initial model mode. "
        "fit should force eval mode during embedding extraction."
    )
