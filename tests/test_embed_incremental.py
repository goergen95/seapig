"""Tests for incremental (batch-wise) fitting in EmbeddingScore."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from seapig.scores.embed import EmbeddingScore
from seapig.scores.knn import EuclideanScore
from seapig.scores.utils import TensorPCA
from seapig.utils.progress import disable as disable_progress

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def approx(t1: torch.Tensor, t2: torch.Tensor, tol: float = 1e-5) -> None:
    assert torch.allclose(t1.float(), t2.float(), atol=tol, rtol=0)


class SimpleScore(EmbeddingScore):
    """Minimal concrete subclass: stores ref_embeddings, L2 scoring."""

    ident = "simple"
    train_required = False
    cal_required = False

    def __init__(self, pca: TensorPCA | None = None) -> None:
        super().__init__(pca=pca)

    def _score_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        assert self.ref_embeddings is not None
        return torch.cdist(X, self.ref_embeddings).min(dim=1).values


class IdentityModel(torch.nn.Module):
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return x


def make_loader(
    data: torch.Tensor, batch_size: int
) -> DataLoader[torch.Tensor]:
    dataset = TensorDataset(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda b: torch.stack([s[0] for s in b]),
    )


# ---------------------------------------------------------------------------
# TensorPCA incremental parity (TensorPCA API unchanged)
# ---------------------------------------------------------------------------


def test_tensor_pca_incremental_parity() -> None:
    """Batch partial_fit/finalize must match single-call fit."""
    disable_progress()
    torch.manual_seed(42)
    X = torch.randn(60, 16).float()

    tpca_full = TensorPCA(n_components=4)
    tpca_full.fit(X)

    tpca_batch = TensorPCA(n_components=4)
    tpca_batch.reset_partial()
    chunk_size = 20
    for start in range(0, X.shape[0], chunk_size):
        tpca_batch.partial_fit(X[start : start + chunk_size])
    tpca_batch.finalize()

    approx(tpca_full.mu, tpca_batch.mu)
    approx(tpca_full.s, tpca_batch.s)
    approx(tpca_full.s_acc, tpca_batch.s_acc)
    # eigenvectors may differ in sign; compare absolute values
    approx(tpca_full.u_q.abs(), tpca_batch.u_q.abs())


# ---------------------------------------------------------------------------
# Pre-computed embeddings: PCA is fitted by the base class
# ---------------------------------------------------------------------------


def test_precomputed_embeddings_fit_pca() -> None:
    """fit(X) with a PCA should call pca.fit on the training embeddings."""
    disable_progress()
    torch.manual_seed(0)
    X = torch.randn(50, 8).float()
    pca = TensorPCA(n_components=4)

    s = SimpleScore(pca=pca)
    s.fit(X=X, incremental="full")

    assert s.ref_embeddings is not None
    approx(s.ref_embeddings, X)
    # PCA should be fitted: u_q must be set (non-empty)
    assert pca.u_q.numel() > 0


def test_precomputed_embeddings_incremental_ignored() -> None:
    """incremental param is ignored for pre-computed embeddings: same result."""
    disable_progress()
    torch.manual_seed(1)
    X = torch.randn(30, 4).float()
    Y = torch.randn(10, 4).float()

    s_full = SimpleScore()
    s_full.fit(X=X, Y=Y, incremental="full")

    s_batch = SimpleScore()
    s_batch.fit(X=X, Y=Y, incremental="batch")

    assert s_full.ref_embeddings is not None
    assert s_batch.ref_embeddings is not None
    approx(s_full.ref_embeddings, s_batch.ref_embeddings)
    assert s_full.cal_embeddings is not None
    assert s_batch.cal_embeddings is not None
    approx(s_full.cal_embeddings, s_batch.cal_embeddings)


# ---------------------------------------------------------------------------
# Full vs batch parity for model + loaders
# ---------------------------------------------------------------------------


def test_embedding_score_full_vs_batch_loaders() -> None:
    """full mode and batch mode produce the same ref_embeddings from loaders."""
    disable_progress()
    torch.manual_seed(2)
    data = torch.randn(20, 4).float()
    val_data = torch.randn(8, 4).float()
    model = IdentityModel()

    s_full = SimpleScore()
    s_full.fit(
        model=model,
        loaders={
            "train": make_loader(data, 4),
            "val": make_loader(val_data, 4),
        },
        incremental="full",
    )

    s_batch = SimpleScore()
    s_batch.fit(
        model=model,
        loaders={
            "train": make_loader(data, 4),
            "val": make_loader(val_data, 4),
        },
        incremental="batch",
    )

    assert s_full.ref_embeddings is not None
    assert s_batch.ref_embeddings is not None
    approx(s_full.ref_embeddings, s_batch.ref_embeddings)
    assert s_full.cal_embeddings is not None
    assert s_batch.cal_embeddings is not None
    approx(s_full.cal_embeddings, s_batch.cal_embeddings)


def test_auto_selects_batch_for_multi_batch_loader() -> None:
    """incremental='auto' with >1 train batches should pick 'batch' mode."""
    disable_progress()
    torch.manual_seed(3)
    data = torch.randn(20, 4).float()  # 5 batches of 4
    model = IdentityModel()

    s_full = SimpleScore()
    s_full.fit(
        model=model, loaders={"train": make_loader(data, 4)}, incremental="full"
    )

    s_auto = SimpleScore()
    s_auto.fit(
        model=model, loaders={"train": make_loader(data, 4)}, incremental="auto"
    )

    assert s_full.ref_embeddings is not None
    assert s_auto.ref_embeddings is not None
    approx(s_full.ref_embeddings, s_auto.ref_embeddings)


def test_auto_selects_full_for_single_batch_loader() -> None:
    """incremental='auto' with a single-batch loader should choose 'full'."""
    disable_progress()
    torch.manual_seed(4)
    data = torch.randn(4, 4).float()  # 1 batch
    model = IdentityModel()

    s_full = SimpleScore()
    s_full.fit(
        model=model, loaders={"train": make_loader(data, 4)}, incremental="full"
    )

    s_auto = SimpleScore()
    s_auto.fit(
        model=model, loaders={"train": make_loader(data, 4)}, incremental="auto"
    )

    assert s_full.ref_embeddings is not None
    assert s_auto.ref_embeddings is not None
    approx(s_full.ref_embeddings, s_auto.ref_embeddings)


# ---------------------------------------------------------------------------
# PCA incremental fitting parity via batch mode
# ---------------------------------------------------------------------------


def test_pca_incremental_during_batch_extraction() -> None:
    """Batch mode with PCA must produce the same result as full mode."""
    disable_progress()
    torch.manual_seed(5)
    data = torch.randn(20, 8).float()
    model = IdentityModel()

    pca_full = TensorPCA(n_components=3)
    s_full = SimpleScore(pca=pca_full)
    s_full.fit(
        model=model, loaders={"train": make_loader(data, 4)}, incremental="full"
    )

    pca_batch = TensorPCA(n_components=3)
    s_batch = SimpleScore(pca=pca_batch)
    s_batch.fit(
        model=model,
        loaders={"train": make_loader(data, 4)},
        incremental="batch",
    )

    # Both should store the same raw embeddings
    assert s_full.ref_embeddings is not None
    assert s_batch.ref_embeddings is not None
    approx(s_full.ref_embeddings, s_batch.ref_embeddings)

    # Both PCA objects should have been fitted (u_q populated)
    assert pca_full.u_q.numel() > 0
    assert pca_batch.u_q.numel() > 0


# ---------------------------------------------------------------------------
# KNN parity: batch mode produces same scores as full mode
# ---------------------------------------------------------------------------


def test_knn_full_vs_batch_mode_parity() -> None:
    """EuclideanScore full and batch modes produce identical scores."""
    disable_progress()
    torch.manual_seed(6)
    ref = torch.randn(30, 8).float()
    query = torch.randn(5, 8).float()
    model = IdentityModel()

    score_full = EuclideanScore(k=2)
    score_full.fit(
        model=model, loaders={"train": make_loader(ref, 10)}, incremental="full"
    )
    scores_full = score_full.score(X=query)

    score_batch = EuclideanScore(k=2)
    score_batch.fit(
        model=model,
        loaders={"train": make_loader(ref, 10)},
        incremental="batch",
    )
    scores_batch = score_batch.score(X=query)

    approx(scores_full, scores_batch)


# ---------------------------------------------------------------------------
# Single-file disk persistence in batch mode
# ---------------------------------------------------------------------------


def test_batch_mode_writes_single_file(tmp_path: Path) -> None:
    """Batch mode with outdir+prefix writes a single embedding file."""
    disable_progress()
    torch.manual_seed(7)
    data = torch.randn(20, 4).float()
    model = IdentityModel()

    s = SimpleScore()
    s.fit(
        model=model,
        loaders={"train": make_loader(data, 4)},
        incremental="batch",
        outdir=tmp_path,
        prefix="run",
    )

    expected_file = tmp_path / "run-embeddings-train.pt"
    assert expected_file.exists(), f"Expected single file {expected_file}"

    # ref_embeddings should match the data
    assert s.ref_embeddings is not None
    approx(s.ref_embeddings, data)


def test_batch_mode_reuses_existing_file(tmp_path: Path) -> None:
    """Batch mode loads from existing file and skips re-embedding."""
    disable_progress()
    torch.manual_seed(8)
    data = torch.randn(20, 4).float()
    model = IdentityModel()

    # First fit: writes file
    s1 = SimpleScore()
    s1.fit(
        model=model,
        loaders={"train": make_loader(data, 4)},
        incremental="batch",
        outdir=tmp_path,
        prefix="cache",
    )
    assert s1.ref_embeddings is not None

    # Second fit: should load from file (emits UserWarning)
    s2 = SimpleScore()
    with pytest.warns(UserWarning, match="Loading pre-existing embeddings"):
        s2.fit(
            model=model,
            loaders={"train": make_loader(data, 4)},
            incremental="batch",
            outdir=tmp_path,
            prefix="cache",
        )
    assert s2.ref_embeddings is not None
    approx(s1.ref_embeddings, s2.ref_embeddings)


def test_full_mode_ref_equals_batch_mode_ref_from_file(tmp_path: Path) -> None:
    """ref_embeddings from batch mode (single file) equal those from full mode."""
    disable_progress()
    torch.manual_seed(9)
    data = torch.randn(15, 6).float()
    model = IdentityModel()

    s_full = SimpleScore()
    s_full.fit(
        model=model, loaders={"train": make_loader(data, 5)}, incremental="full"
    )

    s_batch = SimpleScore()
    s_batch.fit(
        model=model,
        loaders={"train": make_loader(data, 5)},
        incremental="batch",
        outdir=tmp_path,
        prefix="verify",
    )

    assert s_full.ref_embeddings is not None
    assert s_batch.ref_embeddings is not None
    approx(s_full.ref_embeddings, s_batch.ref_embeddings)
