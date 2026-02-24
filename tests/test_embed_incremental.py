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
# TensorPCA incremental parity
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
# EmbeddingScore parity: precomputed X
# ---------------------------------------------------------------------------


def test_embedding_score_full_vs_batch_precomputed() -> None:
    """fit(X, incremental='full') and fit(X, incremental='batch', chunk_size=...)
    must produce identical ref_embeddings."""
    disable_progress()
    torch.manual_seed(0)
    X = torch.randn(50, 8).float()
    Y = torch.randn(10, 8).float()

    s_full = SimpleScore()
    s_full.fit(X=X, Y=Y, incremental="full")

    s_batch = SimpleScore()
    s_batch.fit(X=X, Y=Y, incremental="batch", chunk_size=10)

    assert s_full.ref_embeddings is not None
    assert s_batch.ref_embeddings is not None
    approx(s_full.ref_embeddings, s_batch.ref_embeddings)
    assert s_full.cal_embeddings is not None
    assert s_batch.cal_embeddings is not None
    approx(s_full.cal_embeddings, s_batch.cal_embeddings)


def test_embedding_score_auto_uses_batch_with_chunk_size() -> None:
    """incremental='auto' with chunk_size set should behave like 'batch'."""
    disable_progress()
    torch.manual_seed(1)
    X = torch.randn(30, 4).float()

    s_full = SimpleScore()
    s_full.fit(X=X, incremental="full")

    s_auto = SimpleScore()
    s_auto.fit(X=X, incremental="auto", chunk_size=10)

    assert s_full.ref_embeddings is not None
    assert s_auto.ref_embeddings is not None
    approx(s_full.ref_embeddings, s_auto.ref_embeddings)


# ---------------------------------------------------------------------------
# EmbeddingScore parity: model + loaders
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
    # auto should choose batch (5 batches)
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
# KNN parity via partial_fit / finalize + _fit_impl
# ---------------------------------------------------------------------------


def test_knn_parity_via_partial_fit() -> None:
    """EuclideanScore partial_fit/finalize + _fit_impl must give same scores as fit."""
    disable_progress()
    torch.manual_seed(5)
    ref = torch.randn(30, 8).float()
    query = torch.randn(5, 8).float()

    # full mode
    score_full = EuclideanScore(k=2)
    score_full.fit(X=ref)
    scores_full = score_full.score(X=query)

    # manual incremental via partial_fit / finalize + _fit_impl
    score_inc = EuclideanScore(k=2)
    chunk = 10
    for start in range(0, ref.shape[0], chunk):
        score_inc.partial_fit(X=ref[start : start + chunk])
    score_inc.finalize()
    # finalize sets ref_embeddings; now build the KNN index
    score_inc._fit_impl()

    scores_inc = score_inc.score(X=query)
    approx(scores_full, scores_inc)


# ---------------------------------------------------------------------------
# Batch-write behavior (disk files)
# ---------------------------------------------------------------------------


def test_batch_write_creates_and_removes_files(tmp_path: Path) -> None:
    """With batch_write=True, per-batch files are created and removed after finalize."""
    disable_progress()
    torch.manual_seed(6)
    X = torch.randn(20, 4).float()

    s = SimpleScore()
    s.fit(
        X=X,
        incremental="batch",
        chunk_size=5,
        batch_write=True,
        outdir=tmp_path,
        prefix="test",
        keep_batch_files=False,
    )

    # After finalize, batch files should be gone
    batch_files = list(tmp_path.glob("test-embeddings-train-batch-*.pt"))
    assert len(batch_files) == 0, (
        f"Expected no batch files, found {batch_files}"
    )

    # ref_embeddings should be set and equal to X
    assert s.ref_embeddings is not None
    approx(s.ref_embeddings, X)


def test_batch_write_keeps_files_when_requested(tmp_path: Path) -> None:
    """With keep_batch_files=True, per-batch files remain after finalize."""
    disable_progress()
    torch.manual_seed(7)
    X = torch.randn(20, 4).float()

    s = SimpleScore()
    s.fit(
        X=X,
        incremental="batch",
        chunk_size=5,
        batch_write=True,
        outdir=tmp_path,
        prefix="keep",
        keep_batch_files=True,
    )

    batch_files = sorted(tmp_path.glob("keep-embeddings-train-batch-*.pt"))
    assert len(batch_files) == 4  # 20 samples / 5 chunk_size = 4 batches

    assert s.ref_embeddings is not None
    approx(s.ref_embeddings, X)


def test_batch_write_ref_embeddings_match_full(tmp_path: Path) -> None:
    """Batch-write mode ref_embeddings must equal full-mode ref_embeddings."""
    disable_progress()
    torch.manual_seed(8)
    X = torch.randn(15, 6).float()

    s_full = SimpleScore()
    s_full.fit(X=X, incremental="full")

    s_disk = SimpleScore()
    s_disk.fit(
        X=X,
        incremental="batch",
        chunk_size=5,
        batch_write=True,
        outdir=tmp_path,
        prefix="cmp",
        keep_batch_files=False,
    )

    assert s_full.ref_embeddings is not None
    assert s_disk.ref_embeddings is not None
    approx(s_full.ref_embeddings, s_disk.ref_embeddings)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_finalize_without_partial_fit_raises() -> None:
    """finalize() without prior partial_fit() must raise RuntimeError."""
    s = SimpleScore()
    with pytest.raises(RuntimeError, match="No data provided"):
        s.finalize()


def test_partial_fit_implicit_reset() -> None:
    """partial_fit() called without reset_partial() should start session automatically."""
    disable_progress()
    torch.manual_seed(9)
    X = torch.randn(10, 4).float()

    s = SimpleScore()
    # Call partial_fit without reset_partial first
    s.partial_fit(X=X[:5])
    s.partial_fit(X=X[5:])
    s.finalize()

    assert s.ref_embeddings is not None
    approx(s.ref_embeddings, X)


def test_partial_fit_with_model_and_batch() -> None:
    """partial_fit() with model+batch should embed and accumulate."""
    disable_progress()
    torch.manual_seed(10)
    data = torch.randn(8, 4).float()
    model = IdentityModel()

    s = SimpleScore()
    s.partial_fit(model=model, batch=data[:4])
    s.partial_fit(model=model, batch=data[4:])
    s.finalize()

    assert s.ref_embeddings is not None
    assert s.ref_embeddings.shape == (8, 4)


def test_partial_fit_raises_without_model_or_x() -> None:
    """partial_fit() without X and without model+batch raises ValueError."""
    s = SimpleScore()
    with pytest.raises(ValueError, match="Provide X"):
        s.partial_fit(X=None)


def test_partial_fit_raises_without_batch() -> None:
    """partial_fit() with model but without batch raises ValueError."""
    s = SimpleScore()
    model = IdentityModel()
    with pytest.raises(ValueError, match="Provide X"):
        s.partial_fit(model=model, batch=None)


def test_reset_partial_clears_state() -> None:
    """reset_partial() must clear all accumulators."""
    disable_progress()
    torch.manual_seed(11)
    X = torch.randn(10, 4).float()

    s = SimpleScore()
    s.partial_fit(X=X)
    assert s._n_ref_samples == 10
    assert s._partial_active

    s.reset_partial()
    assert s._n_ref_samples == 0
    assert s._batch_count == 0
    assert not s._partial_active
    assert s._ref_embs_batches == []
    assert s._batch_paths == []
