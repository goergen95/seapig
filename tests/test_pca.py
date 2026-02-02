"""
Unit tests for seapig.scores.pca TensorPCA and PCAScore classes.

These tests verify PCA fitting, reconstruction error behaviour, RFF
transformation handling, and PCAScore integration for scoring and
calibration.
"""

import torch

from seapig.scores.pca import PCAScore
from seapig.scores.utils import TensorPCA


def approx(t1: torch.Tensor, t2: torch.Tensor, tol: float = 1e-6) -> None:
    assert torch.allclose(t1, t2, atol=tol, rtol=0)


def test_l2_normalize_preserves_shape_and_scales() -> None:
    """_l2_normalize should keep shape and normalize rows to unit norm."""
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    tpca = TensorPCA()
    out = tpca._l2_normalize(x)
    assert out.shape == x.shape
    # first row has norm 5 -> normalized to [0.6, 0.8]
    assert torch.allclose(out[0], torch.tensor([0.6, 0.8]), atol=1e-6)
    # second row zero vector should remain zero
    assert torch.allclose(out[1], torch.tensor([0.0, 0.0]), atol=1e-6)


def test_rff_returns_original_when_gamma_or_M_none() -> None:
    x = torch.randn(10, 4)
    out1 = TensorPCA._rff(x, gamma=None, M=None)
    out2 = TensorPCA._rff(x, gamma=3, M=None)
    out3 = TensorPCA._rff(x, gamma=None, M=16)
    # when either param is None, rff should return the original tensor
    assert torch.allclose(out1, x)
    assert torch.allclose(out2, x)
    assert torch.allclose(out3, x)


def test_fit_reconstruct_low_dim() -> None:
    """Fitting and reconstruction should yield low reconstruction error for low-rank data."""
    # create data of rank 1 in 3-d space
    n = 50
    v = torch.tensor([1.0, 2.0, 3.0])
    X = (torch.randn(n, 1) @ v.unsqueeze(0)).float()
    tpca = TensorPCA(exp_var=0.99)
    tpca.fit(X)
    X_rec, err = tpca.reconstruct(X)
    # reconstruction error should be near zero (numerical tolerance)
    assert err.mean() < 1e-5


def test_predict_matches_reconstruct_projection() -> None:
    """predict should return projection onto principal components consistent with reconstruct."""
    X = torch.randn(30, 5)
    tpca = TensorPCA(exp_var=0.90)
    tpca.fit(X)
    X_proj = tpca.predict(X)
    X_rec, _ = tpca.reconstruct(X)
    # projection multiplied back by u_q should approximate reconstruction
    approx((tpca.u_q @ X_proj.T).T, X_rec)


def test_pca_with_rff_changes_dimension() -> None:
    """When RFF is active the intermediate space should have dimension M > D."""
    X = torch.randn(20, 3)
    tpca = TensorPCA(exp_var=0.90, gamma=1.0, M=128)
    # ensure rff increases dimensionality
    X_rff = tpca._rff(X, gamma=tpca.gamma, M=tpca.M)
    assert X_rff.shape[1] == 128


def test_pca_score_fit_and_score_basic() -> None:
    """PCAScore.fit should train and score should produce errors with correct shape."""
    n_train = 100
    n_cal = 20
    dim = 8
    train = torch.randn(n_train, dim)
    cal = torch.randn(n_cal, dim)
    qs = torch.randn(5, dim)

    score = PCAScore(exp_var=0.90, M=16)
    # supply embeddings directly
    score.fit(train, cal)
    # check trained & calibrated flags
    assert score.is_trained()
    assert score.is_calibrated()
    out = score.score(qs)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == qs.shape[0]


def test_q_trimming_in_pca_score_reduces_references() -> None:
    n = 200
    refs = torch.randn(n, 6)
    score = PCAScore(exp_var=0.90, gamma=1.0, M=32)
    score.cal_required = False
    score.ref_embeddings = refs
    original = score.ref_embeddings.shape[0]
    score._fit_impl(q=0.5)
    assert score.ref_embeddings.shape[0] < original
