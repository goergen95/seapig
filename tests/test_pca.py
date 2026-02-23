"""
Unit tests for seapig.scores.utils.TensorPCA and seapig.scores.pca.PCAScore.

Tests cover:
- Basic fit → transform → inverse_transform round-trip
- Whitening (unit-variance outputs on training set)
- n_components and exp_var selection, mutual exclusivity
- Reproducibility via random_state
- partial_fit + finalize equivalence
- state_dict round-trip
- predict / reconstruct compatibility API
- PCAScore integration
"""

import pytest
import torch

from seapig.scores.pca import PCAScore
from seapig.scores.utils import TensorPCA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def approx(t1: torch.Tensor, t2: torch.Tensor, tol: float = 1e-5) -> None:
    """Assert two tensors are element-wise close."""
    assert torch.allclose(t1, t2, atol=tol, rtol=0), (
        f"max abs diff: {(t1 - t2).abs().max().item():.3e}"
    )


def make_low_rank(n: int = 200, rank: int = 3, d: int = 16) -> torch.Tensor:
    """Return a rank-`rank` matrix of shape (n, d)."""
    torch.manual_seed(0)
    U = torch.randn(n, rank)
    V = torch.randn(rank, d)
    return (U @ V).float()


# ---------------------------------------------------------------------------
# Static utility methods (backward compat)
# ---------------------------------------------------------------------------


def test_l2_normalize_preserves_shape_and_scales() -> None:
    """_l2_normalize keeps shape and normalises rows to unit norm."""
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    out = TensorPCA._l2_normalize(x)
    assert out.shape == x.shape
    approx(out[0], torch.tensor([0.6, 0.8]))
    approx(out[1], torch.tensor([0.0, 0.0]))


def test_rff_returns_original_when_gamma_or_M_none() -> None:
    """_rff passes data through unchanged when either param is None."""
    x = torch.randn(10, 4)
    for kwargs in [
        {"gamma": None, "M": None},
        {"gamma": 3, "M": None},
        {"gamma": None, "M": 16},
    ]:
        assert torch.allclose(TensorPCA._rff(x, **kwargs), x)  # type: ignore[arg-type]


def test_pca_with_rff_changes_dimension() -> None:
    """RFF increases feature dimensionality to M."""
    X = torch.randn(20, 3)
    X_rff = TensorPCA._rff(X, gamma=1.0, M=128)
    assert X_rff.shape[1] == 128


# ---------------------------------------------------------------------------
# Mutual exclusivity validation
# ---------------------------------------------------------------------------


def test_mutual_exclusivity_both_raises() -> None:
    """Providing both n_components and exp_var raises ValueError."""
    pca = TensorPCA(n_components=4, exp_var=0.9)
    X = torch.randn(50, 8)
    with pytest.raises(ValueError, match="Only one of"):
        pca.fit(X)


def test_mutual_exclusivity_neither_raises() -> None:
    """Providing neither n_components nor exp_var raises ValueError."""
    pca = TensorPCA()
    X = torch.randn(50, 8)
    with pytest.raises(ValueError, match="Exactly one of"):
        pca.fit(X)


def test_mutual_exclusivity_partial_fit_finalize_raises() -> None:
    """finalize() also validates mutual exclusivity."""
    pca = TensorPCA()
    pca.partial_fit(torch.randn(50, 8))
    with pytest.raises(ValueError, match="Exactly one of"):
        pca.finalize()


# ---------------------------------------------------------------------------
# Basic fit / transform / inverse_transform
# ---------------------------------------------------------------------------


def test_pca_basic_fit_transform_torch() -> None:
    """fit → transform → inverse_transform round-trip is accurate for low-rank data."""
    X = make_low_rank(n=200, rank=3, d=16)
    pca = TensorPCA(n_components=3)
    pca.fit(X)

    proj = pca.transform(X)
    recon = pca.inverse_transform(proj)

    assert proj.shape == (200, 3)
    assert recon.shape == X.shape
    # Low-rank data should be reconstructed nearly exactly
    assert (X - recon).norm(dim=1).mean().item() < 1e-4


def test_fit_reconstruct_low_dim() -> None:
    """Fitting and reconstruction yields near-zero error for rank-1 data."""
    n = 50
    v = torch.tensor([1.0, 2.0, 3.0])
    X = (torch.randn(n, 1) @ v.unsqueeze(0)).float()
    pca = TensorPCA(exp_var=0.99)
    pca.fit(X)
    X_rec, err = pca.reconstruct(X)
    assert err.mean().item() < 1e-4


def test_fit_transform_equivalence() -> None:
    """fit_transform(X) equals fit(X); transform(X)."""
    torch.manual_seed(1)
    X = torch.randn(80, 10)
    pca1 = TensorPCA(n_components=4, random_state=1)
    pca2 = TensorPCA(n_components=4, random_state=1)

    p1 = pca1.fit_transform(X)
    pca2.fit(X)
    p2 = pca2.transform(X)

    # Results may differ in sign; compare absolute values
    approx(p1.abs(), p2.abs(), tol=1e-4)


# ---------------------------------------------------------------------------
# n_components selection
# ---------------------------------------------------------------------------


def test_n_components_choice_torch() -> None:
    """components_ shape matches n_components."""
    X = torch.randn(100, 20)
    pca = TensorPCA(n_components=5)
    pca.fit(X)
    assert pca.components_ is not None
    assert pca.components_.shape == (5, 20)
    assert pca.singular_values_ is not None
    assert pca.singular_values_.shape == (5,)


def test_n_components_cap_warns() -> None:
    """n_components > rank is capped with a warning."""
    X = make_low_rank(n=50, rank=3, d=8)
    pca = TensorPCA(n_components=100)  # impossible rank
    with pytest.warns(UserWarning, match="capping"):
        pca.fit(X)
    assert pca.components_ is not None
    # should be capped at min(N-1, D) = min(49, 8) = 8
    assert pca.components_.shape[0] <= 8


# ---------------------------------------------------------------------------
# exp_var selection
# ---------------------------------------------------------------------------


def test_exp_var_choice_torch() -> None:
    """exp_var selects minimal k with cumulative ratio >= threshold."""
    X = make_low_rank(n=200, rank=5, d=20)
    threshold = 0.90
    pca = TensorPCA(exp_var=threshold)
    pca.fit(X)

    assert pca.explained_variance_ratio_ is not None
    cum = pca.explained_variance_ratio_.cumsum(0)
    k = pca.components_.shape[0]  # type: ignore[union-attr]
    # All but the last selected component should be below the threshold
    if k > 1:
        assert cum[k - 2].item() < threshold
    assert cum[k - 1].item() >= threshold


# ---------------------------------------------------------------------------
# Whitening
# ---------------------------------------------------------------------------


def test_whiten_variance_torch() -> None:
    """Whitened training projections have variance ≈ 1 along each component."""
    torch.manual_seed(42)
    X = torch.randn(500, 20)
    pca = TensorPCA(n_components=8, whiten=True, svd_solver="full")
    pca.fit(X)
    proj = pca.transform(X)
    var = proj.var(dim=0, unbiased=True)
    # Each whitened dimension should have unit variance on the training set
    assert (var - 1.0).abs().max().item() < 0.1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_random_state_reproducible_torch() -> None:
    """Two fits with the same random_state produce identical components_."""
    torch.manual_seed(0)
    X = torch.randn(200, 50)

    pca_a = TensorPCA(n_components=10, svd_solver="randomized", random_state=7)
    pca_b = TensorPCA(n_components=10, svd_solver="randomized", random_state=7)
    pca_a.fit(X)
    pca_b.fit(X)

    assert pca_a.components_ is not None
    assert pca_b.components_ is not None
    # Signs may differ; compare squared components
    approx(pca_a.components_**2, pca_b.components_**2, tol=1e-5)


# ---------------------------------------------------------------------------
# partial_fit + finalize
# ---------------------------------------------------------------------------


def test_partial_fit_equivalence_torch() -> None:
    """partial_fit + finalize approximates full fit on the same dataset."""
    torch.manual_seed(3)
    X = torch.randn(300, 16)

    pca_full = TensorPCA(n_components=4, svd_solver="full")
    pca_full.fit(X)

    pca_incr = TensorPCA(n_components=4, svd_solver="full")
    for chunk in X.split(100):
        pca_incr.partial_fit(chunk)
    pca_incr.finalize()

    assert pca_full.mean_ is not None
    assert pca_incr.mean_ is not None
    # Means should match
    approx(pca_full.mean_, pca_incr.mean_, tol=1e-5)
    # Reconstruction errors should be similar
    _, err_full = pca_full.reconstruct(X)
    _, err_incr = pca_incr.reconstruct(X)
    assert (err_full - err_incr).abs().mean().item() < 1e-3


def test_finalize_raises_if_no_data() -> None:
    """finalize() raises RuntimeError when no partial_fit has been called."""
    pca = TensorPCA(n_components=4)
    with pytest.raises(RuntimeError, match="No data"):
        pca.finalize()


def test_transform_raises_before_finalize() -> None:
    """transform() raises before finalize() is called."""
    pca = TensorPCA(n_components=4)
    pca.partial_fit(torch.randn(50, 8))
    with pytest.raises(RuntimeError, match="not been fitted"):
        pca.transform(torch.randn(5, 8))


# ---------------------------------------------------------------------------
# state_dict round-trip
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip_torch() -> None:
    """save/load state_dict reproduces identical transform outputs."""
    torch.manual_seed(5)
    X = torch.randn(100, 12)
    pca = TensorPCA(n_components=4)
    pca.fit(X)

    state = pca.state_dict()

    pca2 = TensorPCA(n_components=4)
    pca2.load_state_dict(state)

    q1 = pca.transform(X)
    q2 = pca2.transform(X)
    approx(q1, q2)


# ---------------------------------------------------------------------------
# predict / reconstruct compatibility
# ---------------------------------------------------------------------------


def test_predict_returns_error_by_default() -> None:
    """predict(X) returns a 1-D error tensor by default."""
    torch.manual_seed(0)
    X = torch.randn(50, 8)
    pca = TensorPCA(n_components=3)
    pca.fit(X)

    err = pca.predict(X)
    assert isinstance(err, torch.Tensor)
    assert err.shape == (50,)
    assert (err >= 0).all()


def test_predict_returns_reconstruction_when_requested() -> None:
    """predict(X, return_reconstruction=True, return_error=False) returns X_rec."""
    torch.manual_seed(0)
    X = torch.randn(50, 8)
    pca = TensorPCA(n_components=3)
    pca.fit(X)

    X_rec = pca.predict(X, return_reconstruction=True, return_error=False)
    assert isinstance(X_rec, torch.Tensor)
    assert X_rec.shape == X.shape


def test_predict_returns_both_when_requested() -> None:
    """predict(X, return_reconstruction=True) returns (error, X_rec)."""
    torch.manual_seed(0)
    X = torch.randn(50, 8)
    pca = TensorPCA(n_components=3)
    pca.fit(X)

    result = pca.predict(X, return_reconstruction=True, return_error=True)
    assert isinstance(result, tuple)
    err, X_rec = result
    assert err.shape == (50,)
    assert X_rec.shape == X.shape


def test_reconstruct_returns_tuple() -> None:
    """reconstruct(X) returns (X_rec, error) with correct shapes."""
    torch.manual_seed(0)
    X = torch.randn(30, 6)
    pca = TensorPCA(exp_var=0.90)
    pca.fit(X)

    X_rec, err = pca.reconstruct(X)
    assert X_rec.shape == X.shape
    assert err.shape == (30,)
    assert (err >= 0).all()


def test_predict_partial_fit_raises_before_finalize() -> None:
    """predict raises an informative error when finalize has not been called."""
    pca = TensorPCA(n_components=3)
    pca.partial_fit(torch.randn(50, 8))
    with pytest.raises(RuntimeError, match="not been fitted"):
        pca.predict(torch.randn(5, 8))


def test_predict_reconstruct_parity() -> None:
    """predict error matches the norm of (X - reconstruct(X)[0])."""
    torch.manual_seed(0)
    X = torch.randn(40, 10)
    pca = TensorPCA(n_components=5)
    pca.fit(X)

    err_predict = pca.predict(X)
    X_rec, err_reconstruct = pca.reconstruct(X)
    approx(err_predict, err_reconstruct)


# ---------------------------------------------------------------------------
# PCAScore integration
# ---------------------------------------------------------------------------


def test_pca_score_fit_and_score_basic() -> None:
    """PCAScore.fit trains correctly; score returns correct-shaped errors."""
    n_train, n_cal, dim = 100, 20, 8
    train = torch.randn(n_train, dim)
    cal = torch.randn(n_cal, dim)
    qs = torch.randn(5, dim)

    score = PCAScore(pca=TensorPCA(exp_var=0.90))
    score.fit(train, cal)

    assert score.is_trained()
    assert score.is_calibrated()
    out = score.score(qs)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == qs.shape[0]


def test_q_trimming_in_pca_score_reduces_references() -> None:
    """q-trimming in _fit_impl removes high-error outliers from ref_embeddings."""
    n = 200
    refs = torch.randn(n, 6)
    score = PCAScore(pca=TensorPCA(exp_var=0.90))
    score.cal_required = False
    score.ref_embeddings = refs
    original = score.ref_embeddings.shape[0]
    score._fit_impl(q=0.5)
    assert score.ref_embeddings.shape[0] < original
