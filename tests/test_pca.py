"""
Unit tests for seapig.scores.pca TensorPCA and PCAScore classes.

These tests verify PCA fitting, reconstruction error behaviour, RFF
transformation handling, and PCAScore integration for scoring and
calibration.
"""

import pytest
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


def test_transform_matches_inverse_projection() -> None:
    """transform should return projection onto principal components consistent with inverse_transform."""
    X = torch.randn(30, 5)
    tpca = TensorPCA(exp_var=0.90)
    tpca.fit(X)
    X_proj = tpca.transform(X)
    X_rec = tpca.inverse_transform(X_proj)
    # projection multiplied back by u_q should approximate reconstruction
    approx((tpca.u_q @ X_proj.T).T, X_rec)


def test_pca_with_rff_changes_dimension() -> None:
    """When RFF is active the intermediate space should have dimension M > D."""
    X = torch.randn(20, 3)
    tpca = TensorPCA(exp_var=0.90, gamma=1.0, M=128)
    # ensure rff increases dimensionality
    X_rff = tpca._rff(X)
    assert X_rff.shape[1] == 128


def test_pca_score_fit_and_score_basic() -> None:
    """PCAScore.fit should train and score should produce errors with correct shape."""
    n_train = 100
    n_cal = 20
    dim = 8
    train = torch.randn(n_train, dim)
    cal = torch.randn(n_cal, dim)
    qs = torch.randn(5, dim)

    score = PCAScore(pca=TensorPCA(exp_var=0.90, M=16))
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
    score = PCAScore(pca=TensorPCA(exp_var=0.90, gamma=1.0, M=32))
    score.cal_required = False
    score.ref_embeddings = refs
    original = score.ref_embeddings.shape[0]
    score._fit_impl(q=0.5)
    assert score.ref_embeddings.shape[0] < original


def test_n_comp_argument_limits_components() -> None:
    """When n_comp is provided it should limit the number of components used."""
    X = torch.randn(50, 10)
    # request exactly 3 components
    tpca = TensorPCA(n_comp=3)
    tpca.fit(X)
    assert tpca.q == 3
    # reconstruction should use u_q with 3 columns
    assert tpca.u_q.shape[1] == 3


def test_tensorpca_agrees_with_sklearn_mid_size() -> None:
    """TensorPCA should produce reconstructions similar to scikit-learn PCA.

    We create a mid-size dataset, L2-normalise rows (matching TensorPCA
    preprocessing), fit both implementations with the same number of
    components and compare the reconstruction in the centred, preprocessed
    space. This comparison avoids component sign ambiguities.
    """
    pytest.importorskip("sklearn")
    from sklearn.decomposition import PCA as SKPCA

    torch.manual_seed(0)
    n, D = 500, 50
    X = torch.randn(n, D)

    # match TensorPCA preprocessing: row-wise L2 normalisation
    X_norm = TensorPCA._l2_normalize(X)
    X_norm_np = X_norm.cpu().numpy()

    n_comp = 10

    # scikit-learn PCA on the normalised data
    skp = SKPCA(n_components=n_comp, svd_solver="full")
    skp.fit(X_norm_np)
    X_proj_sk = skp.transform(X_norm_np)
    X_rec_sk = skp.inverse_transform(
        X_proj_sk
    )  # reconstructed in original space
    mu_np = X_norm_np.mean(axis=0)
    X_rec_sk_centered = X_rec_sk - mu_np

    # TensorPCA (no RFF) - fit on torch tensors
    tpca = TensorPCA(n_comp=n_comp, gamma=None, M=None)
    tpca.fit(X)
    X_proj_tp = tpca.transform(X)
    X_rec_tp = tpca.inverse_transform(X_proj_tp)

    # compare reconstructions in centred preprocessed space
    rec_sk_t = torch.from_numpy(X_rec_sk_centered).to(X_rec_tp.dtype)
    # ensure same device and dtype
    rec_sk_t = rec_sk_t.to(device=X_rec_tp.device)

    # compute mean squared error between reconstructions
    mse = torch.mean((rec_sk_t - X_rec_tp) ** 2).item()

    # expect close agreement (numerical tolerance due to SVD differences)
    assert mse < 1e-6, (
        f"MSE between scikit-learn and TensorPCA too large: {mse}"
    )

    # compare projections
    proj_sk_t = torch.from_numpy(X_proj_sk).to(
        device=X_proj_tp.device, dtype=X_proj_tp.dtype
    )
    # compute mean squared error between absolute projections (signs can differ due to SVD ambiguities)
    mse = torch.mean((proj_sk_t.abs() - X_proj_tp.abs()) ** 2).item()
    # we expect close agreement in projection magnitudes
    assert mse < 1e-6, "Projections differ between scikit-learn and TensorPCA"


def test_partial_fit_matches_incremental_pca() -> None:
    """Compare TensorPCA.partial_fit/finalize with sklearn IncrementalPCA for linear mode."""
    pytest.importorskip("sklearn")
    from sklearn.decomposition import IncrementalPCA

    torch.manual_seed(1)
    n, D = 400, 30
    X = torch.randn(n, D)

    n_comp = 8
    batch_size = 50

    # preprocess for sklearn: row-wise L2 normalisation
    X_norm = TensorPCA._l2_normalize(X)
    X_norm_np = X_norm.cpu().numpy()

    # sklearn incremental PCA
    ipca = IncrementalPCA(n_components=n_comp)
    for i in range(0, n, batch_size):
        ipca.partial_fit(X_norm_np[i : i + batch_size])
    X_proj_sk = ipca.transform(X_norm_np)
    X_rec_sk = ipca.inverse_transform(X_proj_sk)
    # use the mean estimated by IncrementalPCA (may differ slightly from
    # direct numpy mean due to online updates)
    mu_ipca = ipca.mean_
    X_rec_sk_centered = X_rec_sk - mu_ipca

    # TensorPCA partial fit path (linear)
    tpca = TensorPCA(n_comp=n_comp, gamma=None, M=None)
    for i in range(0, n, batch_size):
        batch = X[i : i + batch_size]
        tpca.partial_fit(batch)
    tpca.finalize()

    X_proj_tp = tpca.transform(X)
    X_rec_tp = tpca.inverse_transform(X_proj_tp)

    rec_sk_t = torch.from_numpy(X_rec_sk_centered).to(X_rec_tp.dtype)
    rec_sk_t = rec_sk_t.to(device=X_rec_tp.device)

    mse = torch.mean((rec_sk_t - X_rec_tp) ** 2).item()
    assert mse < 1e-2, (
        f"MSE between IncrementalPCA and TensorPCA partial path too large: {mse}"
    )

    proj_sk_t = torch.from_numpy(X_proj_sk).to(
        device=X_proj_tp.device, dtype=X_proj_tp.dtype
    )
    mse = torch.mean((proj_sk_t.abs() - X_proj_tp.abs()) ** 2).item()
    assert mse < 1e-1, (
        "Projections differ between IncrementalPCA and TensorPCA partial path"
    )
