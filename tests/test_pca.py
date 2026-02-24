"""
Unit tests for seapig.scores.pca TensorPCA and PCAScore classes.

These tests verify PCA fitting, reconstruction error behaviour, RFF
transformation handling, and PCAScore integration for scoring and
calibration.
"""

import math

import pytest
import torch

from seapig.scores.pca import PCAScore
from seapig.scores.utils import TensorPCA


def approx(t1: torch.Tensor, t2: torch.Tensor, tol: float = 1e-6) -> None:
    assert torch.allclose(t1, t2, atol=tol, rtol=0)


def test_l2_normalize_preserves_shape_and_scales() -> None:
    """_l2_normalize should keep shape and normalize rows to unit norm."""
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    tpca = TensorPCA(exp_var=0.9)
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


def test_fit_transform_returns_lower_dim() -> None:
    """transform should return data with q components, where q < D."""
    X = torch.randn(40, 10)
    tpca = TensorPCA(exp_var=0.90)
    X_proj = tpca.fit_transform(X)
    assert X_proj.shape[1] == tpca.q
    assert tpca.q < X.shape[1]


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


def _assert_state_dicts_equal(sd1: dict, sd2: dict) -> None:
    """Helper to compare two state_dict-like mappings of tensors.

    Compares tensor contents on CPU. Accepts None entries.
    """
    assert set(sd1.keys()) == set(sd2.keys())
    for k in sd1.keys():
        v1 = sd1[k]
        v2 = sd2[k]
        if v1 is None and v2 is None:
            continue
        assert v1 is not None and v2 is not None, (
            f"Mismatch at {k}: one is None"
        )
        t1 = v1.detach().cpu()
        t2 = v2.detach().cpu()
        if t1.dtype == torch.bool:
            assert torch.equal(t1, t2), f"Bool tensor mismatch at {k}"
        else:
            assert torch.allclose(t1, t2, atol=1e-6, rtol=0), (
                f"Tensor mismatch at {k}"
            )


def test_save_load_tensorpca_linear_and_rff(tmp_path) -> None:
    """Saving and loading a fitted TensorPCA should preserve all buffers.

    This test covers both linear and RFF modes. It saves state_dict to disk
    and loads it into a new instance constructed with the same config,
    then compares the resulting state dictionaries.
    """
    torch.manual_seed(42)
    X = torch.randn(120, 10)

    # Linear mode
    tpca_lin = TensorPCA(exp_var=0.95)
    tpca_lin.fit(X)
    path_lin = tmp_path / "tpca_linear.pt"
    torch.save(tpca_lin.state_dict(), path_lin)

    tpca_lin_loaded = TensorPCA(exp_var=0.95)
    sd = torch.load(path_lin)
    tpca_lin_loaded.load_state_dict(sd)

    _assert_state_dicts_equal(
        tpca_lin.state_dict(), tpca_lin_loaded.state_dict()
    )

    # RFF mode (ensure RFF params are created during fit)
    tpca_rff = TensorPCA(exp_var=0.95, gamma=1.0, M=32)
    tpca_rff.fit(X)
    path_rff = tmp_path / "tpca_rff.pt"
    torch.save(tpca_rff.state_dict(), path_rff)

    tpca_rff_loaded = TensorPCA(exp_var=0.95, gamma=1.0, M=32)
    sd2 = torch.load(path_rff)
    tpca_rff_loaded.load_state_dict(sd2)

    _assert_state_dicts_equal(
        tpca_rff.state_dict(), tpca_rff_loaded.state_dict()
    )


def test_constructor_warnings_and_validation() -> None:
    # both exp_var and n_comp provided should warn but not raise
    with pytest.warns(
        UserWarning, match="Both exp_var and n_comp are provided"
    ):
        TensorPCA(exp_var=0.5, n_comp=3)

    # invalid exp_var values raise
    with pytest.raises(ValueError, match="exp_var must be in the interval"):
        TensorPCA(exp_var=0.0)

    with pytest.warns(UserWarning, match="Defaulting to exp_var=0.90"):
        TensorPCA()

    # invalid n_comp values raise
    with pytest.raises(ValueError, match="n_comp must be a positive integer"):
        TensorPCA(n_comp=0)

    # invalid mode string
    with pytest.raises(
        ValueError, match="mode must be either 'linear' or 'rff'"
    ):
        TensorPCA(exp_var=0.5, mode="not-a-mode")


def test_rff_partial_fit_dimension_check() -> None:
    # mode 'rff' with M <= D should raise during partial_fit initialization
    tpca = TensorPCA(mode="rff", gamma=1.0, M=2)
    X = torch.randn(4, 3)
    with pytest.raises(
        ValueError, match="RFF dimension M must be greater than input dim D"
    ):
        tpca.partial_fit(X)


def test__rff_uses_registered_buffers_when_initialized() -> None:
    # construct deterministic buffers and ensure _rff uses them
    D = 3
    M = 8
    gamma = 1.0
    tpca = TensorPCA(mode="rff", gamma=gamma, M=M)

    dtype = torch.float32

    # create explicit buffers (match expected shapes)
    w = math.sqrt(2.0 * float(gamma)) * torch.randn(M, D, dtype=dtype)
    u = 2.0 * math.pi * torch.rand(M, dtype=dtype)

    tpca.register_buffer("_rff_w", w)
    tpca.register_buffer("_rff_u", u)
    # mark initialized
    getattr(tpca, "_rff_initialized").fill_(True)

    X = torch.randn(5, D, dtype=dtype)
    out = tpca._rff(X)

    expected = math.sqrt(2.0 / float(M)) * torch.cos(X @ w.T + u.unsqueeze(0))
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-6)


def test_reset_partial_preserves_rff_and_resets_accumulators() -> None:
    tpca = TensorPCA(gamma=1.0, M=16)
    X = torch.randn(6, 3)
    tpca.partial_fit(X)  # initializes RFF parameters and accumulators

    # rff parameters should be present
    assert getattr(tpca, "_rff_initialized").item() is True
    assert getattr(tpca, "_rff_w").numel() > 0

    # reset partial should clear accumulators but keep RFF params
    tpca.reset_partial()
    assert tpca._n_samples == 0
    assert tpca._sum_X is None and tpca._sum_outer is None
    assert getattr(tpca, "_rff_initialized").item() is True
    assert getattr(tpca, "_rff_w").numel() > 0


def test__rff_returns_input_when_not_in_rff_mode() -> None:
    tpca = TensorPCA()  # default linear mode
    X = torch.randn(3, 4)
    out = tpca._rff(X)
    # should return (possibly) contiguous copy equal to input
    assert out.shape == X.shape
    assert torch.allclose(out, X, atol=0)


def test_finalize_raises_if_no_data() -> None:
    tpca = TensorPCA(exp_var=0.9)
    with pytest.raises(RuntimeError, match="No data provided"):
        tpca.finalize()


def test__load_from_state_dict_registers_missing_buffers() -> None:
    tpca = TensorPCA()
    # remove one of the attributes to trigger the register_buffer path
    if hasattr(tpca, "mu"):
        delattr(tpca, "mu")

    assert not hasattr(tpca, "mu")

    sd = {"mu": torch.tensor([1.0, 2.0], dtype=torch.float64)}

    # call the custom loader — it should register the missing buffer
    tpca._load_from_state_dict(
        sd,
        prefix="",
        local_metadata={},
        strict=True,
        missing_keys=[],
        unexpected_keys=[],
        error_msgs=[],
    )

    # mu should now be registered as a buffer and match the value
    assert hasattr(tpca, "mu")
    assert torch.allclose(
        getattr(tpca, "mu"), torch.tensor([1.0, 2.0], dtype=torch.float64)
    )
