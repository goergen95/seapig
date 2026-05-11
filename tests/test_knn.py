"""
Unit tests for seapig.scores.knn distance and similarity score classes.

These tests exercise Euclidean, Cosine and Mahalanobis scoring implementations
for correct distance/similarity computation, k-nearest handling, statistical
aggregation (min/max/mean/median), index creation, and behavior on singular
covariance matrices.
"""

import pathlib
from collections.abc import Callable

import pytest
import torch

from seapig.scores.knn import CosineScore, EuclideanScore, MahalanobisScore
from seapig.scores.utils import TensorPCA


def approx(t1: torch.Tensor, t2: torch.Tensor, tol: float = 1e-6) -> None:
    assert torch.allclose(t1, t2, atol=tol, rtol=0)


def test_euclidean_distance_simple_nearest() -> None:
    """Verify EuclideanScore returns the correct nearest distance."""
    # Two reference points: (0,0) and (3,4) -> distances to (6,8) -> nearest = (3,4) dist = 5
    ref = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    q = torch.tensor([[6.0, 8.0]])
    score = EuclideanScore(k=1, stat="min")
    score.ref_embeddings = ref
    score._setup_index()
    # offset default 0 => returns distance to k nearest (here k=1)
    out, _ = score._distance(q, offset=0)
    expected = torch.tensor([5.0])
    approx(out, expected)


@pytest.mark.parametrize(
    "stat, expected_fn",
    [
        ("max", lambda ds: ds.max()),
        ("min", lambda ds: ds.min()),
        ("mean", lambda ds: ds.mean()),
        ("median", lambda ds: ds.median()),
    ],
)
def test_euclidean_k_and_stats(
    stat: str, expected_fn: Callable[[torch.Tensor], torch.Tensor]
) -> None:
    """Test EuclideanScore k-nearest selection and aggregation statistic."""
    # create 3 refs, query at origin: distances are simple
    refs = torch.tensor(
        [[3.0, 4.0], [6.0, 8.0], [0.0, 5.0]]
    )  # distances: 5,10,5
    q = torch.tensor([[0.0, 0.0]])
    # k=2 -> pick two nearest squared distances [25,25] -> stat on squared then sqrt
    score = EuclideanScore(k=2, stat=stat)
    score.ref_embeddings = refs
    score._setup_index()
    out, _ = score._distance(q, offset=0)
    # pick two smallest distances: 5 and 5 -> squared are 25 and 25
    two = torch.tensor([5.0, 5.0])
    expected = expected_fn(two)
    approx(out, expected.unsqueeze(0) if expected.dim() == 0 else expected)


def test_cosine_similarity_identical_vector() -> None:
    """Verify CosineScore returns distance ~0.0 for identical vectors."""
    refs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    q = torch.tensor([[1.0, 0.0]])
    score = CosineScore(k=1, stat="max")
    score.ref_embeddings = refs
    score._setup_index()
    out, _ = score._distance(q, offset=0)
    # identical vector should yield cosine distance ~0.0 (1 - similarity of 1.0)
    assert out.shape == (1, 1)
    assert torch.isclose(out[0], torch.tensor(0.0), atol=1e-6)


def test_cosine_k_mean() -> None:
    """Verify CosineScore with k>1 and mean statistic."""
    refs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    # both refs identical to query => distance 0 each -> mean 0
    q = torch.tensor([[0.0, 1.0]])
    score = CosineScore(k=2, stat="mean")
    score.ref_embeddings = refs
    score._setup_index()
    out, _ = score._distance(q, offset=0)
    assert torch.allclose(out, torch.tensor([1.0]), atol=1e-6)


def test_mahalanobis_matches_manual_calculation() -> None:
    """Verify MahalanobisScore against a manual Mahalanobis computation."""
    refs = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    query = torch.tensor([[1.0, 1.0]])
    score = MahalanobisScore(k=1, stat="min")
    score.ref_embeddings = refs
    # call setup to compute vi_zero and populate index
    score._setup_index()
    # compute expected Mahalanobis distances manually
    cov = refs.T.cov()
    cov_inv = torch.linalg.inv(cov)
    expected_list: list[float] = []
    x = query[0]
    for p in refs:
        diff = (x - p).unsqueeze(0)  # 1xD
        val = torch.sqrt((diff @ cov_inv @ diff.T).squeeze())
        expected_list.append(val.item())
    expected = torch.tensor(expected_list)
    expected_min = torch.min(expected).unsqueeze(0)
    out, _ = score._distance(query, offset=0)
    approx(out, expected_min)


def test_mahalanobis_singular_cov_raises() -> None:
    """Ensure setup raises for singular covariance matrices."""
    # identical points -> covariance singular -> cholesky should fail
    refs = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    score = MahalanobisScore(k=1)
    score.ref_embeddings = refs
    with pytest.raises(Exception):
        score._setup_index()


def test_q_trimming_reduces_reference_set() -> None:
    """Test that _fit_impl trimming reduces the number of reference points."""
    n = 100
    refs = torch.randn(n, 5)
    score = EuclideanScore(k=1)
    # avoid calibration requirement for this test
    score.cal_required = False
    score.ref_embeddings = refs.float()
    # run fit-impl trimming (q=0.5 should remove roughly half the points)
    original_count = score.ref_embeddings.shape[0]
    score._fit_impl(q=0.50)
    new_count = score.ref_embeddings.shape[0]
    assert new_count < original_count
    assert new_count >= 1


def test_setup_index_creates_index() -> None:
    """Ensure _setup_index creates an index for each score type."""
    refs = torch.randn(5, 3)
    e = EuclideanScore(k=1)
    e.ref_embeddings = refs
    e._setup_index()
    assert e.index is not None
    # assert isinstance(e.index, nmslib.FloatIndex)

    c = CosineScore(k=1)
    c.ref_embeddings = refs
    c._setup_index()
    assert c.index is not None
    # assert isinstance(e.index, nmslib.FloatIndex)

    m = MahalanobisScore(k=1)
    m.ref_embeddings = refs
    m._setup_index()
    assert m.index is not None
    # assert isinstance(e.index, nmslib.FloatIndex)


def test_pca_reduces_dimension_and_is_applied() -> None:
    """Ensure PCA triggers dimensionality reduction and is applied when scoring."""
    torch.manual_seed(0)
    n, D = 50, 6
    # generate data along a single latent direction with small noise in other dims
    base = torch.randn(n, 1)
    direction = torch.randn(1, D)
    refs = (base @ direction) + 0.01 * torch.randn(n, D)

    q = torch.randn(1, D)

    s_pca = EuclideanScore(k=1, pca=TensorPCA(n_components=0.90))
    s_pca.cal_required = False
    s_pca.ref_embeddings = refs.float()
    s_pca._fit_impl(q=None)

    original_dim = D
    reduced_dim = s_pca.ref_embeddings.shape[1]
    assert reduced_dim < original_dim

    # build a second score that uses the already-projected embeddings
    s_proj = EuclideanScore(k=1, pca=None)
    s_proj.ref_embeddings = s_pca.ref_embeddings.clone()
    s_proj._setup_index()

    # get the PCA-transformed query via the fitted PCA (KNNScore uses pca.predict)
    assert s_pca.pca is not None
    q_proj = s_pca.pca.transform(q)

    out_with_pca = s_pca.score(q)
    out_manual = s_proj.score(q_proj)
    approx(out_with_pca, out_manual)


def test_suggest_params_small_n() -> None:
    """_suggest_index_params returns conservative defaults for very small N."""
    refs = torch.randn(5, 3)
    s = EuclideanScore(k=3)
    params = s._suggest_build_params(refs, k=3)
    assert "M" in params
    assert "efConstruction" in params
    params = s._suggest_query_params(refs, k=3)
    assert "efSearch" in params


@pytest.mark.filterwarnings(r"ignore:.*Loading existing index from disk.*")
def test_build_index_saves_and_loads(tmp_path: pathlib.Path) -> None:
    """_build_index saves index to disk and a second instance loads it."""
    path = tmp_path / "test_index.bin"
    s1 = EuclideanScore(k=1, save_index=path)
    s1.ref_embeddings = torch.randn(10, 3)
    s1._setup_index()
    assert path.exists()

    # second score should load the existing index file
    s2 = EuclideanScore(k=1, save_index=path)
    s2.ref_embeddings = torch.randn(5, 3)
    s2._setup_index()
    assert s2.index is not None


def test_euclidean_distance_returns_L2_distances() -> None:
    """score built with euclidean distances should match manual calculation."""
    torch.manual_seed(0)

    # mid-sized dataset
    N = 20000
    D = 64
    Q = 8

    refs = torch.randn(N, D, dtype=torch.float32)
    queries = torch.randn(Q, D, dtype=torch.float32)

    score = EuclideanScore(k=1, pca=None, save_index=False)

    # inject reference embeddings, build and query index
    score.ref_embeddings = refs
    score._setup_index()
    distances = score.score(queries)  # shape (Q,) for k=1

    # compute exact distances manually
    expected = torch.cdist(queries, refs, p=2.0)
    expected_min = expected.min(dim=1).values

    mae = torch.mean(torch.abs(distances - expected_min))
    assert mae < 0.01, f"Mean absolute error {mae} exceeds tolerance"


def test_query_dimension_mismatch_raises() -> None:
    """Ensure _query_index raises when query dimensionality differs from index dimensionality."""
    refs = torch.randn(5, 16)
    score = EuclideanScore(k=1, stat="min")
    score.ref_embeddings = refs
    score._setup_index()
    query = torch.randn(1, 8)
    with pytest.raises(ValueError) as exc:
        score._query_index(query, offset=0)
    assert "does not match index dimension" in str(exc.value)


def test_knn_search_offset_and_distances() -> None:
    """Verify knn_search returns correct distances and indices with offset.

    Use EuclideanScore with a small reference set where the query matches one
    reference point. With `k=2` and `offset=1` the method should skip the
    self‑match and return the distance to the nearest other point.
    """
    # reference embeddings: origin and two unit points
    refs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    query = torch.tensor([[0.0, 0.0]])
    score = EuclideanScore(k=2, stat="min")
    score.ref_embeddings = refs
    score._setup_index()
    # offset=1 skips the self‑match (distance 0)
    distances, indices = score.knn_search(query, offset=1)
    # Expect a single distance (to the nearest non‑self point) which is 1.0
    expected_distance = torch.tensor([[1.0, 1.0]])
    approx(distances, expected_distance)
    # The returned index should correspond to either of the two unit vectors
    assert indices.shape == (1, 2)
    assert indices[0, :].tolist() == [1, 2]
