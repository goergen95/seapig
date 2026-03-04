"""
Tests for seapig.scores.index — IndexHandler abstraction and nmslib adapter.

Covers:
- Adapter registration and contract
- Aggregation (mean/median/min/max) with full, partial and zero neighbours
- Offset (self-match removal) semantics
- No-zeropad behaviour: warning emitted, available distances aggregated
- Save / load roundtrip and metadata sidecar (atomic write)
- Integration with KNNScore subclasses (EuclideanScore, CosineScore,
  MahalanobisScore) via calibration (offset=1) and test (offset=0) paths
"""

from __future__ import annotations

import math
import pathlib
import warnings

import pytest
import torch

from seapig.scores.index import (
    IndexHandler,
    NmslibHandler,
    get_index_adapter,
    register_index_adapter,
)
from seapig.scores.knn import (
    CosineScore,
    EuclideanScore,
    KNNScore,
    MahalanobisScore,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def approx(t1: torch.Tensor, t2: torch.Tensor, tol: float = 1e-5) -> None:
    assert torch.allclose(t1.float(), t2.float(), atol=tol, rtol=0), (
        f"{t1} != {t2}"
    )


def _simple_handler(k: int = 1) -> NmslibHandler:
    return get_index_adapter("nmslib", k=k)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# test_adapter_contract — registration and contract
# ---------------------------------------------------------------------------


def test_get_index_adapter_returns_nmslib_handler() -> None:
    handler = get_index_adapter("nmslib")
    assert isinstance(handler, NmslibHandler)
    assert isinstance(handler, IndexHandler)


def test_register_and_retrieve_custom_adapter() -> None:
    class DummyAdapter(NmslibHandler):
        pass

    register_index_adapter("dummy_test", DummyAdapter)
    h = get_index_adapter("dummy_test")
    assert isinstance(h, DummyAdapter)


def test_unknown_adapter_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown_xyz"):
        get_index_adapter("unknown_xyz")


def test_nmslib_handler_constructor_initial_state() -> None:
    """NmslibHandler constructor sets k, _index=None, _query_defaults={}."""
    h = NmslibHandler(k=5)
    assert h.k == 5
    assert h._index is None
    assert h._query_defaults == {}


def test_nmslib_handler_implements_all_abstract_methods() -> None:
    """Verify NmslibHandler is instantiable (all abstract methods overridden)."""
    h = NmslibHandler(k=2)
    assert hasattr(h, "_build_impl")
    assert hasattr(h, "_add_impl")
    assert hasattr(h, "_query_impl")
    assert hasattr(h, "_save_impl")
    assert hasattr(h, "_load_impl")


def test_nmslib_handler_metadata_populated_after_build() -> None:
    h = NmslibHandler(k=1)
    embs = torch.randn(10, 4)
    h.build_index(embs, space="l2")
    meta = h.get_metadata()
    assert meta["library"] == "nmslib"
    assert meta["space"] == "l2"
    assert meta["n_items"] == 10
    assert "index_params" in meta


# ---------------------------------------------------------------------------
# test_aggregation — mean/median/min/max with full / partial / zero rows
# ---------------------------------------------------------------------------


def test_aggregate_distances_mean() -> None:
    h = NmslibHandler()
    result = h.aggregate_distances([[1.0, 3.0], [2.0, 4.0]], method="mean")
    assert result.shape == (2,)
    approx(result, torch.tensor([2.0, 3.0]))


def test_aggregate_distances_median() -> None:
    h = NmslibHandler()
    result = h.aggregate_distances([[1.0, 2.0, 3.0]], method="median")
    assert result.shape == (1,)
    approx(result, torch.tensor([2.0]))


def test_aggregate_distances_min() -> None:
    h = NmslibHandler()
    result = h.aggregate_distances([[5.0, 1.0, 3.0]], method="min")
    approx(result, torch.tensor([1.0]))


def test_aggregate_distances_max() -> None:
    h = NmslibHandler()
    result = h.aggregate_distances([[5.0, 1.0, 3.0]], method="max")
    approx(result, torch.tensor([5.0]))


def test_aggregate_distances_empty_row_is_nan() -> None:
    h = NmslibHandler()
    result = h.aggregate_distances([[]], method="max")
    assert result.shape == (1,)
    assert math.isnan(result[0].item())


def test_aggregate_distances_partial_row() -> None:
    """Partial rows (fewer than k) are aggregated over available distances."""
    h = NmslibHandler()
    # Only 1 distance when k=3 was requested: just aggregate what's there
    result = h.aggregate_distances([[7.0]], method="max")
    approx(result, torch.tensor([7.0]))


def test_aggregate_distances_invalid_method_raises() -> None:
    h = NmslibHandler()
    with pytest.raises(ValueError, match="Unsupported"):
        h.aggregate_distances([[1.0]], method="sum")


def test_query_batch_aggregation_shape_is_1d() -> None:
    """query_batch must return aggregated_distances with shape (B,)."""
    h = _simple_handler(k=1)
    embs = torch.randn(20, 4)
    h.build_index(embs, space="l2")
    queries = torch.randn(5, 4)
    indices, aggregated = h.query_batch(queries, k=1, aggregation="max")
    assert aggregated.shape == (5,), f"Expected (5,), got {aggregated.shape}"
    assert len(indices) == 5


@pytest.mark.parametrize("method", ["mean", "median", "min", "max"])
def test_query_batch_all_aggregation_methods(method: str) -> None:
    torch.manual_seed(0)
    h = _simple_handler(k=3)
    embs = torch.randn(30, 8)
    h.build_index(embs, space="l2")
    queries = torch.randn(4, 8)
    _, aggregated = h.query_batch(queries, k=3, aggregation=method)
    assert aggregated.shape == (4,)
    assert not torch.any(torch.isnan(aggregated))


# ---------------------------------------------------------------------------
# test_offset — offset strips leading neighbours correctly
# ---------------------------------------------------------------------------


def test_offset_strips_self_match() -> None:
    """With offset=1 the self-match at distance 0 is discarded."""
    torch.manual_seed(42)
    n, d = 20, 4
    embs = torch.randn(n, d)
    h = _simple_handler(k=2)
    h.build_index(embs, space="l2")

    # Query with the indexed points themselves: nearest neighbour is self → 0.0
    indices_no_offset, dist_no_offset = h.query_batch(embs[:3], k=1, offset=0)
    indices_with_offset, dist_with_offset = h.query_batch(
        embs[:3], k=1, offset=1
    )

    # Without offset, the closest neighbour of a point to itself is 0
    for d_row in dist_no_offset:
        assert d_row.item() == pytest.approx(0.0, abs=1e-5)

    # With offset=1, the self-match is gone and distances are > 0
    for d_row in dist_with_offset:
        assert d_row.item() > 0.0

    # Indices after offset should differ from the no-offset result
    for i_off, i_no in zip(indices_with_offset, indices_no_offset):
        # After stripping self, the first index should not be the self index
        assert i_off != i_no or len(i_off) == 0 or len(i_no) == 0


def test_offset_warning_when_fewer_neighbours_remain() -> None:
    """A warning is emitted when offset leaves fewer than k neighbours."""
    h = _simple_handler(k=3)
    refs = torch.randn(5, 4)
    h.build_index(refs, space="l2")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # k=3, offset=4 → queries for 7 neighbours but only 5 exist;
        # after stripping 4 only ≤1 remain, which is < 3
        h.query_batch(refs[:2], k=3, offset=4)
        assert any("fewer than" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# test_no_zeropad_warning — warning emitted, no padding, correct aggregation
# ---------------------------------------------------------------------------


def test_no_zero_padding_applied() -> None:
    """When fewer neighbours are returned, distances are NOT zero-padded."""
    h = _simple_handler(k=5)
    # Only 2 reference points → at most 2 neighbours returned
    refs = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    h.build_index(refs, space="l2")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, agg = h.query_batch(
            torch.tensor([[0.5, 0.0]]), k=5, aggregation="mean"
        )
        assert any("fewer than" in str(warning.message) for warning in w)

    # mean of [0.25, 0.25] (squared L2 from (0.5,0) to each ref) = 0.25
    approx(agg, torch.tensor([0.25]))


def test_warning_message_does_not_mention_zero_padding() -> None:
    """The warning from the new handler must NOT say 'zero padding'."""
    h = _simple_handler(k=10)
    refs = torch.randn(3, 4)
    h.build_index(refs, space="l2")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        h.query_batch(torch.randn(1, 4), k=10, aggregation="max")
        for warning in w:
            assert "zero padding" not in str(warning.message).lower()


# ---------------------------------------------------------------------------
# test_save_load_atomic — save / load roundtrip and metadata
# ---------------------------------------------------------------------------


def test_save_creates_binary_and_json_sidecar(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "myindex"
    h = _simple_handler(k=1)
    embs = torch.randn(10, 4)
    h.build_index(embs, space="l2")
    h.save_index(path)

    assert path.exists(), "Binary file must exist after save_index"
    assert path.with_suffix(".json").exists(), "JSON sidecar must exist"


def test_save_load_roundtrip(tmp_path: pathlib.Path) -> None:
    """Query results are consistent across a save/load cycle."""
    torch.manual_seed(7)
    path = tmp_path / "idx"
    embs = torch.randn(20, 6)
    queries = torch.randn(3, 6)

    h1 = _simple_handler(k=2)
    h1.build_index(embs, space="l2")
    _, dist_before = h1.query_batch(queries, k=2, aggregation="max")
    h1.save_index(path)

    h2 = _simple_handler(k=2)
    h2.load_index(path)
    _, dist_after = h2.query_batch(queries, k=2, aggregation="max")

    approx(dist_before, dist_after)


def test_load_restores_metadata(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "meta_idx"
    h1 = _simple_handler(k=1)
    embs = torch.randn(8, 3)
    h1.build_index(embs, space="cosinesimil")
    h1.save_index(path, metadata={"custom_key": "hello"})

    h2 = _simple_handler(k=1)
    h2.load_index(path)
    meta = h2.get_metadata()
    assert meta["space"] == "cosinesimil"
    assert meta["n_items"] == 8
    assert meta.get("custom_key") == "hello"


def test_save_atomic_no_partial_write(tmp_path: pathlib.Path) -> None:
    """Atomic save must not leave a .tmp file behind."""
    path = tmp_path / "atomic_idx"
    h = _simple_handler(k=1)
    h.build_index(torch.randn(5, 3), space="l2")
    h.save_index(path, atomic=True)

    tmp_file = path.with_suffix(".tmp")
    assert not tmp_file.exists(), ".tmp must be cleaned up after atomic save"


# ---------------------------------------------------------------------------
# test_integration_knn — KNNScore subclasses with offset=0 and offset=1
# ---------------------------------------------------------------------------


def _make_score(cls: type[KNNScore], k: int = 3, stat: str = "max") -> KNNScore:
    score = cls(k=k, stat=stat)
    return score


@pytest.mark.parametrize(
    "ScoreCls", [EuclideanScore, CosineScore, MahalanobisScore]
)
def test_knn_score_fit_and_score(ScoreCls: type) -> None:
    """KNNScore subclasses fit without error and return (B,) shaped scores."""
    torch.manual_seed(0)
    n, d = 40, 8
    refs = torch.randn(n, d)
    cal = torch.randn(10, d)
    test = torch.randn(5, d)

    score = ScoreCls(k=3)
    score.cal_required = True
    score.ref_embeddings = refs.float()
    score.cal_embeddings = cal.float()
    score._fit_impl(q=None)

    out = score.score(test.float())
    assert out.shape == (5,), f"Expected (5,), got {out.shape}"


@pytest.mark.parametrize(
    "ScoreCls", [EuclideanScore, CosineScore, MahalanobisScore]
)
def test_knn_score_calibration_path_offset_1(ScoreCls: type) -> None:
    """Calibration path (kpn=0, cal_embeddings set) runs without error."""
    torch.manual_seed(1)
    n, d = 30, 6
    refs = torch.randn(n, d)
    cal = torch.randn(8, d)

    score = ScoreCls(k=2)
    score.cal_required = True
    score.ref_embeddings = refs.float()
    score.cal_embeddings = cal.float()
    score._fit_impl(q=None)

    assert score.scores is not None
    assert score.scores.shape == (8,)


@pytest.mark.parametrize(
    "ScoreCls", [EuclideanScore, CosineScore, MahalanobisScore]
)
def test_knn_score_ref_only_path_kpn_1(ScoreCls: type) -> None:
    """Reference-only path (kpn=1, no cal_embeddings) runs without error."""
    torch.manual_seed(2)
    n, d = 25, 5
    refs = torch.randn(n, d)

    score = ScoreCls(k=2)
    score.cal_required = False
    score.ref_embeddings = refs.float()
    score._fit_impl(q=None)

    assert score.scores is not None
    assert score.scores.shape == (n,)


def test_euclidean_distance_consistent_with_direct_query() -> None:
    """EuclideanScore._distance agrees with a direct query_batch call."""
    torch.manual_seed(3)
    refs = torch.randn(15, 4)
    queries = torch.randn(3, 4)

    score = EuclideanScore(k=2, stat="max")
    score.ref_embeddings = refs
    score._setup_index()

    dist_via_score = score._distance(queries, kpn=0)

    assert score.index_handler is not None
    _, dist_direct = score.index_handler.query_batch(
        queries, k=2, aggregation="max", offset=0
    )
    approx(dist_via_score, dist_direct)
