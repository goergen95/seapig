"""Unit tests for FAISS index handler."""

import warnings
from pathlib import Path

import pytest
import torch

from seapig.scores.index_handler import FaissIndexHandler, faiss

pytestmark = pytest.mark.skipif(faiss is None, reason="faiss not installed")


def test_suggest_build_and_query_params() -> None:
    """FAISS parameter suggestions include required keys and valid values."""
    handler = FaissIndexHandler()
    build = handler.suggest_build_params(n_samples=12_000, dim=64, k=7)
    query = handler.suggest_query_params(n_samples=12_000, dim=64, k=7)

    assert {"nlist", "m", "nbits", "use_opq", "use_flat_fallback"} == set(
        build.keys()
    )
    assert build["nlist"] > 0
    assert build["m"] > 0
    assert 64 % build["m"] == 0
    assert build["nbits"] == 8
    assert "nprobe" in query
    assert query["nprobe"] >= 1


def test_build_query_and_aggregate_with_flat_fallback() -> None:
    """Small datasets use flat fallback and return expected query shapes."""
    torch.manual_seed(0)
    refs = torch.randn(50, 8, dtype=torch.float32)
    query = torch.randn(3, 8, dtype=torch.float32)
    handler = FaissIndexHandler()
    handler.build_index(refs, k=2)

    dists = handler.query_index(query=query, k=2, offset=1)
    assert dists.shape == (3, 3)
    assert torch.all(dists >= 0)

    sliced = dists[:, 1:]
    for stat in ["max", "min", "mean", "median"]:
        agg = handler.aggregate_dists(sliced, stat=stat)
        assert agg.shape == (3,)


def test_query_warns_and_zero_pads_when_request_exceeds_available() -> None:
    """Querying more neighbors than available emits warnings and zero-pads."""
    refs = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    query = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
    handler = FaissIndexHandler()
    handler.build_index(refs, k=1)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        dists = handler.query_index(query=query, k=3, offset=0)
    assert dists.shape == (1, 3)
    assert any("fewer than 3 neighbors" in str(w.message) for w in rec)
    assert torch.isclose(dists[0, 2], torch.tensor(0.0))


def test_build_overrides_respected() -> None:
    """build_index merges caller-provided build options."""
    refs = torch.randn(700, 16, dtype=torch.float32)
    handler = FaissIndexHandler()
    handler.build_index(refs, k=4, use_flat_fallback=True)

    assert handler.index_params is not None
    assert handler.index_params["build_defaults"]["use_flat_fallback"] is True


@pytest.mark.filterwarnings(r"ignore:.*Loading existing index from disk.*")
def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Saved FAISS index can be loaded and queried from disk."""
    path = tmp_path / "faiss-index.bin"
    refs = torch.randn(120, 8, dtype=torch.float32)
    query = torch.randn(5, 8, dtype=torch.float32)

    writer = FaissIndexHandler(index_path=path)
    writer.build_index(refs, k=1)
    assert path.exists()

    reader = FaissIndexHandler()
    reader.load_index(path)
    dists = reader.query_index(query=query, k=1, offset=0)
    assert dists.shape == (5, 1)


def test_build_uses_existing_index_path(tmp_path: Path) -> None:
    """Existing on-disk index is loaded on subsequent builds."""
    path = tmp_path / "existing-index.bin"
    refs = torch.randn(80, 8, dtype=torch.float32)

    first = FaissIndexHandler(index_path=path)
    first.build_index(refs, k=1)
    assert path.exists()

    second = FaissIndexHandler(index_path=path)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        second.build_index(torch.randn(30, 8), k=1)
    assert second.index is not None
    assert any("Loading existing index from disk" in str(w.message) for w in rec)
