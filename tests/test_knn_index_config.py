"""
Unit and integration tests for IndexConfig and nmslib index configuration.

Tests cover:
- IndexConfig validation
- Constructor parameter handling (index_config, nms_index, save_index)
- Build/query params forwarding to nmslib
- Metadata save/load with validation
- Passing pre-built nmslib index objects
- End-to-end scoring with default and custom configs
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from seapig.scores.knn import (
    CosineScore,
    EuclideanScore,
    IndexConfig,
    _validate_index_config,
)

# ============================================================================
# Unit tests: IndexConfig validation
# ============================================================================


def test_index_config_default_values() -> None:
    """Verify IndexConfig has correct default values."""
    config = IndexConfig()
    assert config.method == "hnsw"
    assert config.space is None
    assert config.build_params is None
    assert config.query_params is None
    assert config.index_path is None


def test_validate_index_config_valid() -> None:
    """Validate that a valid config passes validation."""
    config = IndexConfig(
        method="hnsw",
        build_params={"M": 16, "efConstruction": 200, "post": 0},
        query_params={"efSearch": 50},
        index_path=Path("test.bin"),
    )
    _validate_index_config(config)  # Should not raise


def test_validate_index_config_invalid_method() -> None:
    """Ensure invalid method raises ValueError."""
    config = IndexConfig(method="invalid")
    with pytest.raises(
        ValueError, match="Only method='hnsw' is currently supported"
    ):
        _validate_index_config(config)


def test_validate_index_config_invalid_index_path_suffix() -> None:
    """Ensure index_path without .bin suffix raises ValueError."""
    config = IndexConfig(index_path=Path("test.txt"))
    with pytest.raises(ValueError, match="must have .bin extension"):
        _validate_index_config(config)


def test_validate_index_config_invalid_build_params() -> None:
    """Ensure invalid build_params keys raise ValueError."""
    config = IndexConfig(build_params={"invalid_key": 1})
    with pytest.raises(ValueError, match="Invalid build_params keys"):
        _validate_index_config(config)


def test_validate_index_config_invalid_query_params() -> None:
    """Ensure invalid query_params keys raise ValueError."""
    config = IndexConfig(query_params={"invalid_key": 1})
    with pytest.raises(ValueError, match="Invalid query_params keys"):
        _validate_index_config(config)


# ============================================================================
# Unit tests: Constructor parameter handling
# ============================================================================


def test_constructor_with_index_config() -> None:
    """Test constructor accepts IndexConfig."""
    config = IndexConfig(index_path=Path("test.bin"))
    score = EuclideanScore(k=1, index_config=config)
    assert score.index_config == config
    assert score.index_path == Path("test.bin")


def test_constructor_backward_compat_save_index_bool() -> None:
    """Test backward compatibility with save_index=True."""
    score = EuclideanScore(k=1, save_index=True)
    assert score.index_config is not None
    assert score.index_path is not None
    assert score.index_path.suffix == ".bin"


def test_constructor_backward_compat_save_index_path() -> None:
    """Test backward compatibility with save_index=Path."""
    path = Path("my_index.bin")
    score = EuclideanScore(k=1, save_index=path)
    assert score.index_config is not None
    assert score.index_path == path


def test_constructor_with_nms_index() -> None:
    """Test constructor accepts pre-built nmslib index."""
    mock_index = MagicMock()
    mock_index.knnQueryBatch = MagicMock()

    score = EuclideanScore(k=1, nms_index=mock_index)
    assert score.index == mock_index
    assert score.index_params is not None


def test_constructor_nms_index_invalid() -> None:
    """Test constructor rejects invalid nms_index."""
    mock_index = MagicMock()
    # Remove knnQueryBatch method
    del mock_index.knnQueryBatch

    with pytest.raises(ValueError, match="must have a knnQueryBatch method"):
        EuclideanScore(k=1, nms_index=mock_index)


# ============================================================================
# Unit tests: Build/query params forwarding (with mocking)
# ============================================================================


@patch("seapig.scores.knn.nmslib")
def test_build_index_uses_custom_build_params(mock_nmslib) -> None:
    """Verify custom build_params are passed to nmslib.createIndex."""
    # Setup mock
    mock_index = MagicMock()
    mock_nmslib.init.return_value = mock_index

    # Create score with custom config
    config = IndexConfig(
        build_params={"M": 32, "efConstruction": 300, "post": 0},
        query_params={"efSearch": 100},
    )
    score = EuclideanScore(k=1, index_config=config)
    score.ref_embeddings = torch.randn(10, 8)

    # Call _setup_index which calls _build_index
    score._setup_index()

    # Verify nmslib.init was called with correct params
    mock_nmslib.init.assert_called_once_with(method="hnsw", space="l2")

    # Verify createIndex was called with custom params
    mock_index.createIndex.assert_called_once_with(
        index_params={"M": 32, "efConstruction": 300, "post": 0}
    )


@patch("seapig.scores.knn.nmslib")
def test_build_index_uses_default_params_when_none(mock_nmslib) -> None:
    """Verify default params from _suggest_index_params are used when config is None."""
    # Setup mock
    mock_index = MagicMock()
    mock_nmslib.init.return_value = mock_index

    # Create score without config
    score = EuclideanScore(k=1)
    score.ref_embeddings = torch.randn(100, 16)

    # Call _setup_index
    score._setup_index()

    # Verify createIndex was called with suggested params
    call_args = mock_index.createIndex.call_args
    assert call_args is not None
    build_params = call_args[1]["index_params"]
    assert "M" in build_params
    assert "efConstruction" in build_params
    assert "post" in build_params


@patch("seapig.scores.knn.nmslib")
def test_query_index_uses_custom_query_params(mock_nmslib) -> None:
    """Verify custom query_params are passed to nmslib.setQueryTimeParams."""
    # Setup mock
    mock_index = MagicMock()
    mock_index.knnQueryBatch.return_value = [
        (torch.tensor([0]), torch.tensor([1.5]))
    ]
    mock_nmslib.init.return_value = mock_index

    # Create score with custom config
    config = IndexConfig(query_params={"efSearch": 200})
    score = EuclideanScore(k=1, index_config=config)
    score.ref_embeddings = torch.randn(10, 8)
    score._setup_index()

    # Query the index
    query = torch.randn(1, 8)
    score._query_index(query, kpn=0)

    # Verify setQueryTimeParams was called with custom params
    mock_nmslib.setQueryTimeParams.assert_called_with(
        mock_index, {"efSearch": 200}
    )


# ============================================================================
# Integration tests: Save/load with metadata
# ============================================================================


def test_save_and_load_with_metadata(tmp_path: Path) -> None:
    """Test saving index with metadata and loading it back."""
    index_path = tmp_path / "test_index.bin"
    metadata_path = tmp_path / "test_index.json"

    # Create and fit score with index_path
    config = IndexConfig(
        build_params={"M": 16, "efConstruction": 200, "post": 0},
        query_params={"efSearch": 50},
        index_path=index_path,
    )
    score1 = EuclideanScore(k=1, index_config=config)
    refs = torch.randn(30, 8)
    score1.ref_embeddings = refs
    score1._setup_index()

    # Verify files exist
    assert index_path.exists()
    assert metadata_path.exists()

    # Verify metadata content
    with open(metadata_path) as f:
        metadata = json.load(f)
    assert metadata["method"] == "hnsw"
    assert metadata["space"] == "l2"
    assert metadata["build_params"] == {
        "M": 16,
        "efConstruction": 200,
        "post": 0,
    }
    assert metadata["query_params"] == {"efSearch": 50}

    # Load with same config - should succeed
    score2 = EuclideanScore(k=1, index_config=config)
    score2.ref_embeddings = torch.randn(5, 8)  # Different embeddings
    score2._setup_index()

    # Verify index was loaded, not rebuilt
    assert score2.index is not None


def test_load_with_mismatched_metadata_raises(tmp_path: Path) -> None:
    """Test that loading with mismatched metadata raises ValueError."""
    index_path = tmp_path / "test_index.bin"

    # Create and save with one config
    config1 = IndexConfig(
        build_params={"M": 16, "efConstruction": 200, "post": 0},
        query_params={"efSearch": 50},
        index_path=index_path,
    )
    score1 = EuclideanScore(k=1, index_config=config1)
    score1.ref_embeddings = torch.randn(30, 8)
    score1._setup_index()

    # Try to load with different config
    config2 = IndexConfig(
        build_params={"M": 32, "efConstruction": 300, "post": 0},  # Different!
        query_params={"efSearch": 50},
        index_path=index_path,
    )
    score2 = EuclideanScore(k=1, index_config=config2)
    score2.ref_embeddings = torch.randn(30, 8)

    with pytest.raises(ValueError, match="Metadata mismatch"):
        score2._setup_index()


def test_load_without_metadata_raises(tmp_path: Path) -> None:
    """Test that loading index without metadata file raises ValueError."""
    index_path = tmp_path / "test_index.bin"

    # Create index file manually (simulate old index without metadata)
    config = IndexConfig(index_path=index_path)
    score1 = EuclideanScore(k=1, index_config=config)
    score1.ref_embeddings = torch.randn(30, 8)
    score1._setup_index()

    # Remove metadata file
    metadata_path = index_path.with_suffix(".json")
    metadata_path.unlink()

    # Try to load
    score2 = EuclideanScore(k=1, index_config=config)
    score2.ref_embeddings = torch.randn(30, 8)

    with pytest.raises(ValueError, match="metadata file.*is missing"):
        score2._setup_index()


# ============================================================================
# Integration tests: End-to-end scoring
# ============================================================================


def test_end_to_end_scoring_default_config() -> None:
    """Test end-to-end scoring with default configuration."""
    torch.manual_seed(42)

    # Create score with defaults
    score = EuclideanScore(k=1)
    refs = torch.randn(30, 8)
    score.ref_embeddings = refs
    score._setup_index()

    # Score some queries
    queries = torch.randn(5, 8)
    scores = score._distance(queries, kpn=0)

    assert scores.shape == (5,)
    assert torch.all(scores >= 0)


def test_end_to_end_scoring_custom_config() -> None:
    """Test end-to-end scoring with custom IndexConfig."""
    torch.manual_seed(42)

    # Create score with custom config
    config = IndexConfig(
        build_params={"M": 16, "efConstruction": 200, "post": 0},
        query_params={"efSearch": 50},
    )
    score = EuclideanScore(k=1, index_config=config)
    refs = torch.randn(30, 8)
    score.ref_embeddings = refs
    score._setup_index()

    # Score some queries
    queries = torch.randn(5, 8)
    scores = score._distance(queries, kpn=0)

    assert scores.shape == (5,)
    assert torch.all(scores >= 0)


@pytest.mark.parametrize("score_class", [EuclideanScore, CosineScore])
def test_default_behavior_matches_suggested_params(score_class) -> None:
    """Test that default behavior uses _suggest_index_params."""
    torch.manual_seed(42)

    # Score with default config
    score1 = score_class(k=2)
    refs = torch.randn(50, 16)
    score1.ref_embeddings = refs
    score1._setup_index()

    # Get suggested params
    suggested = score_class._suggest_index_params(refs, k=2)

    # Verify params match
    assert score1.index_params["build_defaults"] == suggested["build_defaults"]
    assert score1.index_params["query_defaults"] == suggested["query_defaults"]


def test_prebuilt_index_bypasses_building() -> None:
    """Test that passing a pre-built index bypasses index building."""
    # Create a mock index
    mock_index = MagicMock()
    mock_index.knnQueryBatch.return_value = [
        (torch.tensor([0]), torch.tensor([1.5]))
    ]

    # Create score with pre-built index
    score = EuclideanScore(k=1, nms_index=mock_index)
    score.ref_embeddings = torch.randn(10, 8)

    # Call _build_index - should return early
    score._build_index(score.ref_embeddings, space="l2")

    # Verify index wasn't modified
    assert score.index == mock_index
    # Verify addDataPointBatch wasn't called on our mock
    assert not mock_index.addDataPointBatch.called


def test_save_load_round_trip_preserves_scores(tmp_path: Path) -> None:
    """Test that save/load round trip produces identical scores."""
    torch.manual_seed(42)
    index_path = tmp_path / "round_trip.bin"

    # Create, fit, and save
    config = IndexConfig(
        build_params={"M": 16, "efConstruction": 200, "post": 0},
        query_params={"efSearch": 50},
        index_path=index_path,
    )
    refs = torch.randn(30, 8)
    queries = torch.randn(5, 8)

    score1 = EuclideanScore(k=1, index_config=config)
    score1.ref_embeddings = refs
    score1._setup_index()
    scores1 = score1._distance(queries, kpn=0)

    # Load and score again
    score2 = EuclideanScore(k=1, index_config=config)
    score2.ref_embeddings = refs
    score2._setup_index()
    scores2 = score2._distance(queries, kpn=0)

    # Scores should be identical (or very close due to HNSW approximation)
    assert torch.allclose(scores1, scores2, atol=1e-4)


# ============================================================================
# Integration tests: Config space override
# ============================================================================


def test_config_space_overrides_default() -> None:
    """Test that IndexConfig.space overrides the default space."""
    # Note: This is a bit tricky to test without mocking since we'd need
    # embeddings that work with both spaces. We'll just verify the parameter
    # is passed through by checking metadata.
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test.bin"

        # Create config with explicit space
        config = IndexConfig(
            space="cosinesimil",  # Override default "l2"
            index_path=index_path,
        )

        # Create EuclideanScore which normally uses "l2"
        score = EuclideanScore(k=1, index_config=config)
        refs = torch.nn.functional.normalize(torch.randn(30, 8))
        score.ref_embeddings = refs
        score._setup_index()

        # Check metadata
        metadata_path = index_path.with_suffix(".json")
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["space"] == "cosinesimil"
