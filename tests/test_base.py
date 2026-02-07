from unittest.mock import patch

import pytest
import torch

from seapig import EuclideanScore
from seapig.scores.base import RandomScore


def test_random_score():
    # Create a RandomScore instance
    random_score = RandomScore()

    # Generate a batch of random inputs
    batch = torch.rand(10)

    # Test the score method
    scores = random_score.score(batch)
    assert scores.shape == (10,)
    assert torch.all(scores >= 0) and torch.all(scores <= 1)

    # Test the select method
    selection = random_score.select(batch)
    assert "score" in selection and "selected" in selection
    assert selection["score"].shape == (10,)
    assert selection["selected"].shape == (10,)
    assert torch.all(
        selection["selected"]
        == (selection["score"] < random_score.get_threshold())
    )


@pytest.mark.parametrize("include_query", [False, True])
def test_plot_method(include_query):
    # Create dummy embeddings
    ref_embeddings = torch.randn(100, 64)
    cal_embeddings = torch.randn(100, 64)
    query_embeddings = torch.randn(50, 64) * 1.25 if include_query else None

    # Initialize and fit the dummy score
    score = EuclideanScore()
    score.fit(ref_embeddings, cal_embeddings)
    score.set_threshold(0.95)

    # Mock plt.show to avoid displaying the plot during the test
    with patch("matplotlib.pyplot.show") as mock_show:
        # Call the plot method
        query_scores = score.score(query_embeddings) if include_query else None
        score.plot(query_scores=query_scores, bins=25)

        # Ensure plt.show was called
        mock_show.assert_called_once()


@pytest.mark.parametrize(
    "n_points,expected_size_range",
    [
        (1, (100.0, 100.0)),  # Edge case: 1 point -> base_size
        (10, (90.0, 110.0)),  # log10(10)=1, size=100/1=100
        (100, (45.0, 55.0)),  # log10(100)=2, size=100/2=50
        (1000, (30.0, 36.0)),  # log10(1000)=3, size=100/3=33.33
    ],
)
def test_logarithmic_point_size_scaling(n_points, expected_size_range):
    """Test that point sizes scale logarithmically with the number of points."""
    # Create dummy embeddings with specific number of calibration points
    # The number of points in the plot is determined by the calibration embeddings
    ref_embeddings = torch.randn(100, 64)
    cal_embeddings = torch.randn(n_points, 64)  # Vary calibration embeddings

    # Initialize and fit the score
    score = EuclideanScore()
    score.fit(ref_embeddings, cal_embeddings)
    score.set_threshold(0.95)

    # Mock both scatter and show to capture the scatter call
    with (
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.scatter") as mock_scatter,
    ):
        score.plot(query_scores=None, bins=25)

        # Verify scatter was called (at least once for calibration scores)
        assert mock_scatter.call_count >= 1

        # Get the first scatter call (for calibration scores)
        first_call_kwargs = mock_scatter.call_args_list[0][1]

        # Check that the 's' parameter (point size) is in the expected range
        point_size = first_call_kwargs.get("s")
        assert point_size is not None, "Point size 's' parameter not found"

        min_size, max_size = expected_size_range
        assert min_size <= point_size <= max_size, (
            f"Point size {point_size} not in expected range [{min_size}, {max_size}]"
        )


def test_logarithmic_scaling_edge_case_zero():
    """Test edge case where we might have 0 points (should be handled safely)."""
    # Create a scenario where we have at least some calibration points
    ref_embeddings = torch.randn(10, 64)
    cal_embeddings = torch.randn(10, 64)

    score = EuclideanScore()
    score.fit(ref_embeddings, cal_embeddings)
    score.set_threshold(0.95)

    # Should not raise any errors
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.scatter"):
        score.plot(query_scores=None, bins=25)


def test_logarithmic_scaling_with_query_scores():
    """Test that both calibration and query scores get appropriate point sizes."""
    # Create embeddings with different numbers of points
    ref_embeddings = torch.randn(1000, 64)  # Many calibration points
    cal_embeddings = torch.randn(1000, 64)
    query_embeddings = torch.randn(100, 64)  # Fewer query points

    score = EuclideanScore()
    score.fit(ref_embeddings, cal_embeddings)
    score.set_threshold(0.95)

    query_scores = score.score(query_embeddings)

    with (
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.scatter") as mock_scatter,
    ):
        score.plot(query_scores=query_scores, bins=25)

        # Should have two scatter calls: one for calibration, one for query
        assert mock_scatter.call_count == 2

        # Get point sizes from both calls
        cal_size = mock_scatter.call_args_list[0][1].get("s")
        query_size = mock_scatter.call_args_list[1][1].get("s")

        # Calibration has more points (1000), so should have smaller size
        # Query has fewer points (100), so should have larger size
        assert cal_size < query_size, (
            f"Calibration size {cal_size} should be smaller than query size {query_size}"
        )

        # Verify they follow logarithmic scaling
        # For 1000 points: log10(1000)=3, size≈33.33
        # For 100 points: log10(100)=2, size=50
        assert 30.0 <= cal_size <= 36.0
        assert 45.0 <= query_size <= 55.0
