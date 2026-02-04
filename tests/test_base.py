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

    # Mock plt.show to avoid displaying the plot during the test
    with patch("matplotlib.pyplot.show") as mock_show:
        # Call the plot method
        query_scores = score.score(query_embeddings) if include_query else None
        score.plot(query_scores=query_scores, bins=25)

        # Ensure plt.show was called
        mock_show.assert_called_once()
