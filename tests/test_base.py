import torch

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
