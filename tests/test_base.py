from unittest.mock import patch

import pytest
import torch

from seapig.scores import EuclideanScore, RandomScore
from seapig.scores.base import ConfidenceScore


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
    pytest.importorskip("matplotlib")
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

        # Ensure p# pragma: no coverlt.show was called
        mock_show.assert_called_once()


def test_base_select_sets_threshold_and_computes_selection():
    """Ensure ConfidenceScore.select() sets the threshold from calibration
    scores when threshold is None and returns the correct selection mask.
    """

    class Dummy(ConfidenceScore):
        def fit(self, X=None, Y=None):
            return None

        def score(self, X, *args, **kwargs):
            # Return the input values as scores for determinism
            return X.view(-1)

    dummy = Dummy()

    # Calibration scores used to compute the quantile threshold
    cal_scores = torch.tensor([0.1, 0.2, 0.8, 1.0])
    dummy.scores = cal_scores
    dummy.threshold = None  # force select() to call set_threshold()

    # Query scores (we pass them as X so Dummy.score returns them)
    query = torch.tensor([0.05, 0.2, 0.9, 1.2])

    expected_thr = torch.quantile(cal_scores, q=0.99)

    selection = dummy.select(query)

    # threshold should have been set from calibration scores
    assert torch.allclose(dummy.get_threshold(), expected_thr)

    # score returned unchanged and selected equals score < threshold
    assert torch.equal(selection["score"], query)
    assert torch.equal(selection["selected"], selection["score"] < expected_thr)


def test_flag_methods_and_setters() -> None:
    class T(ConfidenceScore):
        def fit(self, X=None, Y=None):
            return None

        def score(self, X, *args, **kwargs):
            return torch.zeros(X.shape[0])

    t = T()
    # defaults
    assert t.requires_training() is False
    assert t.requires_calibration() is False
    assert t.is_trained() is False
    assert t.is_calibrated() is False

    t.set_trained()
    t.set_calibrated()
    assert t.is_trained() is True
    assert t.is_calibrated() is True


def test_set_threshold_invalid_quantile_raises() -> None:
    class Local(ConfidenceScore):
        def fit(self, X=None, Y=None):
            return None

        def score(self, X, *args, **kwargs):
            return torch.zeros(X.shape[0])

    s = Local()
    with pytest.raises(AssertionError):
        s.set_threshold(q=1.0)
    with pytest.raises(AssertionError):
        s.set_threshold(q=0.0)


def test_plot_raises_import_error_when_matplotlib_missing(monkeypatch):
    # Simulate matplotlib not being installed by intercepting imports
    import builtins

    from seapig.scores.base import ConfidenceScore

    class Dummy(ConfidenceScore):
        def fit(self, X=None, Y=None):
            return None

        def score(self, X, *args, **kwargs):
            return torch.zeros(X.shape[0])

    d = Dummy()
    d.scores = torch.tensor([0.1, 0.2])

    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("No module named matplotlib")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        d.plot()

    # monkeypatch will restore the original import automatically


def test_randomscore_select_logs_warning_when_threshold_none(caplog):
    rs = RandomScore()
    rs.threshold = None  # type: ignore[attr-defined]
    X = torch.randn(3, 2)
    caplog.clear()
    with caplog.at_level("WARNING"):
        sel = rs.select(X)
    # warning message expected
    messages = [r.message for r in caplog.records]
    assert any("Trying to set it via `set_threshold()`" in m for m in messages)
    assert sel["score"].shape[0] == X.shape[0]
