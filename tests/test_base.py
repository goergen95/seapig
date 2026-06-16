from typing import Any
from unittest.mock import patch

import pytest
import torch

from seapig.scores import EuclideanScore, RandomScore
from seapig.scores.base import UncertaintyScore


class Dummy(UncertaintyScore):
    def fit(
        self,
        X: torch.Tensor | None = None,
        Y: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        return None  # pragma: no cover

    def score(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return X  # pragma: no cover

    def select(
        self, X: torch.Tensor, *args: Any, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError()  # pragma: no cover


def test_random_score() -> None:
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
        == (selection["score"] < random_score.get_threshold())  # type: ignore[operator, ty:unsupported-operator]
    )


@pytest.mark.parametrize("include_query", [False, True])
def test_plot_method(include_query: bool) -> None:
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


def test_flag_methods_and_setters() -> None:
    dummy = Dummy()
    # defaults
    assert dummy.requires_training() is False
    assert dummy.requires_calibration() is False
    assert dummy.is_trained() is False
    assert dummy.is_calibrated() is False

    dummy.set_trained()
    dummy.set_calibrated()
    assert dummy.is_trained() is True
    assert dummy.is_calibrated() is True


def test_set_threshold_invalid_quantile_raises() -> None:
    s = Dummy()
    with pytest.raises(AssertionError):
        s.set_threshold(q=1.0)
    with pytest.raises(AssertionError):
        s.set_threshold(q=0.0)


def test_plot_raises_import_error_when_matplotlib_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate matplotlib not being installed by intercepting imports
    import builtins

    d = Dummy()
    d.scores = torch.tensor([0.1, 0.2])

    def fake_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("No module named matplotlib")
        return orig_import(
            name, globals, locals, fromlist, level
        )  # pragma: no cover

    orig_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        d.plot()

    # monkeypatch will restore the original import automatically


def test_randomscore_select_logs_warning_when_threshold_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rs = RandomScore()
    rs.threshold = None
    X = torch.randn(3, 2)
    caplog.clear()
    with caplog.at_level("WARNING"):
        sel = rs.select(X)
    # warning message expected
    messages = [r.message for r in caplog.records]
    assert any("Trying to set it via `set_threshold()`" in m for m in messages)
    assert sel["score"].shape[0] == X.shape[0]
