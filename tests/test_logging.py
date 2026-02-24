"""Tests for seapig.logging — the package logging scaffold."""

from __future__ import annotations

import logging
from io import StringIO

import pytest

import seapig
from seapig.utils.logging import configure_logging, get_logger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_logger() -> pytest.FixtureResult[None]:  # type: ignore[type-arg]
    """Reset the seapig logger to its default state after each test."""
    pkg_logger = get_logger("seapig")
    original_level = pkg_logger.level
    original_handlers = list(pkg_logger.handlers)
    original_propagate = pkg_logger.propagate
    yield  # type: ignore[misc]
    pkg_logger.handlers.clear()
    for h in original_handlers:
        pkg_logger.addHandler(h)
    pkg_logger.setLevel(original_level)
    pkg_logger.propagate = original_propagate


# ---------------------------------------------------------------------------
# NullHandler installed by __init__
# ---------------------------------------------------------------------------


def test_package_logger_has_null_handler() -> None:
    """seapig/__init__.py must install a NullHandler on the package logger."""
    # The NullHandler should be present after importing seapig.
    _ = seapig  # noqa: F841 — ensure __init__ has run
    pkg_logger = get_logger("seapig")
    null_handlers = [
        h for h in pkg_logger.handlers if isinstance(h, logging.NullHandler)
    ]
    assert null_handlers, "Expected at least one NullHandler on 'seapig' logger"


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


def test_get_logger_no_name_returns_package_logger() -> None:
    logger = get_logger()
    assert logger.name == "seapig"


def test_get_logger_with_seapig_name_returns_child() -> None:
    logger = get_logger("seapig.scores.knn")
    assert logger.name == "seapig.scores.knn"


def test_get_logger_with_non_seapig_name_returns_package_logger() -> None:
    logger = get_logger("some.other.module")
    assert logger.name == "seapig"


def test_get_logger_with_none_returns_package_logger() -> None:
    logger = get_logger(None)
    assert logger.name == "seapig"


def test_get_logger_returns_logger_instance() -> None:
    logger = get_logger(__name__)
    assert isinstance(logger, logging.Logger)


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


def test_configure_logging_sets_level_by_string() -> None:
    configure_logging(level="DEBUG")
    pkg_logger = get_logger("seapig")
    assert pkg_logger.level == logging.DEBUG


def test_configure_logging_sets_level_by_int() -> None:
    configure_logging(level=logging.INFO)
    pkg_logger = get_logger("seapig")
    assert pkg_logger.level == logging.INFO


def test_configure_logging_default_level_is_warning() -> None:
    configure_logging()
    pkg_logger = get_logger("seapig")
    assert pkg_logger.level == logging.WARNING


def test_configure_logging_attaches_stream_handler_when_none() -> None:
    configure_logging(level="INFO")
    pkg_logger = get_logger("seapig")
    stream_handlers = [
        h for h in pkg_logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert stream_handlers, "Expected a StreamHandler to be attached"


def test_configure_logging_custom_handler() -> None:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    configure_logging(level="INFO", handler=handler)
    pkg_logger = get_logger("seapig")
    assert handler in pkg_logger.handlers


def test_configure_logging_removes_previous_handlers() -> None:
    first = logging.StreamHandler(StringIO())
    configure_logging(level="INFO", handler=first)
    second = logging.StreamHandler(StringIO())
    configure_logging(level="INFO", handler=second)
    pkg_logger = get_logger("seapig")
    assert first not in pkg_logger.handlers
    assert second in pkg_logger.handlers


def test_configure_logging_produces_output() -> None:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    configure_logging(level="DEBUG", handler=handler)
    get_logger("seapig").debug("hello from test")
    assert "hello from test" in stream.getvalue()


def test_configure_logging_suppresses_below_level() -> None:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    configure_logging(level="WARNING", handler=handler)
    get_logger("seapig").debug("should be silent")
    assert stream.getvalue() == ""


# ---------------------------------------------------------------------------
# SEAPIG_LOG_LEVEL environment variable
# ---------------------------------------------------------------------------


def test_seapig_log_level_env_var_overrides_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEAPIG_LOG_LEVEL", "DEBUG")
    configure_logging(level="WARNING")  # env var should win
    pkg_logger = get_logger("seapig")
    assert pkg_logger.level == logging.DEBUG


def test_seapig_log_level_env_var_not_set_uses_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEAPIG_LOG_LEVEL", raising=False)
    configure_logging(level="INFO")
    pkg_logger = get_logger("seapig")
    assert pkg_logger.level == logging.INFO


# ---------------------------------------------------------------------------
# No stray print() output by default
# ---------------------------------------------------------------------------


def test_library_does_not_print_by_default(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Importing seapig and using core objects must not emit print output."""
    import seapig.scores.base  # noqa: F401 — trigger module-level code
    import seapig.scores.embed  # noqa: F401
    import seapig.scores.utils  # noqa: F401

    captured = capfd.readouterr()
    assert captured.out == "", "Library must not write to stdout by default"


def test_no_print_on_random_score_select_without_threshold(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """RandomScore.select() must not print when threshold is auto-set."""
    import torch

    from seapig.scores.base import RandomScore

    score = RandomScore()
    x = torch.randn(5, 4)
    score.select(x)
    captured = capfd.readouterr()
    assert captured.out == ""


def test_no_print_on_tensor_pca_fit(capfd: pytest.CaptureFixture[str]) -> None:
    """TensorPCA.fit() must not print when explaining variance."""
    import torch

    from seapig.scores.utils import TensorPCA

    pca = TensorPCA(n_components=0.9)
    x = torch.randn(50, 16)
    pca.fit(x)
    captured = capfd.readouterr()
    assert captured.out == ""
