"""Tests for seapig.utils.progress — the unified progress subsystem."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

import seapig.utils.progress as _prog_mod
from seapig.utils.progress import (
    disable,
    enable,
    get_backend,
    is_enabled,
    reset,
    set_backend,
    track,
)


@pytest.fixture(autouse=True)
def _reset_state() -> Iterator[None]:
    """Reset global progress state and remove env vars after each test."""
    yield
    reset()
    # Restore module-level _backend default
    _prog_mod._backend = "tqdm"
    # Remove env vars that may have been set
    for var in ("SEAPIG_PROGRESS", "SEAPIG_PROGRESS_BACKEND"):
        os.environ.pop(var, None)


def test_enable_sets_is_enabled_true() -> None:
    enable()
    assert is_enabled() is True


def test_disable_sets_is_enabled_false() -> None:
    disable()
    assert is_enabled() is False


def test_reset_clears_programmatic_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After reset(), auto-detection takes over again."""
    monkeypatch.setattr("sys.stdout", MagicMock(isatty=lambda: False))
    enable()
    assert is_enabled() is True
    reset()
    # No TTY → auto-detection should return False
    assert is_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "on"])
def test_env_var_true_values(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("SEAPIG_PROGRESS", val)
    reset()
    assert is_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "False", "FALSE", "no", "off"])
def test_env_var_false_values(
    monkeypatch: pytest.MonkeyPatch, val: str
) -> None:
    monkeypatch.setenv("SEAPIG_PROGRESS", val)
    reset()
    assert is_enabled() is False


def test_programmatic_override_beats_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A programmatic disable() must override SEAPIG_PROGRESS=1."""
    monkeypatch.setenv("SEAPIG_PROGRESS", "1")
    disable()
    assert is_enabled() is False


def test_auto_detect_non_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    reset()
    monkeypatch.setattr("sys.stdout", MagicMock(isatty=lambda: False))
    # No IPython in builtins
    import builtins

    monkeypatch.delattr(builtins, "get_ipython", raising=False)
    assert is_enabled() is False


def test_auto_detect_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    reset()
    monkeypatch.setattr("sys.stdout", MagicMock(isatty=lambda: True))
    import builtins

    monkeypatch.delattr(builtins, "get_ipython", raising=False)
    assert is_enabled() is True


def test_auto_detect_ipython(monkeypatch: pytest.MonkeyPatch) -> None:
    reset()
    monkeypatch.setattr("sys.stdout", MagicMock(isatty=lambda: False))
    import builtins

    monkeypatch.setattr(
        builtins, "get_ipython", lambda: object(), raising=False
    )
    assert is_enabled() is True


def test_get_backend_default() -> None:
    _prog_mod._backend = "tqdm"
    assert get_backend() == "tqdm"


def test_set_backend_rich() -> None:
    set_backend("rich")
    assert get_backend() == "rich"


def test_set_backend_tqdm() -> None:
    set_backend("rich")
    set_backend("tqdm")
    assert get_backend() == "tqdm"


def test_set_backend_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        set_backend("unknown")


def test_track_disabled_yields_all_items() -> None:
    disable()
    result = list(track([10, 20, 30], desc="test"))
    assert result == [10, 20, 30]


def test_track_disabled_yields_correct_count() -> None:
    disable()
    items = list(range(50))
    assert list(track(items)) == items


def test_track_enabled_tqdm_yields_all_items() -> None:
    enable()
    set_backend("tqdm")
    with patch("tqdm.tqdm") as mock_tqdm:
        mock_tqdm.return_value = iter([1, 2, 3])
        result = list(track([1, 2, 3], desc="t"))
    assert result == [1, 2, 3]


def test_track_enabled_tqdm_passes_kwargs() -> None:
    enable()
    set_backend("tqdm")
    captured: dict[str, object] = {}

    def fake_tqdm(iterable: object, **kw: object) -> Iterator[object]:
        captured.update(kw)
        return iter([])

    with patch("tqdm.tqdm", side_effect=fake_tqdm):
        list(track([], desc="hello", unit="imgs", total=0))

    assert captured["desc"] == "hello"
    assert captured["unit"] == "imgs"
    assert captured["total"] == 0


def test_track_rich_falls_back_to_tqdm_when_rich_missing() -> None:
    """When rich is requested but not installed, fall back to tqdm silently."""
    enable()
    set_backend("rich")

    with patch.dict(sys.modules, {"rich": None, "rich.progress": None}):
        with patch("tqdm.tqdm") as mock_tqdm:
            mock_tqdm.return_value = iter([7, 8, 9])
            result = list(track([7, 8, 9]))

    assert result == [7, 8, 9]


def test_track_rich_uses_rich_when_available() -> None:
    enable()
    set_backend("rich")

    rich_track_mock = MagicMock(return_value=iter(["a", "b"]))
    fake_rich_progress = MagicMock()
    fake_rich_progress.track = rich_track_mock

    with patch.dict(sys.modules, {"rich.progress": fake_rich_progress}):
        with patch("rich.progress.track", rich_track_mock):
            result = list(track(["a", "b"], desc="r"))

    assert result == ["a", "b"]


def test_track_total_none_works() -> None:
    disable()
    assert list(track(range(5), total=None)) == list(range(5))


def test_track_empty_iterable() -> None:
    disable()
    assert list(track([])) == []
