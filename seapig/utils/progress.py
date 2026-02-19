# Copyright (c) seapig Contributors. All rights reserved.
# Licensed under the MIT License.

"""Unified progress-reporting subsystem for seapig.

Provides a single, configurable progress interface that:

- Auto-detects interactive sessions (TTY or IPython) and enables progress
  only in those contexts by default.
- Respects the ``SEAPIG_PROGRESS`` environment variable (``"1"``/``"true"``
  to force-enable, ``"0"``/``"false"`` to force-disable).
- Uses **tqdm** as the default backend.
- Optionally uses **rich** as an alternative backend when it is installed
  and selected via :func:`set_backend` or the ``SEAPIG_PROGRESS_BACKEND``
  environment variable.
- Exposes :func:`track` — a drop-in replacement for ``tqdm(iterable)`` —
  used throughout the rest of the codebase so the backend can be swapped
  and configured centrally.

Examples
--------
Basic iterator usage (mirrors tqdm):

>>> from seapig.utils.progress import track, disable
>>> disable()
>>> list(track(range(3), desc="counting"))
[0, 1, 2]

Programmatic enable/disable:

>>> from seapig.utils.progress import enable, disable, is_enabled
>>> enable()
>>> is_enabled()
True
>>> disable()
>>> is_enabled()
False
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator
from typing import Any

__all__ = [
    "track",
    "enable",
    "disable",
    "reset",
    "is_enabled",
    "set_backend",
    "get_backend",
]

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

#: ``None`` means "auto-detect from session type".
_enabled: bool | None = None

#: Current backend name — ``"tqdm"`` (default) or ``"rich"``.
_backend: str = os.environ.get("SEAPIG_PROGRESS_BACKEND", "tqdm").lower()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_interactive() -> bool:
    """Return ``True`` when running inside an interactive session.

    An interactive session is defined as either:

    * A live IPython / Jupyter kernel (``get_ipython()`` is available and
      returns a non-``None`` object), or
    * A standard terminal whose *stdout* is connected to a TTY.
    """
    # IPython / Jupyter detection
    try:
        import builtins

        get_ipython = getattr(builtins, "get_ipython", None)
        if get_ipython is not None and get_ipython() is not None:
            return True
    except Exception:  # pragma: no cover
        pass

    return sys.stdout.isatty()


# ---------------------------------------------------------------------------
# Public API — enable / disable / backend
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Return whether progress reporting is currently enabled.

    Priority order:

    1. Value set via :func:`enable` / :func:`disable` (programmatic override).
    2. ``SEAPIG_PROGRESS`` environment variable
       (``"1"``, ``"true"``, ``"yes"``, ``"on"`` → enabled;
       ``"0"``, ``"false"``, ``"no"``, ``"off"`` → disabled).
    3. Auto-detection: enabled only when running in an interactive session
       (TTY or IPython/Jupyter).

    Returns
    -------
    bool
        ``True`` if progress bars should be shown, ``False`` otherwise.
    """
    if _enabled is not None:
        return _enabled

    env_val = os.environ.get("SEAPIG_PROGRESS", "").strip().lower()
    if env_val in ("1", "true", "yes", "on"):
        return True
    if env_val in ("0", "false", "no", "off"):
        return False

    return _is_interactive()


def enable() -> None:
    """Globally enable progress reporting.

    This overrides both the ``SEAPIG_PROGRESS`` environment variable and
    the auto-detection logic.  Call :func:`reset` to revert to auto mode.
    """
    global _enabled
    _enabled = True


def disable() -> None:
    """Globally disable progress reporting.

    This overrides both the ``SEAPIG_PROGRESS`` environment variable and
    the auto-detection logic.  Call :func:`reset` to revert to auto mode.
    """
    global _enabled
    _enabled = False


def reset() -> None:
    """Reset progress enable/disable state to automatic detection.

    After calling this function, :func:`is_enabled` will again consult the
    ``SEAPIG_PROGRESS`` environment variable and the TTY/IPython heuristic.
    """
    global _enabled
    _enabled = None


def set_backend(backend: str) -> None:
    """Select the progress-bar backend.

    Parameters
    ----------
    backend:
        ``"tqdm"`` (default) or ``"rich"``.  When ``"rich"`` is requested
        but the ``rich`` package is not installed, :func:`track` silently
        falls back to tqdm.

    Raises
    ------
    ValueError
        If *backend* is not one of the supported values.
    """
    global _backend
    if backend not in ("tqdm", "rich"):
        raise ValueError(
            f"Unknown backend {backend!r}. Supported values: 'tqdm', 'rich'."
        )
    _backend = backend


def get_backend() -> str:
    """Return the name of the currently selected backend.

    Returns
    -------
    str
        Either ``"tqdm"`` or ``"rich"``.
    """
    return _backend


# ---------------------------------------------------------------------------
# Public API — track
# ---------------------------------------------------------------------------


def track[T](
    iterable: Iterable[T],
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    leave: bool = True,
    colour: str | None = None,
    smoothing: float = 0.3,
    **kwargs: Any,
) -> Iterator[T]:
    """Wrap an iterable with a progress bar.

    This is the single entry-point for progress display used throughout
    seapig.  When progress is disabled the iterable is returned as-is with
    zero overhead.

    Parameters
    ----------
    iterable:
        The iterable to wrap.
    total:
        Total number of items (used by tqdm/rich to render a progress bar).
    desc:
        Short description shown to the left of the bar.
    unit:
        Unit label shown after the counter (tqdm only).
    leave:
        Whether to keep the progress bar visible after completion
        (tqdm only; rich always removes it).
    colour:
        Colour of the progress bar as a CSS colour string, e.g. ``"green"``
        (tqdm only).
    smoothing:
        Exponential moving-average smoothing factor for speed estimates
        (tqdm only).
    **kwargs:
        Additional keyword arguments forwarded verbatim to the backend.

    Yields
    ------
    T
        Items from *iterable*, unchanged.

    Examples
    --------
    >>> from seapig.utils.progress import track, disable
    >>> disable()
    >>> list(track([1, 2, 3], desc="items"))
    [1, 2, 3]
    """
    if not is_enabled():
        yield from iterable
        return

    if _backend == "rich":
        try:
            from rich.progress import track as _rich_track

            yield from _rich_track(
                iterable, total=total, description=desc or "Working…", **kwargs
            )
            return
        except ImportError:
            pass  # fall through to tqdm

    from tqdm import tqdm as _tqdm

    yield from _tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        colour=colour,
        smoothing=smoothing,
        **kwargs,
    )
