"""Index handler package — registry, factory, and public exports."""

from __future__ import annotations

from typing import Any

from seapig.scores.index.adapters.nmslib_handler import NmslibHandler
from seapig.scores.index.handler import IndexHandler

_REGISTRY: dict[str, type[IndexHandler]] = {}


def register_index_adapter(name: str, cls: type[IndexHandler]) -> None:
    """Register an :class:`IndexHandler` adapter class under *name*.

    Parameters
    ----------
    name:
        Short identifier used to look up the adapter (e.g. ``"nmslib"``).
    cls:
        The :class:`IndexHandler` subclass to register.
    """
    _REGISTRY[name] = cls


def get_index_adapter(name: str, **kwargs: Any) -> IndexHandler:
    """Instantiate and return a registered :class:`IndexHandler` adapter.

    Parameters
    ----------
    name:
        Adapter name previously passed to :func:`register_index_adapter`.
    **kwargs:
        Keyword arguments forwarded to the adapter constructor.

    Returns
    -------
    IndexHandler
        A freshly constructed adapter instance.

    Raises
    ------
    KeyError
        If *name* has not been registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown index adapter: {name!r}. "
            f"Available adapters: {list(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


# Register built-in adapters.
register_index_adapter("nmslib", NmslibHandler)

__all__ = [
    "IndexHandler",
    "NmslibHandler",
    "get_index_adapter",
    "register_index_adapter",
]
