"""Logging scaffold for the seapig package.

Provides a package-scoped logger and helpers so downstream applications can
opt into structured logging without modifying library code.

By default the `seapig` logger uses a `logging.NullHandler` (added in
`seapig.__init__`) so no output is produced unless the consuming
application configures logging.

Examples
--------
Enable INFO-level logging to the console:

```python
from seapig.utils.logging import configure_logging
configure_logging(level="INFO")
```

Use the `SEAPIG_LOG_LEVEL` environment variable for the same effect:

```bash
SEAPIG_LOG_LEVEL=INFO python my_script.py
```

Retrieve a child logger within a module:

```python
from seapig.utils.logging import get_logger
logger = get_logger(__name__)
logger.info("message")
```
"""

from __future__ import annotations

import logging
import os

__all__ = ["get_logger", "configure_logging"]

_PACKAGE = "seapig"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger scoped to the seapig package.

    Parameters
    ----------
    name:
        Dotted module name, e.g. `__name__`.  When `None` or when *name*
        does not start with `"seapig"`, the package-level logger
        `"seapig"` is returned.

    Returns
    -------
    logging.Logger
        A :class:`logging.Logger` instance.

    Examples
    --------
    >>> from seapig.utils.logging import get_logger
    >>> logger = get_logger(__name__)
    """
    if name and name.startswith(_PACKAGE):
        return logging.getLogger(name)
    return logging.getLogger(_PACKAGE)


def configure_logging(
    level: str | int = "WARNING", handler: logging.Handler | None = None
) -> None:
    """Configure the seapig package logger.

    Sets the log level and attaches *handler* (or a
    `logging.StreamHandler` writing to *stderr* when `None`) to the
    `"seapig"` logger.  Any previously attached handlers are removed first.

    The `SEAPIG_LOG_LEVEL` environment variable, when set, overrides the
    *level* parameter.

    Parameters
    ----------
    level:
        Minimum log level, e.g. `"INFO"`, `"DEBUG"`, or an integer
        constant such as :data:`logging.INFO`.  Defaults to `"WARNING"`.
    handler:
        A custom `logging.Handler`.  When `None` a
        `logging.StreamHandler` (stderr) with a simple formatter is
        used.

    Examples
    --------
    >>> import logging
    >>> from seapig.utils.logging import configure_logging
    >>> configure_logging(level="INFO")
    """
    env_level = os.environ.get("SEAPIG_LOG_LEVEL", "").strip()
    if env_level:
        level = env_level

    pkg_logger = logging.getLogger(_PACKAGE)
    pkg_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates on repeated calls.
    pkg_logger.handlers.clear()

    if handler is None:
        _handler: logging.Handler = logging.StreamHandler()
        _handler.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        )
    else:
        _handler = handler

    pkg_logger.addHandler(_handler)
