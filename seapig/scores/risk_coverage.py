"""Risk-coverage utilities.

Standalone implementation to compute risk-coverage curves.

Functions
---------
calculate_risk_coverage(scores, y_true, y_pred, loss_fn=None, n_points=100)
    Compute risk (loss) at different coverage levels based on ranking by score.

normalize_metric(x)
    Min-max normalize an array-like to the interval [0, 1].
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd


def _default_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default loss function: 0/1 error for discrete predictions, else MAE.

    The function inspects the dtype to choose a reasonable default.
    """
    # If predictions are integers or strings, use error rate
    if np.issubdtype(y_pred.dtype, np.integer) or y_pred.dtype.type is np.str_:
        return float(np.mean(y_true != y_pred))
    # Otherwise use mean absolute error
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_risk_coverage(
    scores: Sequence[float],
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    loss_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_points: int = 100,
) -> dict:
    """Compute a risk-coverage curve.

    Parameters
    ----------
    scores:
        Sequence of confidence/uncertainty scores. Higher means more confident.
    y_true, y_pred:
        Ground truth and predictions. Must be same length as ``scores``.
    loss_fn:
        Optional callable ``(y_true_subset, y_pred_subset) -> float`` to compute
        risk on a subset. If ``None``, a sensible default is chosen.
    n_points:
        Number of coverage points to compute between (1/n, 1].

    Returns
    -------
    dict
        Dictionary with keys ``coverage`` (np.ndarray) and ``risk`` (np.ndarray).
    """
    if loss_fn is None:
        loss_fn = _default_loss

    scores_arr = np.asarray(scores)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if not (len(scores_arr) == len(y_true_arr) == len(y_pred_arr)):
        raise ValueError("`scores`, `y_true` and `y_pred` must have the same length")

    n = len(scores_arr)
    if n == 0:
        return {"coverage": np.array([]), "risk": np.array([])}

    # Sort by descending score (most confident first)
    order = np.argsort(scores_arr)[::-1]
    y_true_sorted = y_true_arr[order]
    y_pred_sorted = y_pred_arr[order]

    # coverage fractions (avoid 0 to have at least one sample)
    ks = np.unique(np.ceil(np.linspace(1, n, min(n, int(n_points)))).astype(int))
    coverage = ks / n

    risks = []
    for k in ks:
        y_true_subset = y_true_sorted[:k]
        y_pred_subset = y_pred_sorted[:k]
        risks.append(loss_fn(y_true_subset, y_pred_subset))

    return {"coverage": np.asarray(coverage), "risk": np.asarray(risks)}


def normalize_metric(x: Iterable[float]) -> np.ndarray:
    """Min-max normalize an array-like to [0, 1].

    If all values are equal, returns zeros.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    mn = arr.min()
    mx = arr.max()
    if mn == mx:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)
