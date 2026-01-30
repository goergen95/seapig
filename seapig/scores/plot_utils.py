"""Plotting utilities using plotnine (ggplot2-like API).

Functions
---------
plot_risk_coverage(coverage, risk)
    Returns a plotnine plot of risk vs coverage.

plot_density(metric, threshold=None)
    Returns a density plot of a metric with an optional threshold line.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    geom_line,
    geom_point,
    geom_density,
    geom_vline,
    ggplot,
    labs,
    theme_minimal,
)


def plot_risk_coverage(coverage: Iterable[float], risk: Iterable[float]):
    """Create a risk-coverage plot (risk vs coverage).

    Returns a `plotnine` ``ggplot`` object.
    """
    cov = np.asarray(coverage)
    risk = np.asarray(risk)
    df = pd.DataFrame({"coverage": cov, "risk": risk})
    p = (
        ggplot(df, aes(x="coverage", y="risk"))
        + geom_line()
        + geom_point()
        + labs(x="Coverage", y="Risk", title="Risk-Coverage")
        + theme_minimal()
    )
    return p


def plot_density(metric: Iterable[float], threshold: float | None = None):
    """Create a density plot for ``metric`` with an optional threshold.

    Returns a `plotnine` ``ggplot`` object.
    """
    arr = np.asarray(metric)
    if arr.size == 0:
        df = pd.DataFrame({"metric": []})
    else:
        df = pd.DataFrame({"metric": arr})

    p = ggplot(df, aes(x="metric")) + geom_density(fill="#4c72b0", alpha=0.4) + labs(
        x="Metric", title="Density"
    ) + theme_minimal()

    if threshold is not None:
        p = p + geom_vline(xintercept=threshold, linetype="dashed", color="#d62728")

    return p
