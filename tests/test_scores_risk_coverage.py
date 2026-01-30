import numpy as np
import pandas as pd
from plotnine.ggplot import ggplot

from seapig.scores.risk_coverage import (
    calculate_risk_coverage,
    normalize_metric,
)
from seapig.scores.plot_utils import plot_risk_coverage, plot_density


def test_calculate_risk_coverage_perfect_predictions():
    y_true = np.array([0, 1, 1, 0])
    y_pred = y_true.copy()
    scores = np.array([0.1, 0.9, 0.8, 0.2])

    out = calculate_risk_coverage(scores, y_true, y_pred, n_points=4)
    assert "coverage" in out and "risk" in out
    # With perfect predictions, risk should be zero for all coverage levels
    assert np.allclose(out["risk"], 0.0)


def test_normalize_metric():
    arr = [5.0, 6.0, 10.0]
    norm = normalize_metric(arr)
    assert np.allclose(norm, [0.0, 0.25, 1.0])

    # all equal values -> zeros
    eq = [1.0, 1.0]
    assert np.allclose(normalize_metric(eq), [0.0, 0.0])


def test_plot_functions_return_type():
    cov = np.linspace(0.25, 1.0, 4)
    risk = np.zeros_like(cov)
    p = plot_risk_coverage(cov, risk)
    assert isinstance(p, ggplot)

    dens = plot_density([0.1, 0.2, 0.3], threshold=0.2)
    assert isinstance(dens, ggplot)
