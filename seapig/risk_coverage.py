"""Risk-coverage utilities for selective prediction.

This module computes risk-coverage curves used to study the trade-off
between coverage (fraction of examples the model accepts) and risk
(error rate) for selective prediction systems.
"""

import torch


class RiskCoverage:
    """Container for risk-coverage results.

    Holds the coverage, score thresholds, empirical and reference risk
    curves, their difference (excess), and AUC metrics.

    Attributes
    ----------
    coverage : torch.Tensor
        Coverage values in `[0, 1]`.
    threshold : torch.Tensor
        Sorted score thresholds used to compute coverage.
    risk : torch.Tensor
        Empirical risk at each coverage level.
    reference : torch.Tensor
        Reference (optimal) risk at each coverage level.
    excess : torch.Tensor
        Excess risk (empirical - reference).
    risk_type : str
        Either `'generalized'` or `'selective'`; see `risk_coverage`.
    auc_empirical : torch.Tensor
        Area under the empirical risk curve (trapezoidal rule).
    auc_reference : torch.Tensor
        Area under the reference risk curve (trapezoidal rule).
    auc_excess : torch.Tensor
        Area under the excess risk curve (trapezoidal rule).

    See Also
    --------
    seapig.risk_coverage.risk_coverage : Function that produces this container.
    seapig.metric.RiskCoverageMetric : Metric wrapper for use with Lightning.
    """

    def __init__(
        self,
        coverage: torch.Tensor,
        threshold: torch.Tensor,
        risk: torch.Tensor,
        reference: torch.Tensor,
        excess: torch.Tensor,
        risk_type: str,
        auc_empirical: torch.Tensor,
        auc_reference: torch.Tensor,
        auc_excess: torch.Tensor,
    ) -> None:
        """Create a `RiskCoverage` container.

        All parameters correspond directly to the attributes of the same name.
        Typically constructed by `risk_coverage` rather than directly.
        """
        self.coverage = coverage
        self.threshold = threshold
        self.risk = risk
        self.reference = reference
        self.excess = excess
        self.risk_type = risk_type
        self.auc_empirical = auc_empirical
        self.auc_reference = auc_reference
        self.auc_excess = auc_excess

    def __repr__(self) -> str:
        """Short representation including AUCs and number of points."""
        return (
            f"RiskCoverage(risk_type='{self.risk_type}', "
            f"n_points={len(self.coverage)}, "
            f"auc_empirical={self.auc_empirical:.4f}, "
            f"auc_reference={self.auc_reference:.4f}, "
            f"auc_excess={self.auc_excess:.4f})"
        )

    def plot(
        self,
        empirical: bool = True,
        reference: bool = True,
        excess: bool = True,
        digits: int = 4,
    ) -> object:
        """Return a matplotlib Figure with the requested curves.

        Parameters
        ----------
        empirical, reference, excess : bool
            Whether to include each curve in the plot.
        digits : int
            Number of decimal places to show for AUC values in the legend.

        Returns
        -------
        matplotlib.figure.Figure
            A figure containing the plotted curves.

        Raises
        ------
        ImportError
            If `matplotlib` is not installed.
        ValueError
            If all curve flags are False.

        Examples
        --------
        ```python
        fig = rc.plot(empirical=True, reference=False)
        ```
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            msg = "matplotlib is required for plotting. Install it with: pip install matplotlib"
            raise ImportError(msg) from e

        if not any([empirical, reference, excess]):
            msg = "At least one of empirical, reference, or excess must be True"
            raise ValueError(msg)

        # Convert to numpy for plotting
        coverage_np = self.coverage.cpu().numpy()
        risk_np = self.risk.cpu().numpy()
        reference_np = self.reference.cpu().numpy()
        excess_np = self.excess.cpu().numpy()

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot selected curves
        colors = {
            "empirical": "#1B9E77",
            "reference": "#7570B3",
            "excess": "#D95F02",
        }
        if empirical:
            ax.plot(
                coverage_np,
                risk_np,
                label=f"empirical: {self.auc_empirical:.{digits}f}",
                color=colors["empirical"],
                linewidth=2,
            )
        if reference:
            ax.plot(
                coverage_np,
                reference_np,
                label=f"reference: {self.auc_reference:.{digits}f}",
                color=colors["reference"],
                linewidth=2,
            )
        if excess:
            ax.plot(
                coverage_np,
                excess_np,
                label=f"excess: {self.auc_excess:.{digits}f}",
                color=colors["excess"],
                linewidth=2,
            )

        # Formatting
        ax.set_xlabel("Coverage", fontsize=12)
        ax.set_ylabel("Risk", fontsize=12)
        ax.set_title(f"{self.risk_type.capitalize()} Risk", fontsize=14)
        ax.legend(title="AUC", loc="best", frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

        plt.tight_layout()
        return fig


def risk_coverage(
    score: torch.Tensor,
    residuals: torch.Tensor,
    risk: str = "generalized",
    n_bins: int = 100,
) -> RiskCoverage:
    """Compute risk-coverage curves and AUCs for a selective predictor.

    Given a score (confidence) and corresponding residuals (prediction
    errors), this function returns a `RiskCoverage` object containing:

    - empirical risk curve (model scores)
    - reference risk curve (using residuals as an ideal score)
    - excess risk (empirical - reference)
    - AUC values for each curve

    Parameters
    ----------
    score : `torch.Tensor`
        Confidence scores. Lower values indicate higher confidence.
        Must have the same length as `residuals`.
    residuals : `torch.Tensor`
        Prediction residuals (errors). Lower values indicate better
        predictions. Same length as `score`.
    risk : {'generalized', 'selective'}, default 'generalized'
        - `'generalized'`: joint probability of failure and acceptance
        - `'selective'`: failure probability conditional on acceptance
    n_bins : int
        If the curve has more than `n_bins` points, it will be downsampled
        to at most `n_bins` points for reporting and plotting.

    Returns
    -------
    `RiskCoverage`
        Container with curves and AUCs.

    Raises
    ------
    TypeError
        If `score` or `residuals` are not `torch.Tensor`.
    ValueError
        If lengths differ or `risk` is not one of the allowed values.

    Examples
    --------
    ```python
    import torch
    score = torch.rand(100)
    residuals = torch.rand(100)
    rc = risk_coverage(score, residuals, risk='generalized')
    ```
    """
    # Validate inputs
    if not isinstance(score, torch.Tensor):
        msg = "score must be a torch.Tensor"
        raise TypeError(msg)
    if not isinstance(residuals, torch.Tensor):
        msg = "residuals must be a torch.Tensor"
        raise TypeError(msg)
    if len(score) != len(residuals):
        msg = f"score and residuals must have the same length, got {len(score)} and {len(residuals)}"
        raise ValueError(msg)
    if risk not in ["generalized", "selective"]:
        msg = f"risk must be 'generalized' or 'selective', got '{risk}'"
        raise ValueError(msg)

    assert risk in ["generalized", "selective"], (
        "risk must be 'generalized' or 'selective'"
    )
    device = score.device
    residuals.to(device=device)

    # Calculate empirical risk-coverage curve
    coverage_emp, threshold_emp, risk_emp = _rc_curve(score, residuals, risk)

    # Calculate reference risk-coverage curve (using residuals as scores)
    coverage_ref, _, risk_ref = _rc_curve(residuals, residuals, risk)

    # Downsample if needed
    if len(coverage_emp) > n_bins:
        (coverage_emp, threshold_emp, risk_emp, risk_ref) = _downsample_curves(
            coverage_emp, threshold_emp, risk_emp, risk_ref, n_bins
        )

    # Calculate excess risk
    excess = risk_emp - risk_ref
    # Calculate AUC using trapezoidal rule
    auc_emp = _trapz(coverage_emp, risk_emp)
    auc_ref = _trapz(coverage_emp, risk_ref)
    auc_exs = _trapz(coverage_emp, excess)

    return RiskCoverage(
        coverage=coverage_emp,
        threshold=threshold_emp,
        risk=risk_emp,
        reference=risk_ref,
        excess=excess,
        risk_type=risk,
        auc_empirical=auc_emp,
        auc_reference=auc_ref,
        auc_excess=auc_exs,
    )


def _rc_curve(
    score: torch.Tensor, residuals: torch.Tensor, risk_type: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a single risk-coverage curve.

    The function sorts examples by `score` (descending), computes coverage
    as the fraction of accepted examples, and returns the cumulative risk.

    Returns (coverage, sorted_score, risk) as `torch.Tensor` objects.
    """
    # Sort by score (descending, so we reject highest scores first)
    device = score.device
    order = torch.argsort(score, descending=True)
    score_sorted = score[order]
    residuals_sorted = residuals[order]

    # Calculate coverage
    n = len(score)
    coverage = torch.arange(1, n + 1, dtype=torch.float32, device=device) / n

    # Calculate cumulative risk
    cumsum_residuals = torch.cumsum(residuals_sorted, dim=0)
    risk = cumsum_residuals / n

    # For selective risk, divide by coverage
    if risk_type == "selective":
        risk = risk / coverage

    return (
        coverage.to(device=device),
        score_sorted.to(device=device),
        risk.to(device=device),
    )


def _downsample_curves(
    coverage: torch.Tensor,
    threshold: torch.Tensor,
    risk_emp: torch.Tensor,
    risk_ref: torch.Tensor,
    n_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce curve points to at most `n_bins` by aggregating within bins.

    Each output bin contains the maximum coverage, threshold and risks
    among points that fall into the bin. This preserves the upper envelope
    of the curves for reporting and plotting.
    """
    # Create bins
    device = coverage.device
    bins = torch.linspace(0, 1, n_bins + 1, device=device)
    bin_indices = torch.searchsorted(bins, coverage, right=False)
    bin_indices = torch.clamp(bin_indices - 1, min=0, max=n_bins - 1)

    # Aggregate by taking max in each bin
    coverage_out = torch.zeros(n_bins, device=device)
    threshold_out = torch.zeros(n_bins, device=device)
    risk_emp_out = torch.zeros(n_bins, device=device)
    risk_ref_out = torch.zeros(n_bins, device=device)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.any():
            coverage_out[i] = coverage[mask].max()
            threshold_out[i] = threshold[mask].max()
            risk_emp_out[i] = risk_emp[mask].max()
            risk_ref_out[i] = risk_ref[mask].max()

    return coverage_out, threshold_out, risk_emp_out, risk_ref_out


def _trapz(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Trapezoidal integration of y with respect to x.

    Both `x` and `y` should be 1-D tensors of the same length.
    """
    dx = x[1:] - x[:-1]
    avg_y = (y[:-1] + y[1:]) / 2
    return (dx * avg_y).sum()
