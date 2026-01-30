"""Risk-Coverage Curve Implementation.

This module implements the risk-coverage curve analysis for selective
prediction systems, based on the R implementation in the CAST package.
"""

import numpy as np
import torch


class RiskCoverage:
    """Risk-Coverage Curve for Selective Prediction.

    The risk-coverage curve describes the performance profile of a selective
    prediction model by showing the trade-off between coverage (fraction of
    samples accepted) and risk (error rate) at different confidence thresholds.

    Attributes
    ----------
    coverage : torch.Tensor
        Coverage values (fraction of samples accepted).
    threshold : torch.Tensor
        Confidence score thresholds.
    risk : torch.Tensor
        Empirical risk values at each coverage level.
    reference : torch.Tensor
        Reference (optimal) risk values at each coverage level.
    excess : torch.Tensor
        Excess risk (empirical - reference) at each coverage level.
    risk_type : str
        Type of risk calculation ('generalized' or 'selective').
    auc_empirical : float
        Area under the empirical risk-coverage curve.
    auc_reference : float
        Area under the reference risk-coverage curve.
    auc_excess : float
        Area under the excess risk-coverage curve (E-AURC).

    Examples
    --------
    ```python
    import torch
    from seapig.risk_coverage import risk_coverage

    # Generate example data
    score = torch.rand(100)  # Lower is more confident
    residuals = torch.rand(100)  # Prediction errors

    # Calculate risk-coverage curve
    rc = risk_coverage(score, residuals, risk="generalized")

    # Access metrics
    print(f"E-AURC: {rc.auc_excess:.4f}")
    ```
    """

    def __init__(
        self,
        coverage: torch.Tensor,
        threshold: torch.Tensor,
        risk: torch.Tensor,
        reference: torch.Tensor,
        excess: torch.Tensor,
        risk_type: str,
        auc_empirical: float,
        auc_reference: float,
        auc_excess: float,
    ) -> None:
        """Initialize RiskCoverage object.

        Parameters
        ----------
        coverage : torch.Tensor
            Coverage values.
        threshold : torch.Tensor
            Threshold values.
        risk : torch.Tensor
            Empirical risk values.
        reference : torch.Tensor
            Reference risk values.
        excess : torch.Tensor
            Excess risk values.
        risk_type : str
            Type of risk ('generalized' or 'selective').
        auc_empirical : float
            AUC of empirical curve.
        auc_reference : float
            AUC of reference curve.
        auc_excess : float
            AUC of excess curve.
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
        """Return string representation."""
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
        """Plot the risk-coverage curves.

        Parameters
        ----------
        empirical : bool, default=True
            Whether to plot the empirical risk-coverage curve.
        reference : bool, default=True
            Whether to plot the reference risk-coverage curve.
        excess : bool, default=True
            Whether to plot the excess risk-coverage curve.
        digits : int, default=4
            Number of digits for AUC values in the legend.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        ValueError
            If all of ``empirical``, ``reference``, and ``excess`` are False.

        Examples
        --------
        ```python
        import torch
        from seapig.risk_coverage import risk_coverage

        score = torch.rand(100)
        residuals = torch.rand(100)
        rc = risk_coverage(score, residuals)

        # Plot all curves
        fig = rc.plot()

        # Plot only empirical curve
        fig = rc.plot(reference=False, excess=False)
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
    score: torch.Tensor | np.ndarray,
    residuals: torch.Tensor | np.ndarray,
    risk: str = "generalized",
    n_bins: int = 100,
) -> RiskCoverage:
    """Calculate the risk-coverage curve.

    Given a confidence score and prediction residuals, this function computes
    the risk-coverage curve that describes the trade-off between coverage
    (fraction of accepted samples) and risk (error rate) in a selective
    prediction system.

    The empirical risk-coverage curve is compared to a reference curve based
    on an optimal confidence score (the residuals themselves). The difference
    between these curves is the excess risk.

    Parameters
    ----------
    score : torch.Tensor or np.ndarray
        Confidence scores. Lower values indicate higher confidence.
        Must have the same length as residuals.
    residuals : torch.Tensor or np.ndarray
        Prediction residuals (errors). Lower values indicate better predictions.
        Must have the same length as score.
    risk : {'generalized', 'selective'}, default='generalized'
        Type of risk to calculate:
        - 'generalized': Joint probability of prediction failure and acceptance
        - 'selective': Conditional probability of failure given acceptance
    n_bins : int, default=100
        Number of coverage bins to use for downsampling. If the number of
        samples is less than n_bins, no downsampling is performed.

    Returns
    -------
    RiskCoverage
        Object containing the risk-coverage curves and AUC metrics.

    Raises
    ------
    ValueError
        If score and residuals have different lengths or if risk type is invalid.
    TypeError
        If score or residuals are not torch.Tensor or np.ndarray.

    Examples
    --------
    ```python
    import torch
    from seapig.risk_coverage import risk_coverage

    # Simulated data
    score = torch.rand(100)
    residuals = torch.rand(100)

    # Calculate generalized risk-coverage curve
    rc_gen = risk_coverage(score, residuals, risk="generalized")
    print(rc_gen)

    # Calculate selective risk-coverage curve
    rc_sel = risk_coverage(score, residuals, risk="selective")
    print(f"Selective E-AURC: {rc_sel.auc_excess:.4f}")
    ```
    """
    # Validate inputs
    if not isinstance(score, torch.Tensor):
        msg = "score must be a torch.Tensor or np.ndarray"
        raise TypeError(msg)
    if not isinstance(residuals, torch.Tensor):
        msg = "residuals must be a torch.Tensor or np.ndarray"
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
        auc_empirical=float(auc_emp),
        auc_reference=float(auc_ref),
        auc_excess=float(auc_exs),
    )


def _rc_curve(
    score: torch.Tensor, residuals: torch.Tensor, risk_type: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate a single risk-coverage curve.

    Parameters
    ----------
    score : torch.Tensor
        Confidence scores.
    residuals : torch.Tensor
        Prediction residuals.
    risk_type : str
        Type of risk ('generalized' or 'selective').

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Coverage, threshold, and risk tensors.
    """
    # Sort by score (descending, so we reject highest scores first)
    order = torch.argsort(score, descending=True)
    score_sorted = score[order]
    residuals_sorted = residuals[order]

    # Calculate coverage
    n = len(score)
    coverage = torch.arange(1, n + 1, dtype=torch.float32) / n

    # Calculate cumulative risk
    cumsum_residuals = torch.cumsum(residuals_sorted, dim=0)
    risk = cumsum_residuals / n

    # For selective risk, divide by coverage
    if risk_type == "selective":
        risk = risk / coverage

    return coverage, score_sorted, risk


def _downsample_curves(
    coverage: torch.Tensor,
    threshold: torch.Tensor,
    risk_emp: torch.Tensor,
    risk_ref: torch.Tensor,
    n_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Downsample risk-coverage curves to n_bins.

    Parameters
    ----------
    coverage : torch.Tensor
        Coverage values.
    threshold : torch.Tensor
        Threshold values.
    risk_emp : torch.Tensor
        Empirical risk values.
    risk_ref : torch.Tensor
        Reference risk values.
    n_bins : int
        Number of bins.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Downsampled coverage, threshold, empirical risk, and reference risk.
    """
    # Create bins
    bins = torch.linspace(0, 1, n_bins + 1)
    bin_indices = torch.searchsorted(bins, coverage, right=False)
    bin_indices = torch.clamp(bin_indices - 1, min=0, max=n_bins - 1)

    # Aggregate by taking max in each bin
    coverage_out = torch.zeros(n_bins)
    threshold_out = torch.zeros(n_bins)
    risk_emp_out = torch.zeros(n_bins)
    risk_ref_out = torch.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.any():
            coverage_out[i] = coverage[mask].max()
            threshold_out[i] = threshold[mask].max()
            risk_emp_out[i] = risk_emp[mask].max()
            risk_ref_out[i] = risk_ref[mask].max()

    return coverage_out, threshold_out, risk_emp_out, risk_ref_out


def _trapz(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Trapezoidal integration.

    Parameters
    ----------
    x : torch.Tensor
        X values.
    y : torch.Tensor
        Y values.

    Returns
    -------
    torch.Tensor
        Integral using trapezoidal rule.
    """
    dx = x[1:] - x[:-1]
    avg_y = (y[:-1] + y[1:]) / 2
    return (dx * avg_y).sum()
