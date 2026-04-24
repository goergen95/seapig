"""Metrics helpers for selective evaluation.

This module provides two utilities used during evaluation when a model
can abstain or return confidence scores:

- `SelectiveMetric`: wraps a `torchmetrics.Metric` (or a
  `torchmetrics.MetricCollection`) and tracks the metric on three
  disjoint subsets: all samples (full), the samples marked as selected,
  and the samples marked as rejected.
- `RiskCoverageMetric`: accumulates per-sample scores and residuals to
  compute a risk-coverage curve using :module:`seapig.risk_coverage`.
"""

from collections.abc import Callable
from typing import final

import torch
from torchmetrics import Metric, MetricCollection

from seapig.risk_coverage import RiskCoverage, risk_coverage


class SelectiveMetric(Metric):
    """Evaluate a metric on full, selected, and rejected subsets.

    Wraps a `torchmetrics.Metric` or `torchmetrics.MetricCollection` and
    keeps three independent copies that are updated separately:

    - `"full"`: all samples passed to `update`.
    - `"selected"`: samples where the provided selection mask is true.
    - `"rejected"`: samples where the selection mask is false.

    The `compute` result is a flat `dict`[`str`, `torch.Tensor`] where
    each underlying metric name is prefixed with ``full/``, ``selected/``,
    or ``rejected/``. If a submetric was never updated, its value is a
    zero `torch.Tensor`.

    Parameters
    ----------
    base : torchmetrics.Metric | torchmetrics.MetricCollection
        Metric (or collection) to wrap. Internally the object is deep-
        copied three times so each subset is tracked independently.

    Notes
    -----
    - The selection mask may be boolean or numeric; numeric values ``> 0``
      are treated as selected.
    - Calls that contain no selected (or no rejected) rows do not update
      the corresponding internal metric for that call.

    Example
    -------
    ```python
    from torchmetrics import Accuracy
    base = Accuracy(task="binary")
    m = SelectiveMetric(base)
    preds = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    target = torch.tensor([0, 1])
    mask = torch.tensor([1, 0], dtype=torch.bool)
    m.update(preds, target, mask)
    results = m.compute()
    # results contains keys like 'full/accuracy', 'selected/accuracy', ...
    ```
    """

    def __init__(self, base: Metric | MetricCollection) -> None:
        super().__init__()
        import copy as _copy

        if isinstance(base, Metric):
            base = MetricCollection(base)

        self.metrics = {
            "full": _copy.deepcopy(base),
            "selected": _copy.deepcopy(base),
            "rejected": _copy.deepcopy(base),
        }

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, selected: torch.Tensor
    ) -> None:
        """Update full, selected, and rejected metrics.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions of shape (B, ...).
        target : torch.Tensor
            Target tensor of shape (B, ...).
        selected : torch.Tensor
            Boolean or binary selection mask of shape (B,). Values > 0 are treated as selected.
        """
        assert preds.shape[0] == target.shape[0] == selected.shape[0], (
            "Batch size of predictions, target, and selection mask must match."
        )

        # Update full with all samples
        self.metrics["full"].update(preds, target)

        # Conditionally update selected/rejected submetrics
        if selected.any():
            self.metrics["selected"].update(preds[selected], target[selected])

        rejected = ~selected
        if rejected.any():
            self.metrics["rejected"].update(preds[rejected], target[rejected])

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute metrics and return a dict with prefixed keys.

        Returns a dictionary where each key is `<scope>/<metric_name>`
        (scope is one of `full`, `selected`, `rejected`). Values are
        `torch.Tensor`s. If a metric instance was never updated, its value is a
        scalar zero tensor.
        """
        total_map = self._to_dict(self.metrics["full"], "full")
        selected_map = self._to_dict(self.metrics["selected"], "selected")
        rejected_map = self._to_dict(self.metrics["rejected"], "rejected")
        return {**total_map, **selected_map, **rejected_map}

    @staticmethod
    def _to_dict(m: MetricCollection, prefix: str) -> dict[str, torch.Tensor]:
        """Convert a `torchmetrics.MetricCollection` to a prefixed dict.

        If a metric has not been updated, returns a zero tensor for that
        metric so the returned mapping always contains all metric names.
        """
        out: dict[str, torch.Tensor] = {}
        for name, metric in m.items():
            if metric.update_called:
                out[f"{prefix}/{name}"] = metric.compute()
            else:
                out[f"{prefix}/{name}"] = torch.tensor(
                    0.0, device=metric.device
                )
        return out

    def reset(self) -> None:
        """Reset the internal metric instances."""
        self.metrics["full"].reset()
        self.metrics["selected"].reset()
        self.metrics["rejected"].reset()


@final
class RiskCoverageMetric(Metric):
    """Build a risk-coverage curve from scores and per-sample errors.

    Collects per-sample `scores` and per-sample `residuals` across
    multiple `update` calls and computes summary area-under-curve
    values using `seapig.risk_coverage.risk_coverage`.

    Parameters
    ----------
    risk : {'generalized', 'selective'}, default 'generalized'
        Which risk definition to use when computing the curve. Must be
        either `'generalized'` or `'selective'`.
    n_bins : int, default 100
        Number of bins used to downsample the curve when computing AUC
        summaries.
    error_fn : Callable | None
        Function `(preds, target) -> residuals` that reduces model
        predictions and targets to a 1-D tensor of per-sample residuals.
        If `None` the default is per-sample mean absolute error.

    The `compute` method returns three tensors: `rc/auc_empirical`,
    `rc/auc_reference`, and `rc/auc_excess`. The last computed
    complete curve object (`seapig.risk_coverage.RiskCoverage`) is
    available via `get_curve`.
    """

    full_state_update: bool = False
    scores: torch.Tensor
    residuals: torch.Tensor

    def __init__(
        self,
        risk: str = "generalized",
        n_bins: int = 100,
        error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
    ) -> None:
        super().__init__()
        assert risk in ["generalized", "selective"], (
            "RiskCoverageMetric risk must be 'generalized' or 'selective'."
        )
        self.risk = risk
        self.n_bins = n_bins
        self._error_fn = error_fn

        # Metric states (concatenate across steps)
        self.add_state(
            "scores",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="cat",
        )
        self.add_state(
            "residuals",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="cat",
        )

        # Last computed curve (non‑tensor; kept for retrieval only)
        self._last_curve: RiskCoverage | None = None

    @staticmethod
    def _default_error_fn(
        preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # Per‑sample mean absolute error across non‑batch dims.
        # Broadcast target to preds if shapes allow.
        residual = torch.abs(preds - target)
        if residual.ndim == 1:
            return residual
        reduce_dims = tuple(range(1, residual.ndim))
        return residual.mean(dim=reduce_dims)

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, scores: torch.Tensor
    ) -> None:
        """Store scores and residuals for later curve computation.

        Parameters
        ----------
        preds, target : torch.Tensor
            Model outputs and targets. These are passed to `error_fn` to
            compute per-sample residuals.
        scores : torch.Tensor
            Per-sample confidence scores (lower values indicate higher confidence).
        """
        device = preds.device
        scores = scores.to(device)
        target = target.to(device)

        err_fn = self._error_fn or self._default_error_fn
        residuals = err_fn(preds, target).to(device)

        # Concatenate into states (keep dtype/device consistent)
        if self.scores.numel() == 0:
            self.scores = scores
        else:
            self.scores = torch.cat([self.scores, scores], dim=0)

        if self.residuals.numel() == 0:
            self.residuals = residuals
        else:
            self.residuals = torch.cat([self.residuals, residuals], dim=0)

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute AUC summaries for the accumulated risk--coverage curve.

        Returns a dict with keys `rc/auc_empirical`, `rc/auc_reference`,
        and `rc/auc_excess`. If no data has been accumulated an all-zero
        mapping is returned on the correct device.
        """
        if self.scores.numel() == 0:
            # No data yet; return zeros on the registered device
            zero = torch.tensor(0.0, device=self.scores.device)
            return {
                "rc/auc_empirical": zero,
                "rc/auc_reference": zero,
                "rc/auc_excess": zero,
            }

        rc = risk_coverage(
            score=self.scores,
            residuals=self.residuals,
            risk=self.risk,
            n_bins=self.n_bins,
        )
        self._last_curve = rc
        device = self.scores.device
        return {
            "rc/auc_empirical": torch.as_tensor(
                rc.auc_empirical, device=device
            ),
            "rc/auc_reference": torch.as_tensor(
                rc.auc_reference, device=device
            ),
            "rc/auc_excess": torch.as_tensor(rc.auc_excess, device=device),
        }

    def get_curve(self) -> RiskCoverage | None:
        """Return the last computed RiskCoverage object (or None if not computed)."""
        return self._last_curve

    def reset(self) -> None:
        """Reset the accumulated scores and residuals."""
        self.scores = torch.tensor(
            [], dtype=torch.float32, device=self.scores.device
        )
        self.residuals = torch.tensor(
            [], dtype=torch.float32, device=self.residuals.device
        )
        self._last_curve = None
