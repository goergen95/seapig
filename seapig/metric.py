"""Selective evaluation metric wrapper."""

from collections.abc import Callable
from typing import final

import torch
from torchmetrics import Metric, MetricCollection

from seapig.risk_coverage import RiskCoverage, risk_coverage


class SelectiveMetric(Metric):
    """Wrap a torchmetrics metric for selective evaluation.

    This wrapper keeps three independent instances of an underlying
    ``Metric`` or ``MetricCollection`` and updates them with (1) all
    samples, (2) only the selected samples, and (3) only the rejected
    samples indicated by a boolean mask inside the provided output dictionary.

    The ``update`` method accepts model predictions, targets, and a selection mask,
    and updates the respective metrics accordingly. The ``compute`` method returns a
    dictionary containing the computed results for the full, selected, and rejected subsets,
    with keys prefixed by "full/", "selected/", and "rejected/" respectively.

    Parameters
    ----------
    base : Metric | MetricCollection
        The metric (or collection) to evaluate. Three deep-copied instances
        are maintained internally for full, selective, and rejected risk computation.

    Notes
    -----
    - If the selection mask is a float/integer tensor, values ``> 0``
      are treated as selected.
    - If no samples are selected or rejected, the respective metric is not updated for
        that call.
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
        device = preds.device
        # Ensure metric collections live on the same device as predictions
        target = target.to(device)
        selected = selected.to(device)
        self.metrics["full"].to(device)
        self.metrics["selected"].to(device)
        self.metrics["rejected"].to(device)

        # Update full with all samples
        self.metrics["full"].update(preds, target)

        # Conditionally update selected/rejected submetrics
        if selected.any():
            self.metrics["selected"].update(preds[selected], target[selected])

        rejected = ~selected
        if rejected.any():
            self.metrics["rejected"].update(preds[rejected], target[rejected])

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute and return results for total, selected, and rejected.

        Returns
        -------
        dict[str, torch.Tensor]
            Prefixed results. Keys are prefixed as ``"full/<name>"``,
            ``"selected/<name>"``, and ``"rejected/<name>"`` for each
            metric in the underlying MetricCollection.
        """
        total_map = self._to_dict(self.metrics["full"], "full")
        selected_map = self._to_dict(self.metrics["selected"], "selected")
        rejected_map = self._to_dict(self.metrics["rejected"], "rejected")
        return {**total_map, **selected_map, **rejected_map}

    @staticmethod
    def _to_dict(m: MetricCollection, prefix: str) -> dict[str, torch.Tensor]:
        """Convert a MetricCollection to a dict with prefixed keys.

        If a metric has not been updated, return 0.0 for that metric.
        """
        out: dict[str, torch.Tensor] = {}
        for name, metric in m.items():
            if metric.update_called:
                out[f"{prefix}/{name}"] = metric.compute()
            else:
                out[f"{prefix}/{name}"] = torch.tensor(0.0)
        return out

    def reset(self) -> None:
        """Reset both internal metric instances."""
        self.metrics["full"].reset()
        self.metrics["selected"].reset()
        self.metrics["rejected"].reset()


@final
class RiskCoverageMetric(Metric):
    """Accumulate scores and residuals, compute a risk‑coverage curve.

    Parameters
    ----------
    risk : {'generalized', 'selective'}, default='generalized'
        Risk definition for the curve.
    n_bins : int, default=100
        Downsampling bins for the curve.
    prediction_key : str, default='predictions'
        Key for model predictions in the outputs dict.
    score_key : str, default='score'
        Key for confidence scores in the outputs dict.
    error_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        Function that reduces (B, ...) preds and targets to per‑sample residuals
        of shape (B,). If None, uses mean absolute error per sample.
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
        """Accumulate scores and residuals from outputs and target."""
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
        """Compute risk-coverage curve metrics."""
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
