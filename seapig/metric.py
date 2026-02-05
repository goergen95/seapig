"""Selective evaluation metric wrapper."""

from collections.abc import Callable, Iterable
from typing import final

import torch
from torchmetrics import Metric, MetricCollection

from seapig.risk_coverage import RiskCoverage, risk_coverage


class SelectiveMetric(Metric):  # type: ignore[misc]
    """Wrap a torchmetrics metric for selective evaluation.

    This wrapper keeps three independent instances of an underlying
    ``Metric`` or ``MetricCollection`` and updates them with (1) all
    samples, (2) only the selected samples, and (3) only the rejected
    samples indicated by a boolean mask inside the provided output dictionary.

    The ``update`` method accepts a mapping like the output of
    :meth:`SelectiveInferenceTask.forward`, containing:

    - model predictions under ``prediction_key`` (shape ``(B, ...)``, default: ``"predictions"``)
    - a selection mask under ``selection_key`` (default: ``"selected"``)

    And the target tensor of shape ``(B, ...)``.

    Parameters
    ----------
    base : Metric | MetricCollection
        The metric (or collection) to evaluate. Three deep-copied instances
        are maintained internally for full, selective, and rejected risk computation.
    prediction_key : str, optional
        Key for predictions inside the outputs dict, by default "predictions".
    selection_key : str, optional
        Key for the selection mask inside the outputs dict, by default
        "selected".

    Notes
    -----
    - If the selection mask is a float/integer tensor, values ``> 0``
      are treated as selected.
    - If no samples are selected or rejected, the respective metric is not updated for
        that call.
    """

    def __init__(
        self,
        base: Metric | MetricCollection,
        prediction_key: str = "predictions",
        selection_key: str = "selected",
    ) -> None:
        super().__init__()
        import copy as _copy

        # Three independent metric instances (full, selection, rejection)
        self._full: Metric | MetricCollection = _copy.deepcopy(base)
        self._selected: Metric | MetricCollection = _copy.deepcopy(base)
        self._rejected: Metric | MetricCollection = _copy.deepcopy(base)

        self.prediction_key = prediction_key
        self.selection_key = selection_key

    def items(self) -> Iterable[tuple[str, torch.Tensor]]:
        """Return items of the computed results."""
        return self.compute().items()

    def keys(self) -> Iterable[str]:
        """Return keys of the computed results."""
        return self.compute().keys()

    def values(self) -> Iterable[torch.Tensor]:
        """Return values of the computed results."""
        return self.compute().values()

    def update(
        self, outputs: dict[str, torch.Tensor], target: torch.Tensor
    ) -> None:
        """Update full, selected, and rejected metrics.

        Parameters
        ----------
        outputs : dict[str, torch.Tensor]
            Mapping containing predictions, a target tensor, and selection
            mask.
        """
        preds = outputs[self.prediction_key]
        mask = outputs[self.selection_key]

        device = preds.device
        # Ensure all metrics are on the right device
        self._full = self._full.to(device)
        self._selected = self._selected.to(device)
        self._rejected = self._rejected.to(device)

        # Update total
        self._full.update(preds, target)

        # Update selected subset (if any)
        if mask.ndim != 1:
            mask = mask.view(-1)
        if mask.any():
            if mask.dtype is not torch.bool:
                mask = mask == 1
            preds_sel = preds[mask]
            target_sel = target[mask]
            self._selected.update(preds_sel, target_sel)

        # Update rejected subset (if any)
        rejected_mask = ~mask
        if rejected_mask.any():
            preds_rej = preds[rejected_mask]
            target_rej = target[rejected_mask]
            self._rejected.update(preds_rej, target_rej)

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute and return results for total, selected, and rejected.

        Returns
        -------
        dict[str, torch.Tensor]
            Prefixed results. For a single metric, keys are
            ``"full_risk"``, ``"selective_risk"``, and ``"rejected_risk"``.
            For a collection, keys are prefixed as ``"full/<name>"``,
            ``"selected/<name>"``, and ``"rejected/<name>"``.
        """

        def _to_mapping(
            m: Metric | MetricCollection, prefix: str
        ) -> dict[str, torch.Tensor]:
            out = m.compute()
            if isinstance(out, dict):
                return {f"{prefix}/{k}": v for k, v in out.items()}
            return {f"{prefix}_risk": out}

        total_map = _to_mapping(self._full, "full")
        selected_map = _to_mapping(self._selected, "selected")
        rejected_map = _to_mapping(self._rejected, "rejected")
        return {**total_map, **selected_map, **rejected_map}

        def _to_mapping(
            m: Metric | MetricCollection, prefix: str
        ) -> dict[str, torch.Tensor]:
            out = m.compute()
            if isinstance(out, dict):
                return {f"{prefix}/{k}": v for k, v in out.items()}
            return {"full_risk" if prefix == "full" else "selective_risk": out}

        total_map = _to_mapping(self._full, "full")
        selected_map = _to_mapping(self._selected, "selected")
        rejected_map = _to_mapping(self._rejected, "rejected")
        return {**total_map, **selected_map}

    def reset(self) -> None:
        """Reset both internal metric instances."""
        self._full.reset()
        self._selected.reset()
        self._rejected.reset()


@final
class RiskCoverageMetric(Metric):  # type: ignore[misc]
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

    def __init__(
        self,
        risk: str = "generalized",
        n_bins: int = 100,
        prediction_key: str = "predictions",
        score_key: str = "score",
        error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
    ) -> None:
        super().__init__()
        assert risk in ["generalized", "selective"], (
            "RiskCoverageMetric risk must be 'generalized' or 'selective'."
        )
        self.risk = risk
        self.n_bins = n_bins
        self.prediction_key = prediction_key
        self.score_key = score_key
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

    def items(self) -> Iterable[tuple[str, torch.Tensor]]:
        """Return items of the computed results."""
        return self.compute().items()

    def keys(self) -> Iterable[str]:
        """Return keys of the computed results."""
        return self.compute().keys()

    def values(self) -> Iterable[torch.Tensor]:
        """Return values of the computed results."""
        return self.compute().values()

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
        self, outputs: dict[str, torch.Tensor], target: torch.Tensor
    ) -> None:
        """Accumulate scores and residuals from outputs and target."""
        preds = outputs.get(self.prediction_key)
        assert isinstance(preds, torch.Tensor), (
            f"RiskCoverageMetric requires '{self.prediction_key}' in outputs."
        )
        scores = outputs.get(self.score_key)
        assert isinstance(scores, torch.Tensor), (
            f"RiskCoverageMetric requires '{self.score_key}' in outputs."
        )

        device = preds.device
        scores = scores.to(device)
        target = target.to(device)

        err_fn = self._error_fn or self._default_error_fn
        residuals = err_fn(preds, target).to(device)

        # Concatenate into states
        if self.scores.numel() == 0:
            self.scores: torch.Tensor = scores
        else:
            self.scores = torch.cat([self.scores, scores], dim=0)

        if self.residuals.numel() == 0:
            self.residuals: torch.Tensor = residuals
        else:
            self.residuals = torch.cat([self.residuals, residuals], dim=0)

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute risk-coverage curve metrics."""
        if self.scores.numel() == 0:
            # No data yet; return zeros
            zero = torch.tensor(0.0, device=self.scores.device)
            return {
                "rc/auc_empirical": zero,
                "rc/auc_reference": zero,
                "rc/e_aurc": zero,
            }

        rc = risk_coverage(
            score=self.scores,
            residuals=self.residuals,
            risk=self.risk,
            n_bins=self.n_bins,
        )
        self._last_curve = rc
        return {
            "rc/auc_empirical": rc.auc_empirical,
            "rc/auc_reference": rc.auc_reference,
            "rc/auc_excess": rc.auc_excess,
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
