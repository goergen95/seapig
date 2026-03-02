"""Selective evaluation metric wrapper."""

from collections.abc import Callable, Iterable

import torch
from torchmetrics import Metric, MetricCollection

from seapig.risk_coverage import RiskCoverage, risk_coverage


class SelectiveMetric(Metric):
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

        # Deep-copy the base metric/collection for independent state
        full = _copy.deepcopy(base)
        selected = _copy.deepcopy(base)
        rejected = _copy.deepcopy(base)

        # Register as submodules so torch/Lightning see and manage them
        self._full = full
        self.add_module("_full", self._full)

        self._selected = selected
        self.add_module("_selected", self._selected)

        self._rejected = rejected
        self.add_module("_rejected", self._rejected)

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
        predictions = outputs[self.prediction_key]
        selected = outputs[self.selection_key].bool()
        rejected = ~selected

        device = predictions.device
        self._full.to(device)
        self._selected.to(device)
        self._rejected.to(device)

        # Update full with all samples
        self._full.update(predictions, target)

        # Conditionally update selected/rejected submetrics
        if selected.any():
            self._selected.update(predictions[selected], target[selected])

        if rejected.any():
            self._rejected.update(predictions[rejected], target[rejected])

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute and return results for total, selected, and rejected.

        Returns
        -------
        dict[str, torch.Tensor]
            Prefixed results. For a single metric, keys are
            ``"full/<metric_name>"``, ``"selected/<metric_name>"``,
            and ``"rejected/<metric_name>"``.
            For a collection, keys are prefixed as ``"full/<name>"``,
            ``"selected/<name>"``, and ``"rejected/<name>"``.
        """

        def _to_mapping(
            m: Metric | MetricCollection, prefix: str
        ) -> dict[str, torch.Tensor]:
            if isinstance(m, MetricCollection):
                # Generate outputs for all metrics in the collection
                out = {}
                for name, metric in m.items():
                    if getattr(
                        metric, "update_called", False
                    ):  # Check if the metric was updated
                        out[f"{prefix}/{name}"] = metric.compute()
                    else:
                        out[f"{prefix}/{name}"] = torch.tensor(0.0)
                return out
            else:
                # Generate output for a single metric
                metric_name = type(m).__name__
                if getattr(
                    m, "update_called", False
                ):  # Check if the metric was updated
                    return {f"{prefix}/{metric_name}": m.compute()}
                else:
                    return {f"{prefix}/{metric_name}": torch.tensor(0.0)}

        total_map = _to_mapping(self._full, "full")
        selected_map = _to_mapping(self._selected, "selected")
        rejected_map = _to_mapping(self._rejected, "rejected")
        return {**total_map, **selected_map, **rejected_map}

    def reset(self) -> None:
        """Reset both internal metric instances."""
        self._full.reset()
        self._selected.reset()
        self._rejected.reset()


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
        self.scores = torch.tensor([], dtype=torch.float32)
        self.add_state(
            "scores",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="cat",
        )
        self.residuals = torch.tensor([], dtype=torch.float32)
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
