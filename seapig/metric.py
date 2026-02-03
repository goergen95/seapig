"""Selective evaluation metric wrapper."""

import torch
from torchmetrics import Metric, MetricCollection


class SelectiveMetric(Metric):  # type: ignore[misc]
    """Wrap a torchmetrics metric for selective evaluation.

    This wrapper keeps two independent instances of an underlying
    ``Metric`` or ``MetricCollection`` and updates them with (1) all
    samples and (2) only the selected samples indicated by a boolean
    mask inside the provided output dictionary.

    The ``update`` method accepts a mapping like the output of
    :meth:`SelectiveInferenceTask.forward`, containing:

    - model predictions under ``prediction_key`` (shape ``(B, ...)``, default: ``"predictions"``)
    - a selection mask under ``selection_key`` (default: ``"selected"``)

    And the target tensor of shape ``(B, ...)``.

    Parameters
    ----------
    base : Metric | MetricCollection
        The metric (or collection) to evaluate. Two deep-copied instances
        are maintained internally for full and selective risk computation.
    prediction_key : str, optional
        Key for predictions inside the outputs dict, by default "predictions".
    selection_key : str, optional
        Key for the selection mask inside the outputs dict, by default
        "selected".

    Notes
    -----
    - If the selection mask is a float/integer tensor, values ``> 0``
      are treated as selected.
    - If no samples are selected, the selected metric is not updated for
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

        # Two independent metric instances (full vs selection)
        self._full: Metric | MetricCollection = _copy.deepcopy(base)
        self._selected: Metric | MetricCollection = _copy.deepcopy(base)

        self.prediction_key = prediction_key
        self.selection_key = selection_key

    def update(
        self, outputs: dict[str, torch.Tensor], target: torch.Tensor
    ) -> None:
        """Update both full and selected metrics.

        Parameters
        ----------
        outputs : dict[str, torch.Tensor]
            Mapping containing predictions, a target tensor, and selection
            mask.
        """
        preds = outputs[self.prediction_key]
        mask = outputs[self.selection_key]

        device = preds.device
        # Ensure both metrics are on the right device
        self._full = self._full.to(device)
        self._selected = self._selected.to(device)

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

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute and return results for total and selected.

        Returns
        -------
        dict[str, torch.Tensor]
            Prefixed results. For a single metric, keys are
            ``"full_risk"`` and ``"selective_risk"``. For a collection,
            keys are prefixed as ``"full/<name>"`` and ``"selected/<name>"``.
        """

        def _to_mapping(
            m: Metric | MetricCollection, prefix: str
        ) -> dict[str, torch.Tensor]:
            out = m.compute()
            if isinstance(out, dict):
                return {f"{prefix}/{k}": v for k, v in out.items()}
            return {"full_risk" if prefix == "full" else "selective_risk": out}

        total_map = _to_mapping(self._full, "full")
        selected_map = _to_mapping(self._selected, "selected")
        return {**total_map, **selected_map}

    def reset(self) -> None:
        """Reset both internal metric instances."""
        self._full.reset()
        self._selected.reset()
