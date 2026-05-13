"""Metrics helpers for selective evaluation.

This module provides two utilities used during evaluation when a model
can abstain or return confidence scores:

- `SelectiveMetric`: wraps a `torchmetrics.Metric` (or a
  `torchmetrics.MetricCollection`) and tracks the metric on three
  disjoint subsets: all samples (full), the samples marked as selected,
  and the samples marked as rejected.
- `RiskCoverageMetric`: accumulates per-sample scores and residuals to
  compute a risk-coverage curve using `scores.risk_coverage`.
"""

import warnings
from collections.abc import Callable

import torch
from torchmetrics import Metric, MetricCollection

from seapig.risk import RiskCoverage, risk_coverage


class SelectiveMetric(Metric):
    """Evaluate a metric on full, selected, and rejected subsets.

    Wraps a `torchmetrics.Metric` or `torchmetrics.MetricCollection` and
    keeps three independent copies that are updated separately:

    - `"full"`: all samples passed to `update`.
    - `"selected"`: samples where the provided selection mask is true.
    - `"rejected"`: samples where the selection mask is false.

    The `compute` result is a flat `dict`[`str`, `torch.Tensor`] where
    each underlying metric name is prefixed with `full/`, `selected/`,
    or `rejected/`. If a submetric was never updated, its value is a
    zero `torch.Tensor`.

    Parameters
    ----------
    base : torchmetrics.Metric | torchmetrics.MetricCollection
        Metric (or collection) to wrap. Internally the object is deep-
        copied three times so each subset is tracked independently.

    Notes
    -----
    - The selection mask may be boolean or numeric; numeric values `> 0`
      are treated as selected.
    - Calls that contain no selected (or no rejected) rows do not update
      the corresponding internal metric for that call.

    Example
    -------
    ```{python}
    import torch
    from torchmetrics import Accuracy
    from seapig import SelectiveMetric

    base = Accuracy(task="binary")
    m = SelectiveMetric(base)
    preds = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    target = torch.tensor([[1.0, 1], [1, 0]])
    selected = torch.tensor([1, 0], dtype=torch.bool)
    m.update(preds, target, selected)
    results = m.compute()
    print(results)
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
        if selected.dtype is not torch.bool:
            bool_mask = selected.to(dtype=torch.bool)
        else:
            bool_mask = selected

        if bool_mask.any():
            self.metrics["selected"].update(preds[bool_mask], target[bool_mask])

        rejected = ~bool_mask
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
    error_metric : torchmetrics.Metric | torchmetrics.MetricCollection | None, default None
        Metric or collection that computes per-sample residuals.
        It must return a 1-D tensor of shape ``(batch,)`` when ``compute`` is called.
    error_fn : callable or None, default None
        Deprecated legacy function ``(preds, target) -> residuals``.
        Use ``error_metric`` instead.


    Notes
    -----
    The `compute` method returns three tensors:
    `rc/auc_empirical`, `rc/auc_reference`, and `rc/auc_excess`.
    The last computed complete curve object (`RiskCoverage`) is
    available via `get_curve`.

    See Also
    --------
    `RiskCoverage` : Container for curve results.

    Examples
    --------
    ```{python}
    import torch
    from seapig.metric import RiskCoverageMetric

    metric = RiskCoverageMetric(risk="generalized")
    preds = torch.rand(50, 1)
    target = torch.rand(50, 1)
    scores = torch.rand(50)
    metric.update(preds, target, scores)
    results = metric.compute()
    print(results)
    _ = metric.get_curve().plot()
    ```
    """

    full_state_update: bool = False
    scores: torch.Tensor
    residuals: torch.Tensor

    def __init__(
        self,
        risk: str = "generalized",
        n_bins: int = 100,
        *,
        error_metric: Metric | MetricCollection | None = None,
        error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
    ) -> None:
        """RiskCoverageMetric computes risk‑coverage curves for a model."""
        super().__init__()
        assert risk in ["generalized", "selective"], (
            "RiskCoverageMetric risk must be 'generalized' or 'selective'."
        )
        self.risk = risk
        self.n_bins = n_bins

        if error_fn is not None and error_metric is not None:
            raise ValueError(
                "Provide either 'error_fn' or 'error_metric', not both."
            )

        self._error_fn = error_fn

        # Normalise metric to a MetricCollection for unified handling.
        # Each metric must output a 1‑D tensor of per‑sample residuals.
        # We store a separate residual stream for each metric.
        self._error_metric = None
        # Store metric names (as strings) for later state creation.
        self._metric_names: list[str] = []
        if error_metric is not None:
            if isinstance(error_metric, Metric):
                error_metric = MetricCollection(error_metric)
            self._error_metric = error_metric
            self._metric_names = [str(k) for k in self._error_metric.keys()]

        self._sanity_check()

        # Metric states (concatenate across steps)
        self.add_state(
            "scores",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="cat",
        )
        # For legacy ``error_fn`` path we keep a single residual stream.
        self.add_state(
            "residuals",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="cat",
        )
        # When using ``error_metric`` we create a separate residual
        # state per metric in the collection.
        for name in getattr(self, "_metric_names", []):
            self.add_state(
                f"residuals_{name}",
                default=torch.tensor([], dtype=torch.float32),
                dist_reduce_fx="cat",
            )

        # Last computed curve (non‑tensor; kept for retrieval only)
        self._last_curve: RiskCoverage | None = None

    def _sanity_check(self) -> None:
        """Sanity check that the provided ``error_metric`` produces valid per‑sample residuals.

        The check is only performed when a ``MetricCollection`` is supplied.  When
        ``error_metric`` is ``None`` (legacy ``error_fn`` path) the method exits
        early – the legacy path does not rely on this validation.
        """
        if self._error_metric is None:
            # No metric collection to validate – rely on ``error_fn`` handling.
            return

        dummy_preds = torch.randn(2, 3)
        dummy_target = torch.randn(2, 3)
        self._error_metric.update(dummy_preds, dummy_target)
        residuals = self._error_metric.compute()
        self._error_metric.reset()

        if isinstance(residuals, dict):
            # Validate each metric's residual tensor individually.
            for name, val in residuals.items():
                residuals[name] = self._validate_tensor(val, name)
        else:
            # Single metric case – validate and replace with flattened version.
            residuals = self._validate_tensor(residuals)

    @staticmethod
    def _validate_tensor(
        tensor: torch.Tensor, name: str | None = None
    ) -> torch.Tensor:
        """Validate that the provided tensor is a 1‑D tensor of per‑sample residuals."""
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Metric{' ' + name if name else ''} must return a torch.Tensor."
            )
        # Allow (B,) or (B,1); flatten to (B,)
        if tensor.ndim == 1:
            return tensor
        if tensor.ndim == 2 and tensor.shape[1] == 1:
            return tensor.squeeze(1)
        raise ValueError(
            f"Metric{' ' + name if name else ''} must return a 1‑D tensor of residuals "
            f"(shape (B,) or (B,1)), got shape {tuple(tensor.shape)}."
        )

    @staticmethod
    def _default_error_fn(
        preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Default error function used when no custom ``error_fn`` or ``error_metric`` is provided."""
        # Cast to float to support integer inputs (e.g., class labels)
        preds_f = preds.to(dtype=torch.float32)
        target_f = target.to(dtype=torch.float32)

        residual = torch.abs(preds_f - target_f)
        # If the residual is already 1‑D (batch only) we can return it directly.
        if residual.ndim == 1:
            return residual
        # Reduce over all non‑batch dimensions.
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
        if preds.ndim == 1:
            preds = preds.unsqueeze(1)
        if target.ndim == 1:
            target = target.unsqueeze(1)

        if self._error_metric is not None:
            # Update the metric collection and compute per‑sample residuals.
            self._error_metric.update(preds, target)
            residuals_raw = self._error_metric.compute()
            # Extract each residual and store it in the corresponding state.
            if isinstance(residuals_raw, dict):
                for name, tensor in residuals_raw.items():
                    tensor = tensor.to(device)
                    self._validate_tensor(tensor, name)
                    state_name = f"residuals_{str(name)}"
                    existing = getattr(self, state_name)
                    if existing.numel() == 0:
                        setattr(self, state_name, tensor)
                    else:
                        setattr(
                            self,
                            state_name,
                            torch.cat([existing, tensor], dim=0),
                        )
                # For backward‑compatibility with single‑metric code paths we also keep
                # the generic ``residuals`` attribute equal to the first metric's values.
                first_tensor = next(iter(residuals_raw.values()))
                residuals = first_tensor.to(device)
        else:
            # Legacy path – emit deprecation warning if needed.
            if self._error_fn is not None:
                warnings.warn(
                    "`error_fn` is deprecated; use `error_metric` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            err_fn = self._error_fn or self._default_error_fn
            residuals = err_fn(preds, target).to(device)

        # Concatenate scores.
        if self.scores.numel() == 0:
            self.scores = scores
        else:
            self.scores = torch.cat([self.scores, scores], dim=0)

        # For the generic residual buffer (used by legacy paths or the first metric)
        if residuals is not None:
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

        device = self.scores.device
        # If no per‑metric names were defined, fall back to the original behaviour.
        if not self._metric_names:
            rc = risk_coverage(
                score=self.scores,
                residuals=self.residuals,
                risk=self.risk,
                n_bins=self.n_bins,
            )
            self._last_curve = rc
            return {
                "rc/auc_empirical": torch.as_tensor(
                    rc.auc_empirical, device=device
                ),
                "rc/auc_reference": torch.as_tensor(
                    rc.auc_reference, device=device
                ),
                "rc/auc_excess": torch.as_tensor(rc.auc_excess, device=device),
            }

        # Multi‑metric case: compute a curve for each metric separately.
        results: dict[str, torch.Tensor] = {}
        for name in self._metric_names:
            residuals = getattr(self, f"residuals_{name}")
            rc = risk_coverage(
                score=self.scores,
                residuals=residuals,
                risk=self.risk,
                n_bins=self.n_bins,
            )
            prefix = f"rc/{name}"
            results[f"{prefix}/auc_empirical"] = torch.as_tensor(
                rc.auc_empirical, device=device
            )
            results[f"{prefix}/auc_reference"] = torch.as_tensor(
                rc.auc_reference, device=device
            )
            results[f"{prefix}/auc_excess"] = torch.as_tensor(
                rc.auc_excess, device=device
            )
        # Store the last curve (from the last metric) for ``get_curve`` compatibility.
        self._last_curve = rc
        return results

    def get_curve(self) -> RiskCoverage | None:
        """Return the last computed RiskCoverage object (or None if not computed)."""
        return self._last_curve

    def reset(self) -> None:
        """Reset all accumulated state.

        The generic ``scores`` and ``residuals`` buffers are cleared, as are any
        per-metric residual buffers created when ``error_metric`` is a
        ``MetricCollection``. After a reset, accessing a per-metric residual
        attribute before the next ``update`` call will yield an empty tensor.
        A ``UserWarning`` is emitted to make this behaviour explicit.
        """
        self.scores = torch.tensor(
            [], dtype=torch.float32, device=self.scores.device
        )
        self.residuals = torch.tensor(
            [], dtype=torch.float32, device=self.residuals.device
        )
        for name in getattr(self, "_metric_names", []):
            setattr(
                self,
                f"residuals_{name}",
                torch.tensor(
                    [], dtype=torch.float32, device=self.scores.device
                ),
            )
        self._last_curve = None
        warnings.warn(
            "RiskCoverageMetric state has been reset; per‑metric residual buffers are now empty.",
            UserWarning,
            stacklevel=2,
        )
