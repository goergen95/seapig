import math
from typing import cast

import pytest
import torch
from torchmetrics import Metric, MetricCollection

from seapig.metric import RiskCoverageMetric
from seapig.risk import RiskCoverage


# Create a collection with two simple custom error metrics.
class AbsErrorMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "res",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="cat",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.res = torch.abs(preds - target).mean(dim=1)

    def compute(self) -> torch.Tensor:
        return self.res


class SqErrorMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "res",
            default=torch.tensor([], dtype=torch.float32),
            dist_reduce_fx="cat",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.res = ((preds - target) ** 2).mean(dim=1)

    def compute(self) -> torch.Tensor:
        return self.res


def test_get_curve_multi_metric():
    coll = MetricCollection({"mae": AbsErrorMetric(), "mse": SqErrorMetric()})
    metric = RiskCoverageMetric(error_metric=coll)
    preds = torch.tensor([[0.5, 0.2], [0.1, 0.9]])
    target = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    scores = torch.tensor([0.1, 0.2])
    metric.update(preds=preds, target=target, scores=scores)
    metric.compute()
    curves = metric.get_curve()
    assert isinstance(curves, dict)
    curves = cast(dict[str, RiskCoverage], curves)
    assert set(curves) == {"mae", "mse"}
    assert all(hasattr(curve, "auc_empirical") for curve in curves.values())
    assert metric.get_curve("mae") is curves["mae"]


def test_get_curve_single_metric():
    metric = RiskCoverageMetric(error_metric=AbsErrorMetric())
    preds = torch.tensor([0.5, 0.6])
    target = torch.tensor([0.0, 1.0])
    scores = torch.tensor([0.2, 0.4])
    metric.update(preds=preds, target=target, scores=scores)
    metric.compute()
    curve = metric.get_curve()
    assert hasattr(curve, "auc_empirical")


def test_get_curve_none_before_compute_and_after_reset():
    metric = RiskCoverageMetric(error_metric=AbsErrorMetric())
    assert metric.get_curve() is None
    preds = torch.tensor([0.5, 0.6])
    target = torch.tensor([0.0, 1.0])
    scores = torch.tensor([0.2, 0.4])
    metric.update(preds=preds, target=target, scores=scores)
    metric.compute()
    assert metric.get_curve() is not None
    metric.reset()
    assert metric.get_curve() is None


def test_compute_output_keys_multi_metric():
    coll = MetricCollection({"mae": AbsErrorMetric(), "mse": SqErrorMetric()})
    metric = RiskCoverageMetric(error_metric=coll)
    preds = torch.tensor([[0.5, 0.2], [0.1, 0.9]])
    target = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    scores = torch.tensor([0.1, 0.2])
    metric.update(preds=preds, target=target, scores=scores)
    out = metric.compute()
    for metric_name in ("mae", "mse"):
        prefix = f"rc/{metric_name}"
        for suffix in ("auc_empirical", "auc_reference", "auc_excess"):
            assert f"{prefix}/{suffix}" in out
            assert isinstance(out[f"{prefix}/{suffix}"], torch.Tensor)


def test_reset_clears_curves_and_buffers():
    coll = MetricCollection({"mae": AbsErrorMetric(), "mse": SqErrorMetric()})
    metric = RiskCoverageMetric(error_metric=coll)
    preds = torch.tensor([[0.5, 0.2], [0.1, 0.9]])
    target = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    scores = torch.tensor([0.1, 0.2])
    metric.update(preds=preds, target=target, scores=scores)
    metric.compute()
    assert metric.get_curve() is not None
    metric.reset()
    assert metric.get_curve() is None
    assert metric.scores.numel() == 0
    assert metric.residuals.numel() == 0
    assert metric.residuals_mae.numel() == 0  # type: ignore[operator, ty:call-non-callable]
    assert metric.residuals_mse.numel() == 0  # type: ignore[operator, ty:call-non-callable]


def test_risk_coverage_metric_empty_with_metric_names():
    metric = RiskCoverageMetric(error_metric=AbsErrorMetric())
    # expect a warning here because compute is called before any updates
    with pytest.warns(UserWarning, match="was called before"):
        result = metric.compute()
    # Should return zero tensors for each metric key
    assert all(v.item() == 0.0 for v in result.values())
    # Should include keys for the metric
    assert set(result.keys()) == {
        "rc/AbsErrorMetric/auc_empirical",
        "rc/AbsErrorMetric/auc_reference",
        "rc/AbsErrorMetric/auc_excess",
    }


@pytest.mark.parametrize("risk", ["generalized", "selective"])
def test_risk_coverage_metric_basic_functionality(risk: str) -> None:
    metric = RiskCoverageMetric(risk=risk)

    preds = torch.tensor([0.9, 0.4, 0.6, 0.8])
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    # Update the metric (new API accepts tensors directly)
    metric.update(preds=preds, target=target, scores=scores)

    # check that scores and residuals are correct
    assert metric.scores.numel() == 4
    assert metric.residuals.numel() == 4

    # Compute the results
    res = metric.compute()
    assert "rc/auc_empirical" in res
    assert "rc/auc_reference" in res
    assert "rc/auc_excess" in res

    # Ensure values are scalars
    for value in res.values():
        assert isinstance(value, torch.Tensor) and value.ndim == 0

    # Reset the metric
    metric.reset()
    assert metric.scores.numel() == 0
    assert metric.residuals.numel() == 0


def test_risk_coverage_metric_missing_keys() -> None:
    metric = RiskCoverageMetric()

    # The RiskCoverageMetric.update now requires explicit tensors: preds, target, scores
    preds = torch.tensor([0.9, 0.4, 0.6, 0.8])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    # missing args should raise a TypeError at call time
    with pytest.raises(TypeError):
        # missing scores
        metric.update(preds=preds, target=target)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        # missing scores (wrong kwarg name)
        metric.update(preds=preds, scores=target)  # type: ignore[call-arg]


def test_risk_coverage_metric_custom_error_function() -> None:
    """Legacy ``error_fn`` path should still work but emit a DeprecationWarning."""

    def custom_error_fn(
        preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.abs(preds - target).sum(dim=1)

    metric = RiskCoverageMetric(error_fn=custom_error_fn)

    preds = torch.tensor([[0.9, 0.1], [0.4, 0.6]])
    scores = torch.tensor([0.1, 0.2])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    # The legacy path emits a DeprecationWarning; assert that it is raised.
    with pytest.warns(DeprecationWarning, match=r"`error_fn` is deprecated"):
        metric.update(preds=preds, target=target, scores=scores)

    res = metric.compute()

    assert "rc/auc_empirical" in res
    assert "rc/auc_reference" in res
    assert "rc/auc_excess" in res


def test_risk_coverage_metric_get_curve() -> None:
    metric = RiskCoverageMetric()

    preds = torch.tensor([0.9, 0.4, 0.6, 0.8])
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    metric.update(preds=preds, target=target, scores=scores)
    _ = metric.compute()

    curve = metric.get_curve()
    assert curve is not None
    assert hasattr(curve, "auc_empirical")
    assert hasattr(curve, "auc_reference")
    assert hasattr(curve, "auc_excess")


def test_risk_coverage_metric_state_concatenation() -> None:
    metric = RiskCoverageMetric()

    preds1 = torch.tensor([0.9, 0.4])
    scores1 = torch.tensor([0.1, 0.2])
    target1 = torch.tensor([1.0, 0.0])

    preds2 = torch.tensor([0.6, 0.8])
    scores2 = torch.tensor([0.3, 0.4])
    target2 = torch.tensor([1.0, 0.0])

    metric.update(preds=preds1, target=target1, scores=scores1)
    metric.update(preds=preds2, target=target2, scores=scores2)

    assert torch.equal(metric.scores, torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert metric.scores.numel() == 4


@pytest.mark.filterwarnings("ignore:.*was called before.*:UserWarning")
def test_risk_coverage_metric_no_data_returns_zeros_and_reset_behavior() -> (
    None
):
    rcm = RiskCoverageMetric()
    # no updates yet -> should return zeros
    empty = rcm.compute()
    assert torch.allclose(empty["rc/auc_empirical"], torch.tensor(0.0))
    assert torch.allclose(empty["rc/auc_reference"], torch.tensor(0.0))
    assert torch.allclose(empty["rc/auc_excess"], torch.tensor(0.0))

    # Provide some data
    preds = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 1.5, 1.5])
    scores = torch.tensor([0.1, 0.4, 0.9])

    rcm.update(preds=preds, target=target, scores=scores)

    # internal buffers should have been concatenated
    assert rcm.scores.numel() == 3
    assert rcm.residuals.numel() == 3

    out = rcm.compute()
    # outputs should be finite tensors
    for k in ("rc/auc_empirical", "rc/auc_reference", "rc/auc_excess"):
        assert k in out
        assert torch.isfinite(out[k]).all()

    curve = rcm.get_curve()
    assert curve is not None
    # the stored curve's AUC should match the compute() output (within tolerance)
    assert math.isclose(
        float(out["rc/auc_empirical"]), float(curve.auc_empirical), rel_tol=1e-6
    )

    # Reset should clear buffers but returns last compute result until next update
    rcm.reset()
    assert rcm.scores.numel() == 0
    assert rcm.residuals.numel() == 0

    curve_after_reset = rcm.get_curve()
    assert curve_after_reset is None

    # expect warnings because update has not been called
    post_reset = rcm.compute()
    assert math.isclose(
        float(post_reset["rc/auc_empirical"]),
        float(curve.auc_empirical),
        rel_tol=1e-6,
    )


def test_risk_coverage_metric_multiple_updates_concatenate_states() -> None:
    rcm = RiskCoverageMetric()
    preds1 = torch.tensor([0.0, 1.0])
    target1 = torch.tensor([0.2, 0.8])
    scores1 = torch.tensor([0.05, 0.2])

    preds2 = torch.tensor([2.0, 3.0, 4.0])
    target2 = torch.tensor([1.5, 1.5, 1.5])
    scores2 = torch.tensor([0.7, 0.8, 0.9])

    rcm.update(preds=preds1, target=target1, scores=scores1)
    rcm.update(preds=preds2, target=target2, scores=scores2)

    # both updates concatenated
    assert rcm.scores.numel() == 5
    assert rcm.residuals.numel() == 5

    # compute should work after concatenation
    out = rcm.compute()
    assert "rc/auc_empirical" in out
    assert torch.isfinite(out["rc/auc_empirical"]).all()


def test_error_fn_and_metric_mutual_exclusion() -> None:
    """Ensure that providing both ``error_fn`` and ``error_metric`` raises."""
    from torchmetrics import MeanAbsoluteError

    mae = MeanAbsoluteError()
    # Both arguments supplied should trigger a ValueError.
    with pytest.raises(ValueError):
        RiskCoverageMetric(
            error_fn=lambda p, t: torch.abs(p - t), error_metric=mae
        )


def test_multi_metric_support_and_sanity_check() -> None:
    """Validate multi‑metric handling and sanity‑check enforcement.

    The ``RiskCoverageMetric`` should accept a ``MetricCollection`` of per‑sample
    residual metrics, create separate residual states for each, and expose
    computed AUC values under ``rc/<metric_name>/`` prefixes. The sanity check
    should raise a ``ValueError`` if a metric returns an invalid residual type
    or shape.
    """
    coll = MetricCollection([AbsErrorMetric(), SqErrorMetric()])

    # The metrics need to be updated before compute; the RiskCoverageMetric will
    # invoke its own sanity check on construction, which should pass.
    metric = RiskCoverageMetric(error_metric=coll)

    # Provide dummy data – the underlying metrics will compute residuals.
    preds = torch.tensor([0.5, 0.7, 0.2])
    target = torch.tensor([0.0, 1.0, 0.0])
    scores = torch.tensor([0.1, 0.2, 0.3])
    metric.update(preds=preds, target=target, scores=scores)

    # Ensure per‑metric residual buffers exist and have the expected shape.
    for name in metric._metric_names:
        residual_buf = getattr(metric, f"residuals_{name}")
        assert isinstance(residual_buf, torch.Tensor)
        assert residual_buf.shape == (3,)

    # Compute should return keys for each metric.
    out = metric.compute()
    for name in metric._metric_names:
        prefix = f"rc/{name}"
        assert f"{prefix}/auc_empirical" in out
        assert f"{prefix}/auc_reference" in out
        assert f"{prefix}/auc_excess" in out


def test_validate_tensor_cases() -> None:
    """Validate that `_validate_tensor` accepts (B,) and (B,1) and rejects others."""

    # instantiate with a dummy metric to access the method
    metric = RiskCoverageMetric()

    # (B,) case – should be returned unchanged
    b_vec = torch.arange(5, dtype=torch.float32)
    assert torch.equal(metric._validate_tensor(b_vec), b_vec)

    # (B,1) case – should be squeezed to (B,)
    b_col = b_vec.unsqueeze(1)
    squeezed = metric._validate_tensor(b_col)
    assert squeezed.ndim == 1 and torch.equal(squeezed, b_vec)

    # Invalid shape (B,2) – should raise ValueError with appropriate message
    bad = torch.stack([b_vec, b_vec], dim=1)
    with pytest.raises(ValueError, match="must return a 1‑D tensor"):
        metric._validate_tensor(bad, name="BadMetric")


def test_multi_metric_support_and_reset() -> None:
    """Validate multi‑metric handling via a MetricCollection."""
    coll = MetricCollection({"mae": AbsErrorMetric(), "mse": SqErrorMetric()})

    metric = RiskCoverageMetric(error_metric=coll)

    preds = torch.tensor([[0.5, 0.2], [0.1, 0.9]])
    target = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    scores = torch.tensor([0.1, 0.2])

    # Perform a single update; residuals for each metric should be stored.
    metric.update(preds=preds, target=target, scores=scores)
    # The per‑metric residual buffers are created during init and filled here.
    assert hasattr(metric, "residuals_mae")
    assert hasattr(metric, "residuals_mse")
    assert metric.residuals_mae.shape == (2,)
    assert metric.residuals_mse.shape == (2,)

    # Compute should return entries for each metric with the proper prefixes.
    out = metric.compute()
    for metric_name in ("mae", "mse"):
        prefix = f"rc/{metric_name}"
        for suffix in ("auc_empirical", "auc_reference", "auc_excess"):
            assert f"{prefix}/{suffix}" in out

    # Reset clears all per‑metric buffers and the generic residual buffer.
    metric.reset()
    # The residual buffers are tensors; mypy/ty may not infer their callable methods correctly.
    assert metric.residuals_mae.numel() == 0  # type: ignore[operator, ty:call-non-callable]
    assert metric.residuals_mse.numel() == 0  # type: ignore[operator, ty:call-non-callable]
    assert metric.residuals.numel() == 0


def test_multi_metric_support_multiple_updates() -> None:
    """Validate multi-metric state accumulation across multiple updates.

    This test ensures that when using a MetricCollection with multiple update() calls,
    the per-metric residual buffers and scores remain aligned and compute() succeeds.
    """
    coll = MetricCollection({"mae": AbsErrorMetric(), "mse": SqErrorMetric()})
    metric = RiskCoverageMetric(error_metric=coll)

    # First update with 2 samples
    preds1 = torch.tensor([[0.5, 0.2], [0.1, 0.9]])
    target1 = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    scores1 = torch.tensor([0.1, 0.2])

    metric.update(preds=preds1, target=target1, scores=scores1)
    assert isinstance(metric.scores, torch.Tensor)
    assert isinstance(metric.residuals_mae, torch.Tensor)
    assert isinstance(metric.residuals_mse, torch.Tensor)
    assert metric.scores.numel() == 2
    assert metric.residuals_mae.numel() == 2
    assert metric.residuals_mse.numel() == 2

    # Second update with 3 samples
    preds2 = torch.tensor([[0.3, 0.7], [0.6, 0.4], [0.8, 0.2]])
    target2 = torch.tensor([[0.5, 0.5], [0.7, 0.3], [0.9, 0.1]])
    scores2 = torch.tensor([0.3, 0.4, 0.5])

    metric.update(preds=preds2, target=target2, scores=scores2)
    # After second update, all buffers should contain 5 samples
    assert metric.scores.numel() == 5
    assert metric.residuals_mae.numel() == 5
    assert metric.residuals_mse.numel() == 5

    # Verify scores and residuals are aligned
    assert metric.scores.shape[0] == metric.residuals_mae.shape[0]
    assert metric.scores.shape[0] == metric.residuals_mse.shape[0]

    # Compute should succeed with accumulated state
    out = metric.compute()
    for metric_name in ("mae", "mse"):
        prefix = f"rc/{metric_name}"
        for suffix in ("auc_empirical", "auc_reference", "auc_excess"):
            key = f"{prefix}/{suffix}"
            assert key in out
            assert isinstance(out[key], torch.Tensor) and out[key].ndim == 0
            assert torch.isfinite(out[key]).all()


def test_risk_coverage_metric_single_error_metric() -> None:
    """Ensure RiskCoverageMetric works when a single Metric is provided as error_metric."""
    # Use a simple custom metric that returns per-sample absolute error.
    metric = RiskCoverageMetric(error_metric=AbsErrorMetric())

    preds = torch.tensor([0.5, 0.6])
    target = torch.tensor([0.0, 1.0])
    scores = torch.tensor([0.2, 0.4])

    # Perform update – should populate per‑metric residual state.
    metric.update(preds=preds, target=target, scores=scores)

    # The metric should have created a residual buffer named after the metric.
    name = metric._metric_names[0]
    residual_buf = getattr(metric, f"residuals_{name}")
    assert isinstance(residual_buf, torch.Tensor)
    assert residual_buf.shape == (2,)

    # Compute should return AUC entries prefixed with the metric name.
    out = metric.compute()
    prefix = f"rc/{name}"
    for suffix in ("auc_empirical", "auc_reference", "auc_excess"):
        assert f"{prefix}/{suffix}" in out
