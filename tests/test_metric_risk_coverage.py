import math

import pytest
import torch

from seapig import RiskCoverageMetric

# Suppress expected UserWarnings emitted when RiskCoverageMetric.reset() clears per‑metric buffers.
pytestmark = pytest.mark.filterwarnings(
    "ignore:RiskCoverageMetric state has been reset.*:UserWarning"
)


@pytest.mark.parametrize("risk", ["generalized", "selective"])
def test_risk_coverage_metric_basic_functionality(risk: str) -> None:
    metric = RiskCoverageMetric(risk=risk)

    preds = torch.tensor([0.9, 0.4, 0.6, 0.8])
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    # Update the metric (new API accepts tensors directly)
    metric.update(preds, target, scores)

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
        metric.update(preds, target, scores)

    res = metric.compute()

    assert "rc/auc_empirical" in res
    assert "rc/auc_reference" in res
    assert "rc/auc_excess" in res


def test_risk_coverage_metric_get_curve() -> None:
    metric = RiskCoverageMetric()

    preds = torch.tensor([0.9, 0.4, 0.6, 0.8])
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    metric.update(preds, target, scores)
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

    metric.update(preds1, target1, scores1)
    metric.update(preds2, target2, scores2)

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

    rcm.update(preds, target, scores)

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

    rcm.update(preds1, target1, scores1)
    rcm.update(preds2, target2, scores2)

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
