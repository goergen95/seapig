import math

import pytest
import torch
from pytorch_lightning import LightningModule
from torchmetrics import (
    Accuracy,
    MeanAbsoluteError,
    MeanMetric,
    MeanSquaredError,
    MetricCollection,
    Precision,
    Recall,
)

from seapig import RiskCoverageMetric, SelectiveInferenceTask, SelectiveMetric
from seapig.scores.base import RandomScore


class DummyTaskTensor(LightningModule):
    """Task that returns a tensor from predict()."""

    test_metrics: MetricCollection = MetricCollection(Accuracy(task="binary"))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)


class DummyTaskDict(LightningModule):
    """Task that returns a mapping from predict()."""

    test_metrics: MetricCollection = MetricCollection(Accuracy(task="binary"))

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"predictions": 3 * x, "extra": x.sum(dim=1)}  # pragma: no cover


def test_selective_metric_binary_accuracy_full_vs_selected() -> None:
    base = Accuracy(task="binary")
    sel = SelectiveMetric(base)

    # Predictions as probabilities; targets are 0/1
    outputs = {
        "predictions": torch.tensor([0.9, 0.4, 0.6, 0.8]),
        "selected": torch.tensor([1, 1, 0, 0]),  # non-bool mask accepted
    }
    target = torch.tensor([1, 0, 1, 0])

    sel.update(outputs, target)
    res = sel.compute()

    # Expect full accuracy: correct on 0.9, 0.4, 0.6; wrong on 0.8 => 3/4 = 0.75
    assert "full/BinaryAccuracy" in res and torch.allclose(
        res["full/BinaryAccuracy"], torch.tensor(0.75)
    )
    # Selected accuracy considers first two only, both correct => 1.0
    assert "selected/BinaryAccuracy" in res and torch.allclose(
        res["selected/BinaryAccuracy"], torch.tensor(1.0)
    )
    assert "rejected/BinaryAccuracy" in res and torch.allclose(
        res["rejected/BinaryAccuracy"], torch.tensor(0.5)
    )


def test_selective_metric_with_metric_collection_prefix_keys() -> None:
    coll = MetricCollection({"acc": Accuracy(task="binary")})
    sel = SelectiveMetric(coll)

    outputs = {
        "predictions": torch.tensor([0.9, 0.4, 0.6, 0.8]),
        "selected": torch.tensor([1, 0, 1, 0]),
    }
    target = torch.tensor([1, 0, 1, 0])

    sel.update(outputs, target)
    res = sel.compute()

    # Keys should be prefixed for collections
    assert any(k.startswith("full/") for k in res.keys())
    assert any(k.startswith("selected/") for k in res.keys())
    # Values are scalars
    for v in res.values():
        assert isinstance(v, torch.Tensor) and v.ndim == 0


def test_selective_metric_end_to_end_with_task_outputs() -> None:
    # Use SelectiveInferenceTask to produce outputs dict, then evaluate SelectiveMetric
    task = DummyTaskTensor()
    score = RandomScore()
    w = SelectiveInferenceTask(task=task, score=score)

    x = torch.tensor([0.2, 0.7, 0.6, 0.1])
    outputs = w.forward(x)
    # Override selection to choose two samples
    outputs["selected"] = torch.tensor([True, True, False, False])
    outputs["predictions"] = torch.tensor([0.0, 1.0, 0.0, 1.0])

    # Binary targets compatible with predictions thresholding at 0.5
    target = torch.tensor([0, 1, 1, 0])

    sel = SelectiveMetric(Accuracy(task="binary"))
    sel.update(outputs, target)
    res = sel.compute()

    # Full accuracy across all 4 predictions: half correct => 0.5
    assert torch.allclose(res["full/BinaryAccuracy"], torch.tensor(0.5))
    # Selected accuracy on first two samples: both correct => 1.0
    assert torch.allclose(res["selected/BinaryAccuracy"], torch.tensor(1.0))


def test_selective_metric_with_metric_collection_output_naming() -> None:
    # Define a MetricCollection with multiple metrics
    metrics = MetricCollection(
        {
            "accuracy": Accuracy(task="binary"),
            "precision": Precision(task="binary"),
            "recall": Recall(task="binary"),
        }
    )
    sel = SelectiveMetric(metrics)

    # Define outputs and targets
    outputs = {
        "predictions": torch.tensor([0.9, 0.4, 0.6, 0.8]),
        "selected": torch.tensor([1, 0, 1, 0]),  # non-bool mask accepted
    }
    target = torch.tensor([1, 0, 1, 0])

    # Update and compute the metrics
    sel.update(outputs, target)
    res = sel.compute()

    # Check that all keys are correctly prefixed
    expected_keys = {
        "full/accuracy",
        "selected/accuracy",
        "full/precision",
        "selected/precision",
        "full/recall",
        "selected/recall",
        "rejected/accuracy",
        "rejected/precision",
        "rejected/recall",
    }
    assert set(res.keys()) == expected_keys

    # Check that all values are scalars
    for value in res.values():
        assert isinstance(value, torch.Tensor) and value.ndim == 0


def test_selective_metric_reset() -> None:
    base = Accuracy(task="binary")
    sel = SelectiveMetric(base)

    # Update the metric with some data
    outputs = {
        "predictions": torch.tensor([0.9, 0.4, 0.6, 0.8]),
        "selected": torch.tensor([1, 0, 1, 0]),
    }
    target = torch.tensor([1, 0, 1, 0])
    sel.update(outputs, target)

    # Compute to initialize internal states
    res = sel.compute()
    # should not raise
    _ = sel._full.compute()

    # Reset the metric
    sel.reset()

    # we test if the base metric state was reset
    with pytest.warns(
        UserWarning, match="was called before the ``update`` method"
    ):
        res = sel._full.compute()
    assert isinstance(res, torch.Tensor)
    assert res == 0.0


@pytest.mark.parametrize("risk", ["generalized", "selective"])
def test_risk_coverage_metric_basic_functionality(risk) -> None:
    metric = RiskCoverageMetric(risk=risk)

    outputs = {
        "predictions": torch.tensor([0.9, 0.4, 0.6, 0.8]),
        "score": torch.tensor([0.1, 0.2, 0.3, 0.4]),
    }
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    # Update the metric
    metric.update(outputs, target)

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

    outputs = {"predictions": torch.tensor([0.9, 0.4, 0.6, 0.8])}
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    with pytest.raises(
        AssertionError, match="RiskCoverageMetric requires 'score'"
    ):
        metric.update(outputs, target)

    outputs = {"score": torch.tensor([0.9, 0.4, 0.6, 0.8])}
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    with pytest.raises(
        AssertionError, match="RiskCoverageMetric requires 'predictions'"
    ):
        metric.update(outputs, target)


def test_risk_coverage_metric_custom_error_function() -> None:
    def custom_error_fn(
        preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.abs(preds - target).sum(dim=1)

    metric = RiskCoverageMetric(error_fn=custom_error_fn)

    outputs = {
        "predictions": torch.tensor([[0.9, 0.1], [0.4, 0.6]]),
        "score": torch.tensor([0.1, 0.2]),
    }
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    metric.update(outputs, target)
    res = metric.compute()

    assert "rc/auc_empirical" in res
    assert "rc/auc_reference" in res
    assert "rc/auc_excess" in res


def test_risk_coverage_metric_get_curve() -> None:
    metric = RiskCoverageMetric()

    outputs = {
        "predictions": torch.tensor([0.9, 0.4, 0.6, 0.8]),
        "score": torch.tensor([0.1, 0.2, 0.3, 0.4]),
    }
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    metric.update(outputs, target)
    _ = metric.compute()

    curve = metric.get_curve()
    assert curve is not None
    assert hasattr(curve, "auc_empirical")
    assert hasattr(curve, "auc_reference")
    assert hasattr(curve, "auc_excess")


def test_risk_coverage_metric_state_concatenation() -> None:
    metric = RiskCoverageMetric()

    outputs1 = {
        "predictions": torch.tensor([0.9, 0.4]),
        "score": torch.tensor([0.1, 0.2]),
    }
    target1 = torch.tensor([1.0, 0.0])

    outputs2 = {
        "predictions": torch.tensor([0.6, 0.8]),
        "score": torch.tensor([0.3, 0.4]),
    }
    target2 = torch.tensor([1.0, 0.0])

    metric.update(outputs1, target1)
    metric.update(outputs2, target2)

    assert torch.equal(metric.scores, torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert metric.scores.numel() == 4


@pytest.mark.parametrize(
    "selection_behavior", ["always_select", "always_reject"]
)
def test_selective_metric_with_mock_metric_collection(selection_behavior):
    # Create a mock metric collection
    base_metrics = MetricCollection({"mean": MeanMetric()})
    selective_metric = SelectiveMetric(base=base_metrics)

    # Generate mock data
    batch_size = 10
    predictions = torch.randn(batch_size)
    target = torch.randn(batch_size)

    if selection_behavior == "always_select":
        selection_mask = torch.ones(batch_size, dtype=torch.bool)
    elif selection_behavior == "always_reject":
        selection_mask = torch.zeros(batch_size, dtype=torch.bool)

    # Update the metric with the mock data
    outputs = {"predictions": predictions, "selected": selection_mask}
    selective_metric.update(outputs, target)

    # Compute the results
    results = selective_metric.compute()

    # Assertions
    if selection_behavior == "always_select":
        # Ensure the selected metric is updated
        assert "selected/mean" in results
        assert results["selected/mean"] != 0
        # Ensure the rejected metric is not updated
        assert "rejected/mean" in results
        assert results["rejected/mean"] == 0
    elif selection_behavior == "always_reject":
        # Ensure the rejected metric is updated
        assert "rejected/mean" in results
        assert results["rejected/mean"] != 0
        # Ensure the selected metric is not updated
        assert "selected/mean" in results
        assert results["selected/mean"] == 0


def _tensor_close(a: torch.Tensor, b: float, tol: float = 1e-6) -> bool:
    return bool(
        torch.allclose(
            a, torch.tensor(b, dtype=a.dtype, device=a.device), atol=tol
        )
    )


# Additional tests to cover branches in metric.py not exercised above


def test_selective_metric_single_metric_selected_and_rejected_values():
    base = MeanAbsoluteError()
    sm = SelectiveMetric(base)

    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([1.0, 3.0, 2.0, 5.0])
    selected = torch.tensor([1, 0, 1, 0], dtype=torch.bool)

    outputs = {"predictions": preds, "selected": selected}
    sm.update(outputs, target)

    res = sm.compute()
    # full MAE: abs diffs = [0,1,1,1] -> mean = 0.75
    assert "full/MeanAbsoluteError" in res
    assert _tensor_close(res["full/MeanAbsoluteError"], 0.75)

    # selected indices (0,2): diffs [0,1] -> mean = 0.5
    assert "selected/MeanAbsoluteError" in res
    assert _tensor_close(res["selected/MeanAbsoluteError"], 0.5)

    # rejected indices (1,3): diffs [1,1] -> mean = 1.0
    assert "rejected/MeanAbsoluteError" in res
    assert _tensor_close(res["rejected/MeanAbsoluteError"], 1.0)

    # items/keys/values should be iterable and consistent
    keys = list(sm.keys())
    items = list(sm.items())
    values = list(sm.values())
    assert len(keys) == len(items) == len(values)


def test_selective_metric_no_selected_updates_metric_not_called():
    # When no samples are selected, the selected submetric should not be updated
    base = MeanAbsoluteError()
    sm = SelectiveMetric(base)

    preds = torch.tensor([0.0, 1.0, 2.0])
    target = torch.tensor([0.5, 0.5, 0.5])
    selected = torch.zeros(3, dtype=torch.bool)  # none selected

    sm.update({"predictions": preds, "selected": selected}, target)
    res = sm.compute()

    # selected metric was never updated -> should yield 0.0 tensor
    assert "selected/MeanAbsoluteError" in res
    assert _tensor_close(res["selected/MeanAbsoluteError"], 0.0)

    # rejected should be computed
    assert "rejected/MeanAbsoluteError" in res
    assert res["rejected/MeanAbsoluteError"].numel() == 1
    assert res["rejected/MeanAbsoluteError"].dtype == torch.float32


def test_selective_metric_with_metric_collection_and_prefixing():
    coll = MetricCollection(
        {"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}
    )
    smc = SelectiveMetric(coll)

    preds = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.2, 1.8, 2.5])
    selected = torch.tensor([1, 1, 0], dtype=torch.bool)

    smc.update({"predictions": preds, "selected": selected}, target)
    out = smc.compute()

    # Ensure both metrics in the collection appear with collection names
    for prefix in ("full", "selected", "rejected"):
        assert f"{prefix}/mae" in out
        assert f"{prefix}/mse" in out

    # Confirm values are tensors
    assert isinstance(out["full/mae"], torch.Tensor)
    assert isinstance(out["full/mse"], torch.Tensor)


@pytest.mark.filterwarnings(
    "ignore",
    category=UserWarning,
    match="was called before the ``update`` method",
)
def test_risk_coverage_metric_no_data_returns_zeros_and_reset_behavior():
    rcm = RiskCoverageMetric()
    # no updates yet -> should return zeros
    empty = rcm.compute()
    assert _tensor_close(empty["rc/auc_empirical"], 0.0)
    assert _tensor_close(empty["rc/auc_reference"], 0.0)
    assert _tensor_close(empty["rc/auc_excess"], 0.0)

    # Provide some data
    preds = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 1.5, 1.5])
    scores = torch.tensor([0.1, 0.4, 0.9])

    rcm.update({"predictions": preds, "score": scores}, target)

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


def test_risk_coverage_metric_multiple_updates_concatenate_states():
    rcm = RiskCoverageMetric()
    preds1 = torch.tensor([0.0, 1.0])
    target1 = torch.tensor([0.2, 0.8])
    scores1 = torch.tensor([0.05, 0.2])

    preds2 = torch.tensor([2.0, 3.0, 4.0])
    target2 = torch.tensor([1.5, 1.5, 1.5])
    scores2 = torch.tensor([0.7, 0.8, 0.9])

    rcm.update({"predictions": preds1, "score": scores1}, target1)
    rcm.update({"predictions": preds2, "score": scores2}, target2)

    # both updates concatenated
    assert rcm.scores.numel() == 5
    assert rcm.residuals.numel() == 5

    # compute should work after concatenation
    out = rcm.compute()
    assert "rc/auc_empirical" in out
    assert torch.isfinite(out["rc/auc_empirical"]).all()
