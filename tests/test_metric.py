import pytest
import torch
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MetricCollection, Precision, Recall

from seapig import SelectiveInferenceTask, SelectiveMetric
from seapig.metric import RiskCoverageMetric
from seapig.scores.base import RandomScore


class DummyScore:
    """Minimal duck-typed score with select() and to()."""

    def __init__(self) -> None:
        self.last_device: str | None = None

    def to(self, device: str) -> "DummyScore":
        self.last_device = device
        return self

    def select(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        return {
            "score": torch.arange(b, dtype=x.dtype, device=x.device),
            "selected": torch.ones(b, dtype=torch.bool, device=x.device),
        }


class DummyTaskTensor(LightningModule):
    """Task that returns a tensor from predict()."""

    test_metrics: MetricCollection = MetricCollection(Accuracy(task="binary"))

    def predict(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return 2 * x


class DummyTaskDict(LightningModule):
    """Task that returns a mapping from predict()."""

    test_metrics: MetricCollection = MetricCollection(Accuracy(task="binary"))

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {"predictions": 3 * x, "extra": x.sum(dim=1)}


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
    assert "full_risk" in res and torch.allclose(
        res["full_risk"], torch.tensor(0.75)
    )
    # Selected accuracy considers first two only, both correct => 1.0
    assert "selective_risk" in res and torch.allclose(
        res["selective_risk"], torch.tensor(1.0)
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
    assert torch.allclose(res["full_risk"], torch.tensor(0.5))
    # Selected accuracy on first two samples: both correct => 1.0
    assert torch.allclose(res["selective_risk"], torch.tensor(1.0))


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
