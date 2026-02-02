import torch
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MetricCollection, Precision, Recall

from seapig import SelectiveInferenceTask, SelectiveMetric


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
    score = DummyScore()
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
