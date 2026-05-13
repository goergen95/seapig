import pytest
import torch
from lightning import LightningModule
from torchmetrics import (
    Accuracy,
    MeanAbsoluteError,
    MeanMetric,
    MeanSquaredError,
    MetricCollection,
    Precision,
    Recall,
)

from seapig import SelectiveMetric


class DummyTaskTensor(LightningModule):
    """Task that returns a tensor from predict()."""

    test_metrics: MetricCollection = MetricCollection(Accuracy(task="binary"))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * x  # pragma: no cover

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return x  # pragma: no cover

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)  # pragma: no cover


class DummyTaskDict(LightningModule):
    """Task that returns a mapping from predict()."""

    test_metrics: MetricCollection = MetricCollection(Accuracy(task="binary"))

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"predictions": 3 * x, "extra": x.sum(dim=1)}  # pragma: no cover


def test_selective_metric_binary_accuracy_full_vs_selected() -> None:
    base = Accuracy(task="binary")
    sel = SelectiveMetric(base)

    preds = torch.tensor([0.9, 0.4, 0.6, 0.8])
    selected = torch.tensor([1, 1, 0, 0])  # non-bool mask accepted
    target = torch.tensor([1, 0, 1, 0])

    sel.update(preds, target, selected)
    res = sel.compute()

    assert "full/BinaryAccuracy" in res and torch.allclose(
        res["full/BinaryAccuracy"], torch.tensor(0.75)
    )
    assert "selected/BinaryAccuracy" in res and torch.allclose(
        res["selected/BinaryAccuracy"], torch.tensor(1.0)
    )
    assert "rejected/BinaryAccuracy" in res and torch.allclose(
        res["rejected/BinaryAccuracy"], torch.tensor(0.5)
    )


def test_selective_metric_with_metric_collection_prefix_keys() -> None:
    coll = MetricCollection({"acc": Accuracy(task="binary")})
    sel = SelectiveMetric(coll)

    preds = torch.tensor([0.9, 0.4, 0.6, 0.8])
    selected = torch.tensor([1, 0, 1, 0])
    target = torch.tensor([1, 0, 1, 0])

    sel.update(preds, target, selected)
    res = sel.compute()

    assert any(k.startswith("full/") for k in res.keys())
    assert any(k.startswith("selected/") for k in res.keys())
    for v in res.values():
        assert isinstance(v, torch.Tensor) and v.ndim == 0


def test_selective_metric_with_metric_collection_output_naming() -> None:
    metrics = MetricCollection(
        {
            "accuracy": Accuracy(task="binary"),
            "precision": Precision(task="binary"),
            "recall": Recall(task="binary"),
        }
    )
    sel = SelectiveMetric(metrics)

    preds = torch.tensor([0.9, 0.4, 0.6, 0.8])
    selected = torch.tensor([1, 0, 1, 0])
    target = torch.tensor([1, 0, 1, 0])

    sel.update(preds, target, selected)
    res = sel.compute()

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
    for value in res.values():
        assert isinstance(value, torch.Tensor) and value.ndim == 0


def test_selective_metric_reset() -> None:
    base = Accuracy(task="binary")
    sel = SelectiveMetric(base)

    preds = torch.tensor([0.9, 0.4, 0.6, 0.8])
    selected = torch.tensor([1, 0, 1, 0])
    target = torch.tensor([1, 0, 1, 0])
    sel.update(preds, target, selected)
    _ = sel.compute()
    _ = sel.metrics["full"].compute()

    sel.reset()
    with pytest.warns(
        UserWarning, match="was called before the ``update`` method"
    ):
        res = sel.metrics["full"].compute()
    assert isinstance(res, dict)
    assert "BinaryAccuracy" in res
    assert res["BinaryAccuracy"] == 0.0


@pytest.mark.parametrize(
    "selection_behavior", ["always_select", "always_reject"]
)
def test_selective_metric_with_mock_metric_collection(
    selection_behavior: str,
) -> None:
    base_metrics = MetricCollection({"mean": MeanMetric()})
    selective_metric = SelectiveMetric(base_metrics)

    batch_size = 10
    predictions = torch.randn(batch_size)
    target = torch.randn(batch_size)
    selection_mask = (
        torch.ones(batch_size, dtype=torch.bool)
        if selection_behavior == "always_select"
        else torch.zeros(batch_size, dtype=torch.bool)
    )

    selective_metric.update(predictions, target, selection_mask)
    results = selective_metric.compute()

    if selection_behavior == "always_select":
        assert "selected/mean" in results
        assert results["selected/mean"] != 0
        assert "rejected/mean" in results
        assert results["rejected/mean"] == 0
    else:
        assert "rejected/mean" in results
        assert results["rejected/mean"] != 0
        assert "selected/mean" in results
        assert results["selected/mean"] == 0


def _tensor_close(a: torch.Tensor, b: float, tol: float = 1e-6) -> bool:
    return bool(
        torch.allclose(
            a, torch.tensor(b, dtype=a.dtype, device=a.device), atol=tol
        )
    )


def test_selective_metric_single_metric_selected_and_rejected_values() -> None:
    base = MeanAbsoluteError()
    sm = SelectiveMetric(base)

    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([1.0, 3.0, 2.0, 5.0])
    selected = torch.tensor([1, 0, 1, 0], dtype=torch.bool)

    sm.update(preds, target, selected)
    res = sm.compute()
    assert "full/MeanAbsoluteError" in res
    assert _tensor_close(res["full/MeanAbsoluteError"], 0.75)
    assert "selected/MeanAbsoluteError" in res
    assert _tensor_close(res["selected/MeanAbsoluteError"], 0.5)
    assert "rejected/MeanAbsoluteError" in res
    assert _tensor_close(res["rejected/MeanAbsoluteError"], 1.0)
    comp = sm.compute()
    keys = list(comp.keys())
    items = list(comp.items())
    values = list(comp.values())
    assert len(keys) == len(items) == len(values)


def test_selective_metric_no_selected_updates_metric_not_called() -> None:
    base = MeanAbsoluteError()
    sm = SelectiveMetric(base)

    preds = torch.tensor([0.0, 1.0, 2.0])
    target = torch.tensor([0.5, 0.5, 0.5])
    selected = torch.zeros(3, dtype=torch.bool)

    sm.update(preds, target, selected)
    res = sm.compute()
    assert "selected/MeanAbsoluteError" in res
    assert _tensor_close(res["selected/MeanAbsoluteError"], 0.0)
    assert "rejected/MeanAbsoluteError" in res
    assert res["rejected/MeanAbsoluteError"].numel() == 1
    assert res["rejected/MeanAbsoluteError"].dtype == torch.float32


def test_selective_metric_with_metric_collection_and_prefixing() -> None:
    coll = MetricCollection(
        {"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}
    )
    smc = SelectiveMetric(coll)

    preds = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.2, 1.8, 2.5])
    selected = torch.tensor([1, 1, 0], dtype=torch.bool)

    smc.update(preds, target, selected)
    out = smc.compute()
    for prefix in ("full", "selected", "rejected"):
        assert f"{prefix}/mae" in out
        assert f"{prefix}/mse" in out
    assert isinstance(out["full/mae"], torch.Tensor)
    assert isinstance(out["full/mse"], torch.Tensor)
