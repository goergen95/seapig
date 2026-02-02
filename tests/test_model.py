import pytest
import torch
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MetricCollection

from seapig import SelectiveInferenceTask


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


def test_init_accepts_default_and_alt_keys() -> None:
    s = DummyScore()
    w = SelectiveInferenceTask(task=DummyTaskTensor(), score=s)
    assert w.input_key == "image" and w.target_key == "label"

    w2 = SelectiveInferenceTask(
        task=DummyTaskTensor(), score=s, input_key="x", target_key="y_true"
    )
    assert w2.input_key == "x" and w2.target_key == "y_true"


@pytest.mark.parametrize(
    "kw, key, value",
    [
        ("input_key", "bad_input", "not-a-key"),
        ("target_key", "bad_target", "also-bad"),
    ],
)
def test_init_rejects_invalid_keys(kw: str, key: str, value: str) -> None:
    kwargs: dict[str, object] = {kw: value}
    with pytest.raises(ValueError):
        SelectiveInferenceTask(
            task=DummyTaskTensor(), score=DummyScore(), **kwargs
        )


def test_forward_wraps_tensor_and_merges_selection() -> None:
    task = DummyTaskTensor()
    score = DummyScore()
    w = SelectiveInferenceTask(task=task, score=score)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = w.forward(x)

    # predictions wrapped and equal to 2*x
    assert "predictions" in out and torch.allclose(out["predictions"], 2 * x)
    # selection merged
    assert set(["score", "selected"]).issubset(out.keys())
    # device propagated to score.to(...)
    assert score.last_device == str(x.device)


def test_forward_keeps_dict_output_and_extra_keys() -> None:
    task = DummyTaskDict()
    score = DummyScore()
    w = SelectiveInferenceTask(task=task, score=score)

    x = torch.tensor([[1.0, 2.0]])
    out = w.forward(x)

    assert torch.allclose(out["predictions"], 3 * x)
    # ensure original extra entries survive merge
    assert "extra" in out and out["extra"].shape[0] == x.shape[0]
    assert "score" in out and "selected" in out


def test_forward_raises_when_predict_not_tensor_or_dict() -> None:
    class BadTask(LightningModule):
        test_metrics: MetricCollection = MetricCollection(
            Accuracy(task="binary")
        )

        def predict(self, x: torch.Tensor):  # type: ignore[override]
            return [x]  # wrong type

    w = SelectiveInferenceTask(task=BadTask(), score=DummyScore())
    with pytest.raises(AssertionError):
        _ = w.forward(torch.randn(2, 3))


def test_predict_step_uses_input_key_and_returns_selection() -> None:
    task = DummyTaskTensor()
    score = DummyScore()
    w = SelectiveInferenceTask(task=task, score=score)

    batch = {"image": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    out = w.predict_step(batch, batch_idx=0)
    assert "predictions" in out and torch.allclose(
        out["predictions"], 2 * batch["image"]
    )  # type: ignore[index]
    assert out["selected"].dtype is torch.bool


def test_predict_step_missing_key_raises_keyerror() -> None:
    w = SelectiveInferenceTask(task=DummyTaskTensor(), score=DummyScore())
    with pytest.raises(KeyError):
        _ = w.predict_step({"not_image": torch.zeros(1, 2)}, batch_idx=0)


def test_test_step_calls_metrics_and_logs(monkeypatch) -> None:
    task = DummyTaskTensor()
    score = DummyScore()
    w = SelectiveInferenceTask(task=task, score=score)

    calls: dict[str, object] = {"metrics_calls": 0, "log_arg": None}

    class DummyMetrics(MetricCollection):
        def __call__(self, outputs, y):  # noqa: ANN001
            calls["metrics_calls"] = int(calls["metrics_calls"]) + 1

    # attach a callable metrics object and stub log_dict to accept any input
    w.test_metrics = DummyMetrics(Accuracy(task="binary"))  # type: ignore[attr-defined]

    def fake_log_dict(arg, batch_size=None):  # noqa: ANN001
        calls["log_arg"] = arg
        return None

    monkeypatch.setattr(w, "log_dict", fake_log_dict)

    batch = {
        "image": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "label": torch.tensor([0, 1]),
    }
    w.test_step(batch, batch_idx=0)

    # metrics called once, and our stub log_dict received the metrics object
    assert calls["metrics_calls"] == 1
    assert calls["log_arg"] is w.test_metrics


def test_test_step_with_alt_keys(monkeypatch) -> None:
    task = DummyTaskTensor()
    score = DummyScore()
    w = SelectiveInferenceTask(
        task=task, score=score, input_key="x", target_key="y"
    )

    class DummyMetrics(MetricCollection):
        def __call__(self, outputs, y):  # noqa: ANN001
            # ensure y is the provided target
            assert torch.equal(y, torch.tensor([1, 1]))

    w.test_metrics = DummyMetrics(Accuracy(task="binary"))  # type: ignore[attr-defined]
    monkeypatch.setattr(w, "log_dict", lambda *a, **k: None)

    batch = {"x": torch.tensor([[1.0, 2.0]]), "y": torch.tensor([1, 1])}
    w.test_step(batch, batch_idx=0)


def test_test_step_missing_keys_raise_keyerror(monkeypatch) -> None:
    w = SelectiveInferenceTask(task=DummyTaskTensor(), score=DummyScore())
    monkeypatch.setattr(w, "log_dict", lambda *a, **k: None)

    with pytest.raises(KeyError):
        w.test_step({"label": torch.tensor([0])}, batch_idx=0)
    with pytest.raises(KeyError):
        w.test_step({"image": torch.zeros(1, 2)}, batch_idx=0)
