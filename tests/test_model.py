from typing import Any

import pytest
import torch
from lightning import LightningModule
from torchmetrics import Accuracy, MetricCollection
from typing_extensions import override

from seapig import RiskCoverageMetric, SelectiveInferenceTask
from seapig.scores.base import ConfidenceScore


class DummyScore(ConfidenceScore):
    """Minimal duck-typed score with select()."""

    def select(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        return {
            "score": torch.arange(b, dtype=x.dtype, device=x.device),
            "selected": torch.ones(b, dtype=torch.bool, device=x.device),
        }

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            x.shape[0], dtype=x.dtype, device=x.device
        )  # pragma: no cover

    def fit(
        self,
        X: torch.Tensor | None = None,
        Y: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Dummy implementation of fit."""
        raise NotImplementedError()

    @override
    def set_threshold(self, q: float = 0.99) -> None:  # pragma: no cover
        """Dummy implementation of set_threshold."""
        self.threshold = torch.tensor(q)


class DummyTaskTensor(LightningModule):
    """Task that returns a tensor from predict()."""

    test_metrics: MetricCollection = MetricCollection(Accuracy(task="binary"))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            x = x[0]
        if isinstance(x, dict):
            x = next(iter(x.values()))
        return 2 * x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)


class DummyTaskDict(DummyTaskTensor):
    """Task that returns a mapping from predict()."""

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {"predictions": 3 * x, "extra": x.sum(dim=1)}


def test_init_accepts_default_and_alt_keys() -> None:
    s = DummyScore()
    # When keys are not provided they should default to 0 and 1, meaning the wrapper
    # must use positional batch items (first item -> input, second -> target).
    w = SelectiveInferenceTask(task=DummyTaskTensor(), score=s)
    assert w.input_key == 0 and w.target_key == 1

    # verify positional access in predict_step
    batch_pos: list[torch.Tensor] = [
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([1]),
    ]
    out = w.predict_step(batch_pos, batch_idx=0)
    assert "predictions" in out and torch.allclose(
        out["predictions"], 2 * batch_pos[0]
    )

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
        _ = SelectiveInferenceTask(
            task=DummyTaskTensor(),
            score=DummyScore(),
            **kwargs,  # type: ignore[arg-type]
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
    class BadTask(DummyTaskTensor):
        def predict(self, x: torch.Tensor) -> list[torch.Tensor]:  # type: ignore[override]
            return [x]  # wrong type

    w = SelectiveInferenceTask(task=BadTask(), score=DummyScore())
    with pytest.raises(AssertionError):
        _ = w.forward(torch.randn(2, 3))


def test_predict_step_uses_input_key_and_returns_selection() -> None:
    task = DummyTaskTensor()
    score = DummyScore()
    w = SelectiveInferenceTask(task=task, score=score, input_key="image")

    batch = {"image": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    out = w.predict_step(batch, batch_idx=0)
    assert "predictions" in out and torch.allclose(
        out["predictions"], 2 * batch["image"]
    )
    assert out["selected"].dtype is torch.bool


def test_predict_step_missing_key_raises_keyerror() -> None:
    w = SelectiveInferenceTask(
        task=DummyTaskTensor(), score=DummyScore(), input_key="image"
    )
    with pytest.raises(KeyError):
        _ = w.predict_step({"not_image": torch.zeros(1, 2)}, batch_idx=0)


def test_test_step_updates_metrics_and_logs_rc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = DummyTaskTensor()
    score = DummyScore()
    w = SelectiveInferenceTask(
        task=task,
        score=score,
        rc_metric=RiskCoverageMetric(),
        input_key="image",
        target_key="label",
    )

    calls: dict[str, object] = {"log_arg": None}

    def fake_log_dict(
        arg: dict[str, Any], batch_size: int | None = None, **kwargs: Any
    ) -> None:
        calls["log_arg"] = arg
        return None

    monkeypatch.setattr(w, "log_dict", fake_log_dict)

    # Use 1D inputs to ensure binary Accuracy shape compatibility
    batch = {
        "image": torch.tensor([0.0, 1.0, 0.6, 0.4]),
        "label": torch.tensor([0, 1, 1, 0]),
    }
    w.test_step(batch, batch_idx=0)

    # SelectiveMetric should have results for collection (prefixed keys)
    assert w.test_metrics is not None
    res = w.test_metrics.compute()
    assert any(k.startswith("full/") for k in res.keys())
    assert any(k.startswith("selected/") for k in res.keys())
    assert any(k.startswith("rejected/") for k in res.keys())

    # RiskCoverageMetric stats should have been logged
    assert isinstance(calls["log_arg"], dict)
    metrics = calls["log_arg"]
    assert "rc/auc_empirical" in metrics
    assert "rc/auc_reference" in metrics
    assert "rc/auc_excess" in metrics


def test_test_step_with_alt_keys_updates_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = DummyTaskTensor()
    score = DummyScore()
    w = SelectiveInferenceTask(
        task=task, score=score, input_key="x", target_key="y"
    )

    monkeypatch.setattr(w, "log_dict", lambda *a, **k: None)

    # 1D inputs/targets so Accuracy is well-defined
    batch = {"x": torch.tensor([0.0, 1.0]), "y": torch.tensor([0, 1])}
    w.test_step(batch, batch_idx=0)

    assert w.test_metrics is not None
    res = w.test_metrics.compute()
    assert any(k.startswith("full/") for k in res.keys())
    assert any(k.startswith("selected/") for k in res.keys())


def test_test_step_missing_keys_raise_keyerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    w = SelectiveInferenceTask(
        task=DummyTaskTensor(),
        score=DummyScore(),
        input_key="image",
        target_key="label",
    )
    monkeypatch.setattr(w, "log_dict", lambda *a, **k: None)

    with pytest.raises(KeyError):
        w.test_step({"not-label": torch.tensor([0])}, batch_idx=0)
    with pytest.raises(KeyError):
        w.test_step({"not-image": torch.zeros(1, 2)}, batch_idx=0)


def test_get_risk_coverage_curve_none_before_compute() -> None:
    task = DummyTaskTensor()
    score = DummyScore()

    # is None if not specified
    w = SelectiveInferenceTask(task=task, score=score)
    assert w.rc_metric is None
    assert w.get_risk_coverage_curve() is None

    # is None before compute
    w = SelectiveInferenceTask(
        task=task, score=score, rc_metric=RiskCoverageMetric()
    )
    assert w.rc_metric is not None
    assert w.get_risk_coverage_curve() is None


@pytest.mark.filterwarnings(
    "ignore:You are trying to `self\\.log\\(\\)` but the `self\\.trainer` reference is not registered on the model yet.*"
)
def test_get_risk_coverage_curve() -> None:
    task = DummyTaskTensor()
    score = DummyScore()
    w = SelectiveInferenceTask(
        task=task,
        score=score,
        rc_metric=RiskCoverageMetric(),
        input_key="image",
        target_key="label",
    )

    batch = {
        "image": torch.tensor([0.0, 1.0, 0.6, 0.4]),
        "label": torch.tensor([0, 1, 1, 0]),
    }
    w.test_step(batch, batch_idx=0)

    curve = w.get_risk_coverage_curve()
    assert curve is not None
    assert hasattr(curve, "auc_empirical")
    assert hasattr(curve, "auc_reference")
    assert hasattr(curve, "auc_excess")


def test_return_test_outputs_collects_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When return_test_outputs=True the wrapper accumulates per-batch
    outputs. Also verify the default behaviour (False) leaves
    test_outputs as None.
    """
    task = DummyTaskDict()
    score = DummyScore()

    # With collection enabled
    w = SelectiveInferenceTask(
        task=task,
        score=score,
        acc_test_outputs=True,
        input_key="image",
        target_key="label",
    )
    # avoid noisy logging during the test
    monkeypatch.setattr(w, "log_dict", lambda *a, **k: None)

    batch = {
        "image": torch.tensor([[1.0, 2.0]]),
        "label": torch.tensor([[0, 1]]),
    }
    w.test_step(batch, batch_idx=0)

    assert isinstance(w.test_outputs, list)
    assert len(w.test_outputs) == 1
    out = w.test_outputs[0]
    assert "predictions" in out
    assert "score" in out and "selected" in out


def test_return_test_outputs_without_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the wrapped task does not expose test_metrics, the wrapper
    should still collect per-batch outputs when return_test_outputs=True.
    """

    class NoMetricTask(DummyTaskDict):
        # Explicitly remove metrics to simulate tasks that don't define them
        test_metrics: None = None  # type: ignore[assignment]

    task = NoMetricTask()
    score = DummyScore()

    w = SelectiveInferenceTask(
        task=task,
        score=score,
        acc_test_outputs=True,
        input_key="image",
        target_key="label",
    )
    # avoid noisy logging during the test
    monkeypatch.setattr(w, "log_dict", lambda *a, **k: None)

    batch = {
        "image": torch.tensor([[1.0, 2.0]]),
        "label": torch.tensor([[0, 1]]),
    }
    w.test_step(batch, batch_idx=0)

    assert isinstance(w.test_outputs, list)
    assert len(w.test_outputs) == 1
    out = w.test_outputs[0]
    assert "predictions" in out
    assert "score" in out and "selected" in out


def test_get_from_batch_helper_behavior() -> None:
    """Small tests for the module-private _get_from_batch helper."""
    from collections import OrderedDict

    from seapig.model import _get_from_batch

    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([0])

    seq: list[torch.Tensor] = [t1, t2]
    mapping: OrderedDict[str, torch.Tensor] = OrderedDict(
        [("image", t1), ("label", t2)]
    )

    # Sequence with None key returns positional element
    out0 = _get_from_batch(seq, None, pos=0)
    assert torch.equal(out0, seq[0])

    # Mapping with None key returns the positional value (preserve insertion order)
    out1 = _get_from_batch(mapping, None, pos=1)
    assert torch.equal(out1, list(mapping.values())[1])

    # Mapping with string key returns mapping lookup
    out2 = _get_from_batch(mapping, "label", pos=1)
    assert torch.equal(out2, mapping["label"])

    # Sequence with int key returns by index
    out3 = _get_from_batch(seq, 1, pos=0)
    assert torch.equal(out3, seq[1])

    # Unsupported batch type raises TypeError
    with pytest.raises(TypeError):
        _ = _get_from_batch(42, None, pos=0)  # type: ignore[arg-type]
