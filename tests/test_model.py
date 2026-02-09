from typing import override

import pytest
import torch
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MetricCollection

from seapig import RiskCoverage, RiskCoverageMetric, SelectiveInferenceTask
from seapig.scores.base import ConfidenceScore


class DummyScore(ConfidenceScore):
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

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    def fit(self, x: torch.Tensor) -> None:
        """Dummy implementation of fit."""
        pass

    @override
    def set_threshold(self, q: float) -> None:
        """Dummy implementation of set_threshold."""
        self.threshold = q


class DummyTaskTensor(LightningModule):
    """Task that returns a tensor from predict()."""

    test_metrics: MetricCollection = MetricCollection(Accuracy(task="binary"))

    def predict(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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
        _ = SelectiveInferenceTask(
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


def test_test_step_updates_metrics_and_logs_rc(monkeypatch) -> None:
    task = DummyTaskTensor()
    score = DummyScore()
    w = SelectiveInferenceTask(
        task=task, score=score, rc_metric=RiskCoverageMetric()
    )

    calls: dict[str, object] = {"log_arg": None}

    def fake_log_dict(arg, batch_size=None, **kwargs):  # noqa: ANN001
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
    res = w.test_metrics.compute()
    assert any(k.startswith("full/") for k in res.keys())
    assert any(k.startswith("selected/") for k in res.keys())
    assert any(k.startswith("rejected/") for k in res.keys())

    # RiskCoverageMetric stats should have been logged
    assert isinstance(calls["log_arg"], RiskCoverageMetric)
    metrics = calls["log_arg"].compute()
    assert "rc/auc_empirical" in metrics
    assert "rc/auc_reference" in metrics
    assert "rc/auc_excess" in metrics
    rc = calls["log_arg"].get_curve()
    assert isinstance(rc, RiskCoverage)


def test_test_step_with_alt_keys_updates_metrics(monkeypatch) -> None:
    task = DummyTaskTensor()
    score = DummyScore()
    w = SelectiveInferenceTask(
        task=task, score=score, input_key="x", target_key="y"
    )

    monkeypatch.setattr(w, "log_dict", lambda *a, **k: None)

    # 1D inputs/targets so Accuracy is well-defined
    batch = {"x": torch.tensor([0.0, 1.0]), "y": torch.tensor([0, 1])}
    w.test_step(batch, batch_idx=0)

    res = w.test_metrics.compute()
    assert any(k.startswith("full/") for k in res.keys())
    assert any(k.startswith("selected/") for k in res.keys())


def test_test_step_missing_keys_raise_keyerror(monkeypatch) -> None:
    w = SelectiveInferenceTask(task=DummyTaskTensor(), score=DummyScore())
    monkeypatch.setattr(w, "log_dict", lambda *a, **k: None)

    with pytest.raises(KeyError):
        w.test_step({"label": torch.tensor([0])}, batch_idx=0)
    with pytest.raises(KeyError):
        w.test_step({"image": torch.zeros(1, 2)}, batch_idx=0)


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


@pytest.mark.filterwarnings("ignore:reference is not registered.*")
def test_get_risk_coverage_curve() -> None:
    task = DummyTaskTensor()
    score = DummyScore()
    w = SelectiveInferenceTask(
        task=task, score=score, rc_metric=RiskCoverageMetric()
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
