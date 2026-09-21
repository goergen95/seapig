from typing import Any

import pytest
import torch
from lightning import LightningModule

from seapig import RiskCoverageMetric, SelectiveInferenceTask
from seapig.scores import UncertaintyScore
from seapig.scores.logits import SoftmaxScore
from tests.fixtures import (
    BadForwardTask,
    BadPredictStepTask,
    DummyScore,
    DummyTaskDict,
    NoMetricTask,
)


class BareScore(UncertaintyScore):
    def fit(self):
        pass

    def select(self, query):
        return query  # pragma: no cover

    def score(self, query):
        return query  # pragma: no cover


class SimpleTask(LightningModule):
    def __init__(self, output):
        super().__init__()
        self._output = output

    def predict(self, batch):  # pragma: no cover
        return self._output


def test_init_requires_predict_method():
    class BadPredictTask(LightningModule):
        predict: int = 123

    with pytest.raises(
        TypeError,
        match="`task` is required to expose a `predict\\(\\)` method.",
    ):
        SelectiveInferenceTask(task=BadPredictTask(), score=BareScore())


def test_init_invalid_test_metrics_type():
    task = SimpleTask(
        output={"prediction": torch.tensor([1]), "label": torch.tensor([0])}
    )
    task.test_metrics = 123  # type: ignore
    with pytest.raises(
        TypeError,
        match="Wrapped task's test_metrics must be a Metric or MetricCollection",
    ):
        SelectiveInferenceTask(task=task, score=BareScore())


def test_init_invalid_rc_metric_type():
    task = SimpleTask(
        output={"prediction": torch.tensor([1]), "label": torch.tensor([0])}
    )
    with pytest.raises(
        TypeError,
        match="rc_metric must be a seapig RiskCoverageMetric instance or None.",
    ):
        SelectiveInferenceTask(
            task=task,
            score=BareScore(),
            rc_metric=object(),  # type: ignore
        )


def test_test_step_missing_prediction_key():

    # ``predict`` returns only ``label`` – ``prediction`` is absent.
    task = SimpleTask(output={"label": torch.tensor([0])})
    inference = SelectiveInferenceTask(task=task, score=BareScore())
    with pytest.raises(KeyError, match="prediction"):
        inference.test_step(batch={}, batch_idx=0)


def test_test_step_missing_label_key():
    # ``predict`` returns only ``prediction`` – ``label`` is absent.
    task = SimpleTask(output={"prediction": torch.tensor([1])})
    inference = SelectiveInferenceTask(task=task, score=BareScore())
    with pytest.raises(KeyError, match="label"):
        inference.test_step(batch={}, batch_idx=0)


def test_predict_step_missing_prediction_key():
    task = SimpleTask(output={"label": torch.tensor([0])})
    inference = SelectiveInferenceTask(task=task, score=BareScore())
    with pytest.raises(KeyError, match="prediction"):
        inference.predict_step(batch={}, batch_idx=0)


def test_init_accepts_default() -> None:
    s = DummyScore()
    w = SelectiveInferenceTask(task=DummyTaskDict(), score=s)

    # verify positional access in predict_step
    batch_pos: list[torch.Tensor] = [
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([1]),
    ]
    out = w.predict_step(batch_pos, batch_idx=0)
    assert "prediction" in out


def test_forward_wraps_tensor_and_merges_selection() -> None:
    task = DummyTaskDict()
    score = DummyScore()
    w = SelectiveInferenceTask(task=task, score=score)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = w.forward(x)

    # predictions wrapped and equal to 2*x
    assert "prediction" in out
    # selection merged
    assert "score" in out and "selected" in out


def test_forward_keeps_dict_output_and_extra_keys() -> None:
    task = DummyTaskDict()
    score = DummyScore()
    w = SelectiveInferenceTask(task=task, score=score)

    x = torch.tensor([[1.0, 2.0]])
    out = w.forward(x)

    # ensure original extra entries survive merge
    assert "extra" in out and out["extra"].shape[0] == x.shape[0]
    assert "score" in out and "selected" in out


def test_predict_step_returns_selection() -> None:
    task = DummyTaskDict()
    score = DummyScore()
    w = SelectiveInferenceTask(task=task, score=score)

    batch = {"image": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    out = w.predict_step(batch, batch_idx=0)
    assert "prediction" in out
    assert out["selected"].dtype is torch.bool


def test_test_step_updates_metrics_and_logs_rc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = DummyTaskDict()
    score = DummyScore()
    w = SelectiveInferenceTask(
        task=task, score=score, rc_metric=RiskCoverageMetric()
    )

    calls: dict[str, object] = {"log_arg": None}

    def fake_log_dict(
        arg: dict[str, Any], batch_size: int | None = None, **kwargs: Any
    ) -> None:
        calls["log_arg"] = arg

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
    assert any(k.startswith("full/") for k in res)
    assert any(k.startswith("selected/") for k in res)
    assert any(k.startswith("rejected/") for k in res)

    # RiskCoverageMetric stats should have been logged
    assert isinstance(calls["log_arg"], dict)
    metrics = calls["log_arg"]
    assert "rc/auc_empirical" in metrics
    assert "rc/auc_reference" in metrics
    assert "rc/auc_excess" in metrics


def test_test_step_with_alt_keys_updates_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = DummyTaskDict()
    score = DummyScore()
    w = SelectiveInferenceTask(task=task, score=score)

    monkeypatch.setattr(w, "log_dict", lambda *a, **k: None)

    # 1D inputs/targets so Accuracy is well-defined
    batch = {"x": torch.tensor([0.0, 1.0]), "y": torch.tensor([0, 1])}
    w.test_step(batch, batch_idx=0)

    assert w.test_metrics is not None
    res = w.test_metrics.compute()
    assert any(k.startswith("full/") for k in res)
    assert any(k.startswith("selected/") for k in res)


def test_get_risk_coverage_curve_none_before_compute() -> None:
    task = DummyTaskDict()
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
    task = DummyTaskDict()
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
    w = SelectiveInferenceTask(task=task, score=score, acc_test_outputs=True)
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
    assert "prediction" in out
    assert "score" in out and "selected" in out


def test_return_test_outputs_without_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the wrapped task does not expose test_metrics, the wrapper
    should still collect per-batch outputs when return_test_outputs=True.
    """

    task = NoMetricTask()
    score = DummyScore()

    w = SelectiveInferenceTask(task=task, score=score, acc_test_outputs=True)
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
    assert "prediction" in out
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
        _ = _get_from_batch(42, None, pos=0)  # type: ignore[arg-type, ty:invalid-argument-type]


def test_forward_raises_type_error_for_invalid_output():
    score = DummyScore()
    w = SelectiveInferenceTask(task=BadForwardTask(), score=score)  # type: ignore
    x = torch.tensor([[1.0, 2.0]])
    with pytest.raises(TypeError):
        w.forward(x)


def test_select_raises_when_logit_key_missing_for_logit_score():
    # SoftmaxScore inherits LogitScore and expects a ``logit`` key.
    score = SoftmaxScore()
    score.fit(torch.randn(100, 2))
    # DummyTaskDict returns ``prediction`` and ``embedding`` but no ``logit``.
    from tests.fixtures import DummyTaskDict

    w = SelectiveInferenceTask(task=DummyTaskDict(), score=score)
    x = torch.tensor([[1.0, 2.0]])
    with pytest.raises(KeyError, match="logit"):
        w.forward(x)


def test_predict_step_asserts_dict_output_from_task_predict_step():
    score = DummyScore()
    # Use default positional keys (0 for input, 1 for target) – target is unused here.
    w = SelectiveInferenceTask(task=BadPredictStepTask(), score=score)
    batch = [torch.tensor([[1.0, 2.0]]), torch.tensor([0])]
    with pytest.raises(TypeError, match="must return a dict"):
        w.predict_step(batch, batch_idx=0)
