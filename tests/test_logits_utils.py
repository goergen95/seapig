"""Tests for seapig.scores.logit_utils"""

import math

import pytest
import torch

import seapig.scores.logits_utils as lu


def test_expect_rank_valid_and_invalid():
    # Valid rank 2 when per_member=False
    logits = torch.randn(5, 3)
    lu._expect_rank(logits, per_member=False, name="test")

    # Valid rank 3 when per_member=True
    logits3 = torch.randn(5, 3, 2)
    lu._expect_rank(logits3, per_member=True, name="test")

    # Invalid rank cases
    with pytest.raises(ValueError) as exc:
        lu._expect_rank(logits, per_member=True, name="test")
    assert "must be 3-D" in str(exc.value)

    with pytest.raises(ValueError) as exc2:
        lu._expect_rank(logits3, per_member=False, name="test")
    assert "must be 2-D" in str(exc2.value)


def test_multiclass_canonicalize_and_labels():
    task = lu.MulticlassTask()
    # per_member=False adds a trailing dimension
    logits = torch.randn(4, 5)
    out = task.canonicalize(logits, per_member=False)
    assert out.shape == (4, 5, 1)
    # per_member=True returns unchanged tensor
    logits_pm = torch.randn(4, 5, 2)
    out2 = task.canonicalize(logits_pm, per_member=True)
    assert out2 is logits_pm

    # Labels handling – squeeze singleton dim
    labels = torch.tensor([[1], [0], [2], [1]])
    proc = task.prepare_labels(labels)
    assert proc.shape == (4,)
    assert proc.dtype == torch.long
    # Incorrect label shape raises
    with pytest.raises(ValueError):
        task.prepare_labels(torch.randn(4, 2))


def test_binary_two_logit_task_errors():
    task = lu.BinaryTwoLogitTask()
    logits = torch.randn(3, 2)
    out = task.canonicalize(logits, per_member=False)
    assert out.shape == (3, 2, 1)
    # Wrong class dimension triggers error
    bad = torch.randn(3, 3)
    with pytest.raises(ValueError) as exc:
        task.canonicalize(bad, per_member=False)
    assert "K=2" in str(exc.value)


def test_multilabel_canonicalize_and_labels():
    task = lu.MultilabelTask()
    logits = torch.randn(2, 4)
    out = task.canonicalize(logits, per_member=False)
    assert out.shape == (2, 4, 1)
    logits_pm = torch.randn(2, 4, 3)
    assert task.canonicalize(logits_pm, per_member=True) is logits_pm

    # Labels must be 2‑D
    labels = torch.randn(2, 4)
    proc = task.prepare_labels(labels)
    assert proc.shape == (2, 4)
    assert proc.dtype == torch.float
    with pytest.raises(ValueError):
        task.prepare_labels(torch.randn(2))


def test_binary_single_logit_canonicalize_shapes():
    task = lu.BinarySingleLogitTask()
    # per_member=True, (N, M) -> (N, 1, M)
    logits = torch.randn(5, 7)
    out = task.canonicalize(logits, per_member=True)
    assert out.shape == (5, 1, 7)
    # Already (N, 1, M)
    logits2 = torch.randn(5, 1, 3)
    assert task.canonicalize(logits2, per_member=True) is logits2
    # per_member=False, (N,) -> (N, 1, 1)
    logits3 = torch.randn(6)
    out3 = task.canonicalize(logits3, per_member=False)
    assert out3.shape == (6, 1, 1)
    # per_member=False, (N, 1) -> (N, 1, 1)
    logits4 = torch.randn(6, 1)
    out4 = task.canonicalize(logits4, per_member=False)
    assert out4.shape == (6, 1, 1)
    # Invalid shape raises
    with pytest.raises(ValueError):
        task.canonicalize(torch.randn(4, 2, 2), per_member=False)

    # Label preparation
    labels = torch.tensor([0, 1, 1, 0])
    proc = task.prepare_labels(labels)
    assert proc.shape == (4, 1)
    assert proc.dtype == torch.float


def test_binary_task_resolve():
    # Two‑logit variant when class dimension = 2
    logits_two = torch.randn(3, 2)
    task = lu.BinaryTask().resolve(logits_two, per_member=False)
    assert isinstance(task, lu.BinaryTwoLogitTask)
    # Otherwise single‑logit variant
    logits_one = torch.randn(3, 1)
    task2 = lu.BinaryTask().resolve(logits_one, per_member=False)
    assert isinstance(task2, lu.BinarySingleLogitTask)


def test_get_task_valid_and_invalid():
    assert isinstance(lu.get_task("multiclass"), lu.MulticlassTask)
    assert isinstance(lu.get_task("binary"), lu.BinaryTask)
    assert isinstance(lu.get_task("multilabel"), lu.MultilabelTask)
    with pytest.raises(ValueError) as exc:
        lu.get_task("unknown")
    assert "Unknown task" in str(exc.value)


def test_temperature_scaler_errors_and_basic_fit():
    scaler = lu.TemperatureScaler(init=1.0, max_iter=10, lr=0.01)
    logits = torch.randn(5, 3)
    labels = torch.tensor([0, 1, 2, 1, 0])
    # Fit using cross‑entropy loss – should succeed
    temp = scaler.fit(logits, labels, lu.F.cross_entropy)
    assert isinstance(temp, float)
    assert temp > 0

    # Empty logits raise
    with pytest.raises(ValueError):
        scaler.fit(torch.empty(0, 3), labels[:0], lu.F.cross_entropy)
    # Mismatched lengths raise
    with pytest.raises(ValueError):
        scaler.fit(logits, torch.tensor([0, 1]), lu.F.cross_entropy)


def test_temperature_scaler_fallback_to_adam(monkeypatch):
    # Replace LBFGS with a stub that always raises RuntimeError
    class BadLBFGS:
        def __init__(self, *_, **__):
            pass

        def zero_grad(self):
            pass

        def step(self, closure):
            raise RuntimeError("forced failure")

    monkeypatch.setattr(lu.torch.optim, "LBFGS", BadLBFGS)
    scaler = lu.TemperatureScaler(init=0.5, max_iter=5, lr=0.1)
    logits = torch.randn(4, 2)
    labels = torch.tensor([0, 1, 0, 1])
    temp = scaler.fit(logits, labels, lu.F.cross_entropy)
    assert math.isfinite(temp) and temp > 0
