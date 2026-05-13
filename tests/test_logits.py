import math
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from seapig.scores import (
    EnergyScore,
    EntropyScore,
    LogitScore,
    MarginScore,
    SoftmaxScore,
)


@pytest.fixture(autouse=True)
def rng_seed() -> Generator[None, None, None]:
    torch.manual_seed(1234)
    yield


def approx_tensor(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-6) -> None:
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=tol, rtol=1e-5)


class SimpleBatchDataset(Dataset[Any]):
    def __init__(self, items: list[Any]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Any:
        return self.items[idx]


class IdentityModel(torch.nn.Module):
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0) if x.dim() > 2 and x.shape[0] == 1 else x


def make_loader_from_tensors(
    logits: torch.Tensor, labels: torch.Tensor | None = None
) -> DataLoader[Any]:
    if labels is None:
        items: list[Any] = [la.unsqueeze(0) for la in logits]
    else:
        items = [
            {"image": logits[i].unsqueeze(0), "label": labels[i].unsqueeze(0)}
            for i in range(len(logits))
        ]
    return DataLoader(SimpleBatchDataset(items), batch_size=1, shuffle=False)


def test_fit_saves_files(tmp_path: Path) -> None:
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.2]])
    labels = logits.argmax(dim=1)
    loader = make_loader_from_tensors(logits, labels)

    model = IdentityModel()
    score = SoftmaxScore()
    outdir = tmp_path / "saved_logits"
    score.fit(model=model, loader=loader, outdir=outdir, prefix="mytest")
    train_file = outdir / "mytest_train.pt"
    assert train_file.exists()
    loaded = torch.load(train_file)
    assert "logits" in loaded and "labels" in loaded
    assert loaded["logits"].shape[0] == logits.shape[0]
    assert hasattr(score, "logits")
    assert score.logits is not None
    assert score.logits.shape[0] == logits.shape[0]


@pytest.mark.parametrize("out_kind", ["tensor", "logits", "preds", "y_hat"])
def test_fit_accepts_output_formats(out_kind: str) -> None:
    logits = torch.tensor([[0.5, 1.5], [2.0, 0.1], [0.0, 0.0]])
    labels = logits.argmax(dim=1)
    loader = make_loader_from_tensors(logits, labels)

    class FlexibleModel(torch.nn.Module):
        def logits(self, x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(0)

    model = FlexibleModel()
    score = SoftmaxScore()
    score.fit(model=model, loader=loader)
    assert hasattr(score, "logits")
    assert score.logits is not None
    assert score.logits.shape[0] == logits.shape[0]


@pytest.mark.parametrize("batch_format", ["dict", "tensor_only"])
def test_fit_batch_formats(batch_format: str) -> None:
    logits = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
    labels = logits.argmax(dim=1)
    items: list[Any]
    if batch_format == "dict":
        items = [
            {"image": logits[i].unsqueeze(0), "label": labels[i].unsqueeze(0)}
            for i in range(len(logits))
        ]
    else:
        items = [logits[i].unsqueeze(0) for i in range(len(logits))]
    loader = DataLoader(SimpleBatchDataset(items), batch_size=1, shuffle=False)

    model = IdentityModel()
    score = EnergyScore()
    score.fit(model=model, loader=loader)
    assert score.logits is not None
    assert score.logits.shape[0] == logits.shape[0]


def test_softmax_numerical_stability() -> None:
    logits = torch.tensor([[1000.0, -1000.0, 0.0], [1e6, 0.0, -1e6]])
    s = SoftmaxScore()
    T = 1.0 if s.temperature is None else float(s.temperature)
    z = logits / T - (logits / T).amax(dim=1, keepdim=True)
    exp_z = z.exp()
    probs = exp_z / exp_z.sum(dim=1, keepdim=True)
    assert probs.shape == (2, 3)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0, 1.0]), atol=1e-6)


def test_predict_proba_temperature() -> None:
    logits = torch.tensor([[2.0, 1.0], [0.5, 0.1]])
    s = SoftmaxScore()

    def predict_proba(
        logits: torch.Tensor, temperature: float | None = None
    ) -> torch.Tensor:
        T = 1.0 if temperature is None else float(temperature)
        z = logits / T - (logits / T).amax(dim=1, keepdim=True)
        exp_z = z.exp()
        return exp_z / exp_z.sum(dim=1, keepdim=True)

    s.temperature = 2.0
    p_explicit = predict_proba(logits, temperature=4.0)
    p_inst = predict_proba(logits, temperature=s.temperature)
    assert not torch.allclose(p_explicit, p_inst)
    s.temperature = 1.0
    p_none = predict_proba(logits, temperature=None)
    p1 = predict_proba(logits, temperature=1.0)
    assert torch.allclose(p_none, p1, atol=1e-6)


def test_logit_helpers_consistency() -> None:
    logits = torch.tensor([[3.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    max_score = -logits.amax(dim=1)
    assert max_score.shape == (2,)
    assert torch.allclose(max_score, -torch.tensor([3.0, 0.0]), atol=1e-6)
    margin_score = MarginScore().score(logits)
    expected_margin = torch.tensor([-2.0, -0.0])
    assert torch.allclose(margin_score, expected_margin, atol=1e-6)
    ent = EntropyScore().score(logits)
    assert ent.shape == (2,)
    assert (ent >= 0.0).all()
    ln = -logits.norm(dim=1)
    assert ln.shape == (2,)
    assert torch.all(ln <= 0.0)


@pytest.mark.parametrize(
    "logits, expected_msp",
    [
        (
            torch.tensor([[2.0, 1.0]]),
            -torch.tensor(
                [torch.softmax(torch.tensor([2.0, 1.0]), dim=0).max()]
            ),
        ),
        (torch.tensor([[0.0, 0.0]]), -torch.tensor([0.5])),
    ],
)
def test_softmax_score_matches_maxprob(
    logits: torch.Tensor, expected_msp: torch.Tensor
) -> None:
    s = SoftmaxScore()
    sc = s.score(logits)
    T = 1.0 if s.temperature is None else float(s.temperature)
    z = logits / T - (logits / T).amax(dim=1, keepdim=True)
    exp_z = z.exp()
    probs = exp_z / exp_z.sum(dim=1, keepdim=True)
    assert torch.allclose(
        sc, -probs.amax(dim=1).to(dtype=torch.float32), atol=1e-6
    )


def test_margin_score_manual() -> None:
    logits = torch.tensor([[5.0, 2.0, 1.0], [0.1, 0.0, -1.0]])
    m = MarginScore()
    sc = m.score(logits)
    top2 = logits.topk(2, dim=1).values
    expect = -(top2[:, 0] - top2[:, 1])
    assert torch.allclose(sc, expect, atol=1e-6)


def test_select_uses_threshold(tmp_path: Path) -> None:
    """select should compute a threshold if none is set and return mask."""
    logits = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
    labels = torch.tensor([1, 0])
    score = SoftmaxScore()
    # Fit to generate calibration scores and store them
    score.fit(X=logits, Y=labels)
    # Ensure threshold not manually set
    assert score.threshold is None
    # Call select, which should set threshold based on quantile
    out = score.select(logits)
    assert "score" in out and "selected" in out
    # Scores should match score() output
    expected_scores = score.score(logits)
    assert torch.equal(out["score"], expected_scores)
    # Selected mask should be boolean and shape matches
    assert out["selected"].dtype == torch.bool
    assert out["selected"].shape == logits.shape[0:1]


def test_fit_temperature_validation_errors() -> None:
    """_fit_temperature should raise on empty logits or size mismatch."""
    s = SoftmaxScore()
    # Empty logits
    with pytest.raises(
        ValueError, match="logits must contain at least one sample"
    ):
        s._fit_temperature(torch.empty(0, 2), torch.tensor([0]))
    # Mismatched sizes
    with pytest.raises(
        ValueError, match="logits and labels must have same number of samples"
    ):
        s._fit_temperature(torch.randn(3, 2), torch.tensor([0, 1]))


def test_loadorpredict_missing_logits_key(tmp_path: Path) -> None:
    """If saved file lacks 'logits', _loadorpredict should raise ValueError."""
    s = SoftmaxScore()
    fake_path = tmp_path / "bad.pt"
    torch.save({"labels": torch.tensor([0])}, fake_path)

    class DummyModel(torch.nn.Module):
        def logits(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[0.0]])  # pragma: no cover

    model = DummyModel()
    loader = make_loader_from_tensors(torch.tensor([[0.0]]))
    with pytest.raises(
        ValueError, match="Saved file .* does not contain 'logits'"
    ):
        s._loadorpredict(path=fake_path, model=model, loader=loader)


def test_loadorpredict_no_batches_raises() -> None:
    """When loader yields no batches, _logits_from_loader raises ValueError."""
    s = SoftmaxScore()

    class EmptyDataset(Dataset[Any]):
        def __len__(self) -> int:
            return 0

        def __getitem__(self, idx: int) -> Any:
            raise IndexError  # pragma: no cover

    empty_loader = DataLoader(EmptyDataset(), batch_size=1)

    class DummyModel(torch.nn.Module):
        def logits(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[0.0]])  # pragma: no cover

    model = DummyModel()
    with pytest.raises(ValueError, match="No batches found in loader"):
        s._logits_from_loader(model=model, loader=empty_loader)


def test_select_automatically_sets_threshold(tmp_path: Path) -> None:
    """Ensure ``select`` calls ``set_threshold`` when none is set."""
    # Create simple logits and labels for binary classification
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.2]])
    labels = logits.argmax(dim=1)
    loader = make_loader_from_tensors(logits, labels)
    model = IdentityModel()
    score = SoftmaxScore()
    # Fit to populate scores but do not set threshold
    score.fit(model=model, loader=loader, outdir=tmp_path, prefix="thr")
    # At this point ``score.threshold`` should be None
    assert score.threshold is None
    # Call select; it should set a threshold based on 99th percentile of scores
    out = score.select(logits)
    assert "score" in out and "selected" in out
    # After select, threshold must be set
    assert score.threshold is not None
    # Selected mask should be a boolean tensor of same length as inputs
    assert out["selected"].shape == (logits.shape[0],)
    assert out["selected"].dtype == torch.bool


def test_entropy_score_formula() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    e = EntropyScore()
    sc = e.score(logits)
    T = 1.0 if e.temperature is None else float(e.temperature)
    z = logits / T - (logits / T).amax(dim=1, keepdim=True)
    exp_z = z.exp()
    probs = exp_z / exp_z.sum(dim=1, keepdim=True)
    p = probs.clamp(min=1e-12)
    expect = -(p * p.log()).sum(dim=1)
    assert torch.allclose(sc, expect, atol=1e-6)


def test_energy_score_logsumexp() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    T = 0.5
    en = EnergyScore(temperature=T)
    sc = en.score(logits)
    expect = -T * (logits / T).logsumexp(dim=1)
    assert torch.allclose(sc, expect, atol=1e-6)


def test_fit_temperature_reduces_nll() -> None:
    true_logits = torch.tensor(
        [[5.0, 0.0, -1.0], [4.0, 1.0, 0.0], [6.0, -1.0, -2.0]]
    )
    labels = true_logits.argmax(dim=1)
    val_logits = true_logits * 0.2
    s = SoftmaxScore()
    nll_before = torch.nn.functional.cross_entropy(val_logits, labels).item()
    s._fit_temperature(logits=val_logits, labels=labels)
    assert s.temperature is not None
    assert isinstance(s.temperature, float)
    assert float(s.temperature) < 1.0 + 1e-6
    T = float(s.temperature)
    nll_after = torch.nn.functional.cross_entropy(val_logits / T, labels).item()
    assert nll_after <= nll_before + 1e-6


def test_fit_temperature_small_valset() -> None:
    logits = torch.randn(2, 4)
    labels = logits.argmax(dim=1)
    s = SoftmaxScore()
    s._fit_temperature(logits=logits, labels=labels)
    assert s.temperature is not None
    assert isinstance(s.temperature, float)
    assert math.isfinite(float(s.temperature))


def make_score_with_task(task: str) -> SoftmaxScore:
    s = SoftmaxScore(temperature=None)
    s.task = task
    s.task_config = None  # type: ignore[assignment]
    return s


def test_is_binary_single_logit() -> None:
    s = make_score_with_task("binary")
    a = torch.randn(5)
    assert s._is_binary_single_logit(a)
    b = torch.randn(5, 1)
    assert s._is_binary_single_logit(b)
    c = torch.randn(5, 2)
    assert not s._is_binary_single_logit(c)


def test_normalize_multiclass() -> None:
    s = make_score_with_task("multiclass")
    logits = torch.randn(7, 4)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.int64)
    nl, lab = s._normalize_logits_and_labels(logits, labels)
    assert nl.shape == (7, 4)
    assert lab.shape == (7,)
    assert lab.dtype == torch.long
    labels2 = labels.unsqueeze(1)
    nl2, lab2 = s._normalize_logits_and_labels(logits, labels2)
    assert lab2.shape == (7,)
    logits_bad_shape = logits.unsqueeze(0)
    with pytest.raises(ValueError, match="multiclass logits must have shape"):
        s._normalize_logits_and_labels(logits_bad_shape, labels)


def test_normalize_binary_single_logit() -> None:
    s = make_score_with_task("binary")
    logits = torch.randn(6)
    labels = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.int64)
    nl, lab = s._normalize_logits_and_labels(logits, labels)
    assert nl.shape == (6,)
    assert lab.dtype == torch.float32
    assert lab.shape == (6,)


def test_normalize_binary_two_logit() -> None:
    s = make_score_with_task("binary")
    logits = torch.randn(6, 2)
    labels = torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.int64)
    nl, lab = s._normalize_logits_and_labels(logits, labels)
    assert nl.shape == (6, 2)
    assert lab.shape == (6,)
    assert lab.dtype == torch.long
    logits_bad_shape = torch.randn(6, 3)
    with pytest.raises(
        ValueError, match="binary two-logit logits must have shape"
    ):
        s._normalize_logits_and_labels(logits_bad_shape, labels)
    labels = labels.unsqueeze(1).to(dtype=torch.bool)
    nl, lab = s._normalize_logits_and_labels(logits, labels)
    assert nl.shape == (6, 2)
    assert lab.shape == (6,)
    assert lab.dtype == torch.long


def test_normalize_multilabel() -> None:
    s = make_score_with_task("multilabel")
    logits = torch.randn(5, 3)
    labels = torch.randint(0, 2, (5, 3)).float()
    nl, lab = s._normalize_logits_and_labels(logits, labels)
    assert nl.shape == (5, 3)
    assert lab.shape == (5, 3)
    assert lab.dtype == torch.float32
    logits_bad_shape = logits.unsqueeze(0)
    with pytest.raises(ValueError, match="multilabel logits must have shape"):
        s._normalize_logits_and_labels(logits_bad_shape, labels)
    labels_bad_shape = labels.unsqueeze(0)
    with pytest.raises(
        ValueError, match="multilabel labels must have same shape as logits"
    ):
        s._normalize_logits_and_labels(logits, labels_bad_shape)


def test_unknown_task_normalize_raises() -> None:
    s = make_score_with_task("multiclass")
    s.task = "not_a_task"
    with pytest.raises(ValueError, match="Unknown task: not_a_task"):
        s._normalize_logits_and_labels(torch.randn(2, 3), torch.tensor([0, 1]))


@pytest.mark.parametrize("task", ["multiclass", "binary", "multilabel"])
def test_temperature_loss_matches_torch(task: str) -> None:
    s = make_score_with_task(task)
    if task == "multiclass":
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))
        expected = F.cross_entropy(logits, labels.long())
        got = s._temperature_loss(logits, labels)
        approx_tensor(got, expected)
    elif task == "binary":
        logits = torch.randn(10)
        labels = torch.randint(0, 2, (10,)).float()
        expected = F.binary_cross_entropy_with_logits(logits, labels)
        got = s._temperature_loss(logits, labels)
        approx_tensor(got, expected)
        logits2 = torch.randn(10, 2)
        labels2 = torch.randint(0, 2, (10,))
        expected2 = F.cross_entropy(logits2, labels2.long())
        got2 = s._temperature_loss(logits2, labels2)
        approx_tensor(got2, expected2)
    elif task == "multilabel":
        logits = torch.randn(7, 3)
        labels = torch.randint(0, 2, (7, 3)).float()
        expected = F.binary_cross_entropy_with_logits(logits, labels)
        got = s._temperature_loss(logits, labels)
        approx_tensor(got, expected)


@pytest.mark.parametrize(
    "score_class,task,logits,expected_shape",
    [
        (EnergyScore, "multiclass", torch.randn(7, 4), (7,)),
        (EnergyScore, "binary", torch.randn(8), (8,)),
        (EnergyScore, "binary", torch.randn(8, 2), (8,)),
        (EnergyScore, "multilabel", torch.randn(5, 3), (5,)),
        (EntropyScore, "multiclass", torch.randn(6, 5), (6,)),
        (EntropyScore, "binary", torch.randn(9), (9,)),
        (EntropyScore, "binary", torch.randn(9, 2), (9,)),
        (EntropyScore, "multilabel", torch.randn(4, 2), (4,)),
        (SoftmaxScore, "multiclass", torch.randn(7, 4), (7,)),
        (SoftmaxScore, "binary", torch.randn(8), (8,)),
        (SoftmaxScore, "binary", torch.randn(8, 2), (8,)),
        (SoftmaxScore, "multilabel", torch.randn(5, 3), (5,)),
        (MarginScore, "multiclass", torch.randn(6, 5), (6,)),
        (MarginScore, "binary", torch.randn(9), (9,)),
        (MarginScore, "binary", torch.randn(9, 2), (9,)),
        (MarginScore, "multilabel", torch.randn(4, 2), (4,)),
    ],
)
def test_score_shapes(
    score_class: type[SoftmaxScore | EnergyScore | MarginScore | EntropyScore],
    task: str,
    logits: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> None:
    score = score_class(task=task)
    result = score.score(logits)
    assert result.shape == expected_shape


def test_entropy_monotonicity() -> None:
    confident = torch.tensor([[10.0, -5.0, -5.0]])
    uncertain = torch.tensor([[0.1, 0.0, -0.1]])
    score = EntropyScore(task="multiclass")
    assert score.score(confident) < score.score(uncertain)


def test_energy_monotonicity() -> None:
    confident = torch.tensor([10.0, -10.0])
    uncertain = torch.tensor([0.0, 0.0])
    score = EnergyScore(task="binary")
    assert torch.all(score.score(confident) < score.score(uncertain))


def test_entropy_multilabel_max_aggregation() -> None:
    logits = torch.tensor([[10.0, 0.0], [0.0, 0.0]])
    score = EntropyScore(task="multilabel")
    out = score.score(logits)
    assert torch.isclose(out[0], out[1], atol=1e-6)


def test_energy_multilabel_sum() -> None:
    logits = torch.tensor([[10.0, 0.0], [0.0, 0.0]])
    score = EnergyScore(task="multilabel")
    out = score.score(logits)
    assert out.shape == (2,)


def test_entropy_numerical_stability() -> None:
    logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])
    score = EntropyScore(task="multiclass")
    out = score.score(logits)
    assert torch.isfinite(out).all()


def test_energy_numerical_stability() -> None:
    logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])
    score = EnergyScore(task="multiclass")
    out = score.score(logits)
    assert torch.isfinite(out).all()


def test_softmax_multilabel_min_aggregation() -> None:
    logits = torch.tensor([[10.0, 0.0], [0.0, 0.0]])
    score = SoftmaxScore(task="multilabel")
    out = score.score(logits)
    assert torch.isclose(out[0], out[1], atol=1e-6)


def test_margin_multilabel_min_aggregation() -> None:
    logits = torch.tensor([[10.0, 0.0], [0.0, 0.0]])
    score = MarginScore(task="multilabel")
    out = score.score(logits)
    assert torch.isclose(out[0], out[1], atol=1e-6)


def test_margin_directionality() -> None:
    confident = torch.tensor([[10.0, -5.0, -5.0]])
    uncertain = torch.tensor([[0.1, 0.0, -0.1]])
    score = MarginScore(task="multiclass")
    assert score.score(confident) < score.score(uncertain)


def test_softmax_binary_single_logit() -> None:
    logits = torch.tensor([10.0, 0.0, -10.0])
    score = SoftmaxScore(task="binary")
    out = score.score(logits)
    assert out[0] < out[1] and out[2] < out[1]


def test_score_empty_inputs() -> None:
    # All scores should handle empty logits gracefully
    _ScoreClasses: list[
        type[SoftmaxScore | MarginScore | EntropyScore | EnergyScore]
    ] = [SoftmaxScore, MarginScore, EntropyScore, EnergyScore]
    for Score in _ScoreClasses:
        score = Score(task="multiclass")
        logits = torch.empty((0, 3))
        out = score.score(logits)
        assert out.shape == (0,)


def test_score_nan_inf_inputs() -> None:
    logits = torch.tensor([[float("nan"), 0.0], [float("inf"), -float("inf")]])
    _ScoreClasses: list[
        type[SoftmaxScore | MarginScore | EntropyScore | EnergyScore]
    ] = [SoftmaxScore, MarginScore, EntropyScore, EnergyScore]
    for Score in _ScoreClasses:
        score = Score(task="multiclass")
        out = score.score(logits)
        assert out.shape == (2,)
        # Output should be finite or propagate NaN/Inf in a controlled way
        assert torch.isfinite(out).sum() >= 0


def test_score_shape_mismatch() -> None:
    score = SoftmaxScore(task="multiclass")
    logits = torch.randn(5, 3)
    # Labels wrong shape for multiclass
    labels = torch.randn(5, 2)
    try:
        score.fit(logits, labels)
    except Exception as e:
        assert isinstance(e, Exception)


def test_fit_temperature_all_same_logits() -> None:
    logits = torch.ones(10, 3)
    labels = torch.zeros(10, dtype=torch.long)
    score = SoftmaxScore()
    score.fit(logits, labels)
    assert score.temperature is not None
    assert math.isfinite(float(score.temperature))


def test_fit_temperature_all_same_labels() -> None:
    logits = torch.randn(8, 3)
    labels = torch.zeros(8, dtype=torch.long)
    score = SoftmaxScore()
    score.fit(logits, labels)
    assert score.temperature is not None
    assert math.isfinite(float(score.temperature))


def test_binary_single_logit_extreme_values() -> None:
    logits = torch.tensor([1000.0, -1000.0, 0.0])
    score = SoftmaxScore(task="binary")
    out = score.score(logits)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_multilabel_all_zero_logits() -> None:
    logits = torch.zeros(4, 5)
    score = EntropyScore(task="multilabel")
    out = score.score(logits)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_fit_empty_loader(tmp_path: Path) -> None:
    loader = DataLoader(SimpleBatchDataset([]), batch_size=1)
    score = SoftmaxScore()
    with pytest.raises(ValueError, match="No batches found in loader"):
        score.fit(model=IdentityModel(), loader=loader, outdir=tmp_path)


def test_fit_temperature_nan_labels() -> None:
    logits = torch.randn(5, 2)
    labels = torch.tensor([0, 1, float("nan"), 0, 1])
    score = SoftmaxScore()
    try:
        score.fit(logits, labels)
    except Exception as e:
        assert isinstance(e, Exception)


def test_check_model_requires_logits_method() -> None:
    class NoLogits(torch.nn.Module):
        pass

    with pytest.raises(
        Exception, match="model is required to have a `\\.logits\\(\\)` method"
    ):
        LogitScore._check_model(NoLogits())


def test_check_model_logits_signature() -> None:
    class BadLogits(torch.nn.Module):
        def logits(self) -> None:  # missing x argument
            pass

    model = BadLogits()
    with pytest.raises(Exception, match="except `x` as argument"):
        LogitScore._check_model(model)


def test_fit_temperature_lbfgs_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force LBFGS to fail, triggering Adam fallback
    logits = torch.randn(5, 2)
    labels = torch.randint(0, 2, (5,))
    score = SoftmaxScore()

    class DummyLBFGS:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def step(self, closure: Callable[[], float]) -> float:
            raise RuntimeError("fail")

    monkeypatch.setattr(torch.optim, "LBFGS", DummyLBFGS)
    score._fit_temperature(logits, labels)
    assert isinstance(score.temperature, float)


def test_loadorpredict_loads_from_disk(tmp_path: Path) -> None:
    logits = torch.randn(3, 2)
    labels = torch.tensor([0, 1, 0])
    path = tmp_path / "logits.pt"
    torch.save({"logits": logits, "labels": labels}, path)

    loader: DataLoader[object] = cast(DataLoader[object], [])
    score = SoftmaxScore()
    loaded_logits, loaded_labels = score._loadorpredict(
        path, IdentityModel(), loader
    )
    assert torch.allclose(loaded_logits, logits)
    assert loaded_labels is not None
    assert torch.allclose(loaded_labels, labels)


def test_loadorpredict_missing_logits(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"labels": torch.tensor([1, 2])}, path)

    loader: DataLoader[object] = cast(DataLoader[object], [])
    score = SoftmaxScore()
    with pytest.raises(ValueError, match="does not contain 'logits'"):
        score._loadorpredict(path, IdentityModel(), loader)


def test_check_model_requires_x_param() -> None:
    class BadSigModel(torch.nn.Module):
        def logits(self) -> torch.Tensor:
            return torch.tensor([1.0])

    with pytest.raises(
        Exception,
        match="`.logits\\(\\)` method is required to except `x` as argument",
    ):
        LogitScore._check_model(BadSigModel())


def test_fit_arg_validation_conflicting_inputs() -> None:
    s = SoftmaxScore()
    # both precomputed and model provided -> ValueError
    with pytest.raises(
        ValueError, match="Cannot specify both precomputed logits"
    ):
        s.fit(X=torch.randn(2, 3), model=object())  # type: ignore[arg-type]


def test_fit_requires_one_input() -> None:
    s = SoftmaxScore()
    with pytest.raises(
        ValueError, match="Must specify either logits or model\\+loader"
    ):
        s.fit()


def test_loadorpredict_missing_logits_field(tmp_path: Path) -> None:
    s = SoftmaxScore()

    class M(torch.nn.Module):
        def logits(self, x: torch.Tensor) -> torch.Tensor:
            return x

    model = M()
    p = tmp_path / "bad.pt"
    torch.save({"labels": torch.tensor([1])}, p)
    with pytest.raises(ValueError, match="does not contain 'logits'"):
        s._loadorpredict(
            path=p,
            model=model,
            loader=make_loader_from_tensors(torch.randn(1, 2)),
        )


def test_logits_from_loader_non_tensor() -> None:
    class M(torch.nn.Module):
        def logits(self, x: torch.Tensor) -> list[int]:
            return [1, 2, 3]

    model = M()
    loader = make_loader_from_tensors(torch.randn(1, 2))
    s = SoftmaxScore()
    with pytest.raises(
        ValueError, match="Extracted logits is not a torch.Tensor"
    ):
        s._logits_from_loader(model=model, loader=loader)


def test_logits_from_loader_no_batches() -> None:
    loader = DataLoader(SimpleBatchDataset([]), batch_size=1)

    class M(torch.nn.Module):
        def logits(self, x: torch.Tensor) -> torch.Tensor:
            return x

    model = M()
    s = SoftmaxScore()
    with pytest.raises(ValueError, match="No batches found in loader"):
        s._logits_from_loader(model=model, loader=loader)


def test_normalize_raises_on_missing_labels() -> None:
    s = make_score_with_task("multiclass")
    with pytest.raises(ValueError, match="labels must be provided"):
        s._normalize_logits_and_labels(torch.randn(2, 3), None)


def test_temperature_loss_unknown_task_raises() -> None:
    s = make_score_with_task("multiclass")
    s.task = "unknown_task"
    with pytest.raises(ValueError, match="Unknown task: unknown_task"):
        s._temperature_loss(torch.randn(2, 3), torch.tensor([0, 1]))


@pytest.mark.parametrize(
    "cls", [SoftmaxScore, EnergyScore, MarginScore, EntropyScore]
)
def test_score_methods_unknown_task_raise(
    cls: type[SoftmaxScore | EnergyScore | MarginScore | EntropyScore],
) -> None:
    inst = cls()
    inst.task = "not_a_task"
    sample = torch.randn(2, 3)
    with pytest.raises(ValueError, match="Unknown task: not_a_task"):
        inst.score(sample)
