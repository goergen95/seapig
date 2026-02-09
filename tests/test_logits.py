import importlib
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from seapig.scores import EnergyScore, EntropyScore, MarginScore, SoftmaxScore


@pytest.fixture(autouse=True)
def rng_seed():
    torch.manual_seed(1234)
    yield

def approx_tensor(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-6) -> None:
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=tol, rtol=1e-5)

class SimpleBatchDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

def make_loader_from_tensors(logits, labels=None):
    if labels is None:
        items = [la.unsqueeze(0) for la in logits]
    else:
        items = [
            {
                "image": logits[i].unsqueeze(0),
                "label": torch.tensor(labels[i]).unsqueeze(0),
            }
            for i in range(len(logits))
        ]
    return DataLoader(SimpleBatchDataset(items), batch_size=1, shuffle=False)

def test_fit_dl_saves_files(tmp_path: Path):
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.2]])
    labels = logits.argmax(dim=1)
    loader = make_loader_from_tensors(logits, labels)
    class IdentityModel(torch.nn.Module):
        def logits(self, x):
            return x.squeeze(0) if x.dim() > 2 and x.shape[0] == 1 else x
    model = IdentityModel()
    score = SoftmaxScore()
    outdir = tmp_path / "saved_logits"
    score.fit_dl(model=model, loader=loader, outdir=outdir, prefix="mytest")
    train_file = outdir / "mytest_train.pt"
    assert train_file.exists()
    loaded = torch.load(train_file)
    assert "logits" in loaded and "labels" in loaded
    assert loaded["logits"].shape[0] == logits.shape[0]
    assert hasattr(score, "logits")
    assert score.logits is not None
    assert score.logits.shape[0] == logits.shape[0]

@pytest.mark.parametrize("out_kind", ["tensor", "logits", "preds", "y_hat"])
def test_fit_dl_accepts_output_formats(out_kind):
    logits = torch.tensor([[0.5, 1.5], [2.0, 0.1], [0.0, 0.0]])
    labels = logits.argmax(dim=1)
    loader = make_loader_from_tensors(logits, labels)
    class FlexibleModel(torch.nn.Module):
        def logits(self, x):
            return x.squeeze(0)
    model = FlexibleModel()
    score = SoftmaxScore()
    score.fit_dl(model=model, loader=loader)
    assert hasattr(score, "logits")
    assert score.logits is not None
    assert score.logits.shape[0] == logits.shape[0]

@pytest.mark.parametrize("batch_format", ["dict", "tensor_only"])
def test_fit_dl_batch_formats(batch_format):
    logits = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
    labels = logits.argmax(dim=1)
    if batch_format == "dict":
        items = [
            {"image": logits[i].unsqueeze(0), "label": labels[i].unsqueeze(0)}
            for i in range(len(logits))
        ]
    else:
        items = [logits[i].unsqueeze(0) for i in range(len(logits))]
    loader = DataLoader(SimpleBatchDataset(items), batch_size=1, shuffle=False)
    class IdentityModel(torch.nn.Module):
        def logits(self, x):
            return x.squeeze(0) if x.dim() > 2 and x.shape[0] == 1 else x
    model = IdentityModel()
    score = EnergyScore()
    score.fit_dl(model=model, loader=loader)
    assert score.logits is not None
    assert score.logits.shape[0] == logits.shape[0]

def test_softmax_numerical_stability():
    logits = torch.tensor([[1000.0, -1000.0, 0.0], [1e6, 0.0, -1e6]])
    s = SoftmaxScore()
    T = 1.0 if s.temperature is None else float(s.temperature)
    z = logits / T - (logits / T).amax(dim=1, keepdim=True)
    exp_z = z.exp()
    probs = exp_z / exp_z.sum(dim=1, keepdim=True)
    assert probs.shape == (2, 3)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0, 1.0]), atol=1e-6)

def test_predict_proba_temperature():
    logits = torch.tensor([[2.0, 1.0], [0.5, 0.1]])
    s = SoftmaxScore()
    def predict_proba(logits, temperature=None):
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

def test_logit_helpers_consistency():
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
def test_softmax_score_matches_maxprob(logits, expected_msp):
    s = SoftmaxScore()
    sc = s.score(logits)
    T = 1.0 if s.temperature is None else float(s.temperature)
    z = logits / T - (logits / T).amax(dim=1, keepdim=True)
    exp_z = z.exp()
    probs = exp_z / exp_z.sum(dim=1, keepdim=True)
    assert torch.allclose(
        sc, -probs.amax(dim=1).to(dtype=torch.float32), atol=1e-6
    )

def test_margin_score_manual():
    logits = torch.tensor([[5.0, 2.0, 1.0], [0.1, 0.0, -1.0]])
    m = MarginScore()
    sc = m.score(logits)
    top2 = logits.topk(2, dim=1).values
    expect = -(top2[:, 0] - top2[:, 1])
    assert torch.allclose(sc, expect, atol=1e-6)

def test_entropy_score_formula():
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

def test_energy_score_logsumexp():
    logits = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    T = 0.5
    en = EnergyScore(temperature=T)
    sc = en.score(logits)
    expect = -T * (logits / T).logsumexp(dim=1)
    assert torch.allclose(sc, expect, atol=1e-6)

def test_fit_temperature_reduces_nll():
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

def test_fit_temperature_small_valset():
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
    s.task_config = None
    return s

def test_is_binary_single_logit():
    s = make_score_with_task("binary")
    a = torch.randn(5)
    assert s._is_binary_single_logit(a)
    b = torch.randn(5, 1)
    assert s._is_binary_single_logit(b)
    c = torch.randn(5, 2)
    assert not s._is_binary_single_logit(c)

def test_normalize_multiclass():
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

def test_normalize_binary_single_logit():
    s = make_score_with_task("binary")
    logits = torch.randn(6)
    labels = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.int64)
    nl, lab = s._normalize_logits_and_labels(logits, labels)
    assert nl.shape == (6,)
    assert lab.dtype == torch.float32
    assert lab.shape == (6,)

def test_normalize_binary_two_logit():
    s = make_score_with_task("binary")
    logits = torch.randn(6, 2)
    labels = torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.int64)
    nl, lab = s._normalize_logits_and_labels(logits, labels)
    assert nl.shape == (6, 2)
    assert lab.shape == (6,)
    assert lab.dtype == torch.long

def test_normalize_multilabel():
    s = make_score_with_task("multilabel")
    logits = torch.randn(5, 3)
    labels = torch.randint(0, 2, (5, 3)).float()
    nl, lab = s._normalize_logits_and_labels(logits, labels)
    assert nl.shape == (5, 3)
    assert lab.shape == (5, 3)
    assert lab.dtype == torch.float32

@pytest.mark.parametrize("task", ["multiclass", "binary", "multilabel"])
def test_temperature_loss_matches_torch(task: str):
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

def test_top_level_score_imports():
    mod = importlib.import_module("seapig")
    for name in ("SoftmaxScore", "MarginScore", "EntropyScore", "EnergyScore"):
        assert hasattr(mod, name)

@pytest.mark.parametrize("score_class,task,logits,expected_shape", [
    (EnergyScore, "multiclass", torch.randn(7, 4), (7,)),
    (EnergyScore, "binary", torch.randn(8), (8,)),
    (EnergyScore, "binary", torch.randn(8, 2), (8,)),
    (EnergyScore, "multilabel", torch.randn(5, 3), (5,)),
    (EntropyScore, "multiclass", torch.randn(6, 5), (6,)),
    (EntropyScore, "binary", torch.randn(9), (9,)),
    (EntropyScore, "binary", torch.randn(9, 2), (9,)),
    (EntropyScore, "multilabel", torch.randn(4, 2), (4,)),
])
def test_score_shapes(score_class, task, logits, expected_shape):
    score = score_class(task=task)
    result = score.score(logits)
    assert result.shape == expected_shape

def test_entropy_monotonicity():
    confident = torch.tensor([[10.0, -5.0, -5.0]])
    uncertain = torch.tensor([[0.1, 0.0, -0.1]])
    score = EntropyScore(task="multiclass")
    assert score.score(confident) < score.score(uncertain)

def test_energy_monotonicity():
    confident = torch.tensor([10.0, -10.0])
    uncertain = torch.tensor([0.0, 0.0])
    score = EnergyScore(task="binary")
    assert torch.all(score.score(confident) < score.score(uncertain))

def test_entropy_multilabel_max_aggregation():
    logits = torch.tensor([[10.0, 0.0], [0.0, 0.0]])
    score = EntropyScore(task="multilabel")
    out = score.score(logits)
    assert torch.isclose(out[0], out[1], atol=1e-6)

def test_energy_multilabel_sum():
    logits = torch.tensor([[10.0, 0.0], [0.0, 0.0]])
    score = EnergyScore(task="multilabel")
    out = score.score(logits)
    assert out.shape == (2,)

def test_entropy_numerical_stability():
    logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])
    score = EntropyScore(task="multiclass")
    out = score.score(logits)
    assert torch.isfinite(out).all()

def test_energy_numerical_stability():
    logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])
    score = EnergyScore(task="multiclass")
    out = score.score(logits)
    assert torch.isfinite(out).all()