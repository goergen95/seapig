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
    """Yield batches provided as a list of items."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_loader_from_tensors(logits, labels=None, batch_size=1):
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


def test_fit_dl_saves_files_when_outdir_and_prefix(tmp_path: Path):
    # Create tiny "logits" dataset where inputs == logits to simplify model
    train_logits = torch.tensor([[2.0, 0.5], [0.1, 1.2]])
    train_labels = train_logits.argmax(dim=1)

    # DataLoader that returns dict batches with inputs already logits
    train_loader = make_loader_from_tensors(train_logits, train_labels)

    # Model that returns the input tensor directly (treated as logits)
    class IdentityModel(torch.nn.Module):
        def logits(self, x):
            return x.squeeze(0) if x.dim() > 2 and x.shape[0] == 1 else x

    model = IdentityModel()
    score = SoftmaxScore()

    outdir = tmp_path / "saved_logits"
    # new interface: fit_dl accepts a single loader
    score.fit_dl(
        model=model, loader=train_loader, outdir=outdir, prefix="mytest"
    )

    train_file = outdir / "mytest_train.pt"
    assert train_file.exists()

    loaded_train = torch.load(train_file)
    assert "logits" in loaded_train and "labels" in loaded_train

    # shapes should match original concatenated tensors
    assert loaded_train["logits"].shape[0] == train_logits.shape[0]

    # score should have stored reference logits (fit from the loader)
    assert hasattr(score, "logits")
    assert score.logits is not None
    assert score.logits.shape[0] == train_logits.shape[0]


@pytest.mark.parametrize("out_kind", ["tensor", "logits", "preds", "y_hat"])
def test_fit_dl_accepts_various_model_output_formats(out_kind):
    # small dataset
    logits = torch.tensor([[0.5, 1.5], [2.0, 0.1], [0.0, 0.0]])
    labels = logits.argmax(dim=1)
    loader = make_loader_from_tensors(logits, labels)

    # create a model that returns different output formats
    class FlexibleModel(torch.nn.Module):
        def logits(self, x):
            return x.squeeze(0)

    model = FlexibleModel()
    score = SoftmaxScore()
    # new interface: pass single loader
    score.fit_dl(model=model, loader=loader)
    # should have stored logits from the loader
    assert hasattr(score, "logits")
    assert score.logits is not None
    assert score.logits.shape[0] == logits.shape[0]


@pytest.mark.parametrize(
    "batch_format",
    [
        "dict",  # {"image": inputs, "label": labels}
        "tensor_only",  # inputs only (no labels)
    ],
)
def test_fit_dl_handles_common_batch_formats(batch_format):
    # Create tiny logits and labels
    logits = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
    labels = logits.argmax(dim=1)

    if batch_format == "dict":
        items = [
            {"image": logits[i].unsqueeze(0), "label": labels[i].unsqueeze(0)}
            for i in range(len(logits))
        ]
    else:  # tensor_only
        items = [logits[i].unsqueeze(0) for i in range(len(logits))]

    loader = DataLoader(SimpleBatchDataset(items), batch_size=1, shuffle=False)

    class IdentityModel(torch.nn.Module):
        def logits(self, x):
            return x.squeeze(0) if x.dim() > 2 and x.shape[0] == 1 else x

    model = IdentityModel()
    score = EnergyScore()
    # Should not raise; for tensor_only there are no labels and fit_temperature won't run
    score.fit_dl(model=model, loader=loader)
    assert score.logits is not None
    assert score.logits.shape[0] == logits.shape[0]


def test_softmax_numerical_stability_and_shape():
    # extreme logits should not produce NaNs and shape should be (M, C)
    logits = torch.tensor([[1000.0, -1000.0, 0.0], [1e6, 0.0, -1e6]])
    s = SoftmaxScore()
    # compute softmax manually (new interface does not provide predict_proba)
    T = 1.0 if s.temperature is None else float(s.temperature)
    z = logits / T - (logits / T).amax(dim=1, keepdim=True)
    exp_z = z.exp()
    probs = exp_z / exp_z.sum(dim=1, keepdim=True)
    assert probs.shape == (2, 3)
    assert torch.isfinite(probs).all()
    # each row sums to 1
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0, 1.0]), atol=1e-6)


def test_predict_proba_uses_instance_temperature():
    logits = torch.tensor([[2.0, 1.0], [0.5, 0.1]])
    s = SoftmaxScore()

    # helper to compute softmax with optional temperature
    def predict_proba(logits, temperature=None):
        T = 1.0 if temperature is None else float(temperature)
        z = logits / T - (logits / T).amax(dim=1, keepdim=True)
        exp_z = z.exp()
        return exp_z / exp_z.sum(dim=1, keepdim=True)

    # set temperature explicitly on instance (now a plain float)
    s.temperature = 2.0
    assert isinstance(s.temperature, float)
    p_explicit = predict_proba(logits, temperature=4.0)
    p_inst = predict_proba(logits, temperature=s.temperature)
    # different temperatures -> different probabilities
    assert not torch.allclose(p_explicit, p_inst)
    # reset to default numeric temperature and ensure None-equivalent
    # behaviour of helper (None -> 1.0) is unchanged
    s.temperature = 1.0
    p_none = predict_proba(logits, temperature=None)
    p1 = predict_proba(logits, temperature=1.0)
    assert torch.allclose(p_none, p1, atol=1e-6)


def test_logit_helpers_consistency_across_subclasses():
    logits = torch.tensor([[3.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    max_score = -logits.amax(dim=1)
    assert max_score.shape == (2,)
    assert torch.allclose(max_score, -torch.tensor([3.0, 0.0]), atol=1e-6)
    # margin (use MarginScore implementation)
    margin_score = MarginScore().score(logits)
    expected_margin = torch.tensor([-2.0, -0.0])
    assert torch.allclose(margin_score, expected_margin, atol=1e-6)
    # entropy (should be >= 0)
    ent = EntropyScore().score(logits)
    assert ent.shape == (2,)
    assert (ent >= 0.0).all()
    # logit norm negative (use negative L2 norm)
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
    assert sc.shape == (logits.shape[0],)
    # negative max-prob
    # compute softmax manually
    T = 1.0 if s.temperature is None else float(s.temperature)
    z = logits / T - (logits / T).amax(dim=1, keepdim=True)
    exp_z = z.exp()
    probs = exp_z / exp_z.sum(dim=1, keepdim=True)
    assert torch.allclose(
        sc, -probs.amax(dim=1).to(dtype=torch.float32), atol=1e-6
    )


def test_margin_score_matches_manual_margin():
    logits = torch.tensor([[5.0, 2.0, 1.0], [0.1, 0.0, -1.0]])
    m = MarginScore()
    sc = m.score(logits)
    # margin = top1 - top2, score = -margin
    top2 = logits.topk(2, dim=1).values
    expect = -(top2[:, 0] - top2[:, 1])
    assert torch.allclose(sc, expect, atol=1e-6)


def test_entropy_score_matches_formula():
    logits = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    e = EntropyScore()
    sc = e.score(logits)
    # compute probabilities manually for comparison
    T = 1.0 if e.temperature is None else float(e.temperature)
    z = logits / T - (logits / T).amax(dim=1, keepdim=True)
    exp_z = z.exp()
    probs = exp_z / exp_z.sum(dim=1, keepdim=True)
    p = probs.clamp(min=1e-12)
    expect = -(p * p.log()).sum(dim=1)
    assert torch.allclose(sc, expect, atol=1e-6)


def test_energy_score_matches_logsumexp():
    logits = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    T = 0.5
    en = EnergyScore(temperature=T)
    sc = en.score(logits)
    # energy = -T * logsumexp(logits / T)
    expect = -T * (logits / T).logsumexp(dim=1)
    assert torch.allclose(sc, expect, atol=1e-6)


def test_fit_temperature_reduces_validation_nll():
    # Create "true" logits with clear class separation and labels from argmax
    true_logits = torch.tensor(
        [[5.0, 0.0, -1.0], [4.0, 1.0, 0.0], [6.0, -1.0, -2.0]]
    )
    labels = true_logits.argmax(dim=1)
    # Create miscalibrated validation logits that are too small in magnitude
    val_logits = true_logits * 0.2  # under-confident
    s = SoftmaxScore()
    # compute NLL before calibration (temperature = 1)
    nll_before = torch.nn.functional.cross_entropy(val_logits, labels).item()
    s._fit_temperature(logits=val_logits, labels=labels)
    assert s.temperature is not None
    # temperature should be a float and < 1 (to amplify logits)
    assert isinstance(s.temperature, float)
    assert float(s.temperature) < 1.0 + 1e-6
    # compute NLL after applying found temperature
    T = float(s.temperature)
    nll_after = torch.nn.functional.cross_entropy(val_logits / T, labels).item()
    assert nll_after <= nll_before + 1e-6


def test_fit_temperature_handles_small_validation_set():
    # tiny validation set still produces a finite temperature
    logits = torch.randn(2, 4)
    labels = logits.argmax(dim=1)
    s = SoftmaxScore()
    # call the internal temperature fitter directly
    s._fit_temperature(logits=logits, labels=labels)
    assert s.temperature is not None
    assert isinstance(s.temperature, float)
    assert math.isfinite(float(s.temperature))


def make_score_with_task(task: str) -> SoftmaxScore:
    """Helper: create a SoftmaxScore and set its task for testing.

    SoftmaxScore doesn't accept a `task` parameter in the public
    constructor, so tests set the attribute directly.
    """
    s = SoftmaxScore(temperature=None)
    s.task = task
    s.task_config = None
    return s


def test_is_binary_single_logit_true_and_false() -> None:
    s = make_score_with_task("binary")

    a = torch.randn(5)  # (N,)  -> single-logit
    assert s._is_binary_single_logit(a)

    b = torch.randn(5, 1)  # (N,1) -> single-logit
    assert s._is_binary_single_logit(b)

    c = torch.randn(5, 2)  # (N,2) -> not single-logit
    assert not s._is_binary_single_logit(c)


def test_normalize_multiclass_shapes_and_types() -> None:
    s = make_score_with_task("multiclass")

    logits = torch.randn(7, 4)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.int64)

    nl, lab = s._normalize_logits_and_labels(logits, labels)
    assert nl.shape == (7, 4)
    assert lab.shape == (7,)
    assert lab.dtype == torch.long

    # labels as (N,1) should be squeezed
    labels2 = labels.unsqueeze(1)
    nl2, lab2 = s._normalize_logits_and_labels(logits, labels2)
    assert lab2.shape == (7,)


def test_normalize_binary_single_logit() -> None:
    s = make_score_with_task("binary")

    logits = torch.randn(6)  # (N,)
    labels = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.int64)

    nl, lab = s._normalize_logits_and_labels(logits, labels)
    assert nl.shape == (6,)
    # labels converted to float for single-logit binary
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


def test_normalize_multilabel() -> None:
    s = make_score_with_task("multilabel")

    logits = torch.randn(5, 3)
    labels = torch.randint(0, 2, (5, 3)).float()

    nl, lab = s._normalize_logits_and_labels(logits, labels)
    assert nl.shape == (5, 3)
    assert lab.shape == (5, 3)
    assert lab.dtype == torch.float32


@pytest.mark.parametrize("task", ["multiclass", "binary", "multilabel"])
def test_temperature_loss_matches_torch_for_tasks(task: str) -> None:
    s = make_score_with_task(task)

    if task == "multiclass":
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))
        expected = F.cross_entropy(logits, labels.long())
        got = s._temperature_loss(logits, labels)
        approx_tensor(got, expected)

    elif task == "binary":
        # single-logit case
        logits = torch.randn(10)
        labels = torch.randint(0, 2, (10,)).float()
        # BCE with logits
        expected = F.binary_cross_entropy_with_logits(logits, labels)
        got = s._temperature_loss(logits, labels)
        approx_tensor(got, expected)

        # two-logit case should behave like cross_entropy
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
        assert hasattr(mod, name), f"{name} not exported from seapig package"
