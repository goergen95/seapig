import pytest
import torch

from seapig.scores.logits import (
    EnergyScore,
    EntropyScore,
    LogitScore,
    MarginScore,
    MutualInformationScore,
    PredictiveVarianceScore,
    SoftmaxScore,
    _bernoulli_entropy,
    _shannon_entropy,
)


def test_bernoulli_entropy_extremes():
    # probabilities near 0 and 1 should yield near 0 entropy after clamping
    probs = torch.tensor([0.0, 1.0, 1e-20, 1 - 1e-20], dtype=torch.float64)
    entropy = _bernoulli_entropy(probs)
    # Values should be finite and very small (<= 1e-4)
    assert torch.isfinite(entropy).all()
    assert (entropy <= 1e-4).all()


def test_shannon_entropy_basic():
    # Simple 2-class distribution
    p = torch.tensor([[0.5, 0.5], [0.9, 0.1]], dtype=torch.float32)
    ent = _shannon_entropy(p, dim=1)
    expected = torch.tensor(
        [
            -(
                0.5 * torch.log(torch.tensor(0.5))
                + 0.5 * torch.log(torch.tensor(0.5))
            ),
            -(
                0.9 * torch.log(torch.tensor(0.9))
                + 0.1 * torch.log(torch.tensor(0.1))
            ),
        ]
    )
    assert torch.allclose(ent, expected, atol=1e-6)


def test_logit_score_resolve_errors():
    score = SoftmaxScore()
    dummy_model = torch.nn.Module()
    dummy_model.logits = lambda x: {"logit": torch.randn(2, 3)}  # type: ignore
    loader = torch.utils.data.DataLoader([torch.randn(2, 3)])  # type: ignore
    with pytest.raises(ValueError):
        score._resolve(
            torch.randn(2, 3),
            dummy_model,
            loader,
            None,
            None,
            want_labels=False,
        )
    with pytest.raises(ValueError):
        score._resolve(None, None, None, None, None, want_labels=False)


def test_logit_score_resolve_precomputed():
    score = SoftmaxScore()
    logits = torch.randn(4, 5)
    out_logits, out_labels = score._resolve(
        logits, None, None, None, None, want_labels=False
    )
    assert out_logits is logits
    assert out_labels is None


def test_flatten_members_and_labels():
    z = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    labels = torch.arange(2 * 3, dtype=torch.float32).view(2, 3)
    flat_z, flat_labels = LogitScore._flatten_members(z, labels)
    assert flat_z.shape == (8, 3)
    assert flat_labels.shape == (8, 3)
    for i in range(2):
        for m in range(4):
            idx = i * 4 + m
            assert torch.allclose(flat_z[idx], z[i, :, m])
            assert torch.allclose(flat_labels[idx], labels[i])


def test_ensemble_score_member_error():
    score = MutualInformationScore()
    logits = torch.randn(3, 4, 1)
    with pytest.raises(ValueError):
        score._score(logits)


def test_temperature_scaling_path():
    torch.manual_seed(0)
    logits = torch.randn(10, 3)
    labels = torch.randint(0, 3, (10,))
    # No scaling – temperature stays None
    score_no_scale = SoftmaxScore()
    score_no_scale.fit(logits, labels, temp_scale=False)
    assert score_no_scale.temperature is None
    # With scaling – temperature set to a positive float
    score_scale = SoftmaxScore()
    score_scale.fit(logits, labels, temp_scale=True)
    assert isinstance(score_scale.temperature, float)
    assert score_scale.temperature > 0


def test_pointwise_scores_consistency():
    torch.manual_seed(1)
    logits = torch.randn(5, 4)
    soft = SoftmaxScore()
    ent = EntropyScore()
    margin = MarginScore()
    energy = EnergyScore()
    for cls in (soft, ent, margin, energy):
        scores = cls.score(logits)
        assert scores.shape == (5,)
        assert torch.isfinite(scores).all()


def _run_pointwise_score(
    score_cls, logits, labels, temp_scale, task="multiclass"
):
    scorer = score_cls(task=task)  # specify task for binary cases
    scorer.fit(logits, labels, temp_scale=temp_scale)
    if temp_scale:
        assert isinstance(scorer.temperature, float) and scorer.temperature > 0
    else:
        assert scorer.temperature is None
    assert scorer.scores.shape == (logits.shape[0],)
    new_scores = scorer.score(logits)
    assert torch.allclose(new_scores, scorer.scores)
    sel = scorer.select(logits)
    assert isinstance(sel, dict)
    assert "score" in sel and "selected" in sel
    assert sel["score"].shape == (logits.shape[0],)
    assert sel["selected"].shape == (logits.shape[0],)
    assert sel["selected"].dtype == torch.bool


@pytest.mark.parametrize(
    "ScoreClass", [SoftmaxScore, EntropyScore, MarginScore, EnergyScore]
)
def test_pointwise_score_categorical(ScoreClass):
    torch.manual_seed(0)
    N, C = 20, 4
    logits = torch.randn(N, C)
    labels = torch.randint(0, C, (N,))
    _run_pointwise_score(ScoreClass, logits, labels, temp_scale=False)
    _run_pointwise_score(ScoreClass, logits, labels, temp_scale=True)


@pytest.mark.parametrize(
    "ScoreClass", [SoftmaxScore, EntropyScore, MarginScore, EnergyScore]
)
def test_pointwise_score_bernoulli_binary(ScoreClass):
    torch.manual_seed(1)
    N = 15
    logits = torch.randn(N)  # single-logit binary
    labels = torch.randint(0, 2, (N,)).float()
    _run_pointwise_score(
        ScoreClass, logits, labels, temp_scale=False, task="binary"
    )
    _run_pointwise_score(
        ScoreClass, logits, labels, temp_scale=True, task="binary"
    )


def _run_ensemble_score(
    score_cls, logits, labels, temp_scale, task="multiclass"
):
    scorer = score_cls(task=task)  # task selection for binary cases
    scorer.fit(logits, labels, temp_scale=temp_scale)
    if temp_scale:
        assert isinstance(scorer.temperature, float) and scorer.temperature > 0
    else:
        assert scorer.temperature is None
    assert scorer.scores.shape == (logits.shape[0],)
    new_scores = scorer.score(logits)
    assert torch.allclose(new_scores, scorer.scores)
    sel = scorer.select(logits)
    assert isinstance(sel, dict)
    assert torch.isfinite(sel["score"]).all()
    assert sel["selected"].shape == (logits.shape[0],)


@pytest.mark.parametrize(
    "ScoreClass", [MutualInformationScore, PredictiveVarianceScore]
)
def test_ensemble_score_categorical(ScoreClass):
    torch.manual_seed(2)
    N, K, M = 12, 3, 5
    logits = torch.randn(N, K, M)
    labels = torch.randint(0, K, (N,))
    _run_ensemble_score(ScoreClass, logits, labels, temp_scale=False)
    _run_ensemble_score(ScoreClass, logits, labels, temp_scale=True)


@pytest.mark.parametrize(
    "ScoreClass", [MutualInformationScore, PredictiveVarianceScore]
)
def test_ensemble_score_bernoulli_binary(ScoreClass):
    torch.manual_seed(3)
    N, M = 10, 4
    logits = torch.randn(N, M)  # binary single-logit with members
    labels = torch.randint(0, 2, (N,)).float()  # per-sample labels
    # task='binary' will resolve to single-logit variant and handle members
    _run_ensemble_score(
        ScoreClass, logits, labels, temp_scale=False, task="binary"
    )
    _run_ensemble_score(
        ScoreClass, logits, labels, temp_scale=True, task="binary"
    )
