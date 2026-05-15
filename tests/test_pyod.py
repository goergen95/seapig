import numpy as np
import pytest
import torch

pytest.importorskip("pyod")

from seapig.scores.pyod import PyODScore
from seapig.scores.utils import TensorPCA


class _MockDetectorBasic:
    """Simple detector that sets decision_scores_ on fit and returns a fixed
    decision_function output."""

    def __init__(self, score_on_call: float = 0.5):
        self.decision_scores_: np.ndarray | None = None
        self._call_score = score_on_call

    def fit(self, X: np.ndarray) -> None:
        # produce deterministic scores based on number of samples
        self.decision_scores_ = np.zeros(X.shape[0])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._call_score)  # pragma: no cover


class _MockDetectorRange:
    """Detector that assigns increasing decision_scores_ during fit.
    Useful for q-trimming tests."""

    def __init__(self) -> None:
        self.decision_scores_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        n = X.shape[0]
        self.decision_scores_ = np.linspace(0.0, 1.0, n)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])  # pragma: no cover


def test_fit_sets_trained_and_scores_without_cal() -> None:
    """When calibration is not required, _fit_impl should set trained state and
    populate scores from detector.decision_scores_."""
    refs = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    det = _MockDetectorBasic(score_on_call=0.1)
    score = PyODScore(detector=det, pca=None)  # type: ignore[invalid-argument-type]
    score.cal_required = False
    score.ref_embeddings = refs

    # run internal fit implementation
    score._fit_impl(q=None)

    assert score.is_trained()
    assert isinstance(score.scores, torch.Tensor)
    # detector produced zeros on fit -> scores should be zeros
    assert torch.allclose(score.scores, torch.zeros(3))


def test_fit_with_calibration_sets_calibrated_and_scores_from_decision_function() -> (
    None
):
    """When calibration embeddings are present, final scores should come from
    detector.decision_function and the score should be marked calibrated."""
    refs = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    cal = torch.tensor([[10.0, 10.0], [20.0, 20.0]])

    class DetCal(_MockDetectorBasic):
        def decision_function(self, X: np.ndarray) -> np.ndarray:
            # return distinct value per calibration sample
            return np.arange(X.shape[0]) + 5.0

    det = DetCal()
    score = PyODScore(detector=det, pca=None)  # type: ignore[invalid-argument-type]
    score.ref_embeddings = refs
    score.cal_embeddings = cal

    score._fit_impl(q=None)

    assert score.is_trained()
    assert score.is_calibrated()
    assert isinstance(score.scores, torch.Tensor)
    assert torch.allclose(score.scores, torch.tensor([5.0, 6.0]))


def test_score_uses_detector_decision_function() -> None:
    """score(X) should call the detector.decision_function and return a
    torch.Tensor with those values."""

    class DetFn(_MockDetectorBasic):
        def decision_function(self, X: np.ndarray) -> np.ndarray:
            # return sum of each row so we can validate the mapping
            result: np.ndarray = np.array(X.sum(axis=1), dtype=np.float64)
            return result

    det = DetFn()
    score = PyODScore(detector=det, pca=None)  # type: ignore[invalid-argument-type]
    # ensure detector present; no need to call _fit_impl for score()
    q = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    out = score.score(q)

    assert isinstance(out, torch.Tensor)
    expected = torch.tensor([3.0, 7.0])
    assert torch.allclose(out, expected)


def test_q_trimming_reduces_reference_set() -> None:
    """When q is provided, _fit_impl should trim reference embeddings based on
    detector.decision_scores_."""
    n = 100
    refs = torch.randn(n, 5)
    det = _MockDetectorRange()
    score = PyODScore(detector=det, pca=None)  # type: ignore[invalid-argument-type]
    score.cal_required = False
    score.ref_embeddings = refs.float()

    original_count = score.ref_embeddings.shape[0]
    score._fit_impl(q=0.50)
    new_count = score.ref_embeddings.shape[0]

    assert new_count < original_count
    assert new_count >= 1


def test_pca_predict_is_applied_before_detector_fit() -> None:
    """If PCA is configured, _fit_impl should call _fit_pca and replace
    ref_embeddings with the PCA.predict result before fitting the detector."""
    refs = torch.randn(100, 64)
    det = _MockDetectorBasic()
    score = PyODScore(detector=det, pca=TensorPCA(n_components=0.90))  # type: ignore[invalid-argument-type]
    score.cal_required = False
    score.ref_embeddings = refs.clone()
    score._fit_impl(q=None)
    assert score.ref_embeddings.shape[1] < refs.shape[1]
