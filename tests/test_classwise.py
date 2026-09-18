import logging

import pytest
import torch
from torch.utils.data import DataLoader

from seapig import scores as sp
from seapig.scores.classwise import ClassWiseMode, ClassWiseScore
from seapig.scores.logits import SoftmaxClassWiseScore
from seapig.scores.utils import TensorPCA
from tests.fixtures import DummyModel

torch.manual_seed(0)


class DummyScore(sp.UncertaintyScore):
    train_required = False
    cal_required = False
    ident = "dummy"

    def __init__(self) -> None:
        super().__init__()
        self.threshold = None

    def fit(
        self,
        X: torch.Tensor | None = None,
        Y: torch.Tensor | None = None,
        **kwargs,
    ):
        return None

    def score(self, X: torch.Tensor) -> torch.Tensor:
        return X.mean(dim=1)

    def set_threshold(self, q: float = 0.99) -> None:
        self.threshold = torch.tensor([0.5])

    def get_threshold(self) -> torch.Tensor:
        assert self.threshold is not None
        return self.threshold

    def select(self, X: torch.Tensor):  # pragma: no cover
        raise NotImplementedError


def make_data(single_label: bool = True):
    X = torch.arange(12, dtype=torch.float32).view(4, 3)
    if single_label:
        y = torch.tensor([0, 1, 0, 1])
    else:
        y = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)
    return X, y


def test_infer_mode():
    # single‑label vector
    y_single = torch.tensor([0, 1, 2])
    mode = ClassWiseScore._infer_mode(y_single)
    assert mode is ClassWiseMode.SINGLE_LABEL
    # multi‑label matrix
    y_multi = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)
    mode2 = ClassWiseScore._infer_mode(y_multi)
    assert mode2 is ClassWiseMode.MULTI_LABEL
    # invalid shape should raise
    with pytest.raises(ValueError):
        ClassWiseScore._infer_mode(torch.randn(2, 2, 2))


def _expected_softmax_single_score(logits, label):
    col = logits[:, label].unsqueeze(1)
    p = torch.sigmoid(col)
    return -torch.maximum(p, 1 - p).squeeze(1)


def test_single_label_fit_and_score():
    X = torch.randn(6, 3)
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    cw = SoftmaxClassWiseScore(task="multilabel")
    cw.fit(X=X, y=y)
    assert cw.mode is ClassWiseMode.SINGLE_LABEL
    scores = cw.score(X=X, y=y)
    assert scores.shape == (6,)
    # verify each entry matches the per‑class SoftmaxScore behaviour
    for i, lbl in enumerate(y.tolist()):
        expected = _expected_softmax_single_score(X, lbl)[i]
        assert torch.allclose(scores[i], expected)
    cw.set_threshold(q=0.5)
    thr = cw.get_threshold()
    assert isinstance(thr, dict)
    assert set(thr.keys()) == {0, 1, 2}
    sel = cw.select(X=X, y=y)
    mask = sel["selected"]
    assert mask.shape == (6,)
    for i, lbl in enumerate(y.tolist()):
        assert mask[i] == (scores[i] < thr[lbl])


@pytest.mark.parametrize(
    "agg, agg_fn",
    [
        (
            "mean",
            lambda s, m: torch.nanmean(s.masked_fill(~m, float("nan")), dim=1),
        ),
        ("max", lambda s, m: s.masked_fill(~m, float("-inf")).amax(dim=1)),
        ("min", lambda s, m: s.masked_fill(~m, float("inf")).amin(dim=1)),
    ],
)
def test_multi_label_aggregation(agg, agg_fn):
    X = torch.randn(4, 3)
    y = torch.tensor(
        [[1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=torch.float32
    )
    cw = SoftmaxClassWiseScore(task="multilabel", aggregation=agg)
    cw.fit(X=X, y=y)
    full = cw._score_full_matrix(X)
    mask = y.to(dtype=torch.bool)
    expected = agg_fn(full, mask)
    scores = cw.score(X=X, y=y)
    assert isinstance(scores, torch.Tensor)
    assert torch.allclose(scores, expected)
    cw.set_threshold(q=0.5)
    assert isinstance(cw.get_threshold(), torch.Tensor)
    sel = cw.select(X=X, y=y)
    assert sel["selected"].shape == (4,)
    overall_thr = cw.get_threshold()
    assert isinstance(overall_thr, torch.Tensor)
    assert torch.all(sel["selected"] == (scores < overall_thr))


def test_fit_errors_and_mode_property():
    X = torch.randn(2, 2)
    y = torch.tensor([0, 1])
    cw = SoftmaxClassWiseScore(task="multilabel")

    dummy = DummyModel()
    # providing both tensors and a model should raise ValueError
    with pytest.raises(ValueError):
        cw.fit(X=X, y=y, model=dummy, loaders={"train": []})  # type: ignore
    # accessing mode before fit should raise RuntimeError
    cw2 = SoftmaxClassWiseScore(task="multilabel")
    with pytest.raises(RuntimeError):
        _ = cw2.mode


def test_unknown_aggregation_raises():
    with pytest.raises(ValueError, match="Unknown aggregation 'invalid'"):
        ClassWiseScore(base_score_cls=sp.EuclideanScore, aggregation="invalid")


def test_make_extractor_branches():
    # KNN branch
    cw = ClassWiseScore(base_score_cls=sp.EuclideanScore)
    # Input
    extractor = cw._make_extractor(labels_from="input")
    assert extractor.output_keys == ("embedding",)
    assert extractor.input_keys == ("image", "label")

    # Output
    extractor = cw._make_extractor(labels_from="output")
    assert extractor.output_keys == ("embedding", "prediction")
    assert extractor.input_keys == ("image",)

    # None
    extractor = cw._make_extractor(labels_from=None)
    assert extractor.output_keys == ("embedding",)
    assert extractor.input_keys == ("image",)

    # Logit branch
    cw = ClassWiseScore(base_score_cls=sp.SoftmaxScore, task="multilabel")
    # Input
    extractor = cw._make_extractor(labels_from="input")
    assert extractor.output_keys == ("logit",)
    assert extractor.input_keys == ("image", "label")

    # Output
    extractor = cw._make_extractor(labels_from="output")
    assert extractor.output_keys == ("logit", "prediction")
    assert extractor.input_keys == ("image",)

    # None
    extractor = cw._make_extractor(labels_from=None)
    assert extractor.output_keys == ("logit",)
    assert extractor.input_keys == ("image",)


def test_model_mode_fit_with_pca_and_validation():
    # Simple dataset: two samples, two classes
    train_data = [
        {"image": torch.tensor([0.0, 0.0]), "label": torch.tensor(0)},
        {"image": torch.tensor([1.0, 1.0]), "label": torch.tensor(1)},
    ]
    val_data = [
        {"image": torch.tensor([0.5, 0.5]), "label": torch.tensor(0)},
        {"image": torch.tensor([1.5, 1.5]), "label": torch.tensor(1)},
    ]
    train_loader = DataLoader(train_data, batch_size=2, shuffle=False)  # type: ignore
    val_loader = DataLoader(val_data, batch_size=2, shuffle=False)  # type: ignore

    pca = TensorPCA(n_components=1)
    cw = sp.EuclideanClassWiseScore(global_pca=pca)
    cw.fit(
        model=DummyModel(), loaders={"train": train_loader, "val": val_loader}
    )
    # After fit, PCA should have reduced dimensionality to 1
    assert cw.pca is pca
    # Verify that class labels were inferred correctly
    assert cw._class_labels == [0, 1]
    # Ensure thresholds are calibrated
    cw.set_threshold()
    thr = cw.get_threshold()
    assert isinstance(thr, dict) and set(thr.keys()) == {0, 1}
    # Full‑matrix scoring via model mode should work
    full_scores = cw.score(
        model=DummyModel(), loader=train_loader, full_matrix=True
    )
    assert isinstance(full_scores, torch.Tensor)
    assert full_scores.shape == (2, 2)


def test_no_training_samples_for_class_raises():
    X = torch.randn(4, 2)
    # Multi‑label matrix with two columns; second class has no positive entries
    y = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]])
    cw = sp.EuclideanClassWiseScore()
    with pytest.raises(
        ValueError, match="No training samples found for class 1"
    ):
        cw.fit(X=X, y=y)


def test_validation_shape_mismatch_raises():
    X_train = torch.randn(2, 2)
    y_train = torch.tensor([0, 1])
    X_val = torch.randn(2, 2)  # mismatched number of rows compared to y_val
    y_val = torch.tensor([0, 1, 0])
    cw = sp.EuclideanClassWiseScore()
    with pytest.raises(AssertionError):
        cw.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)


def test_unknown_label_error_in_single_label_scoring():
    X = torch.randn(4, 2)
    y = torch.tensor([0, 1, 0, 1])
    cw = sp.EuclideanClassWiseScore()
    cw.fit(X=X, y=y)
    cw.set_threshold()
    X_new = torch.randn(2, 2)
    y_invalid = torch.tensor([2, 2])  # label 2 was never seen
    with pytest.raises(ValueError, match=r"Unknown class labels in y: \[2\]"):
        cw.score(X=X_new, y=y_invalid)


def test_plot_propagates_scorer_error():
    class BadPlotScore(sp.UncertaintyScore):
        def fit(self, X=None, Y=None, **kwargs):
            self.set_trained()

        @torch.inference_mode()
        def score(self, X):
            pass  # pragma: no cover

        def select(self, X):
            pass  # pragma: no cover

        def plot(self, query_scores=None, bins=100):
            raise RuntimeError("plot failure")

    cw = ClassWiseScore(base_score_cls=BadPlotScore)
    # Fit a single‑class dataset
    X = torch.randn(3, 2)
    y = torch.tensor([0, 0, 0])
    cw.fit(X=X, y=y)
    with pytest.raises(
        RuntimeError, match="Plot failed for class 0: plot failure"
    ):
        cw.plot()


def test_logit_score_requires_multilabel_task():
    with pytest.raises(
        ValueError, match="Class-wise logit scores require a multilabel task"
    ):
        ClassWiseScore(base_score_cls=sp.SoftmaxScore, task="single_label")


def test_resolve_aggregation_callable():
    agg = lambda s, m: torch.sum(s * m, dim=1)
    cw = ClassWiseScore(
        base_score_cls=sp.SoftmaxScore, aggregation=agg, task="multilabel"
    )
    # Create a tiny multi‑label dataset (2 classes, 3 samples).
    X = torch.randn(3, 2)  # dummy logits
    y = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32)
    cw.fit(X=X, y=y)
    cw.set_threshold(q=0.5)
    scores = cw.score(X=X, y=y)
    assert scores.shape == (3,)


def test_knn_full_matrix_scoring():
    cw = sp.EuclideanClassWiseScore()
    X_train = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y_train = torch.tensor([0, 0, 1, 1])
    cw.fit(X=X_train, y=y_train)
    cw.set_threshold(q=0.9)
    X_test = torch.tensor([[0.5, 0.5], [2.5, 2.5]])
    full = cw.score(X=X_test, full_matrix=True)
    assert isinstance(full, torch.Tensor)
    assert full.shape == (2, 2)  # N=2, C=2 classes


def test_single_label_score_invalid_y_shape():
    cw = sp.EuclideanClassWiseScore()
    X = torch.randn(4, 2)
    y = torch.tensor([0, 1, 0, 1])
    cw.fit(X=X, y=y)
    cw.set_threshold()
    X_new = torch.randn(2, 2)
    y_invalid = torch.tensor([[0, 1], [1, 0]])  # 2‑D instead of 1‑D
    with pytest.raises(
        ValueError,
        match="Model was fit in 'single_label' mode but received labels",
    ):
        cw.score(X=X_new, y=y_invalid)


def test_select_full_matrix_warning_and_mask():
    cw = sp.EuclideanClassWiseScore()
    X_train = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    y_train = torch.tensor([0, 1])
    cw.fit(X=X_train, y=y_train)
    result = cw.select(X=X_train, full_matrix=True)
    # Ensure thresholds are now available.
    assert cw.get_threshold(full_matrix=True) is not None
    scores = result["score"]
    mask = result["selected"]
    assert scores.shape == (2, 2)
    assert mask.shape == (2, 2)
    # Verify mask is computed as scores < per‑class thresholds.
    thr_dict = cw.get_threshold(full_matrix=True)
    assert isinstance(thr_dict, dict)
    assert cw._class_labels is not None
    for i, lbl in enumerate(cw._class_labels):
        expected = scores[:, i] < thr_dict[lbl]
        assert torch.equal(mask[:, i], expected)


def test_mode_property_before_fit_raises():
    cs = ClassWiseScore(base_score_cls=DummyScore)
    with pytest.raises(
        RuntimeError, match=r"fit\(\) must be called before mode"
    ):
        _ = cs.mode


def test_set_threshold_before_fit_raises():
    cs = ClassWiseScore(base_score_cls=DummyScore)
    with pytest.raises(
        RuntimeError, match=r"fit\(\) must be called before setting thresholds"
    ):
        cs.set_threshold()


def test_score_full_matrix_before_fit_raises():
    cs = ClassWiseScore(base_score_cls=DummyScore)
    X, _ = make_data(single_label=True)
    with pytest.raises(
        RuntimeError, match=r"fit\(\) must be called before scoring"
    ):
        cs._score_full_matrix(X)


def test_score_single_label_errors():
    X, y = make_data(single_label=True)
    cs = ClassWiseScore(base_score_cls=DummyScore)
    cs.fit(X=X, y=y)
    y_wrong = y.unsqueeze(1)
    with pytest.raises(ValueError, match="y must be a 1-D tensor"):
        cs._score_single_label(X, y_wrong)
    X_mismatch = X[:3]
    # y retains original length (4), mismatch rows should raise
    with pytest.raises(
        ValueError, match="X and y must have the same number of rows"
    ):
        cs._score_single_label(X_mismatch, y)
    y_unknown = torch.tensor([2, 2, 2, 2])
    with pytest.raises(ValueError, match="Unknown class labels in y"):
        cs._score_single_label(X, y_unknown)


def test_score_multi_label_errors():
    X, y = make_data(single_label=False)
    cs = ClassWiseScore(base_score_cls=DummyScore)
    cs.fit(X=X, y=y)
    # Pass a 1-D tensor to trigger dimension error
    with pytest.raises(ValueError, match="y must be a 2-D binary matrix"):
        cs._score_multi_label(X, y[:, 0])
    X_mismatch = X[:3]
    with pytest.raises(
        ValueError, match="X and y must have the same number of rows"
    ):
        cs._score_multi_label(X_mismatch, y)
    y_bad_cols = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32
    )
    with pytest.raises(ValueError, match="Number of label columns"):
        cs._score_multi_label(X, y_bad_cols)
    y_zero = y.clone()
    y_zero[0] = 0


def test_score_single_label_before_fit():
    cs = ClassWiseScore(base_score_cls=DummyScore)
    X, y = make_data(single_label=True)
    with pytest.raises(RuntimeError, match="fit must be called before scoring"):
        cs._score_single_label(X, y)


def test_score_single_label_continue_branch():
    # Fit on data containing both classes, then score with y that only has one class
    X, y = make_data(single_label=True)
    cs = ClassWiseScore(base_score_cls=DummyScore)
    cs.fit(X=X, y=y)
    y_partial = torch.tensor([0, 0, 0, 0])  # only class 0 present
    scores = cs._score_single_label(X, y_partial)
    # Scores should be computed for class 0 and unchanged for other positions
    assert scores.shape == (4,)
    # Ensure no error raised (continue hit)
    # Compare with direct score from DummyScore for class 0
    expected = DummyScore().score(X)
    assert torch.allclose(scores, expected)


def test_score_before_fit_raises():
    cs = ClassWiseScore(base_score_cls=DummyScore)
    X, _ = make_data(single_label=True)
    with pytest.raises(
        RuntimeError, match=r"fit\(\) must be called before scoring"
    ):
        cs.score(X=X, y=None)


def test_score_model_mode_assigns_y():
    X, y = make_data(single_label=True)
    cs = ClassWiseScore(base_score_cls=DummyScore)
    cs.fit(X=X, y=y)

    dummy_model = DummyModel()
    # Dummy loader (not used)
    dummy_loader = torch.utils.data.DataLoader([{"image": X, "label": y}])  # type: ignore

    class DummyExtractor:
        def __init__(self, out_key):
            self.out_key = out_key

        def extract(self, model, loader, **kwargs):
            return {self.out_key: X, "prediction": y}

    # Patch _make_extractor to return our dummy extractor
    cs._make_extractor = lambda labels_from: DummyExtractor(  # type: ignore
        out_key="embedding" if issubclass(DummyScore, sp.KNNScore) else "logit"
    )
    # Call score in model mode (full_matrix=False) to hit line 538
    result = cs.score(model=dummy_model, loader=dummy_loader)
    assert isinstance(result, torch.Tensor)
    # Ensure shape matches (4,)
    assert result.shape == (4,)


def test_score_model_mode_missing_label_raises():
    X, y = make_data(single_label=True)
    cs = ClassWiseScore(base_score_cls=DummyScore)
    cs.fit(X=X, y=y)

    dummy_model = DummyModel()
    dummy_loader = torch.utils.data.DataLoader([{"image": X}])  # type: ignore

    class DummyExtractorNoLabel:
        def __init__(self, out_key):
            self.out_key = out_key

        def extract(self, model, loader, **kwargs):
            return {self.out_key: X}

    cs._make_extractor = lambda labels_from: DummyExtractorNoLabel(  # type: ignore
        out_key="embedding" if issubclass(DummyScore, sp.KNNScore) else "logit"
    )
    with pytest.raises(ValueError, match="Mode 'single_label' requires labels"):
        cs.score(model=dummy_model, loader=dummy_loader)


def test_full_matrix_scoring_and_select():
    X, y = make_data(single_label=False)
    cs = ClassWiseScore(base_score_cls=DummyScore, aggregation="mean")
    cs.fit(X=X, y=y)
    full = cs.score(X=X, y=y, full_matrix=True)
    assert full.shape == (4, 2)
    cs.set_threshold()
    thr = cs.get_threshold(full_matrix=True)
    assert isinstance(thr, dict)
    result = cs.select(X=X, y=y, full_matrix=True)
    assert result["score"].shape == (4, 2)
    assert result["selected"].shape == (4, 2)


def test_fit_multi_label_with_validation_sets_scores():
    X = torch.randn(4, 3)
    y = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)
    X_val = torch.randn(2, 3)
    y_val = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    cw = sp.SoftmaxClassWiseScore(task="multilabel")
    cw.fit(X=X, y=y, X_val=X_val, y_val=y_val)
    expected = cw._score_multi_label(X_val, y_val)
    assert hasattr(cw, "scores")
    assert cw.scores is not None
    assert torch.allclose(cw.scores, expected)


def test_score_multi_label_before_fit_raises():
    cw = sp.SoftmaxClassWiseScore(task="multilabel")
    X = torch.randn(2, 3)
    y = torch.randn(2, 2)
    with pytest.raises(RuntimeError, match="fit must be called before scoring"):
        cw._score_multi_label(X, y)


def test_score_tensor_and_model_raises():
    cs = ClassWiseScore(base_score_cls=DummyScore)
    X = torch.randn(2, 2)
    with pytest.raises(ValueError, match="Specify either pre-computed tensors"):
        cs.score(X=X, model=DummyModel(), loader=DataLoader([]))  # type: ignore


def test_multi_label_no_positive_labels_filled(caplog):
    # Sample with a row that has no positive labels should be filled with average scores.
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    y = torch.tensor(
        [[1, 0], [0, 0], [0, 1]], dtype=torch.float32
    )  # second sample has no positives
    cw = ClassWiseScore(base_score_cls=DummyScore)  # default aggregation "mean"
    cw.fit(X=X, y=y)
    with caplog.at_level(logging.WARNING):
        scores = cw.score(X=X, y=y)
    full = cw._score_full_matrix(X)
    avg_scores = torch.nanmean(full, dim=1)
    assert isinstance(scores, torch.Tensor)
    assert torch.allclose(scores, avg_scores)
    assert any(
        "Samples with no positive labels encountered" in rec.getMessage()
        for rec in caplog.records
    )


def test_extractor_collects_output_and_prediction():

    class SimpleModel(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            pred = (x > 0).long()
            return {"embedding": x, "prediction": pred}

    data = [
        {"image": torch.tensor([-1.0, 2.0]), "label": torch.tensor(0)},
        {"image": torch.tensor([3.0, -4.0]), "label": torch.tensor(1)},
    ]
    loader = DataLoader(data, batch_size=2, shuffle=False)  # type: ignore

    cw = ClassWiseScore(base_score_cls=sp.EuclideanScore)
    extractor = cw._make_extractor(labels_from="input")
    extracted = extractor.extract(model=SimpleModel(), loader=loader)
    assert "label" in extracted, "The label key should be present"
    extractor = cw._make_extractor(labels_from="output")
    extracted = extractor.extract(model=SimpleModel(), loader=loader)
    assert "prediction" in extracted, "The prediction key should be present"
    # Expected values: 0 where the original input ≤ 0, 1 where > 0.
    # The loader batches the two samples together, so we get a (2, 2) tensor.
    expected = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    assert torch.equal(extracted["prediction"], expected), (
        "Prediction values are incorrect"
    )
