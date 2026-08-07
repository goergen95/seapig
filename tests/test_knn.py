"""Unit tests for seapig.scores.knn."""

import pathlib
import warnings

import faiss
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from seapig.scores.knn import (
    CosineScore,
    EuclideanScore,
    KNNScore,
    MahalanobisScore,
)
from seapig.scores.utils import TensorPCA

# =====================================================================
# Helpers
# =====================================================================


class DummyModel(torch.nn.Module):
    """Identity embedder; counts calls to `embed`."""

    def __init__(self, dim: int = 2) -> None:
        super().__init__()
        # need at least one parameter so `next(model.parameters()).device` works
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.embed_dim = dim
        self.embed_calls = 0

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        self.embed_calls += 1
        return x


class DictDS(Dataset):
    """Yields ``{'image': (D,) tensor, 'label': int scalar}``."""

    def __init__(self, n: int, d: int, num_classes: int = 3) -> None:
        self.n, self.d, self.C = n, d, num_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):  # ty: ignore [invalid-method-override]
        return {
            "image": torch.randn(self.d),
            "label": torch.tensor(idx % self.C, dtype=torch.long),
        }


class TupleDS(Dataset):
    """Yields ``((D,) tensor, int scalar)``."""

    def __init__(self, n: int, d: int, num_classes: int = 3) -> None:
        self.n, self.d, self.C = n, d, num_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):  # ty: ignore [invalid-method-override]
        return torch.randn(self.d), torch.tensor(idx % self.C, dtype=torch.long)


def _dict_loader(n: int, d: int, batch_size: int = 3, C: int = 3) -> DataLoader:
    return DataLoader(
        DictDS(n, d, C),
        batch_size=batch_size,
        collate_fn=lambda b: {
            "image": torch.stack([x["image"] for x in b]),
            "label": torch.stack([x["label"] for x in b]),
        },
    )


def approx(t1: torch.Tensor, t2: torch.Tensor, tol: float = 1e-5) -> None:
    torch.testing.assert_close(t1, t2, atol=tol, rtol=0)


def _small_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced 3-class dataset with 2 samples per class."""
    X = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
            [2.0, 2.0],
            [2.1, 2.0],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    return X, y


# =====================================================================
# Label normalization
# =====================================================================


def test_normalize_labels_single_auto_num_classes():
    labels = torch.tensor([0, 2, 1])
    mh, C, mode = KNNScore._normalize_labels(labels)
    assert mode == "single" and C == 3
    expected = torch.tensor(
        [[True, False, False], [False, False, True], [False, True, False]]
    )
    assert torch.equal(mh, expected)


def test_normalize_labels_single_provided_num_classes():
    mh, C, mode = KNNScore._normalize_labels(
        torch.tensor([0, 2]), num_classes=5
    )
    assert C == 5 and mh.shape == (2, 5) and mode == "single"
    assert mh[0, 0] and mh[1, 2] and int(mh.sum()) == 2


def test_normalize_labels_multi_bool_and_int_equivalent():
    bool_labels = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
    mh_b, C, mode = KNNScore._normalize_labels(bool_labels)
    mh_i, _, mode_i = KNNScore._normalize_labels(
        torch.tensor([[1, 0], [0, 1]], dtype=torch.int)
    )
    assert mode == "multi" and mode_i == "multi" and C == 2
    assert torch.equal(mh_b, bool_labels) and torch.equal(mh_i, bool_labels)


@pytest.mark.parametrize(
    "labels, kwargs, match",
    [
        (torch.tensor([0.1, 0.2]), {}, "1-D labels must be of integer"),
        (
            torch.tensor([[0, 2], [1, 3]]),
            {},
            r"2-D labels must be bool- or \{0,1\}",
        ),
        (torch.randn(2, 2, 2), {}, "labels must be 1-D or 2-D"),
        (
            torch.tensor([[1, 0], [0, 1]], dtype=torch.bool),
            {"num_classes": 3},
            "num_classes=3 does not match",
        ),
    ],
)
def test_normalize_labels_errors(labels, kwargs, match):
    with pytest.raises(ValueError, match=match):
        KNNScore._normalize_labels(labels, **kwargs)


# =====================================================================
# _stat and index parameter helpers
# =====================================================================


@pytest.mark.parametrize(
    "stat, expected",
    [
        ("max", torch.tensor([3.0, 6.0])),
        ("min", torch.tensor([1.0, 4.0])),
        ("mean", torch.tensor([2.0, 5.0])),
        ("median", torch.tensor([2.0, 5.0])),
    ],
)
def test_stat_aggregations(stat, expected):
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    approx(EuclideanScore(stat=stat)._stat(x), expected)


def test_suggest_build_params_small_n():
    embs = torch.randn(5, 60)
    p = KNNScore._suggest_build_params(embs, k=1)
    assert p["M"] == 16 and p["efConstruction"] == max(4 * 16, 128)


def test_suggest_build_params_large_n_dim_gt_64():
    """N > 10_000 & d > 64 triggers the M=24 branch."""
    embs = torch.randn(20_000, 70)
    p = KNNScore._suggest_build_params(embs, k=3)
    assert p["M"] == 24 and p["efConstruction"] >= 4 * 24


def test_suggest_build_params_requires_2d():
    with pytest.raises(ValueError, match="embeddings must be 2D"):
        KNNScore._suggest_build_params(torch.randn(3), k=1)


def test_suggest_query_params_bounds():
    embs = torch.randn(5, 10)
    b = KNNScore._suggest_build_params(embs, k=2)
    q = KNNScore._suggest_query_params(embs, k=2)
    assert q["efSearch"] >= 2 and q["efSearch"] <= b["efConstruction"]


# =====================================================================
# Class-agnostic distances
# =====================================================================


def test_euclidean_distance_simple_nearest():
    ref = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    q = torch.tensor([[6.0, 8.0]])
    s = EuclideanScore(k=1, stat="min")
    s.ref_embeddings = ref
    s._setup_index()
    out, _ = s._distance(q, offset=0)
    approx(out, torch.tensor([[5.0]]))


@pytest.mark.parametrize(
    "stat, fn",
    [
        ("max", lambda d: d.max()),
        ("min", lambda d: d.min()),
        ("mean", lambda d: d.mean()),
        ("median", lambda d: d.median()),
    ],
)
def test_euclidean_k_and_stats(stat, fn):
    refs = torch.tensor([[3.0, 4.0], [6.0, 8.0], [0.0, 5.0]])
    q = torch.tensor([[0.0, 0.0]])
    s = EuclideanScore(k=2, stat=stat)
    s.ref_embeddings = refs
    s._setup_index()
    dists, _ = s._distance(q, offset=0)
    approx(s._stat(dists), fn(torch.tensor([5.0, 5.0])).unsqueeze(0))


def test_cosine_identical_zero_distance():
    s = CosineScore(k=1, stat="max")
    s.ref_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    s._setup_index()
    out, _ = s._distance(torch.tensor([[1.0, 0.0]]), offset=0)
    approx(out.flatten(), torch.tensor([0.0]))


def test_cosine_orthogonal_unit_distance():
    s = CosineScore(k=2, stat="max")
    s.ref_embeddings = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    s._setup_index()
    out, _ = s._distance(torch.tensor([[0.0, 1.0]]), offset=0)
    approx(s._stat(out), torch.tensor([1.0]))


def test_mahalanobis_matches_manual():
    refs = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    query = torch.tensor([[1.0, 1.0]])
    s = MahalanobisScore(k=1, stat="min")
    s.ref_embeddings = refs
    s._setup_index()
    cov = refs.T.cov()
    inv = torch.linalg.inv(cov + 1e-4 * cov.diagonal().mean() * torch.eye(2))
    manual = [
        torch.sqrt(
            (
                (query[0] - p).unsqueeze(0)
                @ inv
                @ (query[0] - p).unsqueeze(0).T
            ).squeeze()
        ).item()
        for p in refs
    ]
    out, _ = s._distance(query, offset=0)
    approx(out.squeeze(), torch.tensor(min(manual)))


def test_mahalanobis_singular_covariance_regularized():
    refs = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    s = MahalanobisScore(k=1)
    s.ref_embeddings = refs
    s._setup_index()  # regularization must prevent Cholesky failure
    assert s.vi_zero is not None
    out, _ = s._distance(refs[:1], offset=0)
    approx(out.squeeze(), torch.tensor(0.0))


@pytest.mark.parametrize("cls", [EuclideanScore, CosineScore, MahalanobisScore])
def test_setup_index_creates_index(cls):
    s = cls(k=1)
    s.ref_embeddings = torch.randn(5, 3)
    s._setup_index()
    assert s.index is not None


# =====================================================================
# _query_index / knn_search
# =====================================================================


def test_query_dimension_mismatch_raises():
    s = EuclideanScore(k=1)
    s.ref_embeddings = torch.randn(5, 16)
    s._setup_index()
    with pytest.raises(ValueError, match="does not match index dimension"):
        s._query_index(torch.randn(1, 8), offset=0)


def test_query_index_hnsw_branch_used_for_large_n():
    """N > 10_000 triggers HNSW index and the efSearch branch in _query_index."""
    s = EuclideanScore(k=3)
    s.ref_embeddings = torch.randn(10_001, 8, dtype=torch.float32)
    s._setup_index()
    assert isinstance(s.index, faiss.IndexHNSW)
    out = s.score(torch.randn(2, 8))
    assert out.shape == (2,)


def test_knn_search_offset_and_indices():
    s = EuclideanScore(k=2, stat="min")
    s.ref_embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    s._setup_index()
    d, idx = s.knn_search(torch.tensor([[0.0, 0.0]]), offset=1)
    approx(d, torch.tensor([[1.0, 1.0]]))
    assert idx.shape == (1, 2) and set(idx[0].tolist()) == {1, 2}


def test_knn_search_with_class_id():
    refs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
    labels = torch.tensor([0, 0, 1, 1])
    s = EuclideanScore(
        k=1, class_wise=True, num_classes=2, min_samples_per_class=1
    )
    s.fit(X=refs, labels=labels)
    q = torch.tensor([[0.1, 0.0]])
    d0, _ = s.knn_search(q, class_id=0)
    d1, _ = s.knn_search(q, class_id=1)
    assert d0.item() < d1.item()


# =====================================================================
# PCA integration
# =====================================================================


def test_pca_reduces_dimension_and_applied_on_score():
    torch.manual_seed(0)
    n, D = 50, 6
    refs = (torch.randn(n, 1) @ torch.randn(1, D)) + 0.01 * torch.randn(n, D)
    q = torch.randn(1, D)

    s_pca = EuclideanScore(k=1, pca=TensorPCA(n_components=0.90))
    s_pca.cal_required = False
    s_pca.ref_embeddings = refs.float()
    s_pca._fit_impl(q=None)
    assert s_pca.ref_embeddings.shape[1] < D

    s_ref = EuclideanScore(k=1)
    s_ref.ref_embeddings = s_pca.ref_embeddings.clone()
    s_ref._setup_index()
    approx(s_pca.score(q), s_ref.score(s_pca.pca.transform(q)))


def test_pca_applied_in_classwise_scoring():
    torch.manual_seed(0)
    n, D = 40, 6
    refs = torch.randn(n, D)
    labels = torch.tensor([0] * 20 + [1] * 20)
    s = EuclideanScore(
        k=1,
        class_wise=True,
        pca=TensorPCA(n_components=0.9),
        min_samples_per_class=1,
    )
    s.fit(X=refs.float(), labels=labels)
    assert s.ref_embeddings.shape[1] < D
    out = s.score(X=torch.randn(3, D))
    assert out.shape == (3,)


# =====================================================================
# `q` outlier trimming
# =====================================================================


def test_q_trimming_reduces_reference_set_global():
    n = 100
    s = EuclideanScore(k=1)
    s.cal_required = False
    s.ref_embeddings = torch.randn(n, 5).float()
    s._fit_impl(q=0.5)
    assert s.ref_embeddings.shape[0] < n


def test_q_trimming_classwise_per_class():
    torch.manual_seed(0)
    refs = torch.randn(40, 3)
    labels = torch.tensor([0] * 20 + [1] * 20)
    s = EuclideanScore(k=1, class_wise=True, min_samples_per_class=1)
    s.fit(X=refs.float(), labels=labels, q=0.5)
    for c in (0, 1):
        assert s.indices_by_class[c].ntotal > 0


# =====================================================================
# Index persistence
# =====================================================================


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_build_index_saves_and_loads(tmp_path: pathlib.Path):
    path = tmp_path / "test_index.bin"
    s1 = EuclideanScore(k=1, save_index=path)
    s1.ref_embeddings = torch.randn(10, 3)
    s1._setup_index()
    assert path.exists()

    s2 = EuclideanScore(k=1, save_index=path)
    s2.ref_embeddings = torch.randn(5, 3)
    with pytest.warns(UserWarning, match="already exists"):
        s2._setup_index()
    assert s2.index is not None


def test_save_index_true_creates_default_bin_path():
    s = EuclideanScore(save_index=True)
    assert isinstance(s.index_path, pathlib.Path)
    assert s.index_path.suffix == ".bin"


def test_save_index_path_wrong_suffix_raises(tmp_path):
    with pytest.raises(AssertionError):
        EuclideanScore(save_index=tmp_path / "bad.idx")


def test_classwise_save_index_true_creates_directory():
    s = EuclideanScore(class_wise=True, save_index=True)
    assert s.index_path.is_dir()


def test_classwise_save_index_custom_directory(tmp_path):
    d = tmp_path / "mydir"
    s = EuclideanScore(class_wise=True, save_index=d)
    assert s.index_path == d and d.is_dir()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_classwise_per_class_indices_saved_and_loaded(tmp_path):
    d = tmp_path / "cls_indices"
    refs = torch.randn(10, 3)
    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    s1 = EuclideanScore(class_wise=True, save_index=d, min_samples_per_class=1)
    s1.fit(X=refs, labels=labels)
    assert (d / "class0.bin").exists() and (d / "class1.bin").exists()

    s2 = EuclideanScore(class_wise=True, save_index=d, min_samples_per_class=1)
    with pytest.warns(UserWarning, match="already exists"):
        s2.fit(X=refs, labels=labels)
    assert 0 in s2.indices_by_class and 1 in s2.indices_by_class


# =====================================================================
# Config validation
# =====================================================================


def test_per_class_agg_with_global_threshold_raises():
    with pytest.raises(ValueError, match="not supported"):
        EuclideanScore(
            class_wise=True,
            class_aggregation="per_class",
            threshold_mode="global",
        )


def test_class_wise_kwargs_ignored_when_flag_false_warns():
    with pytest.warns(
        UserWarning, match="class_wise=False but class-wise kwargs"
    ):
        EuclideanScore(class_wise=False, threshold_mode="per_class")


# =====================================================================
# fit paths (class-agnostic & class-wise)
# =====================================================================


def test_fit_class_agnostic_warns_when_labels_supplied():
    s = EuclideanScore(k=1)
    s.cal_required = False
    with pytest.warns(UserWarning, match="labels/cal_labels provided"):
        s.fit(X=torch.randn(5, 3), labels=torch.tensor([0, 1, 0, 1, 0]))


def test_fit_classwise_precomputed_requires_labels():
    s = EuclideanScore(class_wise=True)
    with pytest.raises(ValueError, match="requires `labels`"):
        s.fit(X=torch.randn(3, 2))


def test_fit_classwise_requires_cal_labels_when_Y_given():
    s = EuclideanScore(class_wise=True, min_samples_per_class=1)
    with pytest.raises(ValueError, match="requires `cal_labels`"):
        s.fit(
            X=torch.randn(3, 2),
            Y=torch.randn(2, 2),
            labels=torch.tensor([0, 1, 0]),
        )


def test_fit_classwise_both_X_and_loaders_raises():
    s = EuclideanScore(class_wise=True)
    with pytest.raises(ValueError, match="Cannot specify both"):
        s.fit(
            X=torch.randn(2, 2),
            labels=torch.tensor([0, 1]),
            model=DummyModel(),
            loaders={"train": None},  # ty: ignore [invalid-argument-type]
        )


def test_fit_classwise_neither_source_raises():
    s = EuclideanScore(class_wise=True)
    with pytest.raises(ValueError, match="Must specify either"):
        s.fit()


def test_fit_classwise_with_loaders_train_and_val():
    d = 4
    loaders = {
        "train": _dict_loader(9, d, batch_size=3),
        "val": _dict_loader(6, d, batch_size=2),
    }
    s = EuclideanScore(
        class_wise=True, min_samples_per_class=1, class_aggregation="min"
    )
    s.fit(model=DummyModel(d), loaders=loaders)
    assert s.ref_embeddings.shape == (9, d)
    assert s.cal_embeddings.shape == (6, d)
    assert s.label_mode == "single"
    assert set(s.indices_by_class.keys()) <= {0, 1, 2}
    assert s.is_calibrated()


# =====================================================================
# Class-wise fit internals
# =====================================================================


def test_empty_and_fallback_classes_state_and_warnings():
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    y = torch.tensor([0, 0, 1, 0])
    s = EuclideanScore(class_wise=True, num_classes=3, min_samples_per_class=5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s.fit(X=X, labels=y)
    msgs = [str(m.message) for m in w]
    assert any("Class 2 has 0 training samples" in m for m in msgs)
    assert any("Class 1 has 1 < min_samples_per_class" in m for m in msgs)
    assert set(s.indices_by_class.keys()) == {0, 1}
    assert s.fallback_classes == {0, 1}  # both have < 5 samples
    assert s.empty_classes == {2}


def test_classwise_calibration_missing_class_warns():
    X, y = _small_dataset()
    Y = torch.tensor([[0.0, 0.0]])  # only class 0
    y_cal = torch.tensor([0])
    s = EuclideanScore(class_wise=True, min_samples_per_class=1, num_classes=3)
    with pytest.warns(UserWarning, match="no calibration samples"):
        s.fit(X=X, Y=Y, labels=y, cal_labels=y_cal)
    assert s.is_calibrated()
    assert s.scores_by_class is not None
    assert 0 in s.scores_by_class
    assert 1 not in s.scores_by_class and 2 not in s.scores_by_class


def test_classwise_self_loo_skips_single_sample_class():
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 0, 1])  # class 1 has 1 sample
    s = EuclideanScore(class_wise=True, min_samples_per_class=1)
    s.fit(X=X, labels=y)
    assert s.scores_by_class is not None
    assert 0 in s.scores_by_class
    assert 1 not in s.scores_by_class


def test_classwise_num_classes_inferred_from_labels():
    X, y = _small_dataset()
    s = EuclideanScore(class_wise=True, min_samples_per_class=1)
    s.fit(X=X, labels=y)
    assert s.num_classes == 3


# =====================================================================
# Class-wise aggregations at scoring time
# =====================================================================


def test_classwise_min_aggregation_selects_close_query():
    X, y = _small_dataset()
    s = EuclideanScore(
        class_wise=True, class_aggregation="min", min_samples_per_class=1
    )
    s.fit(X=X, labels=y)
    s.set_threshold(q=0.99)
    res = s.select(X=torch.tensor([[0.05, 0.0]]))
    assert bool(res["selected"].item()) is True


def test_classwise_mean_aggregation_matches_manual():
    X, y = _small_dataset()
    s = EuclideanScore(
        class_wise=True, class_aggregation="mean", min_samples_per_class=1
    )
    s.fit(X=X, labels=y)
    q = torch.tensor([[0.0, 0.0]])
    out = s.score(X=q)
    per_c = [s._stat(s._distance_for_class(c, q)[0]).item() for c in range(3)]
    approx(out, torch.tensor([sum(per_c) / 3]))


def test_classwise_per_class_aggregation_and_per_class_thresholds():
    X, y = _small_dataset()
    s = EuclideanScore(
        class_wise=True,
        class_aggregation="per_class",
        threshold_mode="per_class",
        min_samples_per_class=1,
    )
    s.fit(X=X, labels=y)
    s.set_threshold(q=0.5)
    assert set(s.threshold_by_class.keys()) == {0, 1, 2}
    out = s.select(X=X)
    assert out["score"].shape == (X.shape[0], 3)
    assert out["selected"].shape == (X.shape[0], 3)
    assert out["selected"].dtype == torch.bool


def test_classwise_target_mean_matches_manual():
    refs = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    s = EuclideanScore(
        class_wise=True,
        class_aggregation="target",
        target_reduce="mean",
        min_samples_per_class=1,
    )
    s.fit(X=refs, labels=labels)
    out = s.score(
        X=torch.tensor([[1.0, 1.0]]), target=torch.tensor([[1, 0, 1]])
    )
    approx(out, torch.sqrt(torch.tensor(2.0)).unsqueeze(0))


# =====================================================================
# score() overrides — class-wise + target error branches
# =====================================================================


def _target_scorer(cal: bool = False) -> EuclideanScore:
    s = EuclideanScore(
        class_wise=True, class_aggregation="target", min_samples_per_class=1
    )
    s.fit(X=torch.randn(6, 2), labels=torch.tensor([0, 1, 0, 1, 0, 1]))
    if cal:
        s.set_threshold()
    return s


def test_score_target_missing_target_with_X_raises():
    s = _target_scorer()
    with pytest.raises(ValueError, match="target labels are required"):
        s.score(X=torch.randn(2, 2))


def test_score_target_via_loader_extracts_labels():
    s = _target_scorer()
    out = s.score(model=DummyModel(2), loader=_dict_loader(6, 2, C=2))
    assert out.shape == (6,)


def test_score_target_loader_with_target_raises():
    s = _target_scorer()
    with pytest.raises(ValueError, match="target must be None"):
        s.score(
            model=DummyModel(2),
            loader=_dict_loader(2, 2, C=2),
            target=torch.tensor([0, 1]),
        )


def test_score_target_both_X_and_loader_raises():
    s = _target_scorer()
    with pytest.raises(ValueError, match="Cannot specify both"):
        s.score(
            X=torch.randn(1, 2),
            model=DummyModel(2),
            loader=_dict_loader(1, 2, C=2),
            target=torch.tensor([0]),
        )


def test_score_target_neither_source_raises():
    s = _target_scorer()
    with pytest.raises(ValueError, match="Must specify either"):
        s.score()


def test_score_warns_when_target_supplied_but_ignored():
    s = EuclideanScore(k=1)
    s.ref_embeddings = torch.randn(3, 2)
    s._setup_index()
    s.set_trained()
    with pytest.warns(UserWarning, match="`target` argument is only used"):
        out = s.score(torch.randn(1, 2), target=torch.tensor([0]))
    assert out.shape == (1,)


# =====================================================================
# _reduce_target
# =====================================================================


def test_reduce_target_variants():
    per_class = torch.tensor(
        [[0.5, 2.0, float("inf")], [1.0, 0.2, 3.0]], dtype=torch.float32
    )
    target = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    approx(
        EuclideanScore(target_reduce="mean", class_wise=True)._reduce_target(
            per_class, target
        ),
        torch.tensor([1.25, 0.6]),
    )
    approx(
        EuclideanScore(target_reduce="min", class_wise=True)._reduce_target(
            per_class, target
        ),
        torch.tensor([0.5, 0.2]),
    )
    approx(
        EuclideanScore(target_reduce="max", class_wise=True)._reduce_target(
            per_class, target
        ),
        torch.tensor([2.0, 1.0]),
    )


# =====================================================================
# select() overrides — per-class threshold branches
# =====================================================================


def test_select_per_class_threshold_min_agg():
    X, y = _small_dataset()
    s = EuclideanScore(
        class_wise=True,
        class_aggregation="min",
        threshold_mode="per_class",
        min_samples_per_class=1,
    )
    s.fit(X=X, labels=y)
    s.set_threshold(q=0.99)
    out = s.select(X=X)
    assert out["selected"].shape == (X.shape[0],)
    assert out["selected"].dtype == torch.bool


def test_select_per_class_threshold_mean_agg():
    X, y = _small_dataset()
    s = EuclideanScore(
        class_wise=True,
        class_aggregation="mean",
        threshold_mode="per_class",
        min_samples_per_class=1,
    )
    s.fit(X=X, labels=y)
    s.set_threshold(q=0.99)
    out = s.select(X=X)
    assert out["selected"].shape == (X.shape[0],)


def test_select_per_class_threshold_target_agg():
    refs = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [2.0, 0.0], [2.1, 0.0], [0.0, 2.0], [0.0, 2.1]]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    s = EuclideanScore(
        class_wise=True,
        class_aggregation="target",
        threshold_mode="per_class",
        min_samples_per_class=1,
    )
    s.fit(X=refs, labels=labels)
    s.set_threshold(q=0.99)
    out = s.select(
        X=torch.tensor([[1.0, 1.0]]), target=torch.tensor([[1, 0, 1]])
    )
    assert out["selected"].shape == (1,)


def test_select_auto_sets_global_threshold_if_missing():
    X, y = _small_dataset()
    s = EuclideanScore(
        class_wise=True, class_aggregation="min", min_samples_per_class=1
    )
    s.fit(X=X, labels=y)
    assert s.get_threshold() is None
    out = s.select(X=X)
    assert s.get_threshold() is not None
    assert out["selected"].shape == (X.shape[0],)


def test_select_auto_sets_per_class_thresholds_if_missing():
    X, y = _small_dataset()
    s = EuclideanScore(
        class_wise=True,
        class_aggregation="min",
        threshold_mode="per_class",
        min_samples_per_class=1,
    )
    s.fit(X=X, labels=y)
    assert s.threshold_by_class is None
    _ = s.select(X=X)
    assert s.threshold_by_class is not None


def test_select_target_missing_target_raises():
    s = _target_scorer(cal=True)
    with pytest.raises(ValueError, match="target labels required"):
        s.select(X=torch.randn(2, 2))


def test_select_target_loader_with_target_raises():
    s = _target_scorer(cal=True)
    with pytest.raises(ValueError, match="target must be None"):
        s.select(
            model=DummyModel(2),
            loader=_dict_loader(1, 2, C=2),
            target=torch.tensor([0]),
        )


def test_select_target_both_X_and_loader_raises():
    s = _target_scorer(cal=True)
    with pytest.raises(ValueError, match="Cannot specify both"):
        s.select(
            X=torch.randn(1, 2),
            model=DummyModel(2),
            loader=_dict_loader(1, 2, C=2),
            target=torch.tensor([0]),
        )


def test_select_target_via_loader_extracts_labels():
    s = _target_scorer(cal=True)
    out = s.select(model=DummyModel(2), loader=_dict_loader(4, 2, C=2))
    assert out["selected"].shape == (4,)


@pytest.mark.parametrize("agg", ["min", "mean"])
def test_select_per_class_all_inf_thresholds_raise(agg):
    X, y = _small_dataset()
    s = EuclideanScore(
        class_wise=True,
        class_aggregation=agg,
        threshold_mode="per_class",
        min_samples_per_class=1,
    )
    s.fit(X=X, labels=y)
    s.set_threshold()
    assert s.threshold_by_class is not None
    s.threshold_by_class = {
        c: torch.tensor(float("inf")) for c in s.threshold_by_class
    }
    with pytest.raises(RuntimeError, match="No finite per-class thresholds"):
        s.select(X=X)


# =====================================================================
# Cosine / Mahalanobis class-wise
# =====================================================================


def test_cosine_classwise_score_shape():
    refs = torch.randn(6, 3)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    s = CosineScore(class_wise=True, min_samples_per_class=1)
    s.fit(X=refs, labels=labels)
    out = s.score(X=torch.randn(2, 3))
    assert out.shape == (2,)


def test_mahalanobis_fallback_for_small_class_uses_shared_cov():
    refs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    s = MahalanobisScore(k=1, class_wise=True, min_samples_per_class=5)
    with pytest.warns(UserWarning, match="may use shared covariance fallback"):
        s.fit(X=refs, labels=labels)
    assert 0 in s.fallback_classes and 1 in s.fallback_classes
    # both classes fall back and share the same inverse Cholesky
    assert s.vi_zero_by_class[1] is s.vi_zero_shared
    assert s.vi_zero_by_class[0] is s.vi_zero_shared


def test_mahalanobis_normal_class_uses_own_covariance():
    torch.manual_seed(0)
    refs = torch.cat([torch.randn(10, 3), torch.randn(10, 3) + 5.0])
    labels = torch.tensor([0] * 10 + [1] * 10)
    s = MahalanobisScore(k=1, class_wise=True, min_samples_per_class=5)
    s.fit(X=refs, labels=labels)
    assert not s.fallback_classes
    assert s.vi_zero_by_class[0] is not s.vi_zero_shared
    assert s.vi_zero_by_class[1] is not s.vi_zero_shared


# =====================================================================
# _embed_and_label_dl
# =====================================================================


def test_embed_and_label_dl_dict_and_tuple_equivalent():
    d = 2
    xs = torch.arange(8, dtype=torch.float32).view(4, d)
    dict_data = [
        {"image": x.unsqueeze(0), "label": torch.tensor([i])}
        for i, x in enumerate(xs)
    ]
    dict_loader = DataLoader(
        dict_data,  # ty: ignore [invalid-argument-type]
        batch_size=2,
        collate_fn=lambda b: {k: torch.cat([d_[k] for d_ in b]) for k in b[0]},
    )
    tuple_data = [(x.unsqueeze(0), torch.tensor([i])) for i, x in enumerate(xs)]
    tuple_loader = DataLoader(
        tuple_data,  # ty: ignore [invalid-argument-type]
        batch_size=2,
        collate_fn=lambda b: (
            torch.cat([d_[0] for d_ in b]),
            torch.cat([d_[1] for d_ in b]),
        ),
    )
    model = DummyModel(d)
    e1, l1 = KNNScore._embed_and_label_dl(model, dict_loader, label_key="label")
    e2, l2 = KNNScore._embed_and_label_dl(
        model, tuple_loader, label_key="label"
    )
    assert e1.shape[0] == 4
    assert torch.equal(e1, e2) and torch.equal(l1, l2)


def test_embed_and_label_dl_restores_training_mode():
    d = 2
    model = DummyModel(d)
    model.train()
    _ = KNNScore._embed_and_label_dl(
        model, _dict_loader(4, d), label_key="label"
    )
    assert model.training is True


def test_embed_and_label_dl_missing_dict_key_raises():
    bad = [{"image": torch.randn(1, 2)}]
    loader = DataLoader(
        bad,  # ty: ignore [invalid-argument-type]
        batch_size=1,
        collate_fn=lambda b: {k: v for d_ in b for k, v in d_.items()},
    )
    with pytest.raises(KeyError, match="does not contain label key 'lbl'"):
        KNNScore._embed_and_label_dl(DummyModel(), loader, label_key="lbl")


def test_embed_and_label_dl_short_tuple_raises():
    loader = DataLoader(
        [(torch.randn(1, 2),)],  # ty: ignore [invalid-argument-type]
        batch_size=1,
        collate_fn=lambda b: (b[0][0],),
    )
    with pytest.raises(ValueError, match=r"Tuple/list batch must contain"):
        KNNScore._embed_and_label_dl(DummyModel(), loader, label_key="label")


def test_embed_and_label_dl_invalid_batch_type_raises():
    # plain-tensor batch: _embed accepts it, but _embed_and_label_dl
    # cannot resolve labels from a non-dict / non-tuple batch.
    loader = DataLoader(
        [torch.randn(2)],  # ty: ignore [invalid-argument-type]
        batch_size=1,
        collate_fn=lambda b: b[0],
    )
    with pytest.raises(TypeError, match="Cannot extract labels"):
        KNNScore._embed_and_label_dl(DummyModel(), loader, label_key="label")


# =====================================================================
# _embed_and_label_from_dict
# =====================================================================


def test_from_dict_missing_key_raises():
    with pytest.raises(KeyError, match="Missing key"):
        KNNScore._embed_and_label_from_dict(
            model=DummyModel(2), loaders={}, key="train"
        )


def test_from_dict_outdir_without_prefix_warns_and_no_cache(tmp_path):
    d = 2
    with pytest.warns(
        UserWarning, match="'outdir' specified but 'prefix' is None"
    ):
        KNNScore._embed_and_label_from_dict(
            model=DummyModel(d),
            loaders={"train": _dict_loader(4, d)},
            key="train",
            outdir=tmp_path,
            prefix=None,
        )
    assert list(tmp_path.iterdir()) == []


def test_from_dict_saves_then_loads_from_disk(tmp_path):
    d = 2
    loader = _dict_loader(4, d)
    model = DummyModel(d)
    e1, l1 = KNNScore._embed_and_label_from_dict(
        model=model,
        loaders={"train": loader},
        key="train",
        outdir=tmp_path,
        prefix="p",
    )
    calls_after_first = model.embed_calls
    assert calls_after_first == len(loader)
    assert (tmp_path / "p-embeddings-train.pt").is_file()
    assert (tmp_path / "p-labels-train.pt").is_file()

    with pytest.warns(UserWarning, match="Loading pre-existing embeddings"):
        e2, l2 = KNNScore._embed_and_label_from_dict(
            model=model,
            loaders={"train": loader},
            key="train",
            outdir=tmp_path,
            prefix="p",
        )
    assert model.embed_calls == calls_after_first  # no recompute
    assert torch.allclose(e1, e2) and torch.equal(l1, l2)


# =====================================================================
# set_threshold overrides
# =====================================================================


def test_set_threshold_global_uses_flat_scores():
    X, y = _small_dataset()
    s = EuclideanScore(
        class_wise=True,
        class_aggregation="min",
        threshold_mode="global",
        min_samples_per_class=1,
    )
    s.fit(X=X, labels=y)
    s.set_threshold(q=0.5)
    assert isinstance(s.threshold, torch.Tensor)


def test_set_threshold_per_class_stores_dict():
    X, y = _small_dataset()
    s = EuclideanScore(
        class_wise=True,
        class_aggregation="min",
        threshold_mode="per_class",
        min_samples_per_class=1,
    )
    s.fit(X=X, labels=y)
    s.set_threshold(q=0.5)
    assert s.threshold_by_class is not None
    assert set(s.threshold_by_class.keys()) == {0, 1, 2}
