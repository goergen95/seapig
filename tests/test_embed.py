import builtins
import pathlib
import re
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from seapig.scores.utils import TensorPCA
from tests.fixtures import DummyModel, MinimalEmbedding

_EmbedLoader = DataLoader[torch.Tensor | dict[str, torch.Tensor]]


def test_pca_correctly_initialized() -> None:
    e = MinimalEmbedding(pca=None)
    assert e.pca is None

    e = MinimalEmbedding(TensorPCA(n_components=0.5))
    assert isinstance(e.pca, TensorPCA)


def test_fit_pca_sets_pca_and_device() -> None:
    # existing test ensures PCA can be fit
    e = MinimalEmbedding(pca=TensorPCA(n_components=0.5))
    e.ref_embeddings = torch.randn(10, 5)
    e._fit_pca()
    assert isinstance(e.pca, TensorPCA)


def test_apply_pca_transforms_cal_embeddings() -> None:
    """Ensure _apply_pca transforms both ref and cal embeddings when PCA is set."""
    e = MinimalEmbedding(pca=TensorPCA(n_components=0.5))
    # create distinct embeddings
    ref = torch.randn(8, 6)
    cal = torch.randn(3, 6)
    e.ref_embeddings = ref.clone()
    e.cal_embeddings = cal.clone()
    # Apply PCA which fits on ref and transforms both
    e._apply_pca()
    # After applying, ref should be transformed
    transformed_ref = e.pca.transform(ref)
    assert torch.allclose(e.ref_embeddings, transformed_ref)
    # Cal should also be transformed
    transformed_cal = e.pca.transform(cal)
    assert torch.allclose(e.cal_embeddings, transformed_cal)
    e = MinimalEmbedding(pca=TensorPCA(n_components=0.5))
    e.ref_embeddings = torch.randn(10, 5)
    e._fit_pca()
    assert isinstance(e.pca, TensorPCA)


def test_set_threshold_and_select_behavior() -> None:
    e = MinimalEmbedding()
    # avoid train/cal checks
    e.train_required = False
    e.cal_required = False
    # provide scores used by set_threshold
    e.scores = torch.tensor([0.0, 1.0, 2.0, 3.0])
    # ensure threshold computed at median (0.5)
    e.set_threshold(q=0.5)
    assert isinstance(e.threshold, torch.Tensor)
    # now test select: supply X and ensure selected mask is returned
    X = torch.tensor([[0.1, 0.1], [2.0, 2.0]])
    res = e.select(X)
    assert "score" in res and "selected" in res
    assert res["score"].shape[0] == X.shape[0]
    assert len(res["score"].shape) == 1
    assert res["selected"].dtype == torch.bool
    assert len(res["selected"].shape) == 1


def test_fit_model_without_embed_raises(tmp_path: pathlib.Path) -> None:
    class NoEmbedModel(torch.nn.Module):
        pass

    loaders: dict[str, _EmbedLoader] = {
        "train": cast(
            _EmbedLoader,
            DataLoader([torch.tensor([0.0, 0.1])], batch_size=1),  # type: ignore[arg-type, ty:invalid-argument-type]
        ),
        "val": cast(
            _EmbedLoader,
            DataLoader([torch.tensor([0.0, 0.1])], batch_size=1),  # type: ignore[arg-type, ty:invalid-argument-type]
        ),
    }

    s = MinimalEmbedding()
    with pytest.raises(
        TypeError,
        match=re.escape(r"`model` is required to have a `embed()` method."),
    ):
        s.fit(model=NoEmbedModel(), loaders=loaders)


def test_score_with_model_loader_writes_and_returns_tensor(
    tmp_path: pathlib.Path,
) -> None:
    samples = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dataset = TensorDataset(samples)
    loader = cast(
        _EmbedLoader,
        DataLoader(
            dataset,
            batch_size=1,
            collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
        ),
    )

    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False

    out = s.score(
        model=DummyModel(), loader=loader, outdir=tmp_path, prefix="pfx"
    )
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 2
    assert (tmp_path / "pfx-embedding.pt").exists()
    # cleanup
    (tmp_path / "pfx-embedding.pt").unlink()


def test_select_with_model_loader_respects_threshold(
    tmp_path: pathlib.Path,
) -> None:
    samples = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    dataset = TensorDataset(samples)
    loader = cast(
        _EmbedLoader,
        DataLoader(
            dataset,
            batch_size=1,
            collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
        ),
    )

    s = MinimalEmbedding()
    s.threshold = torch.tensor(5.0)

    out = s.select(model=DummyModel(), loader=loader, outdir=None, prefix=None)
    assert "score" in out and "selected" in out
    assert out["score"].shape[0] == 2
    assert out["selected"].dtype == torch.bool


def test_visualize_embeddings() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    # Mock embeddings
    ref_embeddings = torch.randn(10, 64)
    cal_embeddings = torch.randn(10, 64)
    query_embeddings = torch.randn(10, 64)

    pca = TensorPCA(n_components=0.75)

    score = MinimalEmbedding(pca=pca)
    score.ref_embeddings = ref_embeddings
    score.cal_embeddings = cal_embeddings
    score._fit_pca()

    # Mock method arguments
    tsne_args = {"perplexity": 5, "random_state": 42}

    with pytest.raises(ValueError):
        score.plot_embs(
            query_embeddings=query_embeddings,
            method="invalid_method",  # type: ignore[arg-type, ty:invalid-argument-type]
            method_args=tsne_args,
        )

    # Ensure no exceptions were raised
    assert True

    # Mock the plotting function to avoid rendering during tests
    with patch.object(plt, "show"):
        # Test with umap
        score.plot_embs(
            query_embeddings=query_embeddings,
            method="tsne",
            method_args=tsne_args,
        )

    # Ensure no exceptions were raised
    assert True


def test_score_with_embeddings_only() -> None:
    """Test that score() works with precomputed embeddings (X parameter)."""
    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False

    # Create some sample embeddings
    embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Call score with embeddings
    scores = s.score(X=embeddings)

    assert isinstance(scores, torch.Tensor)
    assert scores.shape[0] == 3


def test_score_with_model_loader_only() -> None:
    """Test that score() works with model+loader parameters."""
    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False

    # Create a simple dataloader
    samples = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dataset = TensorDataset(samples)
    loader = cast(
        _EmbedLoader,
        DataLoader(
            dataset,
            batch_size=1,
            collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
        ),
    )

    # Call score with model and loader
    scores = s.score(model=DummyModel(), loader=loader)

    assert isinstance(scores, torch.Tensor)
    assert scores.shape[0] == 2


def test_score_rejects_mixed_parameters() -> None:
    """Test that score() rejects mixing embeddings with model+loader."""
    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False

    embeddings = torch.tensor([[1.0, 2.0]])
    samples = torch.tensor([[1.0, 2.0]])
    dataset = TensorDataset(samples)
    loader = cast(_EmbedLoader, DataLoader(dataset, batch_size=1))

    # Should raise ValueError when both X and model are provided
    with pytest.raises(ValueError, match=match):
        s.score(X=embeddings, model=DummyModel(), loader=loader)


def test_score_requires_parameters() -> None:
    """Test that score() requires either X or model+loader."""
    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False

    # Should raise ValueError when no parameters provided
    with pytest.raises(ValueError, match=match):
        s.score()


def test_score_requires_loader_when_model_provided() -> None:
    """Test that score() requires loader when model is provided."""
    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False

    # Should raise ValueError when model provided without loader
    with pytest.raises(ValueError, match=match):
        s.score(model=DummyModel())


def test_select_with_embeddings_only() -> None:
    """Test that select() works with precomputed embeddings (X parameter)."""
    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False
    s.threshold = torch.tensor(5.0)

    # Create some sample embeddings
    embeddings = torch.tensor([[1.0, 2.0], [10.0, 10.0]])

    # Call select with embeddings
    result = s.select(X=embeddings)

    assert "score" in result
    assert "selected" in result
    assert result["score"].shape[0] == 2
    assert result["selected"].dtype == torch.bool


def test_select_with_model_loader_only() -> None:
    """Test that select() works with model+loader parameters."""

    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False
    s.threshold = torch.tensor(5.0)

    # Create a simple dataloader
    samples = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    dataset = TensorDataset(samples)
    loader = cast(
        _EmbedLoader,
        DataLoader(
            dataset,
            batch_size=1,
            collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
        ),
    )

    # Call select with model and loader
    result = s.select(model=DummyModel(), loader=loader)

    assert "score" in result
    assert "selected" in result
    assert result["score"].shape[0] == 2
    assert result["selected"].dtype == torch.bool


def test_select_rejects_mixed_parameters() -> None:
    """Test that select() rejects mixing embeddings with model+loader."""
    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False
    s.threshold = torch.tensor(5.0)

    embeddings = torch.tensor([[1.0, 2.0]])
    samples = torch.tensor([[1.0, 2.0]])
    dataset = TensorDataset(samples)
    loader = cast(_EmbedLoader, DataLoader(dataset, batch_size=1))

    # Should raise ValueError when both X and model are provided
    with pytest.raises(ValueError, match=match):
        s.select(X=embeddings, model=DummyModel(), loader=loader)


def test_select_requires_parameters() -> None:
    """Test that select() requires either X or model+loader."""
    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False
    s.threshold = torch.tensor(5.0)

    # Should raise ValueError when no parameters provided
    with pytest.raises(ValueError, match=match):
        s.select()


match = re.escape(
    "Specify either pre-computed tensors (X and Y) or a model with a loader, but not both."
)


def test_fit_parameter_validation_errors() -> None:
    s = MinimalEmbedding()
    X = torch.randn(2, 4)

    with pytest.raises(ValueError, match=match):
        s.fit(X=X, model=DummyModel(), loaders={"a": 1})  # type: ignore

    # neither provided should raise
    with pytest.raises(ValueError, match=match):
        s.fit()

    # loaders provided but model missing should raise
    dataset = TensorDataset(torch.tensor([[0.0, 0.1]]))
    loaders: dict[str, _EmbedLoader] = {
        "train": cast(_EmbedLoader, DataLoader(dataset, batch_size=1))
    }
    with pytest.raises(AssertionError):
        s.fit(loaders=loaders)


@pytest.fixture(
    params=[
        ("matplotlib", None, "matplotlib is not installed"),
        ("sklearn", "tsne", "t-SNE is not installed"),
        ("umap", "umap", "UMAP is not installed"),
    ],
    ids=["matplotlib", "tsne", "umap"],
)
def missing_library(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> tuple[str | None, str]:
    """Patch imports so the named top-level library raises ImportError."""
    block_prefix, method, expected_msg = request.param
    orig_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.startswith(block_prefix):
            raise ImportError(f"No {block_prefix}")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return method, expected_msg


def test_plot_embs_missing_libraries_raise(
    missing_library: tuple[str | None, str],
) -> None:
    """Unified test: missing library import should raise the expected ImportError."""
    method, expected_msg = missing_library
    e = MinimalEmbedding()
    e.ref_embeddings = torch.randn(3, 4)

    if method is None:
        with pytest.raises(ImportError, match=expected_msg):
            e.plot_embs(query_embeddings=torch.randn(2, 4))
    else:
        with pytest.raises(ImportError, match=expected_msg):
            e.plot_embs(query_embeddings=torch.randn(2, 4), method=method)  # type: ignore[arg-type, ty:invalid-argument-type]


def test_fit_errors_when_both_or_neither_provided() -> None:
    s = MinimalEmbedding()
    emb = torch.randn(3, 4)

    # neither embeddings nor model/loaders
    with pytest.raises(ValueError, match=match):
        s.fit()

    with pytest.raises(ValueError, match=match):
        s.fit(
            X=emb, model=DummyModel(), loaders=cast(dict[str, _EmbedLoader], {})
        )


def test_select_triggers_set_threshold_when_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # existing test
    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False
    s.scores = torch.tensor([0.0, 1.0, 2.0])
    s.threshold = None
    caplog.clear()
    caplog.set_level("WARNING")
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    res = s.select(X=X)
    assert any(
        "Threshold has not been set" in rec.message for rec in caplog.records
    )
    assert s.threshold is not None
    assert "score" in res and "selected" in res


def test_select_and_set_threshold_with_calibrated() -> None:
    """Test select and set_threshold when calibration is required and provided."""
    e = MinimalEmbedding(pca=None)
    e.train_required = False
    e.cal_required = True
    # set embeddings
    e.ref_embeddings = torch.randn(5, 4)
    e.cal_embeddings = torch.randn(3, 4)
    # compute scores for calibration embeddings via dummy _score (sum)
    e.scores = e._score(e.cal_embeddings)
    e.set_calibrated()
    # set threshold based on calibration scores
    e.set_threshold(q=0.5)
    # now select with query embeddings
    X = torch.randn(2, 4)
    result = e.select(X=X)
    assert "score" in result and "selected" in result
    assert result["score"].shape[0] == X.shape[0]
    assert result["selected"].dtype == torch.bool
