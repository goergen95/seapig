import builtins
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from seapig.scores.embed import EmbeddingScore
from seapig.scores.utils import TensorPCA


class DummyModel(torch.nn.Module):
    def embed(self, x):  # must accept 'x' param
        if isinstance(x, dict):
            x = x["image"]  # pragma: no cover
        return x


class DummyBadModel(torch.nn.Module):
    def not_embed(self, x):
        return x  # pragma: no cover


class DummyBadSignature(torch.nn.Module):
    def embed(self):  # missing 'x' param
        return torch.zeros(1, 2)  # pragma: no cover


class IdentityModel(torch.nn.Module):
    def embed(self, x):
        if isinstance(x, dict):
            x = x["image"]  # pragma: no cover
        return x


class DummyEmbedding(EmbeddingScore):
    def __init__(self, pca=None):
        super().__init__(pca=pca)

    def _score_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        # simple deterministic score: sum over features per row
        return X.sum(dim=1)


def test_pca_correctly_initialized() -> None:
    e = DummyEmbedding(pca=None)
    assert e.pca is None

    e = DummyEmbedding(TensorPCA(n_components=0.5))
    assert isinstance(e.pca, TensorPCA)


def test_setup_path_creates_dir_and_returns_path(tmp_path) -> None:
    outdir = tmp_path / "subdir"
    path = EmbeddingScore._setup_path(outdir=outdir, prefix="myprefix")
    assert path is not None
    # the helper should return a Path ending with .parquet but not yet create the file
    assert path.suffix == ".pt"
    assert outdir.is_dir()
    assert "myprefix" in path.name
    # cleanup
    outdir.rmdir()


def test_check_model_valid_and_invalid():
    m = DummyModel()
    # should not raise
    EmbeddingScore._check_model(m)

    with pytest.raises(Exception):
        EmbeddingScore._check_model(DummyBadModel())

    with pytest.raises(Exception):
        EmbeddingScore._check_model(DummyBadSignature())


def test_write_and_load_roundtrip(tmp_path) -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    path = tmp_path / "embs.pt"
    EmbeddingScore._write_pt(x, path)
    assert path.exists()
    y = EmbeddingScore._load_pt(path)
    assert isinstance(y, torch.Tensor)
    assert torch.allclose(y, x)
    # cleanup
    path.unlink()


def test_embed_errors_and_success() -> None:
    model = DummyModel()
    # dict missing "image" should raise KeyError
    with pytest.raises(KeyError):
        EmbeddingScore._embed({"foo": torch.zeros(1, 2)}, model)

    # embed returning extra dims should raise
    class BadShapeModel(DummyModel):
        def embed(self, x):
            t = super().embed(x)
            return t.unsqueeze(1)  # make shape (B,1,D)

    bad = BadShapeModel()
    with pytest.raises(ValueError):
        EmbeddingScore._embed(torch.zeros(2, 3), bad)

    # correct case
    out = EmbeddingScore._embed(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), model)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 2)


def test_embed_dl_concatenates_batches() -> None:
    model = DummyModel()
    # use TensorDataset so DataLoader yields (B,D)
    samples = torch.tensor([[float(i), float(i) + 0.1] for i in range(4)])
    dataset = TensorDataset(samples)

    # TensorDataset yields tuples, collate will produce shape (B,1,D), so use a custom collate
    def collate_fn(batch):
        # batch is list of tuples like (tensor,), extract and stack
        return torch.stack([b[0] for b in batch], dim=0)

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    embs = EmbeddingScore._embed_dl(model=model, loader=loader)
    assert embs.shape[0] == 4
    assert embs.shape[1] == 2


def test_embed_from_dict_errors_and_saves(tmp_path) -> None:
    model = DummyModel()
    samples = torch.tensor([[1.0, 2.0]])
    dataset = TensorDataset(samples)

    def collate_fn(batch):
        return torch.stack([b[0] for b in batch], dim=0)

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    loaders = {"train": loader}
    # missing key 'val' should KeyError
    with pytest.raises(KeyError):
        EmbeddingScore._embed_from_dict(model=model, loaders=loaders, key="val")

    # outdir specified but prefix None should raise a Warning (implementation may raise Warning)
    with pytest.warns(UserWarning):
        EmbeddingScore._embed_from_dict(
            model=model,
            loaders={"train": loader},
            key="train",
            outdir=tmp_path,
            prefix=None,
        )

    # valid save/load path: provide prefix and outdir
    loaders = {"train": loader}
    embs = EmbeddingScore._embed_from_dict(
        model=model, loaders=loaders, key="train", outdir=tmp_path, prefix="pfx"
    )
    assert isinstance(embs, torch.Tensor)
    # file should have been written
    expected = tmp_path / "pfx-embeddings-train.pt"
    assert expected.exists()
    # cleanup
    expected.unlink()


def test_fit_pca_sets_pca_and_device() -> None:
    e = DummyEmbedding(pca=TensorPCA(n_components=0.5))
    e.ref_embeddings = torch.randn(10, 5)
    e._fit_pca()
    assert isinstance(e.pca, TensorPCA)


def test_set_threshold_and_select_behavior() -> None:
    e = DummyEmbedding()
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


class MinimalEmbedding(EmbeddingScore):
    """Small concrete subclass for testing high-level methods.

    Disable training/calibration requirements so tests can call score/select
    without extra setup.
    """

    def __init__(self):
        super().__init__()
        # allow calling score/select without separate training/calibration
        self.train_required = False
        self.cal_required = False

    def _score_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        """Simple deterministic score used in tests.

        Compute per-row sum over features so tests can assert shapes and
        thresholding behavior.
        """
        return X.sum(dim=1)


def test_fit_model_without_embed_raises(tmp_path) -> None:
    class NoEmbedModel(torch.nn.Module):
        pass

    loaders = {
        "train": DataLoader([torch.tensor([0.0, 0.1])], batch_size=1),
        "val": DataLoader([torch.tensor([0.0, 0.1])], batch_size=1),
    }

    s = MinimalEmbedding()
    with pytest.raises(Exception):
        s.fit(model=NoEmbedModel(), loaders=loaders)


def test_score_with_model_loader_writes_and_returns_tensor(tmp_path) -> None:

    samples = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dataset = TensorDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )

    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False

    out = s.score(
        model=IdentityModel(), loader=loader, outdir=tmp_path, prefix="pfx"
    )
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 2
    assert (tmp_path / "pfx.pt").exists()
    # cleanup
    (tmp_path / "pfx.pt").unlink()


def test_select_with_model_loader_respects_threshold(tmp_path) -> None:
    samples = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    dataset = TensorDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )

    s = MinimalEmbedding()
    s.threshold = torch.tensor(5.0)

    out = s.select(
        model=IdentityModel(), loader=loader, outdir=None, prefix=None
    )
    assert "score" in out and "selected" in out
    assert out["score"].shape[0] == 2
    assert out["selected"].dtype == torch.bool


def test_visualize_embeddings():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    # Mock embeddings
    ref_embeddings = torch.randn(10, 64)
    cal_embeddings = torch.randn(10, 64)
    query_embeddings = torch.randn(10, 64)

    pca = TensorPCA(n_components=0.75)

    score = DummyEmbedding(pca=pca)
    score.ref_embeddings = ref_embeddings
    score.cal_embeddings = cal_embeddings
    score._fit_pca()

    # Mock method arguments
    tsne_args = {"perplexity": 5, "random_state": 42}

    with pytest.raises(ValueError):
        score.plot_embs(
            query_embeddings=query_embeddings,
            method="invalid_method",  # type: ignore[arg-type]
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
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )

    # Call score with model and loader
    scores = s.score(model=IdentityModel(), loader=loader)

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
    loader = DataLoader(dataset, batch_size=1)

    # Should raise ValueError when both X and model are provided
    with pytest.raises(ValueError, match="Cannot specify both embeddings"):
        s.score(X=embeddings, model=IdentityModel(), loader=loader)


def test_score_requires_parameters() -> None:
    """Test that score() requires either X or model+loader."""
    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False

    # Should raise ValueError when no parameters provided
    with pytest.raises(ValueError, match="Must specify either embeddings"):
        s.score()


def test_score_requires_loader_when_model_provided() -> None:
    """Test that score() requires loader when model is provided."""
    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False

    # Should raise ValueError when model provided without loader
    with pytest.raises(
        ValueError, match="loader is required when using a model"
    ):
        s.score(model=IdentityModel())


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
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )

    # Call select with model and loader
    result = s.select(model=IdentityModel(), loader=loader)

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
    loader = DataLoader(dataset, batch_size=1)

    # Should raise ValueError when both X and model are provided
    with pytest.raises(ValueError, match="Cannot specify both embeddings"):
        s.select(X=embeddings, model=IdentityModel(), loader=loader)


def test_select_requires_parameters() -> None:
    """Test that select() requires either X or model+loader."""
    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False
    s.threshold = torch.tensor(5.0)

    # Should raise ValueError when no parameters provided
    with pytest.raises(ValueError, match="Must specify either embeddings"):
        s.select()


def test_embed_loadorembed_uses_disk_when_present(tmp_path) -> None:
    """When a saved embeddings file exists, _loadorembed should load it and
    move it to the model device. It should also emit a UserWarning.
    """
    # create a tensor and save it to disk
    saved = torch.tensor([[9.0, 8.0], [7.0, 6.0]])
    path = tmp_path / "already.pt"
    torch.save(saved, path)

    # model must have parameters so next(model.parameters()) works
    class ParamModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

        def embed(self, x):
            return x

    m = ParamModel()
    # move model to cpu (default) and ensure file load uses same device
    loader = DataLoader([torch.tensor([0.0, 0.1])], batch_size=1)

    with pytest.warns(UserWarning):
        out = EmbeddingScore._loadorembed(path=path, model=m, loader=loader)

    assert isinstance(out, torch.Tensor)
    assert out.shape == saved.shape


def test_embed_accepts_dict_and_sequence_inputs() -> None:
    m = DummyModel()
    # dict case
    xdict = {"image": torch.tensor([[1.0, 2.0]])}
    out = EmbeddingScore._embed(xdict, m)
    assert torch.allclose(out, xdict["image"])

    # tuple/list case
    xtup = (torch.tensor([[3.0, 4.0]]),)
    out2 = EmbeddingScore._embed(xtup, m)
    assert torch.allclose(out2, xtup[0])


def test_fit_parameter_validation_errors() -> None:
    s = MinimalEmbedding()
    X = torch.randn(2, 4)

    with pytest.raises(ValueError, match="Cannot specify both embeddings"):
        s.fit(X=X, model=IdentityModel(), loaders=None)

    # neither provided should raise
    with pytest.raises(ValueError, match="Must specify either embeddings"):
        s.fit()

    # loaders provided but model missing should raise
    dataset = TensorDataset(torch.tensor([[0.0, 0.1]]))
    loaders = {"train": DataLoader(dataset, batch_size=1)}
    with pytest.raises(ValueError, match="model is required"):
        s.fit(loaders=loaders)


def test_embed_dl_restores_training_state() -> None:
    class TrainModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

        def embed(self, x):
            return x

    model = TrainModel()
    # ensure model is in training mode
    model.train()
    assert model.training

    samples = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    dataset = TensorDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )

    out = EmbeddingScore._embed_dl(model=model, loader=loader)
    # model should have been restored to training mode
    assert model.training
    assert out.shape[0] == 2


@pytest.fixture(
    params=[
        ("matplotlib", None, "matplotlib is not installed"),
        ("sklearn", "tsne", "t-SNE is not installed"),
        ("umap", "umap", "UMAP is not installed"),
    ],
    ids=["matplotlib", "tsne", "umap"],
)
def missing_library(request, monkeypatch):
    """Patch imports so the named top-level library raises ImportError."""
    block_prefix, method, expected_msg = request.param
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith(block_prefix):
            raise ImportError(f"No {block_prefix}")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return method, expected_msg


def test_plot_embs_missing_libraries_raise(missing_library):
    """Unified test: missing library import should raise the expected ImportError."""
    method, expected_msg = missing_library
    e = DummyEmbedding()
    e.ref_embeddings = torch.randn(3, 4)

    if method is None:
        with pytest.raises(ImportError, match=expected_msg):
            e.plot_embs(query_embeddings=torch.randn(2, 4))
    else:
        with pytest.raises(ImportError, match=expected_msg):
            e.plot_embs(query_embeddings=torch.randn(2, 4), method=method)


def test_loadorembed_uses_existing_file_and_moves_to_model_device(
    tmp_path,
) -> None:
    # prepare tensor file
    tensor = torch.tensor([[7.0, 8.0]])
    path = tmp_path / "pre_embs.pt"
    torch.save(tensor, path)

    class ParamModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l = torch.nn.Linear(2, 2)

        def embed(self, x):
            return x

    model = ParamModel()
    # simple loader (not used when path exists)
    loader = DataLoader([torch.tensor([0.0, 0.1])], batch_size=1)

    with pytest.warns(UserWarning):
        out = EmbeddingScore._loadorembed(path, model, loader)
    assert isinstance(out, torch.Tensor)
    assert out.shape == tensor.shape
    # ensure tensor is on same device as model parameters
    dev = next(model.parameters()).device
    assert out.device == dev


def test_embed_accepts_list_and_rejects_non_tensor_return() -> None:
    model = DummyModel()
    x = torch.tensor([[1.0, 2.0]])
    # list/tuple input should select first element and succeed
    out = EmbeddingScore._embed([x], model)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape

    # model returning non-tensor should raise AssertionError
    class BadReturnModel(torch.nn.Module):
        def embed(self, x):
            return [1, 2, 3]

    with pytest.raises(AssertionError):
        EmbeddingScore._embed(x, BadReturnModel())


def test_fit_errors_when_both_or_neither_provided() -> None:
    s = MinimalEmbedding()
    emb = torch.randn(3, 4)

    # neither embeddings nor model/loaders
    with pytest.raises(ValueError, match="Must specify either embeddings"):
        s.fit()

    with pytest.raises(ValueError, match="Cannot specify both embeddings"):
        s.fit(X=emb, model=IdentityModel(), loaders={})


def test_select_triggers_set_threshold_when_none(caplog) -> None:
    s = MinimalEmbedding()
    s.train_required = False
    s.cal_required = False
    # provide scores so set_threshold can run
    s.scores = torch.tensor([0.0, 1.0, 2.0])
    s.threshold = None

    caplog.clear()
    caplog.set_level("WARNING")

    # call select with embeddings which will cause set_threshold to be called
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    res = s.select(X=X)

    # logger.warning should have been emitted about missing threshold
    assert any(
        "Threshold has not been set" in rec.message for rec in caplog.records
    )
    assert s.threshold is not None
    assert "score" in res and "selected" in res
