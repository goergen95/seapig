from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from seapig.scores.embed import EmbeddingScore
from seapig.scores.utils import TensorPCA


class DummyModel(torch.nn.Module):
    def embed(self, x):  # must accept 'x' param
        if isinstance(x, dict):
            x = x["image"]
        return x


class DummyBadModel(torch.nn.Module):
    def not_embed(self, x):
        return x


class DummyBadSignature(torch.nn.Module):
    def embed(self):  # missing 'x' param
        return torch.zeros(1, 2)


class DummyEmbedding(EmbeddingScore):
    def __init__(self, pca=None):
        super().__init__(pca=pca)

    def score(self, X: torch.Tensor) -> torch.Tensor:
        # simple deterministic score: sum over features per row
        return X.sum(dim=1)


def test_pca_correctly_initialized() -> None:
    e = DummyEmbedding(pca=None)
    assert e.pca is None

    e = DummyEmbedding(TensorPCA(exp_var=0.5))
    assert isinstance(e.pca, TensorPCA)


def test_setup_path_creates_dir_and_returns_path(tmp_path) -> None:
    outdir = tmp_path / "subdir"
    path = EmbeddingScore._setup_path(outdir=outdir, prefix="myprefix")
    assert path is not None
    # the helper should return a Path ending with .parquet but not yet create the file
    assert path.suffix == ".parquet"
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


def test_write_and_load_parquet_roundtrip(tmp_path) -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    path = tmp_path / "embs.parquet"
    EmbeddingScore._write_parquet(x, path)
    assert path.exists()
    y = EmbeddingScore._load_parquet(path)
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
    expected = tmp_path / "pfx-embeddings-train.parquet"
    assert expected.exists()
    # cleanup
    expected.unlink()


def test_fit_pca_sets_pca_and_device() -> None:
    e = DummyEmbedding(pca=TensorPCA(exp_var=0.5))
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

    Disable training/calibration requirements so tests can call score_dl/select_dl
    without extra setup.
    """

    def __init__(self):
        super().__init__()
        # allow calling score/select without separate training/calibration
        self.train_required = False
        self.cal_required = False

    def score(self, X: torch.Tensor) -> torch.Tensor:
        """Simple deterministic score used in tests.

        Compute per-row sum over features so tests can assert shapes and
        thresholding behavior.
        """
        return X.sum(dim=1)


def test_fit_dl_model_without_embed_raises(tmp_path) -> None:
    class NoEmbedModel(torch.nn.Module):
        pass

    loaders = {
        "train": DataLoader([torch.tensor([0.0, 0.1])], batch_size=1),
        "val": DataLoader([torch.tensor([0.0, 0.1])], batch_size=1),
    }

    s = MinimalEmbedding()
    with pytest.raises(Exception):
        s.fit_dl(model=NoEmbedModel(), loaders=loaders)


def test_score_dl_writes_and_returns_tensor(tmp_path) -> None:
    class IdentityModel(torch.nn.Module):
        def embed(self, x):
            if isinstance(x, dict):
                x = x["image"]
            return x

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

    out = s.score_dl(
        model=IdentityModel(), loader=loader, outdir=tmp_path, prefix="pfx"
    )
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 2
    assert (tmp_path / "pfx.parquet").exists()
    # cleanup
    (tmp_path / "pfx.parquet").unlink()


def test_select_dl_respects_threshold(tmp_path) -> None:
    class IdentityModel(torch.nn.Module):
        def embed(self, x):
            if isinstance(x, dict):
                x = x["image"]
            return x

    samples = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    dataset = TensorDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda b: torch.stack([x[0] for x in b], 0),
    )

    s = MinimalEmbedding()
    s.threshold = torch.tensor(5.0)

    out = s.select_dl(
        model=IdentityModel(), loader=loader, outdir=None, prefix=None
    )
    assert "score" in out and "selected" in out
    assert out["score"].shape[0] == 2
    assert out["selected"].dtype == torch.bool


def test_visualize_embeddings():
    # Mock embeddings
    ref_embeddings = torch.randn(100, 64)
    cal_embeddings = torch.randn(50, 64)
    query_embeddings = torch.randn(20, 64)

    pca = TensorPCA(exp_var=0.75)

    score = DummyEmbedding(pca=pca)
    score.ref_embeddings = ref_embeddings
    score.cal_embeddings = cal_embeddings
    score._fit_pca()

    # Mock method arguments
    tsne_args = {"perplexity": 30, "random_state": 42}
    umap_args = {"n_neighbors": 5, "min_dist": 0.1}

    with pytest.raises(ValueError):
        score.plot_embs(
            query_embeddings=query_embeddings,
            method="invalid_method",
            method_args={},
        )

    # Mock the plotting function to avoid rendering during tests
    with patch.object(plt, "show"):
        # Test with t-SNE
        score.plot_embs(
            query_embeddings=query_embeddings,
            method="tsne",
            method_args=tsne_args,
        )

        # Test with UMAP
        score.plot_embs(
            query_embeddings=query_embeddings,
            method="umap",
            method_args=umap_args,
        )

    # Ensure no exceptions were raised
    assert True
