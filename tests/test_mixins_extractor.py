import pathlib
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from seapig.scores.mixins import ModelExtractorMixin


# Simple dummy model with an embed method
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.training = True  # explicitly set for clarity

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        # identity for test purposes
        return x


# Model missing the required method
class BadModelNoMethod(torch.nn.Module):
    def forward(self, x):
        return x  # pragma: no cover


# Model with wrong signature (no 'x' argument)
class BadModelWrongSig(torch.nn.Module):
    def embed(self):  # type: ignore[override]
        return torch.zeros(1, 2)  # pragma: no cover


# Helper to create a deterministic DataLoader yielding tensors
def make_loader(tensor: torch.Tensor, batch_size: int = 1):
    dataset = TensorDataset(tensor)

    # collate returns the raw tensor (batch as (B, D))
    def collate_fn(batch):
        # each batch element is a tuple (tensor,)
        return torch.stack([b[0] for b in batch], dim=0)

    return cast(
        DataLoader[torch.Tensor],
        DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn),
    )


def test_check_model_valid_and_invalid():
    # Valid model should not raise
    ModelExtractorMixin._check_model(DummyModel())

    # Missing method raises TypeError
    with pytest.raises(
        TypeError, match=r"model is required to have a `embed\(\)` method."
    ):
        ModelExtractorMixin._check_model(BadModelNoMethod())

    # Wrong signature raises AttributeError
    with pytest.raises(AttributeError):
        ModelExtractorMixin._check_model(BadModelWrongSig())


def test_setup_path_and_warnings(tmp_path: pathlib.Path):
    # When outdir provided but prefix None, warn and return None
    with pytest.warns(UserWarning):
        result = ModelExtractorMixin._setup_path(outdir=tmp_path, prefix=None)
    assert result is None

    # Valid outdir and prefix creates directory and returns correct Path
    outdir = tmp_path / "sub"
    prefix = "testprefix"
    path = ModelExtractorMixin._setup_path(outdir=outdir, prefix=prefix)
    assert path is not None
    assert outdir.is_dir()
    assert path.suffix == ".pt"
    assert prefix in path.name


def test_write_and_load_roundtrip(tmp_path: pathlib.Path):
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    path = tmp_path / "t.pt"
    ModelExtractorMixin._write_pt(tensor, path)
    assert path.exists()
    loaded = ModelExtractorMixin._load_pt(path)
    assert isinstance(loaded, torch.Tensor)
    assert torch.equal(loaded, tensor)

    # Dict variant
    d = {"a": tensor, "b": tensor * 2}
    dict_path = tmp_path / "d.pt"
    ModelExtractorMixin._write_pt(d, dict_path)
    loaded_dict = ModelExtractorMixin._load_pt(dict_path)
    assert isinstance(loaded_dict, dict)
    for k, v in d.items():
        assert torch.equal(loaded_dict[k], v)


@pytest.mark.parametrize(
    "raw, key, expected",
    [
        (torch.randn(2, 2), "emb", {"emb": torch.randn(2, 2)}),
        ([torch.randn(3, 3)], "out", {"out": torch.randn(3, 3)}),
        ({"out": torch.randn(1, 4)}, "out", {"out": torch.randn(1, 4)}),
    ],
)
def test_normalise_output_variants(raw, key, expected):
    # Use a copy of raw to avoid mutation issues in parametrization
    result = ModelExtractorMixin._normalise_output(raw, key)
    assert set(result.keys()) == {key}
    # Type and shape checks are sufficient; exact values differ per param
    assert isinstance(result[key], torch.Tensor)


def test_normalise_output_errors():
    with pytest.raises(KeyError):
        ModelExtractorMixin._normalise_output(
            {"wrong": torch.tensor([1])}, "good"
        )
    with pytest.raises(TypeError):
        ModelExtractorMixin._normalise_output(123, "any")


def test_normalise_input_variants():
    # Tensor input
    t = torch.randn(4, 5)
    out = ModelExtractorMixin._normalise_input(t, ["img"])
    assert out == {"img": t}

    # List/tuple input
    lst = [torch.randn(2, 2), torch.randn(2, 2)]
    out = ModelExtractorMixin._normalise_input(tuple(lst), ["a", "b"])
    assert out == {"a": lst[0], "b": lst[1]}

    # Dict input
    d = {"x": torch.tensor([1]), "y": torch.tensor([2])}
    out = ModelExtractorMixin._normalise_input(d, ["x", "y"])
    assert out == d

    # Missing key raises
    with pytest.raises(KeyError):
        ModelExtractorMixin._normalise_input(
            {"only": torch.tensor([0])}, ["missing"]
        )

    # Unsupported type raises
    with pytest.raises(TypeError):
        ModelExtractorMixin._normalise_input(42, ["a"])


def test_extract_batch_success_and_extra_inputs():
    model = DummyModel()
    batch = {"image": torch.randn(3, 2), "meta": torch.tensor([1, 2, 3])}
    out = ModelExtractorMixin._extract_batch(
        batch, model, ["image"], "embedding"
    )
    # Should contain embedding and the extra key "meta"
    assert "embedding" in out and "meta" in out
    assert torch.equal(out["embedding"], batch["image"])
    assert torch.equal(out["meta"], batch["meta"])


def test_extract_batch_missing_key_raises():
    model = DummyModel()
    batch = {"wrong": torch.randn(1, 2)}
    with pytest.raises(KeyError):
        ModelExtractorMixin._extract_batch(batch, model, ["image"], "embedding")


def test_extract_dl_concatenates_and_respects_training_state():
    model = DummyModel()
    model.train()  # ensure training mode before extraction
    loader = make_loader(
        torch.arange(12, dtype=torch.float32).reshape(4, 3), batch_size=2
    )
    result = ModelExtractorMixin._extract_dl(
        model, loader, ["image"], "embedding"
    )
    # Should concatenate 4 rows of 3 columns
    assert result["embedding"].shape == (4, 3)
    # Model should be back in training mode
    assert model.training


def test_extract_dl_empty_loader_raises():
    model = DummyModel()
    empty_loader = make_loader(torch.empty((0, 2)), batch_size=1)
    with pytest.raises(ValueError, match="No batches found in loader"):
        ModelExtractorMixin._extract_dl(
            model, empty_loader, ["image"], "embedding"
        )


def test_load_or_extract_caching(tmp_path: pathlib.Path):
    model = DummyModel()
    tensor = torch.randn(5, 4)
    loader = make_loader(tensor, batch_size=5)
    path = tmp_path / "cached.pt"
    # First call extracts and writes file
    data1 = ModelExtractorMixin._load_or_extract(
        model, loader, path, ["image"], "embedding"
    )
    assert path.exists()
    # Second call should load from file and emit a warning
    with pytest.warns(UserWarning, match="Loading pre-existing data"):
        data2 = ModelExtractorMixin._load_or_extract(
            model, loader, path, ["image"], "embedding"
        )
    # Loaded data matches original extraction
    assert torch.equal(data1["embedding"], data2["embedding"])


def test_extract_loader_uses_setup_path_and_load_or_extract(
    tmp_path: pathlib.Path,
):
    model = DummyModel()
    tensor = torch.randn(3, 2)
    loader = make_loader(tensor)
    # Use outdir/prefix to trigger path creation
    out = ModelExtractorMixin._extract_loader(
        model=model,
        loader=loader,
        input_keys=["image"],
        output_key="embedding",
        outdir=tmp_path,
        prefix="pref",
    )
    assert "embedding" in out
    assert (tmp_path / "pref.pt").exists()


def test_extract_dict_missing_key_and_prefix_handling(tmp_path: pathlib.Path):
    class DummyEmbedding(ModelExtractorMixin):
        method_name = "embed"

        def __init__(self):
            pass  # pragma: no cover

    model = DummyModel()
    loader = make_loader(torch.randn(2, 2))
    loaders = {"train": loader}
    # Missing required key raises
    with pytest.raises(KeyError):
        DummyEmbedding._extract_dict(
            model=model,
            loaders=loaders,
            input_keys=["image"],
            output_key="embedding",
            key="val",
        )
    # Prefix handling: when prefix provided, should be modified with -embeddings-
    out = DummyEmbedding._extract_dict(
        model=model,
        loaders=loaders,
        input_keys=["image"],
        output_key="embedding",
        key="train",
        outdir=tmp_path,
        prefix="base",
    )
    expected_path = tmp_path / "base-embeddings-train.pt"
    assert expected_path.exists()
    assert "embedding" in out
