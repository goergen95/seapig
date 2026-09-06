import pathlib
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from seapig.scores.extractor import (
    ModelExtractor,
    _concat,
    _model_device,
    _move,
    _normalise_inputs,
    _normalise_output,
    _resolve_cache_path,
    _resolve_method,
    _to_cpu,
)


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
    _resolve_method(DummyModel(), "embed")

    # Missing method raises TypeError
    with pytest.raises(
        TypeError, match=r"`model` is required to have a `embed\(\)` method."
    ):
        _resolve_method(BadModelNoMethod(), "embed")

    # Wrong signature raises AttributeError
    with pytest.raises(AttributeError):
        _resolve_method(BadModelWrongSig(), "embed")


def test_setup_path_and_warnings(tmp_path: pathlib.Path):
    # When outdir provided but prefix None, warn and return None
    with pytest.warns(UserWarning):
        result = _resolve_cache_path(outdir=tmp_path, prefix=None, tag=None)
    assert result is None

    # Valid outdir and prefix creates directory and returns correct Path
    outdir = tmp_path / "sub"
    prefix = "testprefix"
    path = _resolve_cache_path(outdir=outdir, prefix=prefix, tag=None)
    assert path is not None
    assert outdir.is_dir()
    assert path.suffix == ".pt"
    assert prefix in path.name


def test_write_and_load_roundtrip(tmp_path: pathlib.Path):
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    path = tmp_path / "t.pt"
    torch.save(tensor, path)
    assert path.exists()
    loaded = torch.load(path)
    assert isinstance(loaded, torch.Tensor)
    assert torch.equal(loaded, tensor)

    # Dict variant
    d = {"a": tensor, "b": tensor * 2}
    dict_path = tmp_path / "d.pt"
    torch.save(d, dict_path)
    loaded_dict = torch.load(dict_path)
    assert isinstance(loaded_dict, dict)
    for k, v in d.items():
        assert torch.equal(loaded_dict[k], v)


@pytest.mark.parametrize(
    "raw, key",
    [
        (torch.randn(2, 2), "emb"),
        ([torch.randn(3, 3)], "out"),
        ({"out": torch.randn(1, 4)}, "out"),
    ],
)
def test_normalise_output_variants(raw, key):
    # Use a copy of raw to avoid mutation issues in parametrization
    result = _normalise_output(raw, key)
    assert isinstance(result, torch.Tensor)


def test_normalise_output_errors():
    with pytest.raises(KeyError):
        _normalise_output({"wrong": torch.tensor([1])}, "good")
    with pytest.raises(TypeError):
        _normalise_output(123, "any")


def test_normalise_input_variants():
    # Tensor input
    t = torch.randn(4, 5)
    out = _normalise_inputs(t, ["img"])
    assert out == {"img": t}

    # List/tuple input
    lst = [torch.randn(2, 2), torch.randn(2, 2)]
    out = _normalise_inputs(tuple(lst), ["a", "b"])
    assert out == {"a": lst[0], "b": lst[1]}

    # Dict input
    d = {"x": torch.tensor([1]), "y": torch.tensor([2])}
    out = _normalise_inputs(d, ["x", "y"])
    assert out == d

    # Missing key raises
    with pytest.raises(KeyError):
        _normalise_inputs({"only": torch.tensor([0])}, ["missing"])

    # Unsupported type raises
    with pytest.raises(TypeError):
        _normalise_inputs(42, ["a"])  # type: ignore


def test_extract_with_extra_keys(tmp_path: pathlib.Path):
    # Model returns extra key from batch
    class ModelWithMeta(torch.nn.Module):
        def embed(self, x):
            return x

    model = ModelWithMeta()

    # Custom DataLoader yielding dict batches
    class DictDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            for i in range(2):
                yield {"image": torch.randn(2, 3), "meta": torch.tensor([i])}

    loader = torch.utils.data.DataLoader(DictDataset(), batch_size=1)
    extractor = ModelExtractor(
        method_name="embed",
        output_key="embedding",
        input_keys=("image", "meta"),
    )
    result = extractor.extract(model, loader, outdir=tmp_path, prefix="test")
    # Verify both keys present
    assert "embedding" in result and "meta" in result
    # File should be cached
    assert any(
        p.name.startswith("test") and p.suffix == ".pt"
        for p in tmp_path.iterdir()
    )


def test_extract_missing_input_key_raises():
    model = DummyModel()

    class BadDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield {"wrong": torch.randn(1, 2)}

    loader = torch.utils.data.DataLoader(BadDataset(), batch_size=1)
    extractor = ModelExtractor(
        method_name="embed", output_key="embedding", input_keys=("image",)
    )
    with pytest.raises(KeyError):
        extractor.extract(model, loader)


def test_extract_concatenates_and_respects_training_state():
    model = DummyModel()
    model.train()  # ensure training mode before extraction
    loader = make_loader(
        torch.arange(12, dtype=torch.float32).reshape(4, 3), batch_size=2
    )
    extractor = ModelExtractor(
        method_name="embed", output_key="embedding", input_keys=("image",)
    )
    result = extractor.extract(model, loader)
    # Should concatenate 4 rows of 3 columns
    assert result["embedding"].shape == (4, 3)
    # Model should be back in training mode
    assert model.training


def test_extract_empty_loader_raises():
    model = DummyModel()
    empty_loader = make_loader(torch.empty((0, 2)), batch_size=1)
    extractor = ModelExtractor(
        method_name="embed", output_key="embedding", input_keys=("image",)
    )
    with pytest.raises(ValueError, match="No batches found in loader"):
        extractor.extract(model, empty_loader)


def test_load_or_extract_caching(tmp_path: pathlib.Path):
    model = DummyModel()
    tensor = torch.randn(5, 4)
    loader = make_loader(tensor, batch_size=5)
    path = tmp_path / "cached-embedding.pt"
    extractor = ModelExtractor(
        method_name="embed",
        output_key="embedding",
        input_keys=("image",),
        cache_tag="embedding",
    )
    # First call extracts and writes file
    data1 = extractor.extract(model, loader, outdir=tmp_path, prefix="cached")
    assert path.exists()
    # Second call should load from file and emit a warning
    with pytest.warns(UserWarning, match="Loading pre-existing data"):
        data2 = extractor.extract(
            model, loader, outdir=tmp_path, prefix="cached"
        )
    # Loaded data matches original extraction
    assert torch.equal(data1["embedding"], data2["embedding"])


def test_extract_missing_output_key_raises():
    class BadModel(torch.nn.Module):
        def embed(self, x):
            return {"wrong": torch.tensor([1])}

    model = BadModel()
    loader = make_loader(torch.randn(2, 2))
    extractor = ModelExtractor(
        method_name="embed", output_key="embedding", input_keys=("image",)
    )
    with pytest.raises(KeyError):
        extractor.extract(model, loader)


def test_prefix_tag_handling(tmp_path: pathlib.Path):
    model = DummyModel()
    loader = make_loader(torch.randn(2, 2))
    extractor = ModelExtractor(
        method_name="embed",
        output_key="embedding",
        input_keys=("image",),
        cache_tag="logits",
    )
    result = extractor.extract(model, loader, outdir=tmp_path, prefix="base")
    expected_path = tmp_path / "base-logits.pt"
    assert expected_path.exists()
    assert "embedding" in result


def test_extract_loader_uses_setup_path_and_load_or_extract(
    tmp_path: pathlib.Path,
):
    model = DummyModel()
    tensor = torch.randn(3, 2)
    loader = make_loader(tensor)
    extractor = ModelExtractor(
        method_name="embed",
        output_key="embedding",
        input_keys=("image",),
        cache_tag="embedding",
    )
    out = extractor.extract(model, loader, outdir=tmp_path, prefix="pref")
    assert "embedding" in out
    assert (tmp_path / "pref-embedding.pt").exists()


def test_check_model_non_callable_method():
    # Ensure non-module raises TypeError
    class NotAModule:
        pass

    with pytest.raises(TypeError, match="`model` must be a torch.nn.Module"):
        _resolve_method(NotAModule(), "embed")  # type: ignore

    class BadModelNonCallable(torch.nn.Module):
        embed = 42  # not callable

    with pytest.raises(TypeError, match=r"`model.embed` must be callable"):
        _resolve_method(BadModelNonCallable(), "embed")


def test_normalise_inputs_sequence_too_short():
    # Provide a list shorter than required keys
    with pytest.raises(
        ValueError, match="Batch has 1 elements but 2 input_keys"
    ):
        _normalise_inputs([torch.tensor([1])], ["a", "b"])


def test_normalise_output_empty_sequence():
    # Empty list should raise ValueError
    with pytest.raises(ValueError, match="Model returned an empty sequence"):
        _normalise_output([], "any")


def test_load_invalid_type_raises(tmp_path: pathlib.Path):
    # Save a tensor (not a mapping) and attempt to load via _load
    path = tmp_path / "bad.pt"
    torch.save(torch.tensor([1, 2, 3]), path)
    extractor = ModelExtractor(
        method_name="embed", output_key="emb", input_keys=("x",)
    )
    with pytest.raises(
        TypeError, match="Cached file .* does not contain a dict of tensors"
    ):
        extractor._load(path)


def test_model_device_returns_parameter_device():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2).to("cpu")

        def forward(self, x):
            pass  # pragma: no cover

    model = SimpleModel()
    device = _model_device(model)
    assert device.type == "cpu"

    # Ensure non-module raises TypeError
    class NotAModule:
        pass

    with pytest.raises(TypeError, match="`model` must be a torch.nn.Module"):
        _resolve_method(NotAModule(), "embed")  # type: ignore

    class BadModelNonCallable(torch.nn.Module):
        embed = 42  # not callable

    with pytest.raises(TypeError, match=r"`model.embed` must be callable"):
        _resolve_method(BadModelNonCallable(), "embed")

    class BadModelNonCallable(torch.nn.Module):
        embed = 42  # not callable

    with pytest.raises(TypeError, match=r"`model.embed` must be callable"):
        _resolve_method(BadModelNonCallable(), "embed")


# Helper model without parameters or buffers, that would raise if called
class EmptyModel(torch.nn.Module):
    def embed(self, x):
        raise RuntimeError("Should not be called")  # pragma: no cover


def test_resolve_cache_path_both_none():
    # Both outdir and prefix None should return None without warning
    result = _resolve_cache_path(outdir=None, prefix=None, tag=None)
    assert result is None


def test_resolve_method_non_module():
    class NotAModule:
        pass

    with pytest.raises(TypeError, match="must be a torch.nn.Module"):
        _resolve_method(NotAModule(), "embed")  # type: ignore


def test_model_device_fallback():
    model = EmptyModel()
    # Ensure no parameters or buffers
    assert len(list(model.parameters())) == 0
    assert len(list(model.buffers())) == 0
    device = _model_device(model)
    assert device == torch.device("cpu")


def test_to_cpu_success_and_error():
    tensor = torch.randn(3, 3, device="cpu")
    # Success returns a detached cpu tensor
    out = _to_cpu(tensor)
    assert isinstance(out, torch.Tensor)
    assert out.device.type == "cpu"
    # Error when non-tensor
    with pytest.raises(TypeError):
        _to_cpu([1, 2, 3])


def test_concat_and_move_operations():
    # Prepare collected dict of tensors
    a = torch.arange(4).reshape(2, 2)
    b = torch.arange(4, 8).reshape(2, 2)
    collected = {"first": [a, b]}
    concatenated = _concat(collected)
    assert torch.equal(concatenated["first"], torch.cat([a, b], dim=0))
    # Move to GPU if available, otherwise CPU
    target = torch.device("cpu")
    moved = _move(concatenated, target)
    assert all(t.device == target for t in moved.values())


def test_validated_keys_errors():
    # Empty input_keys should raise
    with pytest.raises(ValueError, match="must not be empty"):
        ModelExtractor(method_name="embed", output_key="out", input_keys=())
    # Collision between output_key and extra input_keys should raise
    with pytest.raises(ValueError, match="collides with input_keys"):
        ModelExtractor(
            method_name="embed", output_key="meta", input_keys=("x", "meta")
        )


def test_load_non_mapping(tmp_path: pathlib.Path):
    path = tmp_path / "bad.pt"
    torch.save(torch.randn(2, 2), path)  # save a tensor, not a dict
    extractor = ModelExtractor(
        method_name="embed", output_key="out", input_keys=("x",)
    )
    with pytest.raises(TypeError, match="does not contain a dict of tensors"):
        extractor._load(path)


def test_validate_data_missing_key(tmp_path: pathlib.Path):
    # Use an extractor that expects an extra key 'meta' which the loader won't provide
    extractor = ModelExtractor(
        method_name="embed", output_key="out", input_keys=("x", "meta")
    )
    # Create a minimal data dict missing the extra 'meta' key
    data = {"out": torch.randn(2, 2)}
    assert "meta" not in data
    with pytest.raises(ValueError, match="missing key"):
        extractor._validate_data(data, ("x", "meta"), None)


def test_load_existing_cache_warn(tmp_path: pathlib.Path):
    # Prepare a cached file with dummy data
    data = {"embedding": torch.randn(2, 3)}
    path = tmp_path / "cached-embedding.pt"
    torch.save(data, path)

    extractor = ModelExtractor(
        method_name="embed",
        output_key="embedding",
        input_keys=("image",),
        cache_tag="embedding",
    )
    # Empty loader (won't be used)
    loader = DataLoader([])  # type: ignore
    with pytest.warns(UserWarning, match="Loading pre-existing data"):
        result = extractor.extract(
            EmptyModel(),
            loader,
            outdir=tmp_path,
            prefix="cached",
            overwrite=False,
        )
    # Result should match cached data
    assert torch.equal(result["embedding"], data["embedding"])


def test_resolve_method_missing():
    class NoMethodModel(torch.nn.Module):
        def forward(self, x):
            return x  # pragma: no cover

    with pytest.raises(
        TypeError, match=r"`model` is required to have a `embed\(\)` method"
    ):
        _resolve_method(NoMethodModel(), "embed")


def test_normalise_inputs_mapping_missing_key():
    with pytest.raises(KeyError, match="Keys \\['b'\\] missing in batch"):
        _normalise_inputs({"a": torch.tensor([1])}, ["a", "b"])


def test_normalise_inputs_sequence_success():
    result = _normalise_inputs(
        [torch.tensor([1]), torch.tensor([2])], ["first", "second"]
    )
    assert list(result.keys()) == ["first", "second"]
    assert torch.equal(result["first"], torch.tensor([1]))
    assert torch.equal(result["second"], torch.tensor([2]))


def test_normalise_output_mapping_missing_key():
    with pytest.raises(KeyError, match="Expected key 'out' in model output"):
        _normalise_output({"wrong": torch.tensor([0])}, "out")
