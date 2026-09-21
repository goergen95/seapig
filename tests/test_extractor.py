import pathlib
import re
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from seapig.scores.extractor import (
    ModelExtractor,
    _concat,
    _model_device,
    _move,
    _resolve_cache_path,
    _resolve_method,
    _to_cpu,
)
from tests.fixtures import (
    BadForwardTask,
    BadModel,
    BadModelWrongSig,
    DummyModel,
    EmptyModel,
)


def test_extract_non_mapping_output_raises():
    model = BadForwardTask()
    # Use a simple loader with a single tensor batch
    tensor = torch.randn(2, 3)
    loader = make_loader(tensor, batch_size=1)
    extractor = ModelExtractor(
        output_keys=("embedding",), input_keys=("image",)
    )
    with pytest.raises(TypeError, match="must return a dict"):
        extractor.extract(model, loader)


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
    _resolve_method(DummyModel())

    with pytest.raises(
        TypeError, match=r"`model` is required to have a `noop\(\)` method."
    ):
        _resolve_method(DummyModel(), "noop")

    with pytest.raises(
        AttributeError,
        match=re.escape(
            "`model.predict()` is required to accept `batch` as argument."
        ),
    ):
        _resolve_method(BadModelWrongSig())


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


def test_extract_with_extra_keys(tmp_path: pathlib.Path):

    model = DummyModel()

    # Custom DataLoader yielding dict batches
    class DictDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            for i in range(2):
                yield {"image": torch.randn(2, 3), "meta": torch.tensor([i])}

    loader = torch.utils.data.DataLoader(DictDataset(), batch_size=1)
    extractor = ModelExtractor(
        output_keys=("embedding",), input_keys=("image", "meta")
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
        output_keys=("embedding",), input_keys=("image",)
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
        output_keys=("embedding",), input_keys=("image",)
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
        output_keys=("embedding",), input_keys=("image",)
    )
    with pytest.raises(ValueError, match="No batches found in loader"):
        extractor.extract(model, empty_loader)


def test_load_or_extract_caching(tmp_path: pathlib.Path):
    model = DummyModel()
    tensor = torch.randn(5, 4)
    loader = make_loader(tensor, batch_size=5)
    path = tmp_path / "cached-embedding.pt"
    extractor = ModelExtractor(
        output_keys=("embedding",), input_keys=("image",), cache_tag="embedding"
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
    model = BadModel()
    loader = make_loader(torch.randn(2, 2))
    extractor = ModelExtractor(
        output_keys=("embedding",), input_keys=("image",)
    )
    with pytest.raises(KeyError):
        extractor.extract(model, loader)


def test_prefix_tag_handling(tmp_path: pathlib.Path):
    model = DummyModel()
    loader = make_loader(torch.randn(2, 2))
    extractor = ModelExtractor(
        output_keys=("embedding",), input_keys=("image",), cache_tag="logits"
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
        output_keys=("embedding",), input_keys=("image",), cache_tag="embedding"
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


def test_load_invalid_type_raises(tmp_path: pathlib.Path):
    # Save a tensor (not a mapping) and attempt to load via _load
    path = tmp_path / "bad.pt"
    torch.save(torch.tensor([1, 2, 3]), path)
    extractor = ModelExtractor(output_keys=("emb",), input_keys=("x",))
    with pytest.raises(
        TypeError, match="Cached file .* does not contain a dict of tensors"
    ):
        extractor._load(path)


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
        ModelExtractor(output_keys=("out",), input_keys=())
    # Collision between output_key and extra input_keys should raise
    with pytest.raises(ValueError, match="collide with input_keys"):
        ModelExtractor(output_keys=("meta",), input_keys=("x", "meta"))


def test_load_non_mapping(tmp_path: pathlib.Path):
    path = tmp_path / "bad.pt"
    torch.save(torch.randn(2, 2), path)  # save a tensor, not a dict
    extractor = ModelExtractor(output_keys=("out",), input_keys=("x",))
    with pytest.raises(TypeError, match="does not contain a dict of tensors"):
        extractor._load(path)


def test_validate_data_missing_key(tmp_path: pathlib.Path):
    # Use an extractor that expects an extra key 'meta' which the loader won't provide
    extractor = ModelExtractor(output_keys=("out",), input_keys=("x", "meta"))
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
        output_keys=("embedding",), input_keys=("image",), cache_tag="embedding"
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
