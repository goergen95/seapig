import pytest
import torch

from seapig.scores.utils import _tensor


def test_tensor_none_returns_none():
    assert _tensor(None, "any_key") is None


def test_tensor_direct_tensor_returns_same():
    t = torch.tensor([1, 2, 3])
    assert _tensor(t, "unused") is t


def test_tensor_dict_with_key_returns_tensor():
    t = torch.tensor([4, 5])
    d = {"my_key": t}
    assert _tensor(d, "my_key") is t


def test_tensor_dict_missing_key_raises_keyerror():
    d = {"other": torch.tensor([0])}
    with pytest.raises(KeyError) as exc:
        _tensor(d, "missing")
    assert "missing" in str(exc.value)


def test_tensor_invalid_type_raises_typeerror():
    with pytest.raises(TypeError) as exc:
        _tensor(123, "key")  # type: ignore
    assert "tensor or dict" in str(exc.value)
