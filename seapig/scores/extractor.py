"""Extractor class enabling model + loader extracting pattern with file cache."""

from __future__ import annotations

import inspect
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

TensorDict = dict[str, torch.Tensor]
Batch = torch.Tensor | Mapping[str, Any] | Sequence[Any]


def _resolve_method(model: torch.nn.Module, name: str) -> Callable[..., Any]:
    """Return `model.<name>` after validating that it can be called with `x`."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"`model` must be a torch.nn.Module, got {type(model)}."
        )

    method = getattr(model, name, None)
    if method is None:
        raise TypeError(f"`model` is required to have a `{name}()` method.")
    if not callable(method):
        raise TypeError(f"`model.{name}` must be callable.")
    if "x" not in inspect.signature(method).parameters:
        raise AttributeError(
            f"`{name}()` is required to accept `x` as argument."
        )
    return method


def _model_device(model: torch.nn.Module) -> torch.device:
    """Device of the first parameter/buffer, falling back to CPU."""
    for tensor in (*model.parameters(), *model.buffers()):
        return tensor.device
    return torch.device("cpu")


def _resolve_cache_path(
    outdir: Path | str | None, prefix: str | None, tag: str | None
) -> Path | None:
    """Build `<outdir>/<prefix>-<tag>.pt`; `None` disables caching."""
    if outdir is None and prefix is None:
        return None
    if outdir is None or prefix is None:
        warnings.warn(
            "Both 'outdir' and 'prefix' must be given to enable caching; "
            "continuing without saving tensors.",
            UserWarning,
            stacklevel=3,
        )
        return None

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}-{tag}" if tag else prefix
    return outdir / f"{name}.pt"


def _normalise_inputs(
    batch: Batch, keys: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Map a batch of any common structure onto `keys`."""
    if isinstance(batch, torch.Tensor):
        return {keys[0]: batch}

    if isinstance(batch, Mapping):
        missing = [key for key in keys if key not in batch]
        if missing:
            raise KeyError(
                f"Keys {missing} missing in batch (got {list(batch)})."
            )
        return {key: batch[key] for key in keys}

    if isinstance(batch, Sequence):
        if len(batch) < len(keys):
            raise ValueError(
                f"Batch has {len(batch)} elements but {len(keys)} input_keys "
                f"were requested ({list(keys)})."
            )
        return {key: batch[i] for i, key in enumerate(keys)}

    raise TypeError(f"Unsupported batch type: {type(batch)}.")


def _normalise_output(raw: Any, key: str) -> torch.Tensor:
    """Reduce whatever the model returned to a single tensor."""
    if isinstance(raw, torch.Tensor):
        return raw
    if isinstance(raw, Mapping):
        if key not in raw:
            raise KeyError(
                f"Expected key '{key}' in model output (got {list(raw)})."
            )
        return raw[key]
    if isinstance(raw, Sequence):
        if not raw:
            raise ValueError("Model returned an empty sequence.")
        return raw[0]
    raise TypeError(f"Unsupported model output type: {type(raw)}.")


def _to_cpu(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor, got {type(value)}.")
    return value.detach().to("cpu")


def _concat(
    collected: Mapping[str, Iterable[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {
        key: torch.cat(list(chunks), dim=0) for key, chunks in collected.items()
    }


def _move(
    data: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in data.items()}


class ModelExtractor:
    """Extract tensors from a model over a `~torch.utils.data.DataLoader`.

    Arguments
    ----------
    `method_name`
        Model method called on every batch, e.g. `"embed"` or `"logits"`.
    `output_key`
        Name under which the method's result is stored.
    `input_keys`
        The first key selects the tensor handed to `method_name`; every
        further key is taken from the batch and appended to the result
        (e.g. `("image", "label")`).
    `cache_tag`
        Suffix used for the cache file name.
    """

    def __init__(
        self,
        method_name: str,
        output_key: str,
        input_keys: tuple[str, ...],
        cache_tag: str | None = None,
    ):

        self.method_name = method_name
        self.output_key = output_key
        self.input_keys = input_keys
        self._validated_keys()
        self.cache_tag = output_key if cache_tag is None else cache_tag

    def extract(
        self,
        model: torch.nn.Module,
        loader: DataLoader[Any],
        *,
        outdir: Path | str | None = None,
        prefix: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return `{output_key: tensor, **extra_input_keys}` on the model device.

        Results are cached as `<outdir>/<prefix>-<cache_tag>.pt` whenever both
        `outdir` and `prefix` are provided.
        """
        path = _resolve_cache_path(outdir, prefix, self.cache_tag)

        if path is not None and path.is_file() and not overwrite:
            warnings.warn(
                f"Loading pre-existing data from {path}.", UserWarning
            )
            data = self._load(path)
        else:
            data = self._extract_loader(model, loader, self.input_keys)
            if path is not None:
                torch.save(data, path)

        self._validate_data(data, self.input_keys, path)
        return _move(data, _model_device(model))

    def _validated_keys(self) -> None:
        keys = tuple(self.input_keys)
        if not keys:
            raise ValueError(f"{self}.input_keys must not be empty.")
        if self.output_key in keys[1:]:
            raise ValueError(
                f"output_key '{self.output_key}' collides with input_keys {keys[1:]}."
            )

    @torch.inference_mode()
    def _extract_loader(
        self,
        model: torch.nn.Module,
        loader: DataLoader[Any],
        keys: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Run the model over all batches and concatenate the results (on CPU)."""
        has_batch = False
        method = _resolve_method(model, self.method_name)
        input_key, *extra_keys = keys

        was_training = model.training
        model.eval()
        collected: dict[str, list[torch.Tensor]] = defaultdict(list)
        try:
            for batch in loader:
                has_batch = True
                inputs = _normalise_inputs(batch, keys)
                output = _normalise_output(
                    method(inputs[input_key]), self.output_key
                )
                collected[self.output_key].append(_to_cpu(output))
                for key in extra_keys:
                    collected[key].append(_to_cpu(inputs[key]))
        finally:
            model.train(was_training)

        if not has_batch:
            raise ValueError("No batches found in loader.")

        return _concat(collected)

    @staticmethod
    def _load(path: Path) -> dict[str, torch.Tensor]:
        data = torch.load(path, map_location="cpu")
        if not isinstance(data, Mapping):
            raise TypeError(
                f"Cached file {path} does not contain a dict of tensors."
            )
        return dict(data)

    def _validate_data(
        self,
        data: dict[str, torch.Tensor],
        keys: Sequence[str],
        path: Path | None,
    ) -> None:
        source = f"Cached file {path}" if path is not None else "Extracted data"
        expected = (self.output_key, *keys[1:])
        missing = [key for key in expected if key not in data]
        if missing:
            raise ValueError(f"{source} is missing key(s) {missing}.")
