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

from seapig.utils.progress import track

TensorDict = dict[str, torch.Tensor]
Batch = torch.Tensor | Mapping[str, Any] | Sequence[Any]
Extract = Callable[..., dict[str, torch.Tensor]]


def _resolve_method(
    model: torch.nn.Module, method_name: str = "predict"
) -> Extract:
    """Resolve a callable method on a torch.nn.Module."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("`model` must be a torch.nn.Module")
    if not hasattr(model, method_name):
        raise TypeError(
            f"`model` is required to have a `{method_name}()` method."
        )
    method = getattr(model, method_name)
    if not callable(method):
        raise TypeError(f"`model.{method_name}` must be callable")
    if "x" not in inspect.signature(method).parameters:
        raise AttributeError(
            f"`model.{method_name}()` is required to accept `x` as argument."
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
    """Extract tensors from a model over a `~torch.utils.data.DataLoader`."""

    def __init__(
        self,
        input_keys: tuple[str, ...] = (),
        output_keys: tuple[str, ...] = (),
        cache_tag: str | None = None,
    ):
        """Create a ModelExtractor.

        Parameters
        ----------
        input_keys: tuple[str, ...]
            Keys to extract from each input batch. The first key selects the tensor
            passed to the model's `forward` method; any additional keys are stored
            as extra inputs alongside the model outputs.
        output_keys: tuple[str, ...]
            Keys to retain from the dictionary returned by the model's `forward`
            method.
        cache_tag: str | None, optional
            Tag used for the cache filename `<prefix>-<cache_tag>.pt`. If `None`
            (the default), the first `output_key` is used.
        """
        self.output_keys = output_keys
        self.input_keys = input_keys
        self._validated_keys()
        self.cache_tag = output_keys[0] if cache_tag is None else cache_tag

    def extract(
        self,
        model: torch.nn.Module,
        loader: DataLoader[Any],
        *,
        outdir: Path | str | None = None,
        prefix: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Extract tensors for the specified `output_keys` from a model using a DataLoader.

        Parameters
        ----------
        model: torch.nn.Module
            The model whose `forward` method will be called. It must accept an
            argument named `x` and return a mapping from output keys to tensors.
        loader: DataLoader[Any]
            An iterator yielding batches. Each batch can be a tensor, a mapping,
            or a sequence; the keys in `input_keys` are used to locate the
            tensor(s) passed to the model.
        outdir: Path | str | None, optional
            Directory in which to store a cache file. If `None` (default), no
            caching is performed.
        prefix: str | None, optional
            Prefix for the cache filename. Required together with `outdir` to
            enable caching.
        overwrite: bool, default `False`
            If `True` and a cache file already exists, it will be overwritten;
            otherwise the existing cache is loaded.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the extracted `output_keys` tensors as well
            as any `extra_input_keys` defined in `input_keys` (excluding the
            first key which is used as the model input). The tensors are placed on
            the same device as the model.

        Notes
        -----
        Cached results are saved to `<outdir>/<prefix>-<cache_tag>.pt` where
        `cache_tag` defaults to the first `output_key` unless overridden in the
        constructor.
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
        # Ensure no output key collides with extra input keys
        intersect = set(self.output_keys).intersection(keys[1:])
        if intersect:
            raise ValueError(
                f"output_keys {self.output_keys} collide with input_keys {keys[1:]}."
            )

    @torch.inference_mode()
    def _extract_loader(
        self,
        model: torch.nn.Module,
        loader: DataLoader[dict[str, torch.Tensor]],
        keys: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Run the model over all batches and concatenate the results (on CPU)."""
        has_batch = False
        method = _resolve_method(model)
        input_key, *extra_keys = keys

        was_training = model.training
        model.eval()
        collected: dict[str, list[torch.Tensor]] = defaultdict(list)
        try:
            for batch in track(loader, desc="Iterating over loader"):
                has_batch = True
                inputs = _normalise_inputs(batch, keys)
                out_dict = method(inputs[input_key])
                if not isinstance(out_dict, Mapping):
                    raise TypeError(
                        f"The model's forward method must return a dict, got {type(out_dict)}."
                    )
                for out_key in self.output_keys:
                    if out_key not in out_dict:
                        raise KeyError(
                            f"Expected key '{out_key}' in model output (got {list(out_dict)})."
                        )
                    collected[out_key].append(_to_cpu(out_dict[out_key]))
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
        expected = (*self.output_keys, *keys[1:])
        missing = [key for key in expected if key not in data]
        if missing:
            raise ValueError(f"{source} is missing key(s) {missing}.")
