"""Mixins for seapig."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader


class ModelExtractorMixin:
    """Generic mixin to extract tensors (embeddings or logits) from a model.

    Subclasses set the class attribute `method_name` to the name of the model
    method that should be called on each batch (e.g. `"embed"` or `"logits"`).
    The mixin provides utilities for path handling, caching, and batch extraction.
    """

    method_name: str = "embed"  # default, overridden by subclasses

    @classmethod
    def _check_model(cls, model: torch.nn.Module) -> None:
        assert isinstance(model, torch.nn.Module)
        if not hasattr(model, cls.method_name):
            raise TypeError(
                f"model is required to have a `{cls.method_name}()` method."
            )
        method = getattr(model, cls.method_name)
        if not callable(method):
            raise TypeError(
                f"`{cls.method_name}()` must be callable on the model."
            )
        sig = inspect.signature(obj=method)
        if "x" not in sig.parameters:
            raise AttributeError(
                f"`{cls.method_name}()` method is required to accept `x` as argument."
            )

    @staticmethod
    def _setup_path(
        outdir: Path | str | None = None, prefix: str | None = None
    ) -> Path | None:
        if outdir is not None and prefix is None:
            warnings.warn(
                "'outdir' has been specified but 'prefix' is None.\n"
                "Consider specifying 'prefix' as well to enable saving tensors.",
                UserWarning,
            )
        if outdir is None or prefix is None:
            return None
        outdir = Path(outdir)
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)
        return outdir / f"{prefix}.pt"

    @staticmethod
    def _write_pt(
        x: torch.Tensor | dict[str, torch.Tensor], path: Path
    ) -> None:
        """Save tensor or dict of tensors to disk."""
        torch.save(x, path)

    @staticmethod
    @torch.inference_mode()
    def _load_pt(path: Path) -> torch.Tensor | dict[str, torch.Tensor]:
        """Load previously saved tensor or dict of tensors."""
        return torch.load(path)

    @classmethod
    def _load_or_extract(
        cls,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
        path: Path | None,
        input_keys: list[str],
        output_key: str,
    ) -> dict[str, torch.Tensor]:
        """Load cached data or extract from ``loader``."""
        if path is not None and path.is_file():
            warnings.warn(
                f"Loading pre-existing data from {path}.", UserWarning
            )
            data = cls._load_pt(path)
        else:
            data = cls._extract_dl(model, loader, input_keys, output_key)
            if path is not None:
                cls._write_pt(data, path)

        if output_key not in data:
            raise ValueError(
                f"Saved file {path} does not contain '{output_key}'"
            )

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

        assert isinstance(data, dict)
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)

        return data

    @classmethod
    def _extract_dl(
        cls,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
        input_keys: list[str],
        output_key: str,
    ) -> dict[str, torch.Tensor]:
        """Extract data from a DataLoader and return a unified dictionary."""
        if len(loader) == 0:
            raise ValueError("No batches found in loader")

        was_training = model.training
        model.eval()

        collected: dict[str, list[torch.Tensor]] = {}
        for batch in loader:
            batch_dict = cls._extract_batch(
                batch, model, input_keys, output_key
            )
            for k, v in batch_dict.items():
                if v is None:
                    continue
                collected.setdefault(k, []).append(v)

        # Concatenate tensors per key
        result: dict[str, torch.Tensor] = {}
        for k, tensors in collected.items():
            result[k] = torch.cat(tensors, dim=0)

        if was_training:
            model.train()

        return result

    @classmethod
    @torch.inference_mode()
    def _extract_batch(
        cls,
        batch: torch.Tensor
        | dict[str, torch.Tensor]
        | list[torch.Tensor]
        | tuple[torch.Tensor, ...],
        model: torch.nn.Module,
        input_keys: list[str],
        output_key: str,
    ) -> dict[str, torch.Tensor]:
        """Extract a single batch and normalise its output."""
        cls._check_model(model)
        inputs = cls._normalise_input(batch, keys=input_keys)
        method = getattr(model, cls.method_name)
        assert callable(method)
        raw_out = method(inputs[input_keys[0]])
        outputs = cls._normalise_output(raw_out, output_key)
        if len(inputs) > 0:
            outputs = outputs | {
                k: v for k, v in inputs.items() if k is not input_keys[0]
            }
        return outputs

    @staticmethod
    def _normalise_output(
        raw: Any, key: str = "embedding"
    ) -> dict[str, torch.Tensor]:
        if isinstance(raw, torch.Tensor):
            return {key: raw}
        if isinstance(raw, (list, tuple)):
            return {key: raw[0]}
        if isinstance(raw, dict):
            if key not in raw:
                raise KeyError(f"Expected key {key} in model output dict.")
            return {key: raw[key]}
        raise TypeError(f"Unsupported output type: {type(raw)}")

    @staticmethod
    def _normalise_input(
        batch: Any, keys: list[str] | None
    ) -> dict[str, torch.Tensor]:
        if keys is None:
            keys = ["image"]
        if isinstance(batch, torch.Tensor):
            return {keys[0]: batch}
        if isinstance(batch, (list, tuple)):
            return {k: batch[i] for i, k in enumerate(keys)}
        if isinstance(batch, dict):
            for key in keys:
                if key not in batch:
                    raise KeyError(f"Expected key {key} not in batch.")
            return dict(batch)
        raise TypeError(f"Unsupported output type: {type(batch)}")

    @classmethod
    def _extract_loader(
        cls,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
        input_keys: list[str],
        output_key: str,
        outdir: Path | str | None = None,
        prefix: str | None = None,
    ) -> dict[str, torch.Tensor]:
        path = cls._setup_path(outdir, prefix)
        data = cls._load_or_extract(model, loader, path, input_keys, output_key)
        return data

    @classmethod
    def _extract_dict(
        cls,
        model: torch.nn.Module,
        loaders: dict[str, DataLoader[torch.Tensor | dict[str, torch.Tensor]]],
        input_keys: list[str],
        output_key: str,
        key: Literal["train", "val"],
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> dict[str, torch.Tensor]:

        if key not in loaders:
            raise KeyError(f"Missing key `{key}` in loaders dictionary.")

        loader = loaders[key]
        assert isinstance(loader, DataLoader)

        if prefix is not None:
            if cls.method_name == "embed":
                prefix = prefix + f"-embeddings-{key}"
            if cls.method_name == "logits":
                prefix = prefix + f"-logits-{key}"

        x = cls._extract_loader(
            model=model,
            loader=loader,
            input_keys=input_keys,
            output_key=output_key,
            outdir=outdir,
            prefix=prefix,
        )

        return x


class FAISSIndexMixin:
    """Encapsulate FAISS index handling.

    Expected attributes on the host class (e.g., `KNNScore`):

    - `self.k`:  number of neighbours.
    - `self.index`: the class-agnostic index (or `None` before building).
    - `self.indices_by_class`: mapping `class_id -> index` for class-wise mode.
    - `self.index_path`: optional `Path` where indexes are persisted.
    """

    def _make_faiss_index(self, embs: torch.Tensor) -> Any:
        """Create a FAISS index appropriate for `embs`.

        Uses a flat index for small datasets (≤10_000 vectors) and an HNSW
        index otherwise. Parameters are suggested by `_suggest_build_params`.
        """
        params = self._suggest_build_params(embs=embs, k=self.k)
        d = int(embs.shape[1])
        N = int(embs.shape[0])
        if N <= 10_000:
            index = faiss.IndexFlatL2(d)  # type: ignore[possibly-missing-attribute]
        else:
            M = params["M"]
            index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)  # type: ignore[possibly-missing-attribute]
            index.hnsw.efConstruction = params["efConstruction"]
        return index

    def _build_index(self, embs: torch.Tensor) -> None:
        """Build the class-agnostic index from ``embs``.

        If `self.index_path` is set and the file already exists, the index is
        loaded from disk instead of rebuilt.
        """
        assert isinstance(embs, torch.Tensor)
        index_path = self.index_path
        index = self._make_faiss_index(embs)
        if index_path is None or not Path(index_path).exists():
            embs_np: np.ndarray = embs.cpu().numpy().astype(np.float32)
            index.add(embs_np)
            if index_path is not None:
                faiss.write_index(index, str(index_path))  # type: ignore[possibly-missing-attribute]
        else:
            warnings.warn(
                f"Index file {index_path} already exists. Loading from disk.",
                UserWarning,
            )
            index = faiss.read_index(str(index_path))  # type: ignore[possibly-missing-attribute]
        self.index = index

    def _build_index_for_class(self, c: int, embs: torch.Tensor) -> None:
        """Build a per-class index for class `c`.

        Persists the index under `self.index_path / f"class{c}.bin"` when an
        `index_path` directory is configured.
        """
        assert isinstance(embs, torch.Tensor)
        index = self._make_faiss_index(embs)
        idx_path: Path | None = None
        if self.index_path is not None:
            idx_path = self.index_path / f"class{c}.bin"

        if idx_path is not None and idx_path.exists():
            warnings.warn(
                f"Class index {idx_path} already exists. Loading from disk.",
                UserWarning,
            )
            index = faiss.read_index(str(idx_path))  # type: ignore[possibly-missing-attribute]
        else:
            embs_np = embs.cpu().numpy().astype(np.float32)
            index.add(embs_np)
            if idx_path is not None:
                faiss.write_index(index, str(idx_path))  # type: ignore[possibly-missing-attribute]

        self.indices_by_class[c] = index

    def _query_index(
        self, query: torch.Tensor, offset: int = 0, *, index: Any | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a KNN search against a FAISS `index`.

        Returns `(distances, indices)`. `offset` discards the nearest
        neighbour when `offset > 0` (useful for excluding the query itself).
        """
        if index is None:
            index = self.index
        assert index is not None, "Index must be built before querying."
        if isinstance(index, faiss.IndexHNSW):  # type: ignore[possibly-missing-attribute]
            params = self._suggest_query_params(query, self.k + offset)
            ef_search = params.get("efSearch", index.hnsw.efSearch)
            index.hnsw.efSearch = ef_search
        if index.d != query.shape[1]:
            raise ValueError(
                f"Query dimension {query.shape[1]} does not match index dimension {index.d}"
            )
        query_np = query.cpu().numpy().astype(np.float32)
        distances, indices = index.search(query_np, self.k + offset)
        distances = torch.from_numpy(distances).to(query.device)
        indices = torch.from_numpy(indices).to(query.device)
        if offset > 0:
            distances = distances[:, offset:]
            indices = indices[:, offset:]
        return distances, indices

    @staticmethod
    def _suggest_build_params(embs: torch.Tensor, k: int = 1) -> dict[str, Any]:
        if embs.dim() != 2:
            raise ValueError("embeddings must be 2D (N, D)")
        n, d = map(int, embs.shape)
        if d <= 64:
            M = 16
        elif d <= 128:
            M = 24
        elif d <= 256:
            M = 32
        elif d <= 512 or d <= 1024:
            M = 48
        else:
            M = 64
        if n > 5_000_000:
            M = max(M, 32)  # pragma: no cover
        if n > 50_000_000:
            M = max(M, 48)  # pragma: no cover
        C = max(4 * M, 128)
        C = min(C, 1024)
        return {"M": M, "efConstruction": C}

    @staticmethod
    def _suggest_query_params(embs: torch.Tensor, k: int = 1) -> dict[str, Any]:
        params = FAISSIndexMixin._suggest_build_params(embs, k)
        S = max(k * 8, 512)
        S = min(S, params["efConstruction"])
        S = max(S, k)
        return {"efSearch": S}
