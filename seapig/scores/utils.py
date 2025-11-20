"""Utilities for Confidence Scores."""

import inspect
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


def check_model(model: torch.nn.Module) -> None:
    """Check model for validity."""
    assert isinstance(model, torch.nn.Module)
    if not callable(model.embed):
        raise Exception("model is required to have a embed method.")
    sig = inspect.signature(obj=model.embed)  # type: ignore [unreachable]
    if "x" not in sig.parameters.keys():
        raise Exception("get_embed method is required to except x as argument.")


def setup_path(
    outdir: Path | None = None, prefix: str | None = None
) -> Path | None:
    """Set the output path for a parquet file."""
    if outdir is None or prefix is None:
        return None
    if not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{prefix}.parquet"


def get_embeddings(
    model: torch.nn.Module,
    loader: DataLoader[Tensor | dict[str, Tensor]],
    path: Path | None = None,
    desc: str = "",
) -> Tensor:
    """Load Embeddings from disk or iterate over DataLoader."""
    if path is not None and path.is_file():
        embeddings = _load_parquet(path)
        return embeddings
    embeddings = _extract_dl(model=model, loader=loader, desc=desc)
    if path is not None:
        _write_parquet(embeddings=embeddings, path=path)
    return embeddings


def _write_parquet(embeddings: Tensor, path: Path) -> None:
    """Write embeddings to Parquet."""
    df = pd.DataFrame(embeddings)
    df.to_parquet(path, index=False)


def _load_parquet(path: Path) -> Tensor:
    """Read Parquet to Tensor."""
    df = pd.read_parquet(path)
    return Tensor(df.values)


@torch.inference_mode()
def _extract_dl(
    model: torch.nn.Module,
    loader: DataLoader[Tensor | dict[str, Tensor]],
    desc: str = "",
) -> Tensor:
    """Extract embeddings for samples in a DataLoader."""
    assert callable(model.embed)
    steps = len(loader)  # type: ignore [unreachable]
    pbar = tqdm(total=steps, desc=f"Embedding {desc}: {steps}", unit="steps")

    embeddings = list()
    step = 0
    for batch in loader:
        step += 1
        if isinstance(batch, dict):
            z = model.embed(batch["inputs"])
        elif isinstance(batch, Tensor):
            z = model.embed(batch)
        else:
            raise TypeError(
                "dataloader is expected to return a Tensor or dict of Tensors with 'inputs' key."
            )
        embeddings.append(z)
        _ = pbar.update(n=step)

    embeddings = torch.cat(embeddings, dim=0)

    if len(embeddings.shape) > 2:
        warnings.warn(
            f"embed method is expected to return (B,N) tensor (batchsize, embedding dim) but got {embeddings.shape}."
        )
        embeddings = embeddings.view(
            embeddings.size(0), embeddings.size(1), -1
        ).mean(2)

    return embeddings
