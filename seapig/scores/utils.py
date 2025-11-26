"""Utilities for Confidence Scores."""

import inspect
import warnings
from pathlib import Path

import pandas as pd
import torch
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
    loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
    path: Path | None = None,
) -> torch.Tensor:
    """Load Embeddings from disk or iterate over DataLoader."""
    if path is not None and path.is_file():
        embeddings = _load_parquet(path)
        return embeddings.to(device=model.device)
    embeddings = _extract_dl(model=model, loader=loader)
    if path is not None:
        _write_parquet(embeddings=embeddings, path=path)
    return embeddings


def _write_parquet(embeddings: torch.Tensor, path: Path) -> None:
    """Write embeddings to Parquet."""
    df = pd.DataFrame(embeddings.cpu())
    df.to_parquet(path, index=False)


def _load_parquet(path: Path) -> torch.Tensor:
    """Read Parquet to torch.Tensor."""
    df = pd.read_parquet(path)
    return torch.Tensor(df.values)


@torch.inference_mode()
def _extract_dl(
    model: torch.nn.Module,
    loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Extract embeddings for samples in a DataLoader."""
    assert callable(model.embed)
    n_batches = len(loader)  # type: ignore [unreachable]
    pbar = tqdm(
        total=n_batches, desc=f"Embedding {n_batches} batches", unit="batches"
    )

    embeddings = list()
    for batch in loader:
        if isinstance(batch, dict):
            z = model.embed(batch["image"].to(device=model.device))
        elif isinstance(batch, torch.torch.Tensor):
            z = model.embed(batch.to(device=model.device))
        else:
            raise TypeError(
                "dataloader is expected to return a torch.Tensor or dict of torch.Tensors with 'inputs' key."
            )
        embeddings.append(z)
        _ = pbar.update(n=1)

    embeddings = torch.cat(embeddings, dim=0)

    if len(embeddings.shape) > 2:
        warnings.warn(
            f"embed method is expected to return (B,N) torch.Tensor (batchsize, embedding dim) but got {embeddings.shape}."
        )
        embeddings = embeddings.view(
            embeddings.size(0), embeddings.size(1), -1
        ).mean(2)

    return embeddings
