"""End-to-end example using TorchGeo helper and seapig

This script trains a tiny CNN on the small Landsat example dataset provided by
`dev.torchgeo`, extracts embeddings, fits a Euclidean KNN score from seapig,
calibrates a coverage threshold on a validation split and evaluates selective
performance on a held-out test set.

Usage:
    python dev/example.py --epochs 5 --q 0.9 --device cuda
    
The script is heavily commented and logs progress. It is intended as a runnable
example for newcomers to understand how to integrate seapig with a Lightning
model and TorchGeo-style dataloaders"""

import argparse
import logging
import os
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Subset
import lightning as pl
from lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

# seapig API
from seapig.scores.knn import EuclideanScore
from seapig.model import SelectiveInferenceTask
from seapig.metric import RiskCoverageMetric
import math
import random
import tempfile
from typing import Iterable, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl
from torchgeo.datasets import Landsat8, CDL
from torchgeo.datasets.utils import download_and_extract_archive
from torchgeo.datasets.utils import stack_samples
from torchgeo.samplers import GridGeoSampler
from torchmetrics import Accuracy, MetricCollection

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("dev.example")


# python
class TinyTask(LightningModule):
    """Pixel-level classification model exposing .embed() and .predict().

    - Encoder outputs a spatial feature map of shape (B, embed_dim, H, W).
    - Classifier is a 1x1 conv producing per-pixel logits (B, num_classes, H, W).
    - embed(x) returns per-pixel embeddings with shape (B, N_pixels, embed_dim).
    """

    def __init__(
        self,
        in_channels: int = 6,
        embed_dim: int = 64,
        num_classes: int = 2,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # multiclass per-pixel accuracy for selective evaluation
        self.test_metrics = MetricCollection(
            {
                "accuracy": Accuracy(
                    task="multiclass", num_classes=self.hparams.num_classes
                )
            }
        )

        # produce spatial embeddings (preserve H,W)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, embed_dim, kernel_size=1),
            nn.ReLU(),
        )

        # 1x1 conv to map embeddings -> per-pixel logits
        self.classifier = nn.Conv2d(self.hparams.embed_dim, self.hparams.num_classes, kernel_size=1)

        # per-pixel cross-entropy
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-pixel logits of shape (B, num_classes, H, W)."""
        emb_map = self.encoder(x)
        logits = self.classifier(emb_map)
        return logits

    def embed(self, x: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        """Return image-level embeddings.

        If flatten=True returns shape (B, embed_dim).
        If flatten=False returns shape (B, embed_dim, H, W).
        """
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
        feat = self.encoder(x)
        if flatten:
            # global average pool to get image-level embedding
            feat = feat.mean(dim=[2, 3])  # (B, embed_dim)
        return feat


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-pixel predicted class indices (B, H, W)."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        x = batch["image"] if isinstance(batch, dict) else batch[0]
        y = batch["mask"] if isinstance(batch, dict) else batch[1]
        # y expected shape: (B, H, W) with integer class labels
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"] if isinstance(batch, dict) else batch[0]
        y = batch["mask"] if isinstance(batch, dict) else batch[1]
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        # simple per-pixel accuracy scalar for logging
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

class _QueryIterableDataset(IterableDataset):
    """IterableDataset that yields dataset[query] for a precomputed list of queries.

    If shuffle=True the query order is randomized each epoch (uses torch seed).
    """

    def __init__(self, base_dataset, queries: Sequence, shuffle: bool = False) -> None:
        super().__init__()
        self.base = base_dataset
        self.queries = list(queries)
        self.shuffle = shuffle

    def __iter__(self) -> Iterator:
        qs = list(self.queries)
        if self.shuffle:
            # use torch generator for reproducibility across workers
            g = torch.Generator()
            g.manual_seed(torch.initial_seed() % (2**31 - 1))
            # perform deterministic shuffle using the generator
            # Generator.shuffle isn't present, emulate by sampling permutation
            perm = torch.randperm(len(qs), generator=g).tolist()
            qs = [qs[i] for i in perm]
        for q in qs:
            yield self.base[q]

    def __len__(self) -> int:
        return len(self.queries)


class LandsatCDLSpatialDataModule(pl.LightningDataModule):
    """DataModule using Landsat8 as predictors and CDL as labels with a spatial split.

    Parameters
    ----------
    landsat_root : str | None
        Directory containing Landsat8 data. If None, a temp dir is used.
    cdl_root : str | None
        Directory containing CDL data. If None, a temp dir is used.
    bands : list[str] | None
        Bands for Landsat8. If None, a sensible default is used.
    tile_size : int
        Side length (pixels) of square spatial tiles used for splitting.
    val_frac : float
        Fraction of tiles assigned to validation.
    test_frac : float
        Fraction of tiles assigned to test.
    batch_size : int
    num_workers : int
    seed : int
        Random seed for reproducible split.
    """

    def __init__(
        self,
        landsat_root: Optional[str] = None,
        cdl_root: Optional[str] = None,
        bands: Optional[Sequence[str]] = None,
        tile_size: int = 128,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.landsat_root = landsat_root or tempfile.gettempdir()
        self.cdl_root = cdl_root or tempfile.gettempdir()
        self.bands = list(bands) if bands is not None else ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
        self.tile_size = int(tile_size)
        self.val_frac = float(val_frac)
        self.test_frac = float(test_frac)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)

        # placeholders populated in setup()
        self.dataset = None
        self.train_queries: List = []
        self.val_queries: List = []
        self.test_queries: List = []

    def prepare_data(self) -> None:
        """Download / extract small example archives if needed (idempotent)."""
        # Example urls used in repo's dev/torchgeo.py helper; adjust if you have your own
        base = "https://hf.co/datasets/torchgeo/tutorials/resolve/ff30b729e3cbf906148d69a4441cc68023898924/"
        landsat8_url = base + "LC08_L2SP_023032_20230831_20230911_02_T1.tar.gz"
        cdl_url = base + "2023_30m_cdls.zip"
        download_and_extract_archive(landsat8_url, self.landsat_root)
        download_and_extract_archive(cdl_url, self.cdl_root)

    def setup(self, stage: Optional[str] = None) -> None:
        """Create composed dataset and compute spatial splits (queries list)."""
        landsat8 = Landsat8(paths=self.landsat_root, bands=self.bands)
        cdl = CDL(paths=self.cdl_root)
        self.dataset = landsat8 & cdl  # composed dataset returns aligned dict samples

        # enumerate non-overlapping grid tiles across the dataset footprint
        grid_sampler = GridGeoSampler(self.dataset, size=self.tile_size, stride=self.tile_size)
        all_queries = list(iter(grid_sampler))  # each entry is a "query" usable as indexing key

        if len(all_queries) == 0:
            raise RuntimeError("GridGeoSampler produced zero tiles: check dataset roots / tile_size")

        # deterministic shuffle of tiles before splitting to avoid spatial adjacency bias
        rng = random.Random(self.seed)
        rng.shuffle(all_queries)

        n = len(all_queries)
        n_test = max(1, int(math.floor(n * self.test_frac)))
        n_val = max(1, int(math.floor(n * self.val_frac)))
        n_train = n - n_val - n_test
        if n_train <= 0:
            # ensure at least one train tile
            n_train = max(1, n - n_val - n_test)
        # partition
        self.train_queries = all_queries[:n_train]
        self.val_queries = all_queries[n_train : n_train + n_val]
        self.test_queries = all_queries[n_train + n_val :]

    def train_dataloader(self) -> DataLoader:
        ds = _QueryIterableDataset(self.dataset, self.train_queries, shuffle=True)
        return DataLoader(ds, batch_size=self.batch_size, collate_fn=stack_samples, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        ds = _QueryIterableDataset(self.dataset, self.val_queries, shuffle=False)
        return DataLoader(ds, batch_size=self.batch_size, collate_fn=stack_samples, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        ds = _QueryIterableDataset(self.dataset, self.test_queries, shuffle=False)
        return DataLoader(ds, batch_size=self.batch_size, collate_fn=stack_samples, num_workers=self.num_workers)


def train_and_evaluate(args: argparse.Namespace) -> None:
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    logger.info("Using device: %s", device)

    pl.seed_everything(42, workers=True)

    model = TinyTask(in_channels=6, embed_dim=64, num_classes=134, lr=args.lr)

    chkpt_dir = Path(".checkpoints")
    chkpt_dir.mkdir(exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=str(chkpt_dir), save_top_k=1, monitor="val_loss", mode="min")

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        logger=False,
        devices=1 if device == "cuda" else None,
        accelerator="gpu" if device == "cuda" else "cpu",
        enable_progress_bar=True,
    )

    # Build DataModule and run training using Landsat + CDL spatial split
    dm = LandsatCDLSpatialDataModule(
        landsat_root=None,
        cdl_root=None,
        bands=None,
        tile_size=128,
        val_frac=0.1,
        test_frac=0.1,
        batch_size=8,
        num_workers=4,
        seed=42,
    )
    # download / prepare data and compute spatial splits
    dm.prepare_data()
    dm.setup()
    logger.info("Starting training for %d epochs", args.epochs)
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    logger.info("Training finished. Best checkpoint: %s", checkpoint_callback.best_model_path)

    if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
        model = TinyTask.load_from_checkpoint(checkpoint_callback.best_model_path)
        logger.info("Loaded best model from checkpoint")

    # Move model to device before embedding extraction via fit_dl/score_dl
    model.to(device)
    model.eval()

    logger.info("Fitting Euclidean KNN score via fit_dl (embeddings extracted on-the-fly)")
    score = EuclideanScore(k=args.k, stat="mean")
    # pass the dataloaders dictionary expected by EmbeddingScore.fit_dl
    score.fit_dl(model=model, loaders={"train": dm.train_dataloader(), "val": dm.val_dataloader()}, outdir=None, prefix=None)

    logger.info("Calibrating threshold to coverage q=%.2f using validation embeddings", args.q)
    score.set_threshold(q=args.q)
    logger.info("Set threshold := %.6f", float(score.threshold))

    # Wrap the trained LightningModule with SelectiveInferenceTask for selective
    # evaluation. The SelectiveInferenceTask delegates embedding extraction and
    # uses the fitted `score` to decide acceptance. We also pass a RiskCoverage
    # metric used internally to report selective risk/coverage statistics.
    selective_task = SelectiveInferenceTask(
        task=model,
        score=score, 
        target_key ="mask"
    )

    # Evaluate selective performance using the Lightning Trainer on the test
    # split. The SelectiveInferenceTask implements `test_step`/`predict_step`
    # to emit selective metrics and (optionally) selection masks.
    logger.info("Evaluating selective performance with SelectiveInferenceTask on test set")
    trainer.test(selective_task, dataloaders=dm.test_dataloader())

    # Optionally obtain predictions and selection masks with predict
    logger.info("Obtaining predictions and selection masks with predict()")
    preds = trainer.predict(selective_task, dataloaders=dm.test_dataloader())
    print("Example predictions and selection masks from predict():")
    for batch in preds[:2]:  # print first 2 batches
        print(batch)

    logger.info("Example run complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("dev/example.py - seapig fit_dl demo")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (small demo)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbours for KNN score")
    parser.add_argument("--q", type=float, default=0.9, help="Target coverage fraction for threshold calibration")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to run model and embedding extraction on")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Running dev/example.py with args: %s", args)
    train_and_evaluate(args)
