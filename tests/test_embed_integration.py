from pathlib import Path
from typing import cast

import pytest
import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from seapig.model import SelectiveInferenceTask
from tests.fixtures import (
    DictDataset,
    DummyModel,
    SimpleDataModule,
    SimpleL2Score,
)

_EmbedLoader = DataLoader[torch.Tensor | dict[str, torch.Tensor]]


@pytest.mark.filterwarnings(
    r"ignore:`isinstance\(treespec, LeafSpec\)` is deprecated.*"
)
def test_datamodule_transform_applied_consistently(tmp_path: Path) -> None:
    torch.manual_seed(0)

    # shapes: (N, C=1, H=2, W=2) -> flattened to 4 features in model/embed
    train_x = torch.arange(0, 40, dtype=torch.float32).reshape(10, 1, 2, 2)
    train_y = torch.zeros(10, dtype=torch.long)
    val_x = torch.arange(40, 64, dtype=torch.float32).reshape(6, 1, 2, 2)
    val_y = torch.zeros(6, dtype=torch.long)
    test_x = torch.arange(64, 84, dtype=torch.float32).reshape(5, 1, 2, 2)
    test_y = torch.zeros(5, dtype=torch.long)

    def transform_fn(x: torch.Tensor) -> torch.Tensor:
        return x + 1.0

    train_ds = DictDataset(train_x, train_y, transform=transform_fn)
    val_ds = DictDataset(val_x, val_y, transform=transform_fn)
    test_ds = DictDataset(test_x, test_y, transform=transform_fn)

    dm = SimpleDataModule(train_ds, val_ds, test_ds, batch_size=2)

    model = DummyModel()
    score = SimpleL2Score()

    # Fit the score by extracting embeddings from the dataloaders
    score.fit(
        model=model,
        loaders=cast(
            dict[str, _EmbedLoader],
            {"train": dm.train_dataloader(), "val": dm.val_dataloader()},
        ),
    )

    # deterministic threshold (median of calibration scores)
    score.set_threshold(q=0.5)

    # wrap and run Trainer.predict using the datamodule (ensures datamodule transforms are used)
    from lightning import LightningModule

    task = SelectiveInferenceTask(
        task=cast(LightningModule, model),
        score=score,
        input_key="image",
        target_key="label",
    )

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    preds = trainer.predict(task, datamodule=dm)
    assert preds is not None

    # collect trainer results
    trainer_scores = torch.cat(
        [
            cast(dict[str, torch.Tensor], p)["score"].detach().cpu()
            for p in preds
        ],
        dim=0,
    )
    trainer_selected = torch.cat(
        [
            cast(dict[str, torch.Tensor], p)["selected"].detach().cpu()
            for p in preds
        ],
        dim=0,
    )

    # compute scores manually using the datamodule's test_dataloader (score embeds on the fly)
    manual_scores = score.score(
        model=model,
        loader=cast(_EmbedLoader, dm.test_dataloader()),
        outdir=None,
        prefix=None,
    )
    threshold = score.get_threshold()
    assert threshold is not None
    manual_selected = manual_scores < threshold

    # assertions: same shape and identical numeric results
    assert trainer_scores.shape == manual_scores.shape
    assert torch.allclose(trainer_scores, manual_scores, atol=1e-6)
    assert torch.equal(trainer_selected, manual_selected)
