from pathlib import Path
from typing import Any, cast

import pytest
import torch
from lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, MetricCollection

from seapig.model import SelectiveInferenceTask
from seapig.scores.embed import EmbeddingScore

_EmbedLoader = DataLoader[torch.Tensor | dict[str, torch.Tensor]]


class SmallDictDataset(Dataset):
    def __init__(
        self, data: torch.Tensor, labels: torch.Tensor, transform: Any = None
    ) -> None:
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):  # type: ignore[override]
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        return {"image": x, "label": self.labels[idx]}


class SimpleDataModule(LightningDataModule):
    """Tiny datamodule-like helper exposing train/val/test dataloaders."""

    def __init__(
        self,
        train_ds: SmallDictDataset,
        val_ds: SmallDictDataset,
        test_ds: SmallDictDataset,
        batch_size: int = 4,
    ) -> None:
        super().__init__()
        self._train = train_ds
        self._val = val_ds
        self._test = test_ds
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return DataLoader(self._train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return DataLoader(self._val, batch_size=self.batch_size)

    def predict_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return DataLoader(self._test, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return DataLoader(self._test, batch_size=self.batch_size)


class DummyModel(torch.nn.Module):
    """Tiny deterministic model providing forward() and embed()."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        torch.manual_seed(0)
        for p in self.lin.parameters():
            torch.nn.init.constant_(p, 0.1)
        # required by SelectiveInferenceTask
        self.test_metrics = MetricCollection(Accuracy(task="binary"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(start_dim=1).mean(dim=1, keepdim=True)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        z = x.flatten(start_dim=1)
        return cast(torch.Tensor, self.lin(z))


class SimpleL2Score(EmbeddingScore):
    """Test-only EmbeddingScore using pure torch cdist for NN distances."""

    ident = "simple-l2"
    train_required = False
    cal_required = False

    def __init__(self) -> None:
        super().__init__(pca=None)

    def _fit_impl(self) -> None:
        assert self.ref_embeddings is not None
        assert self.cal_embeddings is not None
        self.set_trained()
        self.scores = (
            torch.cdist(self.cal_embeddings, self.ref_embeddings)
            .min(dim=1)
            .values
        )
        self.set_calibrated()

    def fit(
        self,
        X: torch.Tensor | None = None,
        Y: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loaders: dict[str, DataLoader[torch.Tensor | dict[str, torch.Tensor]]]
        | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> None:
        super().fit(
            X=X, Y=Y, model=model, loaders=loaders, outdir=outdir, prefix=prefix
        )
        self._fit_impl()

    @torch.inference_mode()
    def _score_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        assert self.ref_embeddings is not None
        dists = torch.cdist(X, self.ref_embeddings)
        return dists.min(dim=1).values


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

    train_ds = SmallDictDataset(train_x, train_y, transform=transform_fn)
    val_ds = SmallDictDataset(val_x, val_y, transform=transform_fn)
    test_ds = SmallDictDataset(test_x, test_y, transform=transform_fn)

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
