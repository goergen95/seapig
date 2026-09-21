"""Common pytest fixtures for the test suite."""

from __future__ import annotations

from typing import Any

import torch
from lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, MetricCollection

from seapig.scores.base import UncertaintyScore
from seapig.scores.embed import EmbeddingScore
from seapig.scores.utils import TensorPCA


class EmptyModel(torch.nn.Module):
    pass


class BadModel(torch.nn.Module):
    def predict(self, x):
        return {"wrong": torch.tensor([1])}


class BadModelWrongSig(torch.nn.Module):
    def predict(self, y):  # type: ignore[override]
        return torch.zeros(1, 2)  # pragma: no cover


class BadForwardTask(torch.nn.Module):
    def predict(self, x: torch.Tensor):
        return [x]


class BadPredictStepTask(LightningModule):
    """Task with a predict_step that returns a list instead of dict/tensor."""

    def predict(self, x: torch.Tensor):
        pass  # pragma: no cover

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        return [batch]


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
        self.test_metrics = MetricCollection(Accuracy(task="binary"))

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "embedding": x.view(x.shape[0], -1),
            "logit": x,
            "prediction": x,
        }


class MinimalEmbedding(EmbeddingScore):
    def __init__(self, pca: TensorPCA | None = None) -> None:
        super().__init__(pca=pca)
        self.train_required = False
        self.cal_required = False

    def _score(self, query: torch.Tensor | None) -> torch.Tensor:
        assert query is not None
        return query.sum(dim=1)

    def _fit(self, q: bool | float = False) -> None:
        pass  # pragma: no cover


class DummyTask(LightningModule):
    """Forward returns predictions; embed returns the input so selection can be driven by input."""

    def __init__(self) -> None:
        super().__init__()
        # base metric required by SelectiveInferenceTask (will be wrapped by SelectiveMetric)
        self.test_metrics = Accuracy(task="binary")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # predictions encoded in second column (0/1)
        return x[:, 1].long()


class DummyTaskTensor(LightningModule):
    """Task returning a tensor from ``predict``."""

    test_metrics: MetricCollection = MetricCollection(Accuracy(task="binary"))

    def forward(self, x: torch.Tensor):
        if isinstance(x, list):
            x = x[0]
        if isinstance(x, dict):
            x = next(iter(x.values()))
        x = 2 * x
        return x

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.forward(x)
        return {"prediction": x, "embedding": x}


class DummyTaskDict(DummyTaskTensor):
    """Task returning a dict from ``predict``."""

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"prediction": 3 * x, "extra": x.sum(dim=1)}


class NoMetricTask(DummyTaskDict):
    test_metrics: None = None  # type: ignore[assignment]


class DictDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor | None = None,
        transform: Any = None,
    ) -> None:
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):  # type: ignore[override, ty:invalid-method-override]
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        return {
            "image": x,
            "label": torch.tensor(1)
            if self.labels is None
            else self.labels[idx],
        }


class SimpleDataModule(LightningDataModule):
    """Tiny datamodule-like helper exposing train/val/test dataloaders."""

    def __init__(
        self,
        train_ds: DictDataset,
        val_ds: DictDataset,
        test_ds: DictDataset,
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


class SimpleL2Score(EmbeddingScore):
    """Test-only EmbeddingScore using pure torch cdist for NN distances."""

    ident = "simple-l2"
    train_required = False
    cal_required = False

    def __init__(self) -> None:
        super().__init__(pca=None)

    def _fit(self, q: float | bool = False) -> None:
        self.set_trained()
        self.scores = (
            torch.cdist(self.cal_embeddings, self.ref_embeddings)
            .min(dim=1)
            .values
        )
        self.set_calibrated()

    def _score(self, query: torch.Tensor | None):
        assert self.ref_embeddings is not None
        dists = torch.cdist(query, self.ref_embeddings)
        return dists.min(dim=1).values


class DummyScore(UncertaintyScore):
    """Minimal duck-typed score with select()."""

    def select(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        return {
            "score": torch.arange(b, dtype=x.dtype, device=x.device),
            "selected": torch.ones(b, dtype=torch.bool, device=x.device),
        }

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return x  # pragma: no cover

    def fit(
        self,
        ref: torch.Tensor | dict[str, torch.Tensor] | None = None,
        cal: torch.Tensor | dict[str, torch.Tensor] | None = None,
        *args,
        **kwargs,
    ) -> None:
        pass  # pragma: no cover
