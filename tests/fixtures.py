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


# Model missing the required method
class BadModelNoMethod(torch.nn.Module):
    def forward(self, x):
        return x  # pragma: no cover


# Model with wrong signature (no 'x' argument)
class BadModelWrongSig(torch.nn.Module):
    def embed(self):  # type: ignore[override]
        return torch.zeros(1, 2)  # pragma: no cover


class EmptyModel(torch.nn.Module):
    def embed(self, x):
        raise RuntimeError("Should not be called")  # pragma: no cover


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
        self.test_metrics = MetricCollection(Accuracy(task="binary"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x
        return x.view(x.shape[0], -1)


class MinimalEmbedding(EmbeddingScore):
    def __init__(self, pca: TensorPCA | None = None) -> None:
        super().__init__(pca=pca)
        self.train_required = False
        self.cal_required = False

    def _score(self, X: torch.Tensor) -> torch.Tensor:
        return X.sum(dim=1)

    def _fit(self, q: bool | float = False) -> None:
        pass  # pragma: no cover


class DummyTaskTensor(LightningModule):
    """Task returning a tensor from ``predict``."""

    test_metrics: MetricCollection = MetricCollection(Accuracy(task="binary"))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            x = x[0]
        if isinstance(x, dict):
            x = next(iter(x.values()))
        return 2 * x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return x  # pragma: no cover

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)


class DummyTaskDict(DummyTaskTensor):
    """Task returning a dict from ``predict``."""

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # ty: ignore[invalid-method-override]
        return {"predictions": 3 * x, "extra": x.sum(dim=1)}


class BadTask(DummyTaskTensor):
    def predict(self, x: torch.Tensor) -> list[torch.Tensor]:  # type: ignore[override, ty:invalid-method-override]
        return [x]  # wrong type


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

    def _score(self, X: torch.Tensor):
        assert self.ref_embeddings is not None
        dists = torch.cdist(X, self.ref_embeddings)
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
        X: torch.Tensor | None = None,
        Y: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> None:
        pass
