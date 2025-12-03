"""KNN-based confidence scores."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

import faiss
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from seapig.scores.base import EmbeddingScore


class KNNScore(EmbeddingScore, ABC):
    """Returns the KNN-distance to the nearest samples.

    Parameters
    ----------
    k:
        An `int`eger indicating the number of neighbors to calculate the distance.
        Defaults to 1, e.g. the distance to the closest neighbor.

    Attributes
    ----------
    embeddings:
        A `torch.Tensor` representing reference embeddings.
    scores:
        A `torch.Tensor` with the confidence scores of the calibration samples.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Defaults to `None`.
    """

    k: int = 1
    train_required: bool = True
    cal_required: bool = True
    threshold: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    index: faiss.IndexFlatL2  # type: ignore [no-any-unimported]

    def __init__(self, k: int = 1) -> None:
        super().__init__()
        self.k = k

    @override
    @torch.inference_mode()
    def score(
        self, batch: Tensor | dict[str, Tensor], model: torch.nn.Module | None
    ) -> Tensor:
        """Compute a confidence score for every sample in a batch.

        Once instantiated, the object can be called to return confidence
        scores based on a batch of inputs and a trained model:

        ```python
        my_score = KNNScore()
        scores = my_score.score(batch, model)
        ```

        Parameters
        ----------
        batch:
            A `dict` with the a subset of the following keys
            ["inputs", "masks", "weights", "labels", "outputs"] or a `torch.Tensor`.
        model:
            A torch.nn.Module representing a trained model.
        """
        assert isinstance(model, torch.nn.Module)
        assert callable(model.embed)
        assert self.embeddings is not None  # type: ignore [unreachable]
        assert self.index is not None

        if isinstance(batch, dict):
            z = model.embed(batch["image"])
        else:
            z = model.embed(batch)
        assert isinstance(z, Tensor)
        distance = self._distance(query=z)
        return distance

    def train(
        self,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> None:
        """Train a confidence score based on samples from a `torch.utils.data.DataLoader`.

        The train method retrieves embeddings for all samples from a DataLoader
        that is expected to represent training samples. These embeddings are
        later used to calculate the KNN-distances for query samples.

        ```python
        my_score = KNNScore(k=2)
        my_score.train(train_loader, model)
        ```

        Parameters
        ----------
        loader:
            DataLoader yielding training samples either as dict or Tensor.
        model:
            A trained model.
        """
        super().train(model, loader, outdir, prefix)
        assert self.embeddings is not None
        self._setup_index()
        self.scores = self._distance(self.embeddings, kpn=1)
        self.set_trained()

    @abstractmethod
    def _setup_index(self) -> None:
        """Prepare an index for KNN search."""
        pass

    @abstractmethod
    def _distance(self, query: torch.Tensor, kpn: int = 0) -> torch.Tensor:
        """Calculate the KNN distance of a query against a populated index."""
        pass


class EuclideanScore(KNNScore):
    """Returns the KNN-distance based on the euclidean distance to the nearest samples.

    Parameters
    ----------
    k:
        An `int`eger indicating the number of neighbors to calculate the distance.
        Defaults to 1, e.g. the distance to the closest neighbor.

    Attributes
    ----------
    embeddings:
        A `torch.Tensor` representing reference embeddings.
    scores:
        A `torch.Tensor` with the confidence scores of the calibration samples.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Defaults to `None`.
    """

    k: int

    def __init__(self, k: int = 1) -> None:
        super().__init__()
        self.k = k

    def _setup_index(self) -> None:
        """Initialize faiss index based on embeddings."""
        assert isinstance(self.embeddings, torch.Tensor)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings.cpu())

    def _distance(self, query: Tensor, kpn: int = 0) -> torch.Tensor:
        dist, _ = self.index.search(query.cpu(), k=self.k + kpn)
        return torch.Tensor(dist[:, kpn:].mean(1))


class CosineScore(KNNScore):
    """Returns the KNN-distance based on the cosine distance to the nearest samples.

    Parameters
    ----------
    k:
        An `int`eger indicating the number of neighbors to calculate the distance.
        Defaults to 1, e.g. the distance to the closest neighbor.
    abs:
        A `bool`ean indicating if the absolute cosine distance should be returned,
        by default `True`.

    Attributes
    ----------
    embeddings:
        A `torch.Tensor` representing reference embeddings.
    scores:
        A `torch.Tensor` with the confidence scores of the calibration samples.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Defaults to `None`.
    """

    k: int = 1
    abs: bool

    def __init__(self, k: int = 1, abs: bool = True) -> None:
        super().__init__()
        self.k = k
        self.abs = abs

    def _setup_index(self) -> None:
        """Initialize faiss index based on embeddings."""
        assert isinstance(self.embeddings, torch.Tensor)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(torch.nn.functional.normalize(self.embeddings).cpu())
        return

    def _distance(self, query: Tensor, kpn: int = 0) -> Tensor:
        query = torch.nn.functional.normalize(query).cpu()
        dist, _ = self.index.search(query, k=self.k + kpn)
        dist = torch.Tensor(dist)
        if self.abs:
            dist = dist.abs()
        return 1 - dist[:, kpn:].mean(1)


class MahalanobisScore(KNNScore):
    """Returns the Mahalanobis distance to the training samples distribution.

    Parameters
    ----------
    k:
        An `int`eger indicating the number of neighbors to calculate the distance.
        Defaults to 1, e.g. the distance to the closest neighbor.

    Attributes
    ----------
    embeddings:
        A `torch.Tensor` representing reference embeddings.
    scores:
        A `torch.Tensor` with the confidence scores of the calibration samples.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Defaults to `None`.
    """

    k: int
    vi_zero: torch.Tensor
    train_required: bool = True
    cal_required: bool = True
    threshold: torch.Tensor | None = None
    scores: torch.Tensor | None = None

    def __init__(self, k: int = 1) -> None:
        super().__init__()
        self.k = k

    def _setup_index(self) -> None:
        """Initialize faiss index based on embeddings."""
        assert isinstance(self.embeddings, torch.Tensor)
        cov_zero = self.embeddings.T.cov()
        self.vi_zero = torch.linalg.inv(torch.linalg.cholesky(cov_zero))
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add((self.embeddings @ self.vi_zero.T).cpu())

    def _distance(self, query: torch.Tensor, kpn: int = 0) -> torch.Tensor:
        query = query.cpu() @ self.vi_zero.T.cpu()
        dist, _ = self.index.search(query, k=self.k + kpn)
        return torch.Tensor(dist[:, kpn:].mean(1))
