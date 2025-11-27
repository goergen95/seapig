"""KNN-based confidence scores."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

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

        if isinstance(batch, dict):
            z = model.embed(batch["image"].to(device=model.device))
        else:
            z = model.embed(batch).to(device=model.device)
        assert isinstance(z, Tensor)
        distance = self._distance(
            query=z, reference=self.embeddings, device=model.device
        )
        return distance

    def _distance(
        self, query: Tensor, reference: Tensor, device: str
    ) -> Tensor:
        dists = self._dist_fun(z1=query, z2=reference, device=device)
        knn_dist = torch.topk(input=dists, k=self.k, largest=False, dim=1)
        return knn_dist.values.mean(dim=1)

    @abstractmethod
    def _dist_fun(self, z1: Tensor, z2: Tensor, device: str) -> Tensor:
        """Calculate the distance between two matrices of embeddings."""
        ...

    @override
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
        self.scores = self._dist_fun(
            z1=self.embeddings, z2=self.embeddings, device=model.device
        )
        self.set_trained()
        self.set_threshold()
        return

    @override
    def set_threshold(self, q: float = 0.99) -> None:
        """Set a threshold based on quantiles on the available confidence scores.

        This method sets the selection threshold based on the quantile on
        the values found in the `scores` attribute. If the confidence score
        is trained, but uncalibrated, this will be based on the K nearest
        neighbors of the training samples, excluding the distance to the
        point itself. If calibrated, the distance of the calibration samples to
        the K-closest training samples are used.

        Parameters
        ----------
        q:
            A `float` indicating the quantile of confidence scores of the
            samples to set the rejection threshold to.
        """
        assert self.is_trained()
        assert self.scores is not None
        with torch.amp.autocast(str(self.scores.device)):
            if not self.is_calibrated():
                scores = torch.topk(
                    input=self.scores, k=self.k + 1, largest=False
                ).values
                self.threshold = scores[:, 1:].float().quantile(q=q)
            else:
                self.threshold = self.scores.float().quantile(q=q)
        return


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

    @override
    def _dist_fun(self, z1: Tensor, z2: Tensor, device: str) -> Tensor:
        with torch.amp.autocast(str(device)):
            dist = torch.cdist(x1=z1, x2=z2, p=2)
        return dist


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

    @override
    def _dist_fun(self, z1: Tensor, z2: Tensor, device: str) -> Tensor:
        with torch.amp.autocast(str(device)):
            sim = pairwise_cosine_similarity(x=z1, y=z2, reduction=None)
        if self.abs:
            sim = sim.abs()
        dist = 1 - sim
        return dist


class PNormScore(KNNScore):
    """Returns the KNN-distance based on the p-norm distance to the nearest training samples.

    Parameters
    ----------
    k:
        An `int`eger indicating the number of neighbors to calculate the distance.
        Defaults to 1, e.g. the distance to the closest neighbor.
    p:
       A `float` indicating the p-norm to calculate via `torch.cdist`, by default 2.0,
       which is equivalent to the euclidean distance.

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
    p: float

    def __init__(self, k: int = 1, p: float = 2) -> None:
        super().__init__()
        self.k = k
        self.p = p

    @override
    def _dist_fun(self, z1: Tensor, z2: Tensor, device: str) -> Tensor:
        with torch.amp.autocast(str(device)):
            dists = torch.cdist(x1=z1, x2=z2, p=self.p)
        return dists
