"""Confidence score based on an arbitrary PyOD model."""

from pathlib import Path

import torch
from pyod.models.base import BaseDetector
from torch.utils.data import DataLoader

from seapig.scores import EmbeddingScore


class PyODScore(EmbeddingScore):
    """Confidence Scores based on detectors supplied by PyOD.

    Parameters
    ----------
    detector:
        An `BaseDetector` instance from PyOD.
    scores:
        A `torch.Tensor` with the confidence scores of the calibration samples.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Defaults to `None`.

    Attributes
    ----------
    embeddings:
        A `torch.Tensor` representing reference embeddings.
    """

    trained: bool = False
    train_required: bool = True
    calibrated: bool = False
    cal_required: bool = True
    detector: BaseDetector

    def __init__(self, detector: BaseDetector) -> None:
        self.detector = detector

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
        self.detector.fit(self.embeddings.cpu().numpy())
        self.set_trained()

    @torch.inference_mode()
    def score(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
        model: torch.nn.Module | None,
    ) -> torch.Tensor:
        """Compute a confidence score for every sample in a batch.

        Once instantiated, the object can be called to return confidence
        scores based on a batch of inputs and a trained model:

        ```python
        my_score = PyODScore()
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
            z = model.embed(batch["image"])
            device = str(batch["image"].device)
        else:
            z = model.embed(batch)
            device = str(batch.device)
        assert isinstance(z, torch.Tensor)
        score = self.detector.decision_function(z.cpu().numpy())
        return torch.Tensor(score).to(device=device)
