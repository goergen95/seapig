"""Distribution based confidence Scores."""

from pathlib import Path
from typing import override

import torch
from torch.utils.data import DataLoader

from seapig.scores.base import EmbeddingScore


class MahalanobisScore(EmbeddingScore):
    """Returns the Mahalanobis distance to the training samples distribution.

    Attributes
    ----------
    embeddings:
        A `torch.Tensor` with the embeddings of trainings samples.
    scores:
            A `torch.Tensor` with the confidence scores of the calibration samples.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Defaults to `None`.
    """

    mu_zero: torch.Tensor
    cov_zero: torch.Tensor
    vi_zero: torch.Tensor
    train_required: bool = True
    cal_required: bool = True
    threshold: torch.Tensor | None = None
    scores: torch.Tensor | None = None

    @override
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
        my_score = MahalanobisScore()
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
        assert model is not None
        assert self.embeddings is not None
        assert callable(model.embed)

        if isinstance(batch, dict):  # type: ignore [unreachable]
            z = model.embed(batch["image"].to(device=model.device))
        else:
            z = model.embed(batch.to(device=model.device))
        assert isinstance(z, torch.Tensor)
        distance = self._distance(query=z, device=model.device)
        return distance

    def _distance(self, query: torch.Tensor, device: str) -> torch.Tensor:
        with torch.amp.autocast(str(device)):
            delta = query - self.mu_zero
            score = torch.diag(
                input=torch.sqrt(input=((delta @ self.vi_zero) @ delta.T))
            )
        return score

    @override
    def train(
        self,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> None:
        """Train a confidence score based on samples from a `torch.utils.data.DataLoader`.

        Iterates over the loader to extract embeddings for samples.
        These are later used to retrieve confidence scores for query samples.

        ```python
        my_score = MahalanobisScore()
        my_score.train(train_loader, model)
        ```

        Parameters
        ----------
        model:
            A torch.nn.Module representing a trained model instance. It is
            required to have an `.embed()` method.
        loader:
            A DataLoader returning a `torch.Tensor` or a `dict` of `torch.Tensor`s.
        outdir:
            A `pathlib.Path` object pointing towards a directory, by default None.
        prefix:
            A `str`ing used as filename prefix to save embeddings, by default
            None.
        """
        super().train(model=model, loader=loader, outdir=outdir, prefix=prefix)
        assert isinstance(self.embeddings, torch.Tensor)
        self.mu_zero = self.embeddings.mean(dim=0)
        self.cov_zero = self.embeddings.T.cov()
        self.vi_zero = torch.linalg.inv(self.cov_zero)
        self.scores = self._distance(
            query=self.embeddings, device=str(model.device)
        )
        self.set_trained()
        self.set_threshold()

    @override
    def set_threshold(self, q: float = 0.99) -> None:
        """Set a threshold based on quantiles on the available confidence scores.

        This method sets the selection threshold based on the quantile on
        the values found in the `scores` attribute. If the confidence score
        is trained, but uncalibrated, this will be based on the distances of
        the training samples. If calibrated, the distances of the calibration
        samples are used.

        Parameters
        ----------
        q:
            A `float` indicating the quantile of confidence scores of the
            samples to set the rejection threshold to.
        """
        assert self.is_trained()
        assert self.scores is not None
        self.threshold = self.scores.float().quantile(q=q)
