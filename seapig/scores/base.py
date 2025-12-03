"""Base Classes for Confidence Scores."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seapig.scores.utils import (
    _load_parquet,
    _write_parquet,
    check_model,
    get_embeddings,
    setup_path,
)


class ConfidenceScore(ABC):
    """Abstract Base Class for Confidence Scores.

    Attributes
    ----------
    trained:
        A `bool`ean indicating if the score has been trained. Defaults to `FALSE`.
    train_required:
        A `bool`ean indicating if the score requires training. Defaults to `FALSE`.
    cal_required:
        A `bool`ean indicating if the score requires calibration. Defaults to `FALSE`.
    calibrated:
        A `bool`ean indicating if the score has been calibrated. Defaults to `FALSE`.
    scores:
        A `torch.Tensor` with the confidence scores of the calibration samples.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Defaults to `None`.
    device:
        A `str`ing indicating to which device internal `torch.Tensor`s are put.
        Default is `"cpu"`.
    """

    trained: bool = False
    train_required: bool = False
    calibrated: bool = False
    cal_required: bool = False
    scores: torch.Tensor | None = None
    threshold: torch.Tensor | None = None

    def __init__(self) -> None:
        return

    def requires_training(self) -> bool:
        """Return boolean indicating if the score requires training."""
        return self.train_required

    def requires_calibration(self) -> bool:
        """Return boolean indicating if the score requires calibration."""
        return self.cal_required

    def is_trained(self) -> bool:
        """Return boolean indicating if the score is already trained."""
        return self.trained

    def is_calibrated(self) -> bool:
        """Return boolean indicating if the score is already calibrated."""
        return self.calibrated

    def set_trained(self) -> None:
        """Set a boolean that the score is already trained."""
        self.trained = True

    def set_calibrated(self) -> None:
        """Set a boolean that the score is already calibrated."""
        self.calibrated = True

    def set_threshold(self, q: float = 0.99) -> None:
        """Set a threshold based on a specific quantile on the available scores."""
        ...

    def get_threshold(self) -> torch.Tensor | None:
        """Get the current threshold value."""
        return self.threshold

    @abstractmethod
    @torch.inference_mode()
    def score(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
        model: torch.nn.Module | None,
    ) -> torch.Tensor:
        """Compute a confidence score for every sample in a batch."""
        ...

    @abstractmethod
    @torch.inference_mode()
    def train(
        self,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> None:
        """Train a confidence score based on samples from a `torch.utils.data.DataLoader`."""
        pass

    @torch.inference_mode()
    def calibrate(
        self,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
        model: torch.nn.Module,
        q: float = 0.99,
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> None:
        """Calibrates a rejection threshold based on samples from a `torch.utils.data.DataLoader`.

        Iterates over the loader to calculate confidence scores for the samples
        yielded by the `DataLoader`. These are used to fix the rejection threshold
        for future query samples. For most cases, it is expected that the samples
        represent the validation split.

        ```python
        my_score = ConfidenceScore(k=2)
        my_score.calibrate(val_loader, model)
        ```

        Parameters
        ----------
        q:
            A `float` indicating the quantile of confidence scores on the
            calibration samples to set the rejection threshold to. Inputs with scores
            below or equal to the threshold value will be accepted for model prediction.
            Inputs with scores above the threshold will be rejected. Defaults to `0.99`.
        model:
            A `torch.nn.Module` representing a trained model instance. It is
            required to have an `.embed()` method.
        loader:
            A `DataLoader`` returning a `torch.Tensor` or a `dict` of `torch.Tensor`s.
        outdir:
            A `pathlib.Path` object pointing towards a directory, by default None.
        prefix:
            A `str`ing used as filename prefix to save embeddings, by default
            None.
        """
        if not self.requires_calibration():
            pass
        check_model(model)
        path = setup_path(outdir, prefix)
        assert isinstance(loader, DataLoader)

        if path is not None and path.is_file():
            self.scores = _load_parquet(path)
            self.threshold = self.scores.float().quantile(q=q)
            self.set_calibrated()
            return

        scores_ls: list[torch.Tensor] = []
        n_batches = len(loader)
        pbar = tqdm(
            total=n_batches,
            desc=f"Embedding {n_batches} batches",
            unit="batches",
        )
        for batch in loader:
            scores = self.score(batch=batch, model=model)
            scores_ls.append(scores)
            _ = pbar.update(n=1)
        self.scores = torch.cat(scores_ls, dim=0)
        if path is not None:
            _write_parquet(embeddings=self.scores, path=path)
        self.threshold = self.scores.float().quantile(q=q)
        self.set_calibrated()

    @torch.inference_mode()
    def select(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Select samples for prediction based on their confidence score.

        Samples are selected for prediction based on their confidence score compared
        to a threshold. It is expected that the threshold was previously
        calibrated on, e.g. validation samples.

        ```python
        my_score = ConfidenceScore()
        scores = my_score.select(batch, model)
        ```

        Parameters
        ----------
        model:
            A `torch.nn.Module` representing a trained model instance. It is
            required to have an `.embed()` method.
        batch:
            A batch of samples in the form of a `torch.Tensor` or a `dict` of `torch.Tensor`s
            with an `"image"` key.

        """
        if self.threshold is None:
            print(
                "Threshold has not been set. Trying to set it via `set_threshold()`."
            )
            self.set_threshold()
        assert self.threshold is not None
        score = self.score(batch=batch, model=model)
        return {"score": score, "selected": score < self.threshold}


class EmbeddingScore(ConfidenceScore, ABC):
    """Base Class for Embedding-based Confidence Scores.

    Attributes
    ----------
    embeddings:
        A `torch.Tensor` with the embeddings of trainings samples. Defaults to `None`.
    scores:
        A `torch.Tensor` with the confidence scores of the calibration samples.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Defaults to `None`.
    """

    embeddings: torch.Tensor | None
    train_required: bool = True
    cal_required: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = None

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
        my_score = EmbeddingScore(k=2)
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
        check_model(model)
        path = setup_path(outdir, prefix)
        assert isinstance(loader, DataLoader)
        self.embeddings = get_embeddings(model=model, loader=loader, path=path)

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
        self.threshold = self.scores.float().quantile(q=q)


class RandomScore(ConfidenceScore):
    """Returns random confidence scores per sample.

    This score returns a random float in the range `[0,1]` for each sample
    in a batch. By default, it selects samples below values of `0.99`.

    Examples
    --------
    ```{python}
    import torch
    from seapig import RandomScore
    my_score = RandomScore()
    batch = {"image": torch.rand(4)}
    my_score.score(batch)
    ```
    """

    train_required: bool = False
    cal_required: bool = False
    threshold: torch.Tensor | None = torch.Tensor([0.099])

    def train(
        self,
        model: torch.nn.Module | None = None,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]]
        | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> None:
        """Unused."""
        pass

    def score(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """Compute a confidence score for every sample in a batch.

        Once instantiated, the object can be called to return confidence
        scores based on a batch of inputs and a trained model:

        ```python
        my_score = RandomScore()
        scores = my_score(batch)
        ```

        Parameters
        ----------
        batch:
            A `dict` with the a subset of the following keys
            ["image", "masks", "weights", "labels", "outputs"] or a `torch.Tensor`.
        model:
            Unused, because the scores are random.
        """
        if isinstance(batch, dict):
            return torch.rand(batch["image"].shape[0])
        return torch.rand(batch.shape[0])

    @override
    def set_threshold(self, q: float = 0.99) -> None:
        self.threshold = torch.Tensor([q])
