"""Base Classes for Confidence Scores."""

from abc import ABC, abstractmethod
from typing import Any, override

import torch


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
    ident:
        A `str`ing identifying the confidence score.
    """

    trained: bool = False
    train_required: bool = False
    calibrated: bool = False
    cal_required: bool = False
    scores: torch.Tensor | None = None
    threshold: torch.Tensor | None = None
    ident: str

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

    def get_threshold(self) -> torch.Tensor | None:
        """Get the current threshold value."""
        return self.threshold

    @abstractmethod
    def to(self, device: str | torch.device = "cpu") -> None:
        """Move all tensors to the specified device."""
        pass

    @staticmethod
    def _to(
        tensor: torch.Tensor | None, device: str | torch.device = "cpu"
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.to(device=device)

    @abstractmethod
    def set_threshold(self, q: float = 0.99) -> None:
        """Set a threshold based on a specific quantile on the available scores."""
        pass

    @abstractmethod
    def fit(self, X: Any, Y: Any | None, *args: Any, **kwargs: Any) -> None:
        """Fit a confidence score.

        Here, `X` is used as training samples to fit the downstream method,
        while `Y` as an optional parameter that can be used to calculate
        references scores for the decision threshold.
        """
        pass

    @abstractmethod
    def score(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Calculate the confidence score for an tensor of samples."""
        pass

    @abstractmethod
    def select(
        self, X: torch.Tensor, *args: Any, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Select samples for prediction based on their confidence score."""
        pass


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
    batch = torch.rand(4)
    my_score.score(batch)
    ```
    """

    train_required: bool = False
    cal_required: bool = False
    threshold: torch.Tensor | None = torch.Tensor([0.099])
    ident = "random"

    @override
    def to(self, device: str | torch.device = "cpu") -> None:
        self.threshold = self._to(self.threshold, device=device)

    @override
    def fit(
        self, X: torch.Tensor | None = None, Y: torch.Tensor | None = None
    ) -> None:
        """Unused."""
        pass

    @override
    @torch.inference_mode()
    def score(self, X: torch.Tensor) -> torch.Tensor:
        """Compute a confidence score for every sample in a batch.

        Once instantiated, the object can be called to return a tensor of
        random confidence scores based on a batch of inputs:

        ```python
        my_score = RandomScore()
        scores = my_score.score(batch)
        ```

        Parameters
        ----------
        X:
            A `torch.Tensor`.
        """
        self.to(device=X.device)
        return torch.rand(X.shape[0])

    @override
    @torch.inference_mode()
    def select(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Select samples for prediction based on their confidence score.

        Once instantiated, the object can be called to return a tensor of
        random confidence scores and selection decision based on a batch of inputs:

        ```python
        my_score = RandomScore()
        selection = my_score.select(batch)
        ```

        Parameters
        ----------
        X:
            A `torch.Tensor`.
        """
        if self.get_threshold() is None:
            print("Trying to set it via `set_threshold()`.")
            self.set_threshold()
        assert self.threshold is not None
        score = self.score(X=X)
        return {"score": score, "selected": score < self.threshold}

    @override
    def set_threshold(self, q: float = 0.99) -> None:
        self.threshold = torch.Tensor([q])
