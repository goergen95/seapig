"""Base Classes for Confidence Scores."""

from abc import ABC, abstractmethod
from typing import Any, override

import numpy as np
import torch


class ConfidenceScore(torch.nn.Module, ABC):  # type: ignore[misc]
    """Abstract Base Class for Confidence Scores.

    Confidence scores quantify the deviation of query samples from the training
    distribution. Low scores indicate likely inliers (samples similar to training),
    while high scores indicate likely outliers (samples deviating from training).
    Samples with scores exceeding the threshold are excluded from prediction.

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
        Low scores indicate likely inliers, high scores indicate likely outliers.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Samples with scores higher
        than this threshold are excluded from prediction. Defaults to `None`.
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
    scores: torch.Tensor | None
    threshold: torch.Tensor | None
    ident: str

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("threshold", None)
        self.register_buffer("scores", None, persistent=False)

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
    def set_threshold(self, q: float = 0.99) -> None:
        """Set a threshold based on a specific quantile on the available scores.

        Samples with scores higher than this threshold are excluded from prediction.
        """
        pass

    @abstractmethod
    def fit(
        self, X: torch.Tensor, Y: torch.Tensor | None, *args: Any, **kwargs: Any
    ) -> None:
        """Fit a confidence score.

        Here, `X` is used as training samples to fit the downstream method,
        while `Y` as an optional parameter that can be used to calculate
        reference scores for the decision threshold.
        """
        pass

    @abstractmethod
    def score(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Calculate the confidence score for a tensor of samples.

        Returns scores where low values indicate likely inliers and high values
        indicate likely outliers.
        """
        pass

    @abstractmethod
    def select(
        self, X: torch.Tensor, *args: Any, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Select samples for prediction based on their confidence score.

        Samples with scores lower than the threshold are selected for prediction,
        while samples with scores higher than the threshold are excluded.
        """
        pass

    def plot(
        self, query_scores: torch.Tensor | None = None, bins: int = 100
    ) -> None:
        """Plot densities for confidence scores.

        By default, this method plots densities for the confidence scores.
        Optionally, it can also plot densities for `query_scores`.

        Parameters
        ----------
        query_scores:
            A `torch.Tensor` representing query scores to include in the plot. Defaults to `None`.
        bins:
            An `int` indicating the number of bins to use for density estimation. Defaults to `100`.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is not installed. Please install it with `pip install matplotlib`."
            )
        assert self.scores is not None, (
            "Calibration scores (scores) must be available to plot."
        )

        # Convert tensors to numpy arrays for plotting
        scores = self.scores.cpu().numpy()
        q_scores = (
            query_scores.cpu().numpy() if query_scores is not None else None
        )

        # Flatten embeddings for density plotting
        scores = scores.flatten()
        q_scores = q_scores.flatten() if q_scores is not None else None

        # Define a function to compute density
        def compute_density(
            data: np.ndarray, bins: int = 100
        ) -> tuple[np.ndarray, np.ndarray]:
            density, edges = np.histogram(data, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            return centers, density

        # Calculate rejected samples
        if self.threshold is not None:
            threshold_value = self.threshold.cpu().item()
            rejected_calibration = np.sum(scores > threshold_value)
            rejected_query = (
                np.sum(q_scores > threshold_value)
                if q_scores is not None
                else 0
            )
        else:
            rejected_calibration = 0
            rejected_query = 0

        total_calibration = len(scores)
        total_query = len(q_scores) if q_scores is not None else 0

        # Initialize the plot
        plt.figure(figsize=(10, 6))

        # Plot reference embeddings density
        centers, density = compute_density(scores, bins=bins)
        plt.fill_between(
            centers,
            density,
            alpha=0.5,
            color="steelblue",
            label=f"Calibration Scores (N={total_calibration}, Rejected={rejected_calibration})",
        )

        # Plot query embeddings density if provided
        if q_scores is not None:
            query_centers, query_density = compute_density(q_scores, bins=bins)
            plt.fill_between(
                query_centers,
                query_density,
                alpha=0.5,
                color="darkorange",
                label=f"Query Scores (N={total_query}, Rejected={rejected_query})",
            )

        # Add a vertical line for the threshold if it exists
        if self.threshold is not None:
            plt.axvline(
                x=self.threshold.cpu().item(),
                color="black",
                linestyle="--",
                label=f"Threshold ({self.threshold.cpu().item():.2f})",
            )

        # Add labels and legend
        plt.title("Confidence Score Densities")
        plt.xlabel("Confidence Score")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        # Show the plot
        plt.show()


class RandomScore(ConfidenceScore):
    """Returns random confidence scores per sample.

    This score returns a random float in the range `[0,1]` for each sample
    in a batch. Low scores indicate likely inliers, high scores indicate likely
    outliers. By default, samples with scores below 0.99 are selected for prediction.

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
    ident: str = "random"

    def __init__(self) -> None:
        super().__init__()
        self.set_threshold(q=0.99)

    @override
    def fit(
        self, X: torch.Tensor | None = None, Y: torch.Tensor | None = None
    ) -> None:
        """Unused."""
        pass

    @override
    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def score(self, X: torch.Tensor) -> torch.Tensor:
        """Compute a confidence score for every sample in a batch.

        Returns random scores where low values indicate likely inliers and
        high values indicate likely outliers.

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
        return torch.rand(X.shape[0])

    @override
    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def select(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Select samples for prediction based on their confidence score.

        Samples with scores lower than the threshold are selected for prediction.

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
