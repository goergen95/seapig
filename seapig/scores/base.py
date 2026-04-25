"""Base Classes for Confidence Scores."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from typing_extensions import override

from seapig.utils import get_logger

logger = get_logger(__name__)


class ConfidenceScore(torch.nn.Module, ABC):
    """Abstract Base Class for Confidence Scores.

    Confidence scores quantify the deviation of query samples from the training
    distribution. Low scores indicate likely inliers (samples similar to training),
    while high scores indicate likely outliers (samples deviating from training).
    Samples with scores exceeding the threshold are excluded from prediction.

    Attributes
    ----------
    trained : bool
        Whether the score has been trained. Defaults to `False`.
    train_required : bool
        Whether training is required before scoring. Defaults to `False`.
    cal_required : bool
        Whether calibration is required before selecting. Defaults to `False`.
    calibrated : bool
        Whether the score has been calibrated. Defaults to `False`.
    scores : torch.Tensor or None
        Confidence scores of the calibration samples. Low scores indicate
        likely inliers, high scores indicate likely outliers.
    threshold : torch.Tensor or None
        Rejection threshold. Samples with scores higher than this value are
        excluded from prediction.
    device : str
        Device to which internal tensors are put. Defaults to `"cpu"`.
    ident : str
        String identifying the confidence score implementation.

    See Also
    --------
    seapig.scores.knn.EuclideanScore : KNN-based score using Euclidean distance.
    seapig.scores.knn.CosineScore : KNN-based score using cosine distance.
    seapig.scores.pca.PCAScore : PCA reconstruction error score.
    seapig.scores.logits.SoftmaxScore : Softmax probability score.
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

    def set_threshold(self, q: float = 0.99) -> None:
        """Set a threshold based on a specific quantile on the available scores.

        Samples with scores higher than this threshold are excluded from prediction.

        Parameters
        ----------
        q : float
            Quantile in the interval `(0, 1)` used to compute the threshold
            from the stored calibration scores. Defaults to `0.99`.

        Raises
        ------
        ValueError
            If no calibration scores are available yet.
        """
        assert 0.0 < q < 1.0, "Quantile (q) must be between 0 and 1."
        q = float(q)
        if self.scores is None:
            raise ValueError(
                "Calibration scores (scores) must be available to set a threshold."
            )
        qval = torch.quantile(self.scores, q=q)
        self.threshold = qval
        self.set_calibrated()

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> None:
        """Fit a confidence score on training data.

        `X` is used as training samples to fit the underlying method,
        while `Y` is an optional parameter that can be used to compute
        reference scores for the decision threshold (calibration set).

        Subclasses define the exact parameter signatures and accepted
        input modes (precomputed tensors or model + DataLoader).
        """

    @abstractmethod
    def score(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Calculate the confidence score for a tensor of samples.

        Returns scores where low values indicate likely inliers and high values
        indicate likely outliers.
        """

    @abstractmethod
    def select(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Select samples for prediction based on their confidence score.

        Samples with scores lower than the threshold are selected for prediction,
        while samples with scores higher than the threshold are excluded.

        Returns
        -------
        dict[str, torch.Tensor]
            A dict with keys `'score'` (raw confidence scores) and
            `'selected'` (boolean selection mask).
        """

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
            rejected_calibration = int(np.sum(scores > threshold_value))
            rejected_query = (
                int(np.sum(q_scores > threshold_value))
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

    This score assigns a random float in `[0, 1]` to each sample.
    It is useful as a baseline or for testing purposes. Low scores
    indicate likely inliers, high scores indicate likely outliers.
    By default, the threshold is set to `0.99`, so approximately
    99% of samples are selected.

    See Also
    --------
    seapig.scores.base.ConfidenceScore : Abstract base class.
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
        raise NotImplementedError()

    @override
    @torch.inference_mode()
    def score(self, X: torch.Tensor) -> torch.Tensor:
        """Compute a random confidence score for every sample in a batch.

        Returns random scores where low values indicate likely inliers and
        high values indicate likely outliers.

        Parameters
        ----------
        X : torch.Tensor
            Input batch of shape `(B, ...)`. Only the batch size is used.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape `(B,)` with uniform random scores in `[0, 1]`.

        Examples
        --------
        ```python
        import torch
        from seapig.scores import RandomScore
        score = RandomScore()
        scores = score.score(torch.zeros(4, 10))
        ```
        """
        return torch.rand(X.shape[0])

    @override
    @torch.inference_mode()
    def select(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Select samples for prediction based on their random confidence score.

        Samples with scores lower than the threshold are selected for prediction.

        Parameters
        ----------
        X : torch.Tensor
            Input batch of shape `(B, ...)`. Only the batch size is used.

        Returns
        -------
        dict[str, torch.Tensor]
            A dict with keys `'score'` (random scores) and `'selected'`
            (boolean mask where `True` means the sample is selected).

        Examples
        --------
        ```python
        import torch
        from seapig.scores import RandomScore
        score = RandomScore()
        result = score.select(torch.zeros(4, 10))
        # result['selected'] is a boolean tensor of shape (4,)
        ```
        """
        if self.get_threshold() is None:
            logger.warning(
                "No threshold set. Trying to set it via `set_threshold()`."
            )
            self.set_threshold()
        assert self.threshold is not None
        score = self.score(X=X)
        return {"score": score, "selected": score < self.threshold}

    @override
    def set_threshold(self, q: float = 0.99) -> None:
        self.threshold = torch.Tensor([q])
