"""Uncertainty Scores supplied by seapig."""

from seapig.scores.base import RandomScore, UncertaintyScore
from seapig.scores.embed import EmbeddingScore
from seapig.scores.knn import (
    CosineScore,
    EuclideanScore,
    KNNScore,
    MahalanobisScore,
)
from seapig.scores.logits import (
    EnergyScore,
    EntropyScore,
    LogitScore,
    MarginScore,
    SoftmaxScore,
)
from seapig.scores.pca import PCAScore
from seapig.scores.residual import ResidualScore

__all__ = [
    "UncertaintyScore",
    "EmbeddingScore",
    "KNNScore",
    "LogitScore",
    "RandomScore",
    "EuclideanScore",
    "CosineScore",
    "MahalanobisScore",
    "PCAScore",
    "SoftmaxScore",
    "EnergyScore",
    "EntropyScore",
    "MarginScore",
    "ResidualScore",
]

try:
    from seapig.scores.pyod import PyODScore

    __all__ += ["PyODScore"]
except ImportError:  # pragma: no cover
    pass
