"""Uncertainty Scores supplied by seapig."""

from seapig.scores.base import RandomScore, UncertaintyScore
from seapig.scores.classwise import ClassWiseScore
from seapig.scores.embed import EmbeddingScore
from seapig.scores.knn import (
    CosineClassWiseScore,
    CosineScore,
    EuclideanClassWiseScore,
    EuclideanScore,
    KNNScore,
    MahalanobisClassWiseScore,
    MahalanobisScore,
)
from seapig.scores.logits import (
    EnergyScore,
    EntropyScore,
    LogitScore,
    MarginScore,
    MutualInformationScore,
    PredictiveVarianceScore,
    SoftmaxScore,
)
from seapig.scores.pca import PCAScore

__all__ = [
    "ClassWiseScore",
    "CosineClassWiseScore",
    "CosineScore",
    "EmbeddingScore",
    "EnergyScore",
    "EntropyScore",
    "EuclideanClassWiseScore",
    "EuclideanScore",
    "KNNScore",
    "LogitScore",
    "MahalanobisClassWiseScore",
    "MahalanobisScore",
    "MarginScore",
    "MutualInformationScore",
    "PCAScore",
    "PredictiveVarianceScore",
    "RandomScore",
    "SoftmaxScore",
    "UncertaintyScore",
]

try:
    from seapig.scores.pyod import PyODScore

    __all__ += ["PyODScore"]
except ImportError:  # pragma: no cover
    pass
