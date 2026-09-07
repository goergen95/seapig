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
    EnergyClassWiseScore,
    EnergyScore,
    EntropyClassWiseScore,
    EntropyScore,
    LogitScore,
    MarginClassWiseScore,
    MarginScore,
    MutualInformationClassWiseScore,
    MutualInformationScore,
    PredictiveVarianceClassWiseScore,
    PredictiveVarianceScore,
    SoftmaxClassWiseScore,
    SoftmaxScore,
)
from seapig.scores.pca import PCAScore

__all__ = [
    "ClassWiseScore",
    "CosineClassWiseScore",
    "CosineScore",
    "EmbeddingScore",
    "EnergyClassWiseScore",
    "EnergyScore",
    "EntropyClassWiseScore",
    "EntropyScore",
    "EuclideanClassWiseScore",
    "EuclideanScore",
    "KNNScore",
    "LogitClassWiseScore",
    "LogitScore",
    "MahalanobisClassWiseScore",
    "MahalanobisScore",
    "MarginClassWiseScore",
    "MarginScore",
    "MutualInformationClassWiseScore",
    "MutualInformationScore",
    "PCAScore",
    "PredictiveVarianceClassWiseScore",
    "PredictiveVarianceScore",
    "RandomScore",
    "SoftmaxClassWiseScore",
    "SoftmaxScore",
    "UncertaintyScore",
]

try:
    from seapig.scores.pyod import PyODScore

    __all__ += ["PyODScore"]
except ImportError:  # pragma: no cover
    pass
