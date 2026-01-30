"""Confidence Scores supplied by seapig."""

from seapig.scores.base import ConfidenceScore, RandomScore
from seapig.scores.embed import EmbeddingScore
from seapig.scores.knn import (
    CosineScore,
    EuclideanScore,
    KNNScore,
    MahalanobisScore,
)
from seapig.scores.pca import PCAScore
from seapig.scores.pyod import PyODScore
from seapig.scores.risk_coverage import calculate_risk_coverage, normalize_metric
from seapig.scores.plot_utils import plot_risk_coverage, plot_density

__all__ = [
    "ConfidenceScore",
    "EmbeddingScore",
    "KNNScore",
    "RandomScore",
    "EuclideanScore",
    "CosineScore",
    "MahalanobisScore",
    "PyODScore",
    "PCAScore",
    "calculate_risk_coverage",
    "normalize_metric",
    "plot_risk_coverage",
    "plot_density",
]
