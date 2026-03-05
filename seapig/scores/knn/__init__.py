"""KNN-based scores for evaluating the quality of generated samples."""

from seapig.scores.knn.knn import (
    CosineScore,
    EuclideanScore,
    KNNScore,
    MahalanobisScore,
)

__all__ = ["KNNScore", "EuclideanScore", "CosineScore", "MahalanobisScore"]
