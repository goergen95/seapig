# Copyright (c) seapig Contributors. All rights reserved.
# Licensed under the MIT License.

"""seapig: Confidence based selection of compatible inputs.

This library supplies classes and methods for latent-space analysis used
to derive confidence scores to be applied for selective prediction systems.
"""

__author__ = "Darius A. Görgen"
__version__ = "0.0.1"

from seapig.model import SelectiveModel
from seapig.scores.base import RandomScore
from seapig.scores.knn import CosineScore, EuclideanScore, MahalanobisScore
from seapig.scores.pyod import PyODScore

__all__ = [
    "SelectiveModel",
    "RandomScore",
    "EuclideanScore",
    "CosineScore",
    "MahalanobisScore",
    "PyODScore",
]
