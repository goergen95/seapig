# Copyright (c) seapig Contributors. All rights reserved.
# Licensed under the MIT License.

"""seapig: Confidence based selection of compatible inputs.

This library supplies classes and methods for latent-space analysis used
to derive confidence scores to be applied for selective prediction systems.
"""

__author__ = "Darius A. Görgen"
__version__ = "0.2.0.dev0"

import logging

from seapig import scores, utils
from seapig.metric import RiskCoverageMetric, SelectiveMetric
from seapig.model import SelectiveInferenceTask
from seapig.risk import RiskCoverage

logging.getLogger("seapig").addHandler(logging.NullHandler())


__all__ = [
    "RiskCoverage",
    "RiskCoverageMetric",
    "SelectiveInferenceTask",
    "SelectiveMetric",
    "scores",
    "utils",
]
