"""Selective Model Class."""

from typing import override

import torch

from seapig.scores.base import ConfidenceScore


class SelectiveModel(torch.nn.Module):
    """Wrap a model to apply selection during inference."""

    model: torch.nn.Module
    csf: ConfidenceScore

    def __init__(
        self, model: torch.nn.Module, confidence_score: ConfidenceScore
    ) -> None:
        super().__init__()
        self.model = model
        self.csf = confidence_score

    @override
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        preds: torch.Tensor | dict[str, torch.Tensor] = self.model(x)
        if isinstance(preds, torch.Tensor):
            preds = {"predictions": preds}
        assert isinstance(preds, dict)
        scores: dict[str, torch.Tensor] = self.csf.select(
            batch=x, model=self.model
        )
        return preds | scores
