"""Selective inference wrapper combining a LightningModule with a UncertaintyScore.

This module provides SelectiveInferenceTask, a thin wrapper that runs a pre-trained
LightningModule in inference mode, computes a uncertainty score from the
model's embeddings using a UncertaintyScore, and returns predictions augmented with
selection results. The wrapper can also update and log selective metrics during
testing.
"""

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from lightning import LightningModule
from torchmetrics import Metric, MetricCollection

from seapig.metric import RiskCoverageMetric, SelectiveMetric
from seapig.risk import RiskCoverage
from seapig.scores import UncertaintyScore

INPUT_KEYS = Literal["image", "input", "images", "inputs", "x"]
TARGET_KEYS = Literal[
    "mask", "label", "masks", "labels", "targets", "target", "y", "y_true"
]


class SelectiveInferenceTask(LightningModule):
    """Wrap a trained `LightningModule` to attach selection results during inference.

    The wrapper calls the wrapped model in inference mode and combines its
    predictions with selection outputs produced by a provided `UncertaintyScore`.

    Examples
    --------
    ```python
    from seapig import SelectiveInferenceTask
    from seapig.scores import EuclideanScore

    score = EuclideanScore()
    # score.fit(X=train_embeddings)  # fit before wrapping
    selective_task = SelectiveInferenceTask(task=model, score=score)
    ```
    """

    test_outputs: list[dict[str, Any]] | None = None

    def __init__(
        self,
        task: LightningModule,
        score: UncertaintyScore,
        acc_test_outputs: bool = False,
        rc_metric: RiskCoverageMetric | None = None,
    ) -> None:
        """Create a SelectiveInferenceTask.

        Parameters
        ----------
        task: LightningModule
            A trained LightningModule that provides a `forward(x)` method returning
            a dict with predictions and any required fields for the score (e.g.
            `embedding` or `logit``). The task is deep‑copied and set to eval mode
            to avoid side-effects during inference.
        score: UncertaintyScore
            An instance implementing `select` to compute a selection mask. It can
            be a generic `UncertaintyScore` or one of the concrete subclasses
            `EmbeddingScore` / `LogitScore` which operate on specific model
            outputs.
        acc_test_outputs: bool, default `False``
            If `True` the per-batch outputs (predictions merged with selection
            results) are stored in `self.test_outputs` for later inspection.
        rc_metric: RiskCoverageMetric | None, optional
            Optional metric to track risk-coverage during testing. If supplied it
            will be updated on each test step and the final curve can be retrieved
            via :meth:`get_risk_coverage_curve`.
        """
        super().__init__()
        self.task = copy.deepcopy(task)
        self.task.eval()

        assert isinstance(score, UncertaintyScore), (
            "score must be a seapig UncertaintyScore instance"
        )
        if not hasattr(self.task, "predict") and not callable(
            self.task.predict
        ):
            raise TypeError(
                "`task` is required to expose a `predict()` method."
            )
        self.score = score

        task_metric = getattr(task, "test_metrics", None)
        self.test_metrics = None
        if task_metric is not None:
            if not isinstance(task_metric, (MetricCollection, Metric)):
                raise TypeError(
                    "Wrapped task's test_metrics must be a Metric or MetricCollection"
                )
            self.test_metrics = SelectiveMetric(base=task_metric)

        if rc_metric is not None and not isinstance(
            rc_metric, RiskCoverageMetric
        ):
            raise TypeError(
                "rc_metric must be a seapig RiskCoverageMetric instance or None."
            )
        self.rc_metric = rc_metric
        self.test_outputs = [] if acc_test_outputs else None

    @torch.inference_mode()
    def forward(
        self, batch: Mapping[str, Any] | Sequence[Any] | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Run the wrapped model and attach selection results.

        Parameters
        ----------
        batch :
            Input batch passed directly to the task's `predict` method.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the model's predictions merged with the
            selection outputs produced by the configured `UncertaintyScore`.
            The selection dictionary always includes `'score'` (the raw
            uncertainty values) and `'selected'` (a boolean mask). If the wrapped
            model returns a `torch.Tensor` instead of a mapping, it is wrapped
            under the `'prediction'` key before merging.
        """
        assert callable(self.task.predict)
        outputs = self.task.predict(batch)
        if not isinstance(outputs, dict):
            raise TypeError(
                f"Wrapped task must return a dict, got {type(outputs).__name__}"
            )
        selection = self.score.select(query=outputs)
        return outputs | selection

    @torch.inference_mode()
    def test_step(
        self,
        batch: Mapping[str, Any] | Sequence[Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Execute a test step, updating metrics with selection information.

        Parameters
        ----------
        batch : Mapping[str, Any] | Sequence[Any]
            A batch from the test DataLoader. It is passed directly to the wrapped
            model's `predict` method via `self.forward`.
        batch_idx : int
            Index of the current batch (required by Lightning but not used here).
        dataloader_idx : int, optional
            Index of the DataLoader when multiple loaders are used. Defaults to `0`.

        Notes
        -----
        The method runs `self.forward` to obtain predictions together with the
        `score` and `selected` masks produced by the configured
        `UncertaintyScore`. It then updates any attached `SelectiveMetric` and
        `RiskCoverageMetric` (if supplied) and optionally records the full
        selection dict in `self.test_outputs` when `acc_test_outputs=True`.
        Metrics are logged via Lightning's `log_dict` mechanism.
        """
        selection = self.forward(batch)
        if "prediction" not in selection:
            raise KeyError(
                "`test_step()` of SelectiveInferenceTask score requires `prediction` key in output of `predict()`."
            )
        if "label" not in selection:
            raise KeyError(
                "`test_step()` of SelectiveInferenceTask score requires `label` key in output of `predict()`."
            )

        if self.test_metrics is not None:
            self.test_metrics.update(
                preds=selection["prediction"],
                target=selection["label"],
                selected=selection["selected"],
            )
            self.log_dict(self.test_metrics.compute(), sync_dist=True)

        # Update risk-coverage metric; final values are logged in on_test_epoch_end
        if self.rc_metric is not None:
            self.rc_metric.update(
                preds=selection["prediction"],
                target=selection["label"],
                scores=selection["score"],
            )
            self.log_dict(self.rc_metric.compute(), sync_dist=True)

        if self.test_outputs is not None:
            self.test_outputs.append(selection)

    def on_test_epoch_end(self) -> None:
        """Log final computed test metrics once at the end of testing."""
        if self.test_metrics is not None:
            self.log_dict(self.test_metrics.compute(), sync_dist=True)
        if self.rc_metric is not None:
            self.log_dict(self.rc_metric.compute(), sync_dist=True)

    @torch.inference_mode()
    def predict_step(
        self,
        batch: Mapping[str, Any] | Sequence[Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Perform prediction and return predictions with selection outputs.

        The wrapper calls `self.forward` and returns a mapping containing the
        model's `prediction` together with the `score` and `selected` masks.
        """
        selection = self.forward(batch)
        if "prediction" not in selection:
            raise KeyError(
                "`predict_step()` of SelectiveInferenceTask score requires `prediction` key in output of `predict()`."
            )
        keys = ["prediction", "score", "selected"]
        outputs = {k: v for k, v in selection.items() if k in keys}
        return outputs

    def get_risk_coverage_curve(
        self,
    ) -> RiskCoverage | dict[str, RiskCoverage] | None:
        """Return the latest computed risk-coverage curve(s), or None if not available."""
        if self.rc_metric is None:
            return None
        return self.rc_metric.get_curve()


def _get_from_batch(
    batch: Mapping[str, Any] | Sequence[Any], key: str | int | None, pos: int
) -> Any:
    """Return item by key or by positional index `pos` when key is None."""
    if key is None:
        if isinstance(batch, Sequence):
            return batch[pos]
        if isinstance(batch, Mapping):
            values = list(batch.values())
            return values[pos]
        raise TypeError("Unsupported batch type")
    if isinstance(batch, Mapping):
        assert isinstance(key, str)
        return batch[key]  # type: ignore[arg-type]
    else:
        assert isinstance(key, int)
        return batch[key]
    raise TypeError("Unsupported batch type")
