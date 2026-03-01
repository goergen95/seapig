"""Selective inference task wrapper.

Wraps a ``pytorch_lightning.LightningModule`` and a ``seapig`` confidence
score so that the model’s output is automatically combined with the score’s
selection results.
"""

import copy
from typing import Any, Literal, get_args

import torch
from lightning import LightningModule
from torchmetrics import Metric, MetricCollection

from seapig.metric import RiskCoverageMetric, SelectiveMetric
from seapig.risk_coverage import RiskCoverage
from seapig.scores.base import ConfidenceScore

INPUT_KEYS = Literal["image", "input", "images", "inputs", "x"]
TARGET_KEYS = Literal[
    "mask", "label", "masks", "labels", "targets", "target", "y", "y_true"
]


class SelectiveInferenceTask(LightningModule):
    """Wrap a LightningModule to attach a confidence‑score based selector.

    This task is designed to be compatible with ``torchgeo`` based tasks and
    models. It is meant to be used in inference mode only, so you are required
    to bring a trained model. This wrapper supplies extended  methods for
    the `test_step()` and `predict_step()` of the LightningModule that
    automatically attach selection results from the provided confidence score.
    Combined with the seapig `selectiveMetric`s, this allows for easy
    evaluation of selective performance during testing and prediction with a
    pytorch-lightning `Trainer`.

    Parameters
    ----------
    task : LightningModule
        The model (or any LightningModule) whose ``forward`` method produces
        predictions.  The returned value may be a tensor or a mapping of
        tensors.
    score : ConfidenceScore
        A ``seapig`` confidence‑score object providing a ``select`` method.
    input_key : input_keys, optional
        The key in the input batch dictionary corresponding to the model
        inputs, by default "image".
    target_key : target_keys, optional
        The key in the input batch dictionary corresponding to the model
        targets, by default "label".
    acc_test_outputs : bool, optional
        Whether to accumulate the outputs of the wrapped task’s ``test_step`` method
        (with selection results attached). This is useful if you want to analyse
        the selection results of test samples. By default, this is set to False,
        meaning that the wrapper will log the metrics from the wrapped task’s
        ``test_step`` method as usual.
        If set to True, the wrapper will accumulate the outputs of the wrapped task’s
        combined with te selection results to the ``test_outputs`` attribute,
        which can be accessed after testing is complete.
    """

    test_metrics: SelectiveMetric | None
    rc_metric: RiskCoverageMetric | None
    test_outputs: list[dict[str, Any]] | None = None

    def __init__(
        self,
        task: Any,
        score: ConfidenceScore,
        input_key: str = "image",
        target_key: str = "label",
        rc_metric: RiskCoverageMetric | None = None,
        acc_test_outputs: bool = False,
    ) -> None:
        super().__init__()
        assert callable(getattr(task, "embed", None)), (
            "Wrapped task must have an embed() method"
        )
        self.task = copy.deepcopy(task)
        self.task.eval()  # Ensure the wrapped task is in eval mode
        assert isinstance(score, ConfidenceScore), (
            "score must be a seapig ConfidenceScore instance"
        )
        self.score = score
        if input_key not in get_args(INPUT_KEYS):
            raise ValueError(
                f"input_key must be one of {get_args(INPUT_KEYS)}; got {input_key!r}"
            )
        self.input_key = input_key
        if target_key not in get_args(TARGET_KEYS):
            raise ValueError(
                f"target_key must be one of {get_args(TARGET_KEYS)}; got {target_key!r}"
            )
        self.target_key = target_key

        self.test_metrics = None
        if hasattr(task, "test_metrics") and task.test_metrics is not None:
            assert isinstance(task.test_metrics, (MetricCollection, Metric)), (
                "Wrapped task's test_metrics must be a Metric or MetricCollection"
            )
            self.test_metrics = SelectiveMetric(
                base=task.test_metrics,
                prediction_key="predictions",
                selection_key="selected",
            )

        assert rc_metric is None or isinstance(rc_metric, RiskCoverageMetric), (
            "rc_metric must be a seapig RiskCoverageMetric instance or None"
        )
        self.rc_metric = rc_metric

        if acc_test_outputs:
            self.test_outputs = []

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the wrapped model and attach selection scores.

        The wrapped ``task`` is called first.  Its output is normalised to a
        ``dict`` under the key ``"predictions"`` when a plain tensor is
        returned.  The confidence‑score ``select`` method is then evaluated on
        the same input batch, moved to the device of the model’s predictions,
        and merged with the prediction mapping.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the original predictions together with the
            selection scores.
        """
        preds = self.task(x)
        if isinstance(preds, torch.Tensor):
            preds = {"predictions": preds}
        assert isinstance(preds, dict)

        assert callable(self.task.embed)
        embs: torch.Tensor = self.task.embed(x)
        selection = self.score.select(embs)

        return preds | selection

    @torch.inference_mode()
    def test_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Perform a test step with selection results attached.

        This method calls the wrapped task’s ``test_step`` method, but
        replaces the output predictions with those from this wrapper’s
        ``forward`` method.  This ensures that selection results are included
        in the output.

        Parameters
        ----------
        batch : dict[str, Any]
            A batch of data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        dict[str, Any]
            The output of the wrapped task’s ``test_step``, but with
            predictions replaced by those from this wrapper.
        """
        x = batch[self.input_key]
        y = batch[self.target_key]

        outputs = self.forward(x)

        if self.test_metrics is not None:
            self.test_metrics.update(outputs, y)
            self.log_dict(self.test_metrics.compute())

        # Update risk‑coverage metric; final values are logged in on_test_epoch_end
        if self.rc_metric is not None:
            self.rc_metric.update(outputs, y)
            self.log_dict(self.rc_metric.compute())

        if self.test_outputs is not None:
            self.test_outputs.append(outputs)

    def on_test_epoch_end(self) -> None:
        """Log final computed test metrics once (avoid per-batch aggregation)."""
        if self.test_metrics is not None:
            self.log_dict(self.test_metrics.compute(), sync_dist=True)
        if self.rc_metric is not None:
            self.log_dict(self.rc_metric.compute(), sync_dist=True)

    @torch.inference_mode()
    def predict_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, torch.Tensor]:
        """Perform a predict step with selection results attached.

        This method calls the wrapped task’s ``predict_step`` method, but
        replaces the output predictions with those from this wrapper’s
        ``forward`` method.  This ensures that selection results are included
        in the output.

        Parameters
        ----------
        batch : dict[str, Any]
            A batch of data.
        batch_idx : int
            The index of the batch.
        dataloader_idx : int, optional
            The index of the dataloader, by default 0.

        Returns
        -------
        dict[str, torch.Tensor]
            The output of the wrapped task’s ``predict_step``, but with
            predictions replaced by those from this wrapper.
        """
        x = batch[self.input_key]
        outputs: dict[str, torch.Tensor] = self.forward(x)
        return outputs

    def get_risk_coverage_curve(self) -> RiskCoverage | None:
        """Return the latest computed risk‑coverage curve (if any)."""
        if self.rc_metric is None:
            return None
        return self.rc_metric.get_curve()
