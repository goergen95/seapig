"""Selective inference task wrapper.

Wraps a ``pytorch_lightning.LightningModule`` and a ``seapig`` confidence
score so that the model’s output is automatically combined with the score’s
selection results.
"""

from typing import Any, Literal, get_args

import torch
from pytorch_lightning import LightningModule
from torchmetrics import Metric, MetricCollection

from seapig.metric import RiskCoverageMetric, SelectiveMetric
from seapig.risk_coverage import RiskCoverage
from seapig.scores.base import ConfidenceScore

INPUT_KEYS = Literal["image", "input", "images", "inputs", "x"]
TARGET_KEYS = Literal[
    "mask", "label", "masks", "labels", "targets", "target", "y", "y_true"
]


class SelectiveInferenceTask(LightningModule):  # type: ignore[misc]
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
    """

    def __new__(cls, *args, **kwargs) -> LightningModule:
        # Create the instance of the class
        instance = super().__new__(cls)
        # Initialize the instance
        instance.__init__(*args, **kwargs)
        # Return the modified task object instead of the instance
        return instance.task

    def __init__(
        self,
        task: LightningModule,
        score: ConfidenceScore,
        input_key: INPUT_KEYS = "image",
        target_key: TARGET_KEYS = "label",
        rc_metric: RiskCoverageMetric | None = None,
    ) -> None:
        super().__init__()
        assert callable(getattr(task, "embed", None)), (
            "Wrapped task must have an embed() method"
        )
        assert hasattr(task, "test_metrics"), (
            "Wrapped task must have test_metrics"
        )
        assert isinstance(task.test_metrics, (MetricCollection, Metric)), (
            "Wrapped task's test_metrics must be a Metric or MetricCollection"
        )
        self.task = task
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

        self.test_metrics = SelectiveMetric(
            base=task.test_metrics,
            prediction_key="predictions",
            selection_key="selected",
        )

        # self.rc_metric = rc_metric or RiskCoverageMetric(
        #    risk="generalized",
        #    n_bins=100,
        #    prediction_key="predictions",
        #    score_key="score",
        # )
        self._org_forward = task.forward
        self.task.forward = self.forward
        self.task.test_step = self.test_step
        self.task.predict_step = self.predict_step
        # self.task.get_risk_coverage_curve = self.get_risk_coverage_curve

    @torch.inference_mode()  # type: ignore[untyped-decorator]
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
        preds = self._org_forward(x)
        if isinstance(preds, torch.Tensor):
            preds = {"predictions": preds}
        assert isinstance(preds, dict)

        embs: torch.Tensor = self.task.embed(x)
        selection = self.score.select(embs)

        return preds | selection

    @torch.inference_mode()  # type: ignore[untyped-decorator]
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
        batch_size = x.shape[0]

        outputs = self.forward(x)

        # Update metrics:
        self.test_metrics(outputs, y)
        self.task.log_dict(self.test_metrics, batch_size=batch_size)

        # Update risk‑coverage metric and log AUCs
        # self.rc_metric(outputs, y)
        # self.log_dict(self.rc_metric, batch_size=batch_size)

    @torch.inference_mode()  # type: ignore[untyped-decorator]
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

    # def get_risk_coverage_curve(self) -> RiskCoverage | None:
    #    """Return the latest computed risk‑coverage curve (if any)."""
    #    return self.rc_metric.get_curve()
