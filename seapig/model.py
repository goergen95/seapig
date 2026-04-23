"""Selective inference task wrapper.

Wraps a `LightningModule` and a `ConfidenceScore`
so that the model’s output is automatically combined with the score’s
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
    """Make a selective inference task by combining a `LightningModule` with a `ConfidenceScore`.

    This wrapper returns a selective inference task. It extends the `test_step()`
    and `predict_step()` of the `LightningModule` and automatically attaches
    selection results from the provided `ConfidenceScore`. Currently, the
    specified task is expected to be operate in inference mode (we enable inference
    mode if that is not the case). Combined with the `SelectiveMetric` class,
    this allows for easy evaluation of selective performance during testing and
    prediction with a `Trainer` (see Examples below). If a `RiskCoverageMetric` is
    provided, the wrapper will also compute and log risk‑coverage metrics during testing.

    Parameters
    ----------
    task : `LightningModule`
        A `LightningModule` whose `forward` method produces
        predictions.  The returned value may be a tensor or a mapping of
        tensors.
    score : `ConfidenceScore`
        A seapig `ConfidenceScore` object providing a `ConfidenceScore.select` method.
    input_key : input_keys, optional
        The key in the input batch dictionary corresponding to the model
        inputs, by default `"image"`.
    target_key : target_keys, optional
        The key in the input batch dictionary corresponding to the model
        targets, by default `"label"`.
    acc_test_outputs : bool, optional
        Whether to accumulate the outputs of the wrapped `task`’s `test_step()` method
        (with selection results attached). This is useful if you want to analyse
        the selection results of test samples. By default, this is set to `False`,
        meaning that the wrapper will log the metrics from the wrapped `task`’s
        `test_step()` method as usual.
        If set to `True`, the wrapper will accumulate the outputs of the wrapped `task`’s
        combined with te selection results to the `test_outputs` attribute,
        which can be accessed after testing is complete.

    Examples
    --------
    ```python
    import torch
    from lightning import Trainer, LightningModule
    from torchmetrics import Accuracy
    from seapig import SelectiveInferenceTask
    from seapig.scores import EuclideanScore

    # Define a simple LightningModule for demonstration
    class SimpleModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 2)
            self.test_metrics = Accuracy(task="binary")
        def forward(self, x):
            return self.layer(x)
        def embed(self, x):
            return self.layer(x)  # Just for demonstration; typically this would be a different embedding
        def test_step(self, batch, batch_idx):
            x, y = batch
            preds = self(x)
            self.test_metrics.update(preds, y)
            return {"predictions": preds}
        def predict_step(self, batch, batch_idx):
            x, _ = batch
            preds = self(x)
            return {"predictions": preds}

    # Create a confidence score and risk-coverage metric
    score = EuclideanScore()
    # Wrap the model with SelectiveInferenceTask
    selective_task = SelectiveInferenceTask(
        task=SimpleModel(),
        score=score,
        input_key="x",
        target_key="y",
        acc_test_outputs=True,  # Set to True to accumulate test outputs for analysis
    )
    # Create a Trainer and test the selective task
    trainer = Trainer()
    # return the test metrics and selection results in the test step outputs
    trainer.test(
        selective_task,
        test_dataloader=test_dataloader, # Replace with actual dataloader
    )
    # returns prediction with attached selection results
    trainer.predict(
        selective_task,
        dataloaders=predict_dataloader, # Replace with actual dataloader
    )
    ```
    """

    test_outputs: list[dict[str, Any]] | None = None

    def __init__(
        self,
        task: LightningModule,
        score: ConfidenceScore,
        input_key: INPUT_KEYS = "image",
        target_key: TARGET_KEYS = "label",
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
            self.test_metrics = SelectiveMetric(base=task.test_metrics)

        assert rc_metric is None or isinstance(rc_metric, RiskCoverageMetric), (
            "rc_metric must be a seapig RiskCoverageMetric instance or None"
        )
        self.rc_metric = rc_metric

        if acc_test_outputs:
            self.test_outputs = []

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the wrapped model and attach selection scores.

        The wrapped `task` is called first.  Its output is normalised to a
        `dict` under the key `"predictions"` when a plain tensor is
        returned.  The `ConfidenceScore.select` method is then evaluated on
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

        This method calls the wrapped task’s `test_step` method, but
        replaces the output predictions with those from this wrapper’s
        `forward` method.  This ensures that selection results are included
        in the output. If `rc_metric` is provided, the risk‑coverage metric is
        updated with the selection scores and predictions, and the current metric
        values are logged at each step. If `acc_test_outputs` is set to `True`,
        the outputs of the wrapped task’s `test_step()` method are combined with
        the selection results are accumulated in the `test_outputs` attribute for
        later analysis. Otherwise, the wrapper will log the metrics from the wrapped
        task’s `test_step()` method as usual.

        Parameters
        ----------
        batch : dict[str, Any]
            A batch of data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        dict[str, Any]
            The output of the wrapped task’s `test_step`, but with
            predictions replaced by those from this wrapper.
        """
        x = batch[self.input_key]
        y = batch[self.target_key]

        outputs = self.forward(x)

        if self.test_metrics is not None:
            self.test_metrics.update(
                outputs["predictions"], y, outputs["selected"]
            )
            self.log_dict(self.test_metrics.compute(), sync_dist=True)

        # Update risk‑coverage metric; final values are logged in on_test_epoch_end
        if self.rc_metric is not None:
            self.rc_metric.update(outputs["predictions"], y, outputs["score"])
            self.log_dict(self.rc_metric.compute(), sync_dist=True)

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

        This method calls the wrapped task’s `predict_step` method, but
        replaces the output predictions with those from this wrapper’s
        `forward` method.  This ensures that selection results are included
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
