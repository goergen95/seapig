"""Selective inference wrapper combining a LightningModule with a UncertaintyScore.

This module provides SelectiveInferenceTask, a thin wrapper that runs a pre-trained
LightningModule in inference mode, computes a uncertainty score from the
model's embeddings using a UncertaintyScore, and returns predictions augmented with
selection results. The wrapper can also update and log selective metrics during
testing.
"""

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Literal, get_args

import torch
from lightning import LightningModule
from torchmetrics import Metric, MetricCollection

from seapig.metric import RiskCoverageMetric, SelectiveMetric
from seapig.risk import RiskCoverage
from seapig.scores import EmbeddingScore, LogitScore, UncertaintyScore

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
        input_key: INPUT_KEYS | None = None,
        target_key: TARGET_KEYS | None = None,
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
            If `True` the per‑batch outputs (predictions merged with selection
            results) are stored in `self.test_outputs` for later inspection.
        input_key: INPUT_KEYS | None, optional
            Key or positional index used to extract the input tensor from a batch.
            `None` (default) selects the first element (position `0``). When a
            string is supplied it must be one of the literals defined in
            `INPUT_KEYS``.
        target_key: TARGET_KEYS | None, optional
            Similar to `input_key` but for the target/label tensor. `None``
            selects the second element (position `1``). Must be a member of the
            `TARGET_KEYS` literals when provided.
        rc_metric: RiskCoverageMetric | None, optional
            Optional metric to track risk‑coverage during testing. If supplied it
            will be updated on each test step and the final curve can be retrieved
            via :meth:`get_risk_coverage_curve`.
        """
        super().__init__()
        self.task = copy.deepcopy(task)
        self.task.eval()  # Keep the wrapped task in evaluation mode
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
        if input_key is not None and input_key not in get_args(INPUT_KEYS):
            raise ValueError(
                f"input_key must be one of {get_args(INPUT_KEYS)}; got {input_key!r}"
            )
        self.input_key = 0 if input_key is None else input_key
        if target_key is not None and target_key not in get_args(TARGET_KEYS):
            raise ValueError(
                f"target_key must be one of {get_args(TARGET_KEYS)}; got {target_key!r}"
            )
        self.target_key = 1 if target_key is None else target_key

        self.test_metrics: SelectiveMetric | None = None
        task_metric = getattr(task, "test_metrics", None)
        if task_metric is not None:
            assert isinstance(task_metric, (MetricCollection, Metric)), (
                "Wrapped task's test_metrics must be a Metric or MetricCollection"
            )
            self.test_metrics = SelectiveMetric(base=task_metric)

        self.rc_metric: RiskCoverageMetric | None = None
        if rc_metric is not None:
            assert isinstance(rc_metric, RiskCoverageMetric), (
                "rc_metric must be a seapig RiskCoverageMetric instance or None"
            )
            self.rc_metric = rc_metric

        # Initialize per‑batch output collection if requested
        if acc_test_outputs:
            self.test_outputs = []
        else:
            self.test_outputs = None

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the wrapped model and attach selection results.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor passed directly to the underlying `task` model.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the model's predictions merged with the
            selection outputs produced by the configured `UncertaintyScore``.
            The selection dictionary always includes `'score'` (the raw
            uncertainty values) and `'selected'` (a boolean mask). If the
            wrapped model returns a `torch.Tensor` instead of a mapping, it is
            wrapped under the `'prediction'` key before merging.
        """
        assert callable(self.task.predict)
        preds = self.task.predict(x)
        if isinstance(preds, torch.Tensor):
            preds = {"prediction": preds}
        if not isinstance(preds, dict):
            raise TypeError(
                f"Wrapped task must return a dict or torch.Tensor, got {type(preds).__name__}"
            )
        assert "prediction" in preds, "Missing 'prediction' key in task output"
        if isinstance(preds["prediction"], dict):
            inner = preds.pop("prediction")
            preds.update(inner)
        selection = self._select(preds, x)
        return preds | selection

    def _select(
        self, preds: dict[str, torch.Tensor], x_input: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute selection mask from inputs."""
        if isinstance(self.score, EmbeddingScore):
            assert "embedding" in preds, (
                "Embedding score requires `embedding` key in model's output dict."
            )
            _x = preds["embedding"]
        elif isinstance(self.score, LogitScore):
            assert "logit" in preds, (
                "Logit score requires `logit` key in model's output dict."
            )
            _x = preds["logit"]
        else:
            # Generic UncertaintyScore expects the original model input tensor.
            _x = x_input
        selection = self.score.select(_x)
        return selection

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
        batch: Mapping[str, Any] | Sequence[Any]
            A batch from the test DataLoader. The input tensor and target are
            extracted using `self.input_key` and `self.target_key``.
        batch_idx: int
            Index of the current batch (required by Lightning but not used here).
        dataloader_idx: int, optional
            Index of the DataLoader when multiple loaders are used. Defaults to
            `0``.

        Notes
        -----
        The method extracts `x` and `y` from the batch, runs `self.forward``
        to obtain predictions together with `score` and `selected` masks, and
        updates any attached `SelectiveMetric` and `RiskCoverageMetric``.
        Per-batch outputs are optionally stored in `self.test_outputs` when the
        instance was created with `acc_test_outputs=True``. No value is returned;
        metrics are logged via Lightning's `log_dict` mechanism.

        """
        x = _get_from_batch(batch, self.input_key, pos=0)
        y = _get_from_batch(batch, self.target_key, pos=1)

        outputs = self.forward(x)

        if self.test_metrics is not None:
            self.test_metrics.update(
                preds=outputs["prediction"],
                target=y,
                selected=outputs["selected"],
            )
            self.log_dict(self.test_metrics.compute(), sync_dist=True)

        # Update risk-coverage metric; final values are logged in on_test_epoch_end
        if self.rc_metric is not None:
            self.rc_metric.update(
                preds=outputs["prediction"], target=y, scores=outputs["score"]
            )
            self.log_dict(self.rc_metric.compute(), sync_dist=True)

        if self.test_outputs is not None:
            self.test_outputs.append(outputs)

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

        The wrapper calls `forward(x)` and returns the combined mapping produced by
        the wrapped model and the score. This mapping typically contains the
        model's predictions and the selection outputs (e.g. `score` and `selected`).
        """
        x = _get_from_batch(batch, self.input_key, pos=0)

        # if task has no predict_step we simply call forward
        if not hasattr(self.task, "predict_step") or not callable(
            self.task.predict_step
        ):
            return self.forward(x)

        # otherwise we call the task's predict_step and merge with selection results
        preds = self.task.predict_step(batch, batch_idx, dataloader_idx)
        if isinstance(preds, torch.Tensor):
            preds = {"prediction": preds}
        assert isinstance(preds, dict)
        selection = self._select(preds, x)
        return preds | selection

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
