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
from seapig.scores.base import UncertaintyScore

INPUT_KEYS = Literal["image", "input", "images", "inputs", "x"]
TARGET_KEYS = Literal[
    "mask", "label", "masks", "labels", "targets", "target", "y", "y_true"
]


class SelectiveInferenceTask(LightningModule):
    """Wrap a trained `LightningModule` to attach selection results during inference.

    The wrapper calls the wrapped model in inference mode and combines its
    predictions with selection outputs produced by a provided `UncertaintyScore`.

    Key behavior:

    - The wrapped task must provide an `.embed(x)` method. The wrapper calls
      `task.embed(x)` to produce embeddings used by the score.
    - The wrapped task is copied and set to `eval()` during initialization
      to avoid accidental training side effects.
    - If the wrapped task defines `test_metrics` (a `Metric` or `MetricCollection`),
      it will be wrapped by `SelectiveMetric` so metrics are computed only on
      selected examples.
    - If `rc_metric` (a `RiskCoverageMetric`) is provided, the wrapper will
      update it during test steps; the final risk-coverage values are available via
      `get_risk_coverage_curve()`.

    Parameters
    ----------
    task
        A trained `LightningModule` whose `forward(x)` returns predictions. The
        module must implement `embed(x)` to produce embeddings for scoring.
    score
        A seapig `UncertaintyScore` instance providing the `UncertaintyScore.select` method.
    input_key
        Key used to extract inputs from an incoming batch. If `None` (default),
        the first element of the batch is used (positional index 0). When a
        string is given it must be one of: `'image'`, `'input'`,
        `'images'`, `'inputs'`, `'x'`.
    target_key
        Key used to extract targets from an incoming batch. If `None` (default),
        the second element of the batch is used (positional index 1). When a
        string is given it must be one of: `'mask'`, `'label'`, `'masks'`,
        `'labels'`, `'targets'`, `'target'`, `'y'`, `'y_true'`.
    acc_test_outputs
        If `True`, per-batch outputs (predictions merged with selection results)
        are accumulated in the `test_outputs` list for later inspection. If
        `False` (default), outputs are not accumulated and metrics are logged as usual.
    rc_metric
        Optional `RiskCoverageMetric` that will be updated during testing.

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
        super().__init__()
        assert callable(getattr(task, "embed", None)), (
            "Wrapped task must have an embed() method"
        )
        self.task = copy.deepcopy(task)
        self.task.eval()  # Keep the wrapped task in evaluation mode
        assert isinstance(score, UncertaintyScore), (
            "score must be a seapig UncertaintyScore instance"
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

        # Initialize per-batch output collection if requested
        if acc_test_outputs:
            self.test_outputs = []
        else:
            self.test_outputs = None

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the wrapped model and attach selection results.

        Steps performed:

        - Calls the wrapped model. If a `torch.Tensor` is returned it
        is placed under the key `'predictions'`.
        - Computes embeddings with `task.embed(x)` and call `score.select(embs)`.
        - Merges prediction mapping and selection mapping and return the result.

        Returns
        -------
        dict[str, torch.Tensor]
            A `dict` containing the model predictions and the selection
            outputs returned by the score (`'score'` and `'selected'`).
        """
        preds = self.task(x)
        if isinstance(preds, torch.Tensor):
            preds = {"predictions": preds}
        assert isinstance(preds, dict)

        selection = self._select(x)

        return preds | selection

    def _select(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute selection mask from inputs."""
        assert callable(self.task.embed)
        embs = self.task.embed(x)
        assert isinstance(embs, torch.Tensor)
        selection = self.score.select(embs)
        return selection

    @torch.inference_mode()
    def test_step(
        self,
        batch: Mapping[str, Any] | Sequence[Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform a test step and include selection outputs.

        Behavior:

        - Extracts inputs and targets from the batch using `input_key`/`target_key`.
        - Calls `forward(x)` to get predictions augmented with selection results.
        - If a `SelectiveMetric` was created (from the wrapped task's `test_metrics`),
        it is updated with (`predictions`, `targets`, `selected_mask`) and logged.
        - If `rc_metric` is provided, it is updated with (`predictions`, `targets`, `score`)
        and its values are logged; final rc values are logged at `on_test_epoch_end` step.
        - If `test_outputs` was enabled at construction, the per-batch outputs are
        appended to the `test_outputs` list for later inspection.

        Notes
        -----
        This method does not return a value; metrics are updated and logged via
        `Lightning`'s logging utilities.
        """
        x = _get_from_batch(batch, self.input_key, pos=0)
        y = _get_from_batch(batch, self.target_key, pos=1)

        outputs = self.forward(x)

        if self.test_metrics is not None:
            self.test_metrics.update(
                preds=outputs["predictions"],
                target=y,
                selected=outputs["selected"],
            )
            self.log_dict(self.test_metrics.compute(), sync_dist=True)

        # Update risk‑coverage metric; final values are logged in on_test_epoch_end
        if self.rc_metric is not None:
            self.rc_metric.update(
                preds=outputs["predictions"], target=y, scores=outputs["score"]
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
            getattr(self.task, "predict_step")
        ):
            return self.forward(x)

        # otherwise we call the task's predict_step and merge with selection results
        preds = self.task.predict_step(batch, batch_idx, dataloader_idx)
        if isinstance(preds, torch.Tensor):
            preds = {"predictions": preds}
        assert isinstance(preds, dict)

        selection = self._select(x)
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
        return batch[key]  # type: ignore[arg-type, ty:invalid-argument-type]
    else:
        assert isinstance(key, int)
        return batch[key]
    raise TypeError("Unsupported batch type")
