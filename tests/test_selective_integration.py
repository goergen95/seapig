import pytest
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy

from seapig import RiskCoverageMetric
from seapig.model import SelectiveInferenceTask
from seapig.scores.base import ConfidenceScore


class DummyTask(LightningModule):
    """Forward returns predictions; embed returns the input so selection can be driven by input."""

    def __init__(self):
        super().__init__()
        # base metric required by SelectiveInferenceTask (will be wrapped by SelectiveMetric)
        self.test_metrics = Accuracy(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # predictions encoded in second column (0/1)
        return x[:, 1].long()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        # expose input so the score can decide selection from the first column
        return x


class FlagScore(ConfidenceScore):
    """Select if the first element of embedding equals 1.

    Minimal implementation for tests: implements the abstract `fit` and
    `score` methods from ConfidenceScore as no-ops / simple stubs so the
    class can be instantiated. The test's selection logic still relies on
    `select()` which inspects embeddings directly.
    """

    def fit(
        self,
        ref_embeddings: torch.Tensor,
        val_embeddings: torch.Tensor | None = None,
    ) -> None:
        # no-op: store reference embeddings for completeness
        self.ref_embeddings = ref_embeddings  # pragma: no cover

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        # return a dummy score vector (lower is better). Not used by select().
        return torch.zeros(
            embeddings.shape[0], dtype=torch.float32
        )  # pragma: no cover

    def select(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        # boolean selection
        selected = embeddings[:, 0].to(torch.bool)
        # numeric score (lower is better) — include in outputs for RiskCoverageMetric
        score = (1.0 - embeddings[:, 0].to(torch.float32)).reshape(-1)
        return {"selected": selected, "score": score}


class SmallDataset(Dataset):
    def __init__(self):
        # Each sample: [selected_flag, predicted_value], label
        self.samples = [
            ([1, 1], 1),  # selected, correct
            ([1, 0], 0),  # selected, correct
            ([0, 1], 0),  # rejected, incorrect
            ([0, 0], 1),  # rejected, incorrect
            ([0, 0], 0),  # rejected, correct
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return {
            "image": torch.tensor(x, dtype=torch.float32),
            "label": torch.tensor(y, dtype=torch.long),
        }


class DM(LightningDataModule):
    def test_dataloader(self):
        # batch_size=3 -> produces batches of sizes 3 and 2 (uneven)
        return DataLoader(SmallDataset(), batch_size=3, shuffle=False)


def _find_metric(results: dict, suffix: str):
    for k, v in results.items():
        if k.endswith(suffix):
            return float(v)


def _tensor_dict_to_floats(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        out[k] = float(v.item()) if hasattr(v, "item") else float(v)
    return out


@pytest.mark.filterwarnings(
    r"ignore:`isinstance\(treespec, LeafSpec\)` is deprecated.*"
)
def test_selective_inference_trainer_integration(tmp_path):
    task = DummyTask()
    score = FlagScore()
    sel_model = SelectiveInferenceTask(
        task, score, input_key="image", target_key="label"
    )

    trainer = Trainer(
        logger=False, enable_checkpointing=False, accelerator="cpu", devices=1
    )

    results = trainer.test(sel_model, datamodule=DM())

    # trainer.test returns a list (one entry per test dataloader)
    assert isinstance(results, list) and len(results) == 1
    res = results[0]

    # locate keys (lightning may prefix them with "test/")
    reported_full = _find_metric(res, "full/BinaryAccuracy")
    reported_selected = _find_metric(res, "selected/BinaryAccuracy")
    reported_rejected = _find_metric(res, "rejected/BinaryAccuracy")

    # Final computed dict from the SelectiveMetric inside the wrapper
    assert sel_model.test_metrics is not None
    metric_dict = sel_model.test_metrics.compute()
    metric_floats = _tensor_dict_to_floats(metric_dict)

    # Ensure trainer-reported values match the metric.compute() values
    assert (
        abs(
            reported_full
            - metric_floats.get("full/BinaryAccuracy", reported_full)
        )
        < 1e-6
    )
    assert (
        abs(
            reported_selected
            - metric_floats.get("selected/BinaryAccuracy", reported_selected)
        )
        < 1e-6
    )
    assert (
        abs(
            reported_rejected
            - metric_floats.get("rejected/BinaryAccuracy", reported_rejected)
        )
        < 1e-6
    )

    # Manual expected values for the dataset
    # expected: selected accuracy = 2/2 = 1.0, full = 3/5 = 0.6, rejected = 1/3 ~= 0.333...
    assert abs(metric_floats["selected/BinaryAccuracy"] - 1.0) < 1e-6
    assert abs(metric_floats["full/BinaryAccuracy"] - 0.6) < 1e-6
    assert abs(metric_floats["rejected/BinaryAccuracy"] - (1.0 / 3.0)) < 1e-6

    # sanity: selected != full and rejected > 0
    assert (
        metric_floats["selected/BinaryAccuracy"]
        != metric_floats["full/BinaryAccuracy"]
    )
    assert metric_floats["rejected/BinaryAccuracy"] > 0.0


@pytest.mark.filterwarnings(
    r"ignore:`isinstance\(treespec, LeafSpec\)` is deprecated.*"
)
def test_risk_coverage_integration_via_trainer(tmp_path):
    task = DummyTask()
    score = FlagScore()
    sel_model = SelectiveInferenceTask(
        task,
        score,
        rc_metric=RiskCoverageMetric(risk="selective"),
        input_key="image",
        target_key="label",
    )

    trainer = Trainer(
        logger=False, enable_checkpointing=False, accelerator="cpu", devices=1
    )

    results = trainer.test(sel_model, datamodule=DM())

    # trainer.test returns a list (one entry per test dataloader)
    assert isinstance(results, list) and len(results) == 1
    res = results[0]

    # find reported values (Lightning may prefix them, so search by suffix)
    reported_emp = _find_metric(res, "rc/auc_empirical")
    reported_ref = _find_metric(res, "rc/auc_reference")
    reported_excess = _find_metric(res, "rc/auc_excess")

    # final computed dict from the RiskCoverageMetric on the model
    assert sel_model.rc_metric is not None
    metric_dict = sel_model.rc_metric.compute()
    metric_floats = _tensor_dict_to_floats(metric_dict)

    # Ensure trainer-reported values match metric.compute() values
    assert (
        abs(reported_emp - metric_floats.get("rc/auc_empirical", reported_emp))
        < 1e-6
    )
    assert (
        abs(reported_ref - metric_floats.get("rc/auc_reference", reported_ref))
        < 1e-6
    )
    assert (
        abs(
            reported_excess
            - metric_floats.get("rc/auc_excess", reported_excess)
        )
        < 1e-6
    )

    # Basic sanity checks on numeric ranges and presence
    assert "rc/auc_empirical" in metric_floats
    assert "rc/auc_reference" in metric_floats
    assert "rc/auc_excess" in metric_floats

    emp = metric_floats["rc/auc_empirical"]
    ref = metric_floats["rc/auc_reference"]
    excess = metric_floats["rc/auc_excess"]

    assert 0.0 <= emp <= 1.0
    assert 0.0 <= ref <= 1.0
    # auc_excess can be slightly negative due to numeric differences; allow tiny tolerance
    assert excess >= -1e-6
