"""Concrete logit-based uncertainty scores."""

from __future__ import annotations

import abc
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing_extensions import override

from seapig.scores.base import UncertaintyScore
from seapig.scores.logits_utils import Task, TemperatureScaler, get_task
from seapig.scores.mixins import ModelExtractorMixin

EPS = 1e-12
Batch = torch.Tensor | dict[str, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_AGG: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "max": lambda t: t.amax(dim=1),
    "min": lambda t: t.amin(dim=1),
    "sum": lambda t: t.sum(dim=1),
}


def _bernoulli_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(EPS, 1 - EPS)
    return -(p * p.log() + (1 - p) * torch.log1p(-p))


def _shannon_entropy(p: torch.Tensor, dim: int = 1) -> torch.Tensor:
    p = p.clamp(EPS, 1 - EPS)
    return -(p * p.log()).sum(dim=dim)


class LogitScore(UncertaintyScore, ModelExtractorMixin, abc.ABC):
    """Base class for logit-based uncertainty scores.

    Supports multiclass, binary (single/two-logit), and multilabel tasks.
    Handles temperature fitting and input normalization for all cases.

    The `per_member` flag enables handling of logits that contain multiple stochastic
    members per sample (e.g. ensembles or MC-dropout). When `True`, score methods
    compute the metric for each member and return the mean across the member axis.

    Parameters
    ----------
    temperature : float or None, default None
        Optional temperature to apply to logits. If `None`, no temperature
        scaling is applied until :meth:`fit` is called.
    task : {'multiclass', 'binary', 'multilabel'}, default 'multiclass'
        Type of classification task. Determines score computation and
        temperature fitting loss.
    per_member : bool, default False
        If `True`, logits are expected to have a member dimension (e.g. for ensembles or MC-dropout).
        Score methods will compute the score for each member and return the mean across members

    Notes
    -----
    Input shapes and label formats by task:

    - `multiclass`: logits `(N, C)`, labels `(N,)` long
    - `binary` single-logit: logits `(N,)` or `(N, 1)`, labels `(N,)` float/long
    - `binary` two-logit: logits `(N, 2)`, labels `(N,)` long
    - `multilabel`: logits `(N, C)`, labels `(N, C)` float

    See Also
    --------
    `scores.SoftmaxScore`
    `scores.EntropyScore`
    `scores.EnergyScore`
    `scores.MarginScore`

    Examples
    --------
    ```python
    import torch
    from seapig.scores.logits import SoftmaxScore
    logits = torch.randn(4, 3)
    score = SoftmaxScore()
    score.score(logits)
    ```
    """

    method_name: str = "logits"
    ident: str

    logits: torch.Tensor | None
    labels: torch.Tensor | None

    def __init__(
        self,
        temperature: float | None = None,
        task: str = "multiclass",
        per_member: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("logits", None)
        self.register_buffer("labels", None)
        self.temperature = None if temperature is None else float(temperature)
        self.task = task
        self._task: Task = get_task(task)
        self.per_member = bool(per_member)

    def fit(
        self,
        X: torch.Tensor | None = None,
        Y: torch.Tensor | None = None,
        temp_scale: bool = False,
        model: torch.nn.Module | None = None,
        loader: DataLoader[Batch] | None = None,
        outdir: Path | str | None = None,
        prefix: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Fit the score on reference logits.

        This method supports two usage modes:

        1. **Precomputed logits**: Supply logits directly via `X`, with optional
           labels via `Y` for temperature fitting.
        2. **On-the-fly extraction**: Supply a `model` with a `.logits()` method
           and a `DataLoader` to extract logits automatically.

        You must use either logits OR model+loader, but not both.

        Parameters
        ----------
        X : torch.Tensor or None
            Reference logits. Shape depends on task (see class docstring).
            Required when not using `model` and `loader`.
        Y : torch.Tensor or None
            Optional labels for temperature fitting. Shape/type depends on task.
        temp_scale: bool
            Boolean indicating if temperature scaling is to be applied. Defaults to
            `False`. If set to `True` labels are required.
        model : torch.nn.Module or None
            Model with a `.logits(x)` method. Required when not using
            precomputed logits.
        loader : DataLoader or None
            DataLoader yielding batches for inference. Required when using `model`.
        outdir : Path or str or None
            Optional directory to save/load logits. Only used with `model` and `loader`.
        prefix : str or None
            Optional prefix for saved files. Only used with `model` and `loader`.

        Notes
        -----
        Labels are required for temperature fitting to minimize NLL for the task.
        """
        logits, extracted_labels = self._resolve(
            X, model, loader, outdir, prefix, want_labels=True
        )
        labels = Y if Y is not None else extracted_labels

        self.logits, self.labels = logits, labels
        if labels is not None and temp_scale:
            self.temperature = self._fit_temperature(logits, labels)
        self.scores = self._score(logits)

    @override
    def score(
        self,
        query_logits: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loader: DataLoader[Batch] | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> torch.Tensor:
        """Compute uncertainty scores for query logits.

        This method supports two usage modes:

        1. **Precomputed logits**: Supply query logits via `query_logits`.
        2. **On-the-fly extraction**: Supply a `model` with an `.logits()` method
           and a `DataLoader` to extract logits automatically.

        You must use either logits (query_logits) OR model+loader, but not both.

        ```python
        # Mode 1: Precomputed logits
        from seapig.scores import EntropyScore
        my_score = EntropyScore()
        scores = my_score.score(query_logits=test_logits)

        # Mode 2: On-the-fly extraction
        my_score = EntropyScore()
        scores = my_score.score(model=model, loader=test_dl)
        ```

        Parameters
        ----------
        query_logits : torch.Tensor
            Logits for samples to score. Shape depends on task.
        model:
            A `torch.nn.Module` with an `.embed()` method.
            Required when not using `X`.
        loader:
            A `torch.utils.data.DataLoader` returning `torch.Tensor`s or
            dicts with the `"image"` key. Required when using `model`.
        outdir:
            A `pathlib.Path` pointing to a directory for saving/loading embeddings.
            Only used with `model` and `loader`.
        prefix:
            A `str` used as filename prefix for saved embeddings.
            Only used with `model` and `loader`.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape `(N,)`. Lower values indicate lower uncertainty.
        """
        logits, _ = self._resolve(
            query_logits, model, loader, outdir, prefix, want_labels=False
        )
        return self._score(logits)

    def select(
        self,
        query_logits: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loader: DataLoader[Batch] | None = None,
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Select samples for prediction based on their uncertainty score.

        Samples with scores lower than the threshold are selected for prediction,
        while samples with scores higher than the threshold are excluded.

        Parameters
        ----------
        query_logits : torch.Tensor
            Logits for samples to select. Shape depends on task.
        model:
                A `torch.nn.Module` with an `.embed()` method.
            Required when not using `X`.
        loader:
            A `torch.utils.data.DataLoader` returning `torch.Tensor`s or
            dicts with the `"image"` key. Required when using `model`.
        outdir:
            A `pathlib.Path` pointing to a directory for saving/loading embeddings.
            Only used with `model` and `loader`.
        prefix:
            A `str` used as filename prefix for saved embeddings.
            Only used with `model` and `loader`.

        Returns
        -------
        dict[str, torch.Tensor]
            A dict with keys `'score'` (uncertainty scores) and `'selected'`
            (boolean mask where `True` means the sample is selected).
        """
        if self.threshold is None:
            self.set_threshold()
        assert self.threshold is not None
        scores = self.score(query_logits, model, loader, outdir, prefix)
        return {"score": scores, "selected": scores < self.threshold}

    def _resolve(
        self,
        logits: torch.Tensor | None,
        model: torch.nn.Module | None,
        loader: DataLoader[Batch] | None,
        outdir: Path | str | None,
        prefix: str | None,
        *,
        want_labels: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        model_mode = model is not None or loader is not None
        if (logits is not None) == model_mode:
            raise ValueError(
                "Specify either pre-computed logits or a model with a loader, "
                "but not both."
            )
        if logits is not None:
            return logits, None
        if model is None or loader is None:
            raise ValueError("`model` and `loader` must be given together.")

        out = self._extract_loader(
            model=model,
            loader=loader,
            input_keys=["image", "label"] if want_labels else ["image"],
            output_key="logit",
            outdir=outdir,
            prefix=prefix,
        )
        return out["logit"], out.get("label")

    @property
    def T(self) -> float:
        """Temperature property."""
        return 1.0 if self.temperature is None else float(self.temperature)

    def _prepare(self, logits: torch.Tensor) -> tuple[Task, torch.Tensor]:
        """Resolve the task variant and canonicalise shape to `(N, K, M)`."""
        task = self._task.resolve(logits, self.per_member)
        return task, task.canonicalize(logits, self.per_member)

    @staticmethod
    def _flatten_members(
        z: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(N, K, M)`` -> ``(N*M, K)``, repeating labels along members."""
        n, k, m = z.shape
        return z.permute(0, 2, 1).reshape(n * m, k), labels.repeat_interleave(
            m, dim=0
        )

    def _fit_temperature(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> float:
        task, z = self._prepare(logits)
        flat, flat_labels = self._flatten_members(
            z, task.prepare_labels(labels)
        )
        return TemperatureScaler(init=self.T).fit(flat, flat_labels, task.nll)

    def _score(self, query_logits: torch.Tensor) -> torch.Tensor:
        task, z = self._prepare(query_logits)
        return self._compute(z / self.T, family=task.family)

    @abc.abstractmethod
    def _compute(self, z: torch.Tensor, family: str) -> torch.Tensor:
        """``(N, K, M)`` temperature-scaled logits -> `(N,)` scores."""


class PointwiseLogitScore(LogitScore, abc.ABC):
    """Score computed independently per member, then averaged over members."""

    # how per-label (bernoulli) scores are aggregated across K
    aggregate: Literal["max", "min", "sum"] = "max"

    @override
    def _compute(self, z: torch.Tensor, family: str) -> torch.Tensor:
        if family == "categorical":
            per_member = self._categorical(z)  # (N, M)
        else:
            per_member = _AGG[self.aggregate](self._bernoulli(z))  # (N, M)
        return per_member.mean(dim=1)

    @abc.abstractmethod
    def _categorical(self, z: torch.Tensor) -> torch.Tensor:
        """`(N, K, M)` -> `(N, M)`; softmax over the K axis."""

    @abc.abstractmethod
    def _bernoulli(self, z: torch.Tensor) -> torch.Tensor:
        """`(N, K, M)` -> `(N, K, M)`; independent sigmoid per unit."""


class EnsembleLogitScore(LogitScore, abc.ABC):
    """Score defined across members. Requires `M > 1`."""

    def __init__(
        self, temperature: float | None = None, task: str = "multiclass"
    ) -> None:
        super().__init__(temperature=temperature, task=task, per_member=True)

    @override
    def _compute(self, z: torch.Tensor, family: str) -> torch.Tensor:
        if z.shape[-1] < 2:
            raise ValueError(
                f"{type(self).__name__} needs >1 member, got {z.shape[-1]}"
            )
        if family == "categorical":
            return self._categorical(z)  # (N,)
        return self._bernoulli(z).amax(dim=1)  # (N, K) -> (N,)

    @abc.abstractmethod
    def _categorical(self, z: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def _bernoulli(self, z: torch.Tensor) -> torch.Tensor: ...


class SoftmaxScore(PointwiseLogitScore):
    """Negative maximum predicted probability."""

    ident = "softmax"

    @override
    def _categorical(self, z):
        return -z.softmax(dim=1).amax(dim=1)

    @override
    def _bernoulli(self, z):
        p = z.sigmoid()
        return -torch.maximum(p, 1 - p)


class EntropyScore(PointwiseLogitScore):
    """Predictive entropy (worst label for multilabel)."""

    ident = "entropy"

    @override
    def _categorical(self, z):
        return _shannon_entropy(z.softmax(dim=1), dim=1)

    @override
    def _bernoulli(self, z):
        return _bernoulli_entropy(z.sigmoid())


class MarginScore(PointwiseLogitScore):
    """Negative top-two logit margin."""

    ident = "margin"

    @override
    def _categorical(self, z):
        top2 = z.topk(k=2, dim=1).values
        return -(top2[:, 0] - top2[:, 1])

    @override
    def _bernoulli(self, z):
        return -z.abs()


class EnergyScore(PointwiseLogitScore):
    """Free energy of the logit distribution."""

    ident = "energy"
    aggregate = "sum"  # energies are extensive over labels

    @override
    def _categorical(self, z):
        return -self.T * z.logsumexp(dim=1)

    @override
    def _bernoulli(self, z):
        return -self.T * F.softplus(z)


class MutualInformationScore(EnsembleLogitScore):
    r"""Mutual information (BALD) uncertainty score for ensembles / MC-dropout."""

    ident = "mutual_information"

    @override
    def _categorical(self, z):
        p = z.softmax(dim=1)
        return _shannon_entropy(p.mean(dim=-1), dim=1) - _shannon_entropy(
            p, dim=1
        ).mean(dim=-1)

    @override
    def _bernoulli(self, z):
        p = z.sigmoid()
        return _bernoulli_entropy(p.mean(dim=-1)) - _bernoulli_entropy(p).mean(
            dim=-1
        )


class PredictiveVarianceScore(EnsembleLogitScore):
    """Variance of predicted probabilities across members."""

    ident = "predictive_variance"

    @override
    def _categorical(self, z):
        return z.softmax(dim=1).var(dim=-1, unbiased=False).sum(dim=1)

    @override
    def _bernoulli(self, z):
        return z.sigmoid().var(dim=-1, unbiased=False)
