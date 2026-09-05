"""Helpers for LogitScore including task adapters and scalar temperature scaling."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

T_MIN, T_MAX = 1e-12, 1e12
Family = Literal["categorical", "bernoulli"]


# ----------------------- Resolving Tasks
class Task(abc.ABC):
    """Encapsulates everything task-specific: shapes, labels, and NLL.

    Canonical logit form is always `(N, K, M)`:
    N samples, K units (classes or labels), M stochastic members.
    """

    name: str
    family: Family

    def resolve(self, logits: torch.Tensor, per_member: bool) -> Task:
        """Return the concrete task variant implied by `logits`' shape."""
        return self

    @abc.abstractmethod
    def canonicalize(
        self, logits: torch.Tensor, per_member: bool
    ) -> torch.Tensor:
        """Reshape raw logits to `(N, K, M)`; raise on unexpected rank."""

    @abc.abstractmethod
    def prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Coerce labels to canonical per-sample form and dtype."""

    @abc.abstractmethod
    def nll(
        self, flat_logits: torch.Tensor, flat_labels: torch.Tensor
    ) -> torch.Tensor:
        """Mean NLL on flattened `(N*M, K)` logits."""


def _expect_rank(logits: torch.Tensor, per_member: bool, name: str) -> None:
    want = 3 if per_member else 2
    if logits.ndim != want:
        raise ValueError(
            f"{name} logits with per_member={per_member} must be "
            f"{want}-D, got shape {tuple(logits.shape)}"
        )


class MulticlassTask(Task):
    """Multiclass task."""

    name, family = "multiclass", "categorical"

    def canonicalize(self, logits, per_member):
        """Reshape raw logits to `(N, K, M)`; raise on unexpected rank."""
        _expect_rank(logits, per_member, "multiclass")
        return logits if per_member else logits.unsqueeze(-1)

    def prepare_labels(self, labels):
        """Coerce labels to canonical per-sample form and dtype."""
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        if labels.ndim != 1:
            raise ValueError("multiclass labels must be 1-D class indices")
        return labels.long()

    def nll(self, flat_logits, flat_labels):
        """Mean NLL on flattened `(N*M, K)` logits."""
        return F.cross_entropy(flat_logits, flat_labels)


class BinaryTwoLogitTask(MulticlassTask):
    """Binary two logit task."""

    name, family = "binary", "categorical"

    def canonicalize(self, logits, per_member):
        """Reshape raw logits to `(N, K, M)`; raise on unexpected rank."""
        z = super().canonicalize(logits, per_member)
        if z.shape[1] != 2:
            raise ValueError("binary two-logit logits must have K=2")
        return z


class BernoulliTask(Task):
    """Shared implementation for multilabel and single-logit binary."""

    family: Family = "bernoulli"

    def prepare_labels(self, labels):
        """Coerce labels to canonical per-sample form and dtype."""
        return labels.float()

    def nll(self, flat_logits, flat_labels):
        """Mean NLL on flattened `(N*M, K)` logits."""
        return F.binary_cross_entropy_with_logits(flat_logits, flat_labels)


class MultilabelTask(BernoulliTask):
    """Multilabel task."""

    name = "multilabel"

    def canonicalize(self, logits, per_member):
        """Reshape raw logits to `(N, K, M)`; raise on unexpected rank."""
        _expect_rank(logits, per_member, "multilabel")
        return logits if per_member else logits.unsqueeze(-1)

    def prepare_labels(self, labels):
        """Coerce labels to canonical per-sample form and dtype."""
        if labels.ndim != 2:
            raise ValueError("multilabel labels must have shape (N, C)")
        return labels.float()


class BinarySingleLogitTask(BernoulliTask):
    """Binary single logit type task."""

    name = "binary"

    def canonicalize(self, logits, per_member):
        """Reshape raw logits to `(N, K, M)`; raise on unexpected rank."""
        if per_member:
            if logits.ndim == 2:  # (N, M)
                return logits.unsqueeze(1)
            if logits.ndim == 3 and logits.shape[1] == 1:
                return logits
        else:
            if logits.ndim == 1:  # (N,)
                return logits[:, None, None]
            if logits.ndim == 2 and logits.shape[1] == 1:
                return logits.unsqueeze(-1)
        raise ValueError(
            "binary single-logit logits must be (N,)/(N,1) or (N,M) "
            f"with per_member={per_member}, got {tuple(logits.shape)}"
        )

    def prepare_labels(self, labels):
        """Coerce labels to canonical per-sample form and dtype."""
        return labels.reshape(-1, 1).float()


class BinaryTask(Task):
    """Dispatches to the single- or two-logit variant based on shape."""

    name, family = "binary", "categorical"  # overridden after resolve()

    def resolve(self, logits, per_member):
        """Return the concrete task variant implied by `logits`' shape."""
        class_dim = 1 if logits.ndim > (2 if per_member else 1) else None
        two = class_dim is not None and logits.shape[1] == 2
        return BinaryTwoLogitTask() if two else BinarySingleLogitTask()

    def canonicalize(self, logits, per_member):
        """Reshape raw logits to `(N, K, M)`; raise on unexpected rank."""
        raise RuntimeError("call resolve() first")

    def prepare_labels(self, labels):  # will be overridden after resolve
        """Coerce labels to canonical per-sample form and dtype."""

    def nll(self, flat_logits, flat_labels):  # will be overridden after resolve
        """Mean NLL on flattened `(N*M, K)` logits."""


TASKS: dict[str, type[Task]] = {
    "multiclass": MulticlassTask,
    "binary": BinaryTask,
    "multilabel": MultilabelTask,
}


def get_task(name: str) -> Task:
    """Resolve the task based on a string."""
    try:
        return TASKS[name]()
    except KeyError:
        raise ValueError(
            f"Unknown task: {name!r}. Choose from {sorted(TASKS)}."
        ) from None


# ----------------------- Temperature Scaling


class TemperatureScaler:
    """Fit a single positive temperature by minimising a validation NLL.

    Optimises ``log T`` with LBFGS, falling back to Adam if LBFGS fails.
    """

    def __init__(
        self, init: float = 1.0, max_iter: int = 200, lr: float = 0.1
    ) -> None:
        self.init = float(init)
        self.max_iter = max_iter
        self.lr = lr

    def fit(
        self, logits: torch.Tensor, labels: torch.Tensor, loss_fn: LossFn
    ) -> float:
        """Fit the temperature scaler."""
        if logits.shape[0] == 0:
            raise ValueError("logits must contain at least one sample")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("logits and labels must have same length")

        logits = logits.detach().clone()
        labels = labels.detach().to(logits.device)

        def objective(log_t: torch.Tensor) -> torch.Tensor:
            temp = log_t.exp().clamp(T_MIN, T_MAX)
            return loss_fn(logits / temp, labels)

        log_t = self._new_param(logits.device)
        try:
            opt = torch.optim.LBFGS(
                [log_t], max_iter=self.max_iter, line_search_fn="strong_wolfe"
            )

            def closure() -> torch.Tensor:
                opt.zero_grad()
                loss = objective(log_t)
                loss.backward()
                return loss

            opt.step(closure)
        except RuntimeError:
            log_t = self._new_param(logits.device)
            opt = torch.optim.Adam([log_t], lr=self.lr)
            for _ in range(self.max_iter):
                opt.zero_grad()
                objective(log_t).backward()
                opt.step()

        return float(log_t.detach().exp().clamp(T_MIN, T_MAX).item())

    def _new_param(self, device: torch.device) -> torch.nn.Parameter:
        return torch.nn.Parameter(
            torch.tensor([self.init], device=device).log()
        )
