# python
"""Logit-derived confidence score base class.

Provides helpers for scores computed from model logits (pre-softmax
outputs): stable softmax, entropy, margin, max-logit, and
temperature scaling calibration.
"""

from __future__ import annotations

import abc
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from seapig.scores.base import ConfidenceScore


@dataclass
class TaskConfig:
    """Configuration for task-specific score computation.

    Parameters
    ----------
    task : {"binary", "multiclass", "multilabel"}
        Type of classification task.
    multilabel_agg : {"mean", "min", "max", "sum"}
        Aggregation strategy for multilabel scores (default: "min").
    binary_format : {"auto", "single", "two"}
        How to interpret binary logits when ambiguous (default: "auto").
    """

    task: Literal["binary", "multiclass", "multilabel"]
    multilabel_agg: Literal["mean", "min", "max", "sum"] = "min"
    binary_format: Literal["auto", "single", "two"] = "auto"


class LogitScore(ConfidenceScore, abc.ABC):
    """Abstract base for logit-derived confidence scores.

    Parameters
    ----------
    temperature : float | None
        Optional temperature to apply to logits. If None no temperature
        scaling is applied until :meth:`fit_temperature` is called.
    task : {"multiclass", "binary", "multilabel"}, default "multiclass"
        Task type which determines loss and input shapes used for
        temperature fitting.
    task_config : TaskConfig | None
        Optional TaskConfig to further control multilabel aggregation and
        binary format handling. If provided its `task` field overrides
        the `task` argument.
    """

    logits: torch.Tensor | None
    labels: torch.Tensor | None
    temperature: float | None
    task: str
    task_config: TaskConfig | None

    def __init__(
        self,
        temperature: float | None = None,
        task: str = "multiclass",
        task_config: TaskConfig | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("logits", None)
        self.register_buffer("labels", None)
        self.temperature: float | None = (
            None if temperature is None else float(temperature)
        )
        # task and optional configuration
        if task_config is not None:
            self.task_config = task_config
            self.task = task_config.task
        else:
            self.task_config = None
            self.task = task

    @staticmethod
    def _check_model(model: torch.nn.Module) -> None:
        """Check a model for compatibility with logits-based confidence scores."""
        assert isinstance(model, torch.nn.Module)
        if not callable(model.logits):
            raise Exception("model is required to have a `.logits()` method.")
        sig = inspect.signature(obj=model.logits)
        if "x" not in sig.parameters.keys():
            raise Exception(
                "`.logits()` method is required to except `x` as argument."
            )

    def fit(
        self, logits: torch.Tensor, labels: torch.Tensor | None = None
    ) -> None:
        """Fit the score on reference logits.

        Parameters
        ----------
        logits : torch.Tensor
            Reference logits of shape (N, C).
        labels : torch.Tensor | None
            Optional integer labels (N,) for temperature scaling.
        """
        self.logits = logits
        self.labels = labels
        if self.labels is not None:
            self._fit_temperature(logits=self.logits, labels=self.labels)
        self.scores = self.score(self.logits)

    @abc.abstractmethod
    def score(self, query_logits: torch.Tensor) -> torch.Tensor:
        """Compute scores for query logits.

        Parameters
        ----------
        query_logits : torch.Tensor
            Logits of shape (M, C).

        Returns
        -------
        torch.Tensor
            1-D tensor of shape (M,) with scores (lower == more confident).
        """
        pass

    def _fit_temperature(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """Fit a scalar temperature T by minimizing validation NLL.

        Optimization is performed on log(T) for stability. Result stored
        in self.temperature.
        Bounds: T in [1e-3, 1e3].
        """
        if logits.shape[0] == 0:
            raise ValueError("logits must contain at least one sample")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                "logits and labels must have same number of samples"
            )

        # Normalize inputs according to the declared task
        logits, labels = self._normalize_logits_and_labels(logits, labels)

        device = logits.device
        labels = labels.to(device=device)

        init_t = 1.0 if self.temperature is None else float(self.temperature)
        log_t = torch.nn.Parameter(torch.tensor([init_t], device=device).log())

        optimizer = torch.optim.LBFGS(
            [log_t], max_iter=200, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            T = log_t.exp().clamp(min=1e-3, max=1e3)
            # clone logits to ensure we don't use inference-mode tensors
            scaled = logits.clone() / T
            loss = self._temperature_loss(scaled, labels)
            loss.backward()  # type: ignore [no-untyped-call]
            return loss

        try:
            optimizer.step(closure)  # type: ignore [no-untyped-call]
        except Exception:
            # fallback to Adam on a fresh leaf Parameter if LBFGS fails
            log_t = torch.nn.Parameter(
                torch.tensor([init_t], device=device).log()
            )
            opt = torch.optim.Adam([log_t], lr=0.1)
            for _ in range(200):
                opt.zero_grad()
                T = log_t.exp().clamp(min=1e-3, max=1e3)
                scaled = logits.clone() / T
                loss = self._temperature_loss(scaled, labels)
                loss.backward()  # type: ignore [no-untyped-call]
                opt.step()

        T_final = log_t.exp().clamp(min=1e-3, max=1e3).detach()
        self.temperature = T_final.item()


    def _is_binary_single_logit(self, logits: torch.Tensor) -> bool:
        """Return True when task=="binary" and logits are single-logit.

        Single-logit binary formats are (N,) or (N,1).
        """
        if self.task != "binary":
            return False
        return logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1)

    def _normalize_logits_and_labels(
        self, logits: torch.Tensor, labels: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize shapes for temperature fitting and validate inputs.

        Returns normalized (logits, labels) suitable for loss computation.
        """
        if labels is None:
            raise ValueError("labels must be provided to fit temperature")

        # Multiclass: logits (N,C), labels (N,)
        if self.task == "multiclass":
            if logits.ndim != 2:
                raise ValueError("multiclass logits must have shape (N, C)")
            if labels.dim() == 2 and labels.size(1) == 1:
                labels = labels.squeeze(1)
            if labels.dim() != 1:
                raise ValueError("multiclass labels must be 1-D class indices")
            if labels.dtype != torch.long:
                labels = labels.to(dtype=torch.long)
            return logits, labels

        # binary task: single-logit (N,) or two-logit (N,2)
        if self.task == "binary":
            if self._is_binary_single_logit(logits):
                # squeeze to (N,)
                logits_n = logits.squeeze()
                lab = labels.squeeze()
                lab = lab.to(dtype=torch.float)
                return logits_n, lab
            else:
                # expect (N,2) logits and integer labels
                if logits.ndim != 2 or logits.size(1) != 2:
                    raise ValueError(
                        "binary two-logit logits must have shape (N, 2)"
                    )
                if labels.dim() == 2 and labels.size(1) == 1:
                    labels = labels.squeeze(1)
                if labels.dtype != torch.long:
                    labels = labels.to(dtype=torch.long)
                return logits, labels

        # multilabel: logits (N,C), labels (N,C) floats
        if self.task == "multilabel":
            if logits.ndim != 2:
                raise ValueError("multilabel logits must have shape (N, C)")
            if labels.ndim != 2 or labels.shape != logits.shape:
                raise ValueError(
                    "multilabel labels must have same shape as logits (N, C)"
                )
            return logits, labels.to(dtype=torch.float)

        raise ValueError(f"Unknown task: {self.task}")

    def _temperature_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute task-appropriate loss on temperature-scaled logits.

        Assumes `logits` are already scaled by 1/T (i.e. logits / T).
        """
        if self.task == "multiclass":
            return F.cross_entropy(logits, labels.long())

        if self.task == "binary":
            if self._is_binary_single_logit(logits):
                # logits and labels should be 1-D here
                logits_1d = logits.squeeze()
                labels_1d = labels.squeeze().float()
                return F.binary_cross_entropy_with_logits(logits_1d, labels_1d)
            else:
                return F.cross_entropy(logits, labels.long())

        if self.task == "multilabel":
            return F.binary_cross_entropy_with_logits(logits, labels.float())

        raise ValueError(f"Unknown task: {self.task}")
    
    def _loadorpredict(
        self,
        path: Path | None,
        model: torch.nn.Module,
        loader: DataLoader[object],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Load logits and labels from disk or predict from model and loader."""
        self._check_model(model=model)
        if path is not None and path.exists():
            data = torch.load(path, map_location="cpu")
            logits = data.get("logits", None)
            labels = data.get("labels", None)
            if logits is None:
                raise ValueError(f"Saved file {path} does not contain 'logits'")
        else:
            logits, labels = self._logits_from_loader(
                model=model, loader=loader
            )
            if logits is None:
                raise ValueError("Failed to extract logits from loader")
            if path is not None:
                torch.save(
                    {
                        "logits": logits.cpu(),
                        "labels": labels.cpu() if labels is not None else None,
                    },
                    path,
                )

        return logits, labels

    def _logits_from_loader(
        self, model: torch.nn.Module, loader: DataLoader[object]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Iterate over a DataLoader and collect logits and optional labels.

        Returns (logits, labels) where labels may be None if the loader does not
        provide them.
        """
        self._check_model(model=model)
        pbar = tqdm(
            total=len(loader),
            desc=f"Forward {len(loader)} batches",
            unit="batches",
        )
        logits_ls: list[torch.Tensor] = []
        labels_ls: list[torch.Tensor] = []
        device = None
        try:
            device = next(model.parameters()).device
        except StopIteration:
            try:
                device = next(model.buffers()).device
            except StopIteration:
                device = torch.device("cpu")
        for batch in loader:
            if isinstance(batch, torch.Tensor):
                x = batch
                y = None
            elif isinstance(batch, dict):
                x = batch.get("image", None)
                y = batch.get("label", None)

            assert callable(model.logits)
            logits = model.logits(x=x.to(device))
            if not isinstance(logits, torch.Tensor):
                raise ValueError("Extracted logits is not a torch.Tensor")
            logits_ls.append(logits)
            if y is not None:
                assert y.shape[0] == logits.shape[0], (
                    "Batch size of labels must match logits"
                )
                labels_ls.append(y.to(device))
            if pbar is not None:
                _ = pbar.update(n=1)
        if len(logits_ls) == 0:
            raise ValueError("No batches found in loader")
        logits = torch.cat(logits_ls, dim=0)
        labels = torch.cat(labels_ls, dim=0) if len(labels_ls) > 0 else None
        return logits, labels

    def fit_dl(
        self,
        model: torch.nn.Module,
        loader: DataLoader[object],
        outdir: Path | str | None = None,
        prefix: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Fit the score by extracting logits from a single DataLoader.

        If outdir is provided, this method will first construct the output
        path and attempt to load pre-extracted logits from
        {outdir}/{prefix or 'score'}_train.pt. If that file does not exist the
        logits and labels are extracted from the provided DataLoader and the
        results are optionally saved to disk under the same path.

        Parameters
        ----------
        model : torch.nn.Module
            Model to extract logits from.
        loader : DataLoader
            Single DataLoader to extract logits/labels from.
        outdir : Path | str | None
            Optional directory to save/load extracted logits and labels.
        prefix : str | None
            Optional prefix for saved files. If None or empty, "logits" is
            used as prefix.
        """
        assert isinstance(loader, DataLoader)
        assert isinstance(model, torch.nn.Module)

        # prepare output path if requested
        path: Path | None = None
        if outdir is not None:
            out_path = Path(outdir)
            out_path.mkdir(parents=True, exist_ok=True)
            base = prefix if (prefix and prefix.strip()) else "logits"
            path = out_path / f"{base}_train.pt"

        logits, labels = self._loadorpredict(
            path=path, model=model, loader=loader
        )

        self.fit(logits=logits, labels=labels)


class SoftmaxScore(LogitScore):
    """Maximum softmax probability confidence score.

    Parameters
    ----------
    temperature : float | None
        Optional initial temperature. If None no temperature scaling is
        applied until :meth:`fit` calls :meth:`fit_temperature`.
    """

    ident: str = "softmax"

    def __init__(self, temperature: float | None = None) -> None:
        super().__init__(temperature=temperature)

    def score(self, query_logits: torch.Tensor) -> torch.Tensor:
        """Compute -max_softmax_probability for each sample.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape (M,) with scores (lower == more confident).
        """
        T = 1.0 if self.temperature is None else float(self.temperature)
        probs = F.softmax(query_logits / T, dim=1)
        maxp = probs.amax(dim=1)
        return -maxp


class EnergyScore(LogitScore):
    """Energy-based confidence score.

    Parameters
    ----------
    temperature : float | None
        Optional initial temperature. If None no temperature scaling is
        applied until :meth:`fit` calls :meth:`fit_temperature`.
    """

    ident: str = "energy"

    def __init__(self, temperature: float | None = None, task: str = "multiclass") -> None:
        super().__init__(temperature=temperature, task=task)

    def score(self, query_logits: torch.Tensor) -> torch.Tensor:
        """Compute energy for query logits (task-aware).

        Returns a 1-D tensor of shape (M,) where lower values are more
        confident.
        """
        T = 1.0 if self.temperature is None else float(self.temperature)
        task = self.task
        logits = query_logits
        if task == "multiclass":
            # -T * logsumexp(logits / T)
            return -(logits / T).logsumexp(dim=1) * T
        elif task == "binary":
            if self._is_binary_single_logit(logits):
                # we take abs(logits) to ensure energy is low for confident 
                # predictions in  either direction
                return -T * F.softplus(torch.abs(logits) / T)
            else:
                # two-logit: same as multiclass
                return -(logits / T).logsumexp(dim=1) * T
        elif task == "multilabel":
            # Sum of per-label free energies: -T * sum(softplus(logit/T))
            return -T * F.softplus(logits / T).sum(dim=1)
        else:
            raise ValueError(f"Unknown task: {task}")


class MarginScore(LogitScore):
    """Top-two margin confidence score.

    Parameters
    ----------
    temperature : float | None
        Optional initial temperature. If None no temperature scaling is
        applied until :meth:`fit` calls :meth:`fit_temperature`.
    """

    ident: str = "margin"

    def __init__(self, temperature: float | None = None) -> None:
        super().__init__(temperature=temperature)

    def score(self, query_logits: torch.Tensor) -> torch.Tensor:
        """Compute negative top-two margin for each sample.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape (M,) with scores (lower == more confident).
        """
        T = 1.0 if self.temperature is None else float(self.temperature)
        scaled = query_logits / T
        if scaled.size(1) < 2:
            raise ValueError(
                "MarginScore requires at least two classes in logits (C>=2)."
            )
        top2 = scaled.topk(k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        return -margin


class EntropyScore(LogitScore):
    """Entropy-based confidence score.

    Parameters
    ----------
    temperature : float | None
        Optional initial temperature. If None no temperature scaling is
        applied until :meth:`fit` calls :meth:`fit_temperature`.
    """

    ident: str = "entropy"

    def __init__(self, temperature: float | None = None, task: str = "multiclass") -> None:
        super().__init__(temperature=temperature, task=task)

    def score(self, query_logits: torch.Tensor) -> torch.Tensor:
        """Compute predictive entropy for each sample (task-aware).

        Returns
        -------
        torch.Tensor
            1-D tensor of shape (M,) with entropy scores (lower == more confident).
        """
        T = 1.0 if self.temperature is None else float(self.temperature)
        task = self.task
        logits = query_logits
        EPS = 1e-12
        if task == "multiclass":
            probs = F.softmax(logits / T, dim=1)
            p = probs.clamp(min=EPS)
            entropy = -(p * p.log()).sum(dim=1)
            return entropy
        elif task == "binary":
            if self._is_binary_single_logit(logits):
                p = torch.sigmoid(logits / T)
                p = p.clamp(min=EPS, max=1-EPS)
                entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                return entropy
            else:
                # two-logit: use softmax, then Bernoulli entropy on class 1 prob
                probs = F.softmax(logits / T, dim=1)
                p = probs[:, 1].clamp(min=EPS, max=1-EPS)
                entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                return entropy
        elif task == "multilabel":
            p = torch.sigmoid(logits / T)
            p = p.clamp(min=EPS, max=1-EPS)
            per_label_entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            # MAX aggregation: highest uncertainty across labels
            entropy = per_label_entropy.max(dim=1).values
            return entropy
        else:
            raise ValueError(f"Unknown task: {task}")


