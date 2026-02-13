# python
"""Logit-derived confidence score base class.

Provides helpers for scores computed from model logits (pre-softmax
outputs): stable softmax, entropy, margin, max-logit, and
temperature scaling calibration.
"""

from __future__ import annotations

import abc
import inspect
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from seapig.scores.base import ConfidenceScore


class LogitScore(ConfidenceScore, abc.ABC):
    """
    Base class for logit-based confidence scores.

    Supports multiclass, binary (single/two-logit), and multilabel tasks.
    Handles temperature fitting and input normalization for all cases.

    Parameters
    ----------
    temperature : float or None
        Optional temperature to apply to logits. If None, no temperature
        scaling is applied until :meth:`fit` or :meth:`fit_temperature` is called.
    task : {"multiclass", "binary", "multilabel"}, default="multiclass"
        Type of classification task. Determines score computation and
        temperature fitting loss. Default is multiclass for backwards compatibility.

    Notes
    -----
    Input shapes and label formats by task:

    - multiclass: logits (N, C), labels (N,) long
    - binary single-logit: logits (N,) or (N, 1), labels (N,) float/long
    - binary two-logit: logits (N, 2), labels (N,) long
    - multilabel: logits (N, C), labels (N, C) float

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

    logits: torch.Tensor | None
    labels: torch.Tensor | None
    temperature: float | None
    task: str

    def __init__(
        self, temperature: float | None = None, task: str = "multiclass"
    ) -> None:
        super().__init__()
        self.register_buffer("logits", None)
        self.register_buffer("labels", None)
        self.temperature: float | None = (
            None if temperature is None else float(temperature)
        )
        self.task = task

    @staticmethod
    def _check_model(model: torch.nn.Module) -> None:
        """
        Check if the model is compatible with logits-based confidence scores.

        Parameters
        ----------
        model : torch.nn.Module
            Model to check. Must have a callable `.logits(x)` method.
        """
        assert isinstance(model, torch.nn.Module)
        if not hasattr(model, "logits") or not callable(model.logits):
            raise Exception("model is required to have a `.logits()` method.")
        sig = inspect.signature(obj=model.logits)
        if "x" not in sig.parameters.keys():
            raise Exception(
                "`.logits()` method is required to except `x` as argument."
            )

    def fit(
        self, logits: torch.Tensor, labels: torch.Tensor | None = None
    ) -> None:
        """
        Fit the score on reference logits.

        Parameters
        ----------
        logits : torch.Tensor
            Reference logits. Shape depends on task (see class docstring).
        labels : torch.Tensor or None
            Optional labels for temperature fitting. Shape/type depends on task.

        Notes
        -----
        If labels are provided, temperature is fitted to minimize NLL for the task.
        """
        self.logits = logits
        self.labels = labels
        if self.labels is not None:
            self._fit_temperature(logits=self.logits, labels=self.labels)
        self.scores = self.score(self.logits)

    @abc.abstractmethod
    def score(self, query_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence scores for query logits.

        Parameters
        ----------
        query_logits : torch.Tensor
            Logits for samples to score. Shape depends on task.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape (M,). Lower is more confident.
        """
        pass

    def _fit_temperature(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """
        Fit scalar temperature by minimizing validation NLL.

        Parameters
        ----------
        logits : torch.Tensor
            Logits for temperature fitting.
        labels : torch.Tensor
            Corresponding labels.
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
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
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
                loss.backward()
                opt.step()

        T_final = log_t.exp().clamp(min=1e-3, max=1e3).detach()
        self.temperature = T_final.item()

    def _is_binary_single_logit(self, logits: torch.Tensor) -> bool:
        """
        Determine if logits are single-logit binary format.

        Parameters
        ----------
        logits : torch.Tensor
            Input logits.

        Returns
        -------
        bool
            True if single-logit binary, else False.
        """
        if self.task != "binary":
            return False
        is_single_logit: bool = logits.ndim == 1 or (
            logits.ndim == 2 and logits.shape[1] == 1
        )
        return is_single_logit

    def _normalize_logits_and_labels(
        self, logits: torch.Tensor, labels: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize logits and labels for temperature fitting.

        Parameters
        ----------
        logits : torch.Tensor
            Input logits.
        labels : torch.Tensor or None
            Input labels.

        Returns
        -------
        tuple of torch.Tensor
            Normalized logits and labels.
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

    def _temperature_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for temperature scaling based on task.

        Parameters
        ----------
        logits : torch.Tensor
            Temperature-scaled logits.
        labels : torch.Tensor
            Corresponding labels.

        Returns
        -------
        torch.Tensor
            Loss value.
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
        """
        Load logits and labels from disk or compute from model.

        Parameters
        ----------
        path : Path or None
            Path to saved logits file.
        model : torch.nn.Module
            Model for prediction.
        loader : DataLoader
            DataLoader for inference.

        Returns
        -------
        tuple of torch.Tensor, torch.Tensor or None
            Logits and labels.
        """
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
        """
        Extract logits and labels from a DataLoader.

        Parameters
        ----------
        model : torch.nn.Module
            Model for inference.
        loader : DataLoader
            DataLoader yielding batches.

        Returns
        -------
        tuple of torch.Tensor or None, torch.Tensor or None
            Logits and labels.
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
        """
        Fit the score by extracting logits from a DataLoader.

        Loads logits/labels from disk if available, else computes from model.

        Parameters
        ----------
        model : torch.nn.Module
            Model with a `.logits(x)` method.
        loader : DataLoader
            DataLoader yielding batches for inference.
        outdir : Path or str or None
            Optional directory to save/load logits.
        prefix : str or None
            Optional prefix for saved files.

        Examples
        --------
        ```python
        # Minimal example (pseudo-code)
        score = SoftmaxScore()
        score.fit_dl(model, loader)
        ```
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
    """
    Maximum softmax probability confidence score.

    Supports multiclass, binary (single/two-logit), and multilabel tasks.

    Parameters
    ----------
    temperature : float or None
        Optional initial temperature. If None, temperature is fitted if labels are provided.
    task : {"multiclass", "binary", "multilabel"}, default="multiclass"
        Task type for score computation.

    Examples
    --------
    ```python
    import torch
    from seapig.scores.logits import SoftmaxScore
    logits = torch.randn(2, 4)
    SoftmaxScore().score(logits)
    ```
    """

    ident: str = "softmax"

    def __init__(
        self, temperature: float | None = None, task: str = "multiclass"
    ) -> None:
        super().__init__(temperature=temperature, task=task)

    def score(self, query_logits: torch.Tensor) -> torch.Tensor:
        """Compute task-aware softmax-based confidence score.

        For multiclass: -max softmax probability.
        For binary single-logit: -sigmoid(|logit|).
        For binary two-logit: -max softmax probability.
        For multilabel: -min(max(p, 1-p)), where p = sigmoid(logit).

        Returns
        -------
        torch.Tensor
            1-D tensor of shape (M,) with scores (lower == more confident).
        """
        T = 1.0 if self.temperature is None else float(self.temperature)
        task = self.task
        logits = query_logits
        if task == "multiclass":
            probs = F.softmax(logits / T, dim=1)
            return -probs.amax(dim=1)
        elif task == "binary":
            if self._is_binary_single_logit(logits):
                p = torch.sigmoid(logits.abs() / T)
                return -p
            else:
                probs = F.softmax(logits / T, dim=1)
                return -probs.amax(dim=1)
        elif task == "multilabel":
            p = torch.sigmoid(logits / T)
            max_p = torch.maximum(p, 1 - p)
            return -max_p.min(dim=1).values
        else:
            raise ValueError(f"Unknown task: {task}")


class EnergyScore(LogitScore):
    """
    Energy-based confidence score.

    Supports multiclass, binary, and multilabel tasks.

    Parameters
    ----------
    temperature : float or None
        Optional initial temperature. If None, temperature is fitted if labels are provided.

    Examples
    --------
    ```python
    import torch
    from seapig.scores.logits import EnergyScore
    logits = torch.randn(2, 3)
    EnergyScore().score(logits)
    ```
    """

    ident: str = "energy"

    def __init__(
        self, temperature: float | None = None, task: str = "multiclass"
    ) -> None:
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
    """
    Top-two margin confidence score.

    Supports multiclass, binary (single/two-logit), and multilabel tasks.

    Parameters
    ----------
    temperature : float or None
        Optional initial temperature. If None, temperature is fitted if labels are provided.
    task : {"multiclass", "binary", "multilabel"}, default="multiclass"
        Task type for score computation.

    Examples
    --------
    ```python
    import torch
    from seapig.scores.logits import MarginScore
    logits = torch.randn(2, 3)
    MarginScore().score(logits)
    ```
    """

    ident: str = "margin"

    def __init__(
        self, temperature: float | None = None, task: str = "multiclass"
    ) -> None:
        super().__init__(temperature=temperature, task=task)

    def score(self, query_logits: torch.Tensor) -> torch.Tensor:
        """Compute task-aware margin-based confidence score.

        For multiclass: negative top-two margin.
        For binary single-logit: negative absolute logit.
        For binary two-logit: negative top-two margin.
        For multilabel: negative min(|logit|).

        Returns
        -------
        torch.Tensor
            1-D tensor of shape (M,) with scores (lower == more confident).
        """
        T = 1.0 if self.temperature is None else float(self.temperature)
        task = self.task
        logits = query_logits
        scaled = logits / T
        if task == "multiclass":
            top2 = scaled.topk(k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
            return -margin
        elif task == "binary":
            if self._is_binary_single_logit(logits):
                return -logits.abs()
            else:
                top2 = scaled.topk(k=2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]
                return -margin
        elif task == "multilabel":
            per_label_margin = logits.abs()
            return -per_label_margin.min(dim=1).values
        else:
            raise ValueError(f"Unknown task: {task}")


class EntropyScore(LogitScore):
    """
    Entropy-based confidence score.

    Supports multiclass, binary, and multilabel tasks.

    Parameters
    ----------
    temperature : float or None
        Optional initial temperature. If None, temperature is fitted if labels are provided.

    Examples
    --------
    ```python
    import torch
    from seapig.scores.logits import EntropyScore
    logits = torch.randn(2, 3)
    EntropyScore().score(logits)
    ```
    """

    ident: str = "entropy"

    def __init__(
        self, temperature: float | None = None, task: str = "multiclass"
    ) -> None:
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
                p = p.clamp(min=EPS, max=1 - EPS)
                entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                return entropy
            else:
                # two-logit: use softmax, then Bernoulli entropy on class 1 prob
                probs = F.softmax(logits / T, dim=1)
                p = probs[:, 1].clamp(min=EPS, max=1 - EPS)
                entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                return entropy
        elif task == "multilabel":
            p = torch.sigmoid(logits / T)
            p = p.clamp(min=EPS, max=1 - EPS)
            per_label_entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            # MAX aggregation: highest uncertainty across labels
            entropy = per_label_entropy.max(dim=1).values
            return entropy
        else:
            raise ValueError(f"Unknown task: {task}")
