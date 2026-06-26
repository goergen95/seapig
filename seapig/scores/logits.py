# python
"""Logit-derived uncertainty score base class.

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
from typing_extensions import override

from seapig.scores.base import UncertaintyScore
from seapig.utils.progress import track


class LogitScore(UncertaintyScore, abc.ABC):
    """Base class for logit-based uncertainty scores.

    Supports multiclass, binary (single/two-logit), and multilabel tasks.
    Handles temperature fitting and input normalization for all cases.

    The ``per_member`` flag enables handling of logits that contain multiple stochastic
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

    logits: torch.Tensor | None
    labels: torch.Tensor | None
    temperature: float | None
    task: str

    def __init__(
        self,
        temperature: float | None = None,
        task: str = "multiclass",
        per_member: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("logits", None)
        self.register_buffer("labels", None)
        self.temperature: float | None = (
            None if temperature is None else float(temperature)
        )
        self.task = task
        self.per_member: bool = bool(per_member)

    @staticmethod
    def _check_model(model: torch.nn.Module) -> None:
        """Check if the model is compatible with logits-based uncertainty scores.

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
        self,
        X: torch.Tensor | None = None,
        Y: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loader: DataLoader[object] | None = None,
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
        If labels are provided, temperature is fitted to minimize NLL for the task.
        """
        # For backward compatibility, also support 'logits' and 'labels' as kwargs
        logits = X if X is not None else None
        labels = Y if Y is not None else None

        # Validate parameter combinations
        using_precomputed = logits is not None
        using_model = model is not None or loader is not None

        if using_precomputed and using_model:
            raise ValueError(
                "Cannot specify both precomputed logits and model+loader. "
                "Use either precomputed logits OR on-the-fly extraction."
            )

        if not using_precomputed and not using_model:
            raise ValueError(
                "Must specify either logits or model+loader for fitting."
            )

        if using_precomputed:
            # Mode 1: Use precomputed logits
            self.logits = logits
            self.labels = labels
            if self.labels is not None:
                assert self.logits is not None
                self._fit_temperature(logits=self.logits, labels=self.labels)
            assert self.logits is not None
            self.scores = self.score(self.logits)
        else:
            # Mode 2: Extract logits on-the-fly
            if model is None:
                raise ValueError(
                    "model is required when not using precomputed logits."
                )
            if loader is None:
                raise ValueError("loader is required when using a model.")

            assert isinstance(loader, DataLoader)
            assert isinstance(model, torch.nn.Module)

            # prepare output path if requested
            path: Path | None = None
            if outdir is not None:
                out_path = Path(outdir)
                out_path.mkdir(parents=True, exist_ok=True)
                base = prefix if (prefix and prefix.strip()) else "logits"
                path = out_path / f"{base}_train.pt"

            extracted_logits, extracted_labels = self._loadorpredict(
                path=path, model=model, loader=loader
            )

            self.logits = extracted_logits
            self.labels = extracted_labels
            if self.labels is not None:
                self._fit_temperature(logits=self.logits, labels=self.labels)
            self.scores = self.score(self.logits)

    @abc.abstractmethod
    def score(self, query_logits: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty scores for query logits.

        Parameters
        ----------
        query_logits : torch.Tensor
            Logits for samples to score. Shape depends on task.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape `(M,)`. Lower values indicate lower uncertainty.
        """
        raise NotImplementedError()

    def select(self, query_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        """Select samples for prediction based on their uncertainty score.

        Samples with scores lower than the threshold are selected for prediction,
        while samples with scores higher than the threshold are excluded.

        Parameters
        ----------
        query_logits : torch.Tensor
            Logits for samples to select. Shape depends on task.

        Returns
        -------
        dict[str, torch.Tensor]
            A dict with keys `'score'` (uncertainty scores) and `'selected'`
            (boolean mask where `True` means the sample is selected).
        """
        if self.threshold is None:
            self.set_threshold()
        assert self.threshold is not None
        scores = self.score(query_logits)
        selected = scores < self.threshold
        return {"score": scores, "selected": selected}

    def _expand_per_member(
        self, logits: torch.Tensor, labels: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Expand per-member logits and repeat labels.

        When ``self.per_member`` is ``True`` the logits are expected to have an
        extra member dimension (e.g. ``(N, C, M)`` for multiclass, ``(N, M)`` for
        binary single-logit, or ``(N, 2, M)`` for binary two-logit). This helper
        reshapes the tensors so that the member axis is merged into the batch
        axis, yielding a flat view suitable for temperature fitting. Labels are
        repeated ``M`` times to align with the expanded batch dimension.
        """
        if not self.per_member:
            return logits, labels
        # Determine member count M based on logits shape
        if logits.ndim == 3:
            # (N, *, M) – the middle dimension is either C or 2
            N, dim, M = logits.shape
            # Move member axis to second position then flatten batch and member
            logits_exp = logits.permute(0, 2, 1).reshape(-1, dim)
        elif logits.ndim == 2:
            # Could be binary single‑logit per‑member (N, M) where M>1
            if self.task == "binary":
                N, M = logits.shape
                logits_exp = logits.reshape(-1)
            else:
                raise ValueError(
                    "per_member=True but logits shape is not supported for the current task"
                )
        else:
            raise ValueError(
                "per_member=True but logits shape is not supported for the current task"
            )
        # Expand labels if provided
        if labels is None:
            return logits_exp, None
        # Number of members is the size of the last dimension of the original logits
        M = logits.shape[-1] if logits.ndim >= 2 else 1
        if labels.dim() == 1:
            labels_exp = labels.repeat_interleave(M)
        elif labels.dim() == 2:
            # For multilabel or binary two‑logit labels, repeat rows
            labels_exp = labels.repeat_interleave(M, dim=0)
        else:
            raise ValueError(
                "labels have unsupported shape for per_member expansion"
            )
        return logits_exp, labels_exp

    def _fit_temperature(
        self, logits: torch.Tensor, labels: torch.Tensor | None
    ) -> None:
        """Fit scalar temperature by minimizing validation NLL.

        Parameters
        ----------
        logits : torch.Tensor
            Logits for temperature fitting.
        labels : torch.Tensor
            Corresponding labels.
        """
        # If per_member, flatten tensors so temperature is fitted on all members
        logits, labels = self._expand_per_member(logits, labels)

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
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        try:
            optimizer.step(closure)  # type: ignore[no-untyped-call]
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
                loss.backward()  # type: ignore[no-untyped-call]
                opt.step()

        T_final = log_t.exp().clamp(min=1e-3, max=1e3).detach()
        self.temperature = T_final.item()

    def _is_binary_single_logit(self, logits: torch.Tensor) -> bool:
        """Determine if logits are single-logit binary format.

        Parameters
        ----------
        logits : torch.Tensor
            Input logits.

        Returns
        -------
        bool
            `True` if single-logit binary format, else `False`.
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
        """Normalize logits and labels for temperature fitting.

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
        """Compute loss for temperature scaling based on task.

        Parameters
        ----------
        logits : torch.Tensor
            Temperature-scaled logits.
        labels : torch.Tensor
            Corresponding labels.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
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
        """Load logits and labels from disk or compute from model.

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
        tuple of (torch.Tensor, torch.Tensor or None)
            Logits and labels (labels may be `None` if not provided by loader).
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
        """Extract logits and labels from a DataLoader.

        Parameters
        ----------
        model : torch.nn.Module
            Model for inference.
        loader : DataLoader
            DataLoader yielding batches.

        Returns
        -------
        tuple of (torch.Tensor or None, torch.Tensor or None)
            Logits and labels.
        """
        self._check_model(model=model)
        pbar_desc = f"Forward {len(loader)} batches"
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
        for batch in track(
            loader, total=len(loader), desc=pbar_desc, unit="batches"
        ):
            x: torch.Tensor
            y: torch.Tensor | None
            if isinstance(batch, torch.Tensor):
                x = batch
                y = None
            elif isinstance(batch, dict):
                _x = batch.get("image")
                assert isinstance(_x, torch.Tensor)
                x = _x
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
        if len(logits_ls) == 0:
            raise ValueError("No batches found in loader")
        logits = torch.cat(logits_ls, dim=0)
        labels = torch.cat(labels_ls, dim=0) if len(labels_ls) > 0 else None
        return logits, labels


class SoftmaxScore(LogitScore):
    """Maximum softmax probability uncertainty score.

    Supports multiclass, binary (single/two-logit), and multilabel tasks.
    Higher maximum softmax probability indicates higher uncertainty (higher score).

    Parameters
    ----------
    temperature : float or None, default None
        Optional initial temperature. If `None`, temperature is fitted if
        labels are provided to `fit`.
    task : {'multiclass', 'binary', 'multilabel'}, default 'multiclass'
        Task type for score computation.

    Examples
    --------
    ```python
    import torch
    from seapig.scores.logits import SoftmaxScore
    logits = torch.randn(2, 4)
    SoftmaxScore().score(logits)
    ```

    See Also
    --------
    `scores.LogitScore`
    `scores.EntropyScore`
    `scores.EnergyScore`
    `scores.MarginScore`
    """

    ident: str = "softmax"

    def __init__(
        self,
        temperature: float | None = None,
        task: str = "multiclass",
        per_member: bool = False,
    ) -> None:
        super().__init__(
            temperature=temperature, task=task, per_member=per_member
        )

    @override
    def score(self, query_logits: torch.Tensor) -> torch.Tensor:
        """Compute task-aware softmax-based uncertainty score.

        For multiclass: -max softmax probability.
        For binary single-logit: -sigmoid(|logit|).
        For binary two-logit: -max softmax probability.
        For multilabel: -min(max(p, 1-p)), where p = sigmoid(logit).

        Parameters
        ----------
        query_logits : torch.Tensor
            Logits for samples to score. Shape depends on task.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape `(M,)`. Lower values indicate lower uncertainty.
        """
        T = 1.0 if self.temperature is None else float(self.temperature)
        task = self.task
        logits = query_logits
        if self.per_member:
            if logits.ndim == 3:
                # Shape (N, C, M)
                if task == "multiclass" or task == "binary":
                    probs = F.softmax(logits / T, dim=1)
                    scores = -probs.amax(dim=1)  # (N, M)
                    return scores.mean(dim=1)
                elif task == "multilabel":
                    p = torch.sigmoid(logits / T)
                    max_p = torch.maximum(p, 1 - p)
                    scores = -max_p.min(dim=1).values  # (N, M)
                    return scores.mean(dim=1)
            elif logits.ndim == 2 and task == "binary":
                p = torch.sigmoid(logits.abs() / T)
                scores = -p  # (N, M)
                return scores.mean(dim=1)
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
    """Energy-based uncertainty score.

    Computes the free energy of the logit distribution. Lower energy indicates
    lower uncertainty. Supports multiclass, binary, and multilabel tasks.

    Parameters
    ----------
    temperature : float or None, default None
        Optional initial temperature. If `None`, temperature is fitted if
        labels are provided to `fit`.
    task : {'multiclass', 'binary', 'multilabel'}, default 'multiclass'
        Task type for score computation.

    Examples
    --------
    ```python
    import torch
    from seapig.scores.logits import EnergyScore
    logits = torch.randn(2, 3)
    EnergyScore().score(logits)
    ```

    See Also
    --------
    `scores.LogitScore`
    `scores.SoftmaxScore`
    `scores.EntropyScore`
    `scores.MarginScore`
    """

    ident: str = "energy"

    def __init__(
        self,
        temperature: float | None = None,
        task: str = "multiclass",
        per_member: bool = False,
    ) -> None:
        super().__init__(
            temperature=temperature, task=task, per_member=per_member
        )

    @override
    def score(self, query_logits: torch.Tensor) -> torch.Tensor:
        """Compute energy for query logits (task-aware).

        Parameters
        ----------
        query_logits : torch.Tensor
            Logits for samples to score. Shape depends on task.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape `(M,)`. Lower values indicate lower uncertainty.
        """
        T = 1.0 if self.temperature is None else float(self.temperature)
        task = self.task
        logits = query_logits
        if self.per_member:
            if logits.ndim == 3:
                # (N, C, M)
                energies = -(logits / T).logsumexp(dim=1) * T  # (N, M)
                return energies.mean(dim=1)
            elif logits.ndim == 2 and task == "binary":
                energies = -T * F.softplus(torch.abs(logits) / T)  # (N, M)
                return energies.mean(dim=1)
        if task == "multiclass":
            return -(logits / T).logsumexp(dim=1) * T
        elif task == "binary":
            if self._is_binary_single_logit(logits):
                return -T * F.softplus(torch.abs(logits) / T)
            else:
                # two-logit: same as multiclass
                return -(logits / T).logsumexp(dim=1) * T
        elif task == "multilabel":
            return -T * F.softplus(logits / T).sum(dim=1)
        else:
            raise ValueError(f"Unknown task: {task}")


class MarginScore(LogitScore):
    """Top-two margin uncertainty score.

    Computes the difference between the top-two logits. A larger margin
    indicates lower uncertainty. Supports multiclass, binary (single/two-logit),
    and multilabel tasks.

    Parameters
    ----------
    temperature : float or None, default None
        Optional initial temperature. If `None`, temperature is fitted if
        labels are provided to `fit`.
    task : {'multiclass', 'binary', 'multilabel'}, default 'multiclass'
        Task type for score computation.

    Examples
    --------
    ```python
    import torch
    from seapig.scores.logits import MarginScore
    logits = torch.randn(2, 3)
    MarginScore().score(logits)
    ```

    See Also
    --------
    `scores.LogitScore`
    `scores.SoftmaxScore`
    `scores.EntropyScore`
    `scores.EnergyScore`
    """

    ident: str = "margin"

    def __init__(
        self,
        temperature: float | None = None,
        task: str = "multiclass",
        per_member: bool = False,
    ) -> None:
        super().__init__(
            temperature=temperature, task=task, per_member=per_member
        )

    @override
    def score(self, query_logits: torch.Tensor) -> torch.Tensor:
        """Compute task-aware margin-based uncertainty score.

        For multiclass: negative top-two margin.
        For binary single-logit: negative absolute logit.
        For binary two-logit: negative top-two margin.
        For multilabel: negative min(|logit|).

        Parameters
        ----------
        query_logits : torch.Tensor
            Logits for samples to score. Shape depends on task.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape `(M,)`. Lower values indicate lower uncertainty.
        """
        T = 1.0 if self.temperature is None else float(self.temperature)
        task = self.task
        logits = query_logits
        scaled = logits / T
        if self.per_member:
            if logits.ndim == 3:
                # (N, C, M)
                top2 = scaled.topk(k=2, dim=1).values  # (N, 2, M)
                margin = top2[:, 0, :] - top2[:, 1, :]  # (N, M)
                return -margin.mean(dim=1)
            elif logits.ndim == 2 and self._is_binary_single_logit(logits):
                return -logits.abs().mean(dim=1)  # (N, M)
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
    """Entropy-based uncertainty score.

    Computes the predictive entropy of the output distribution. Lower entropy
    indicates lower uncertainty. Supports multiclass, binary,
    and multilabel tasks.

    Parameters
    ----------
    temperature : float or None, default None
        Optional initial temperature. If `None`, temperature is fitted if
        labels are provided to `fit`.
    task : {'multiclass', 'binary', 'multilabel'}, default 'multiclass'
        Task type for score computation.

    Examples
    --------
    ```python
    import torch
    from seapig.scores.logits import EntropyScore
    logits = torch.randn(2, 3)
    EntropyScore().score(logits)
    ```

    See Also
    --------
    `scores.LogitScore`
    `scores.SoftmaxScore`
    `scores.EnergyScore`
    `scores.MarginScore`
    """

    ident: str = "entropy"

    def __init__(
        self,
        temperature: float | None = None,
        task: str = "multiclass",
        per_member: bool = False,
    ) -> None:
        super().__init__(
            temperature=temperature, task=task, per_member=per_member
        )

    @override
    def score(self, query_logits: torch.Tensor) -> torch.Tensor:
        """Compute predictive entropy for each sample (task-aware).

        Parameters
        ----------
        query_logits : torch.Tensor
            Logits for samples to score. Shape depends on task.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape `(M,)`. Lower values indicate lower uncertainty.
        """
        T = 1.0 if self.temperature is None else float(self.temperature)
        task = self.task
        logits = query_logits
        EPS = 1e-12
        if self.per_member:
            if logits.ndim == 3:
                # (N, C, M)
                if task == "multiclass" or task == "binary":
                    probs = F.softmax(logits / T, dim=1)  # (N, C, M)
                    if task == "binary":
                        p = probs[:, 1, :].clamp(min=EPS, max=1 - EPS)
                        ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                    else:
                        p = probs.clamp(min=EPS)
                        ent = -(p * p.log()).sum(dim=1)  # (N, M)
                    return ent.mean(dim=1)
                elif task == "multilabel":
                    p = torch.sigmoid(logits / T)
                    p = p.clamp(min=EPS, max=1 - EPS)
                    per_label_entropy = -(
                        p * torch.log(p) + (1 - p) * torch.log(1 - p)
                    )
                    ent = per_label_entropy.max(dim=1).values  # (N, M)
                    return ent.mean(dim=1)
            elif logits.ndim == 2 and task == "binary":
                p = torch.sigmoid(logits / T).clamp(min=EPS, max=1 - EPS)
                ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                return ent.mean(dim=1)
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
