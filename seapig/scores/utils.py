"""Utilities accessed by several modules."""

import math
import warnings
from typing import final

import torch

from seapig.utils.logging import get_logger

logger = get_logger(__name__)


@final
class TensorPCA(torch.nn.Module):  # type: ignore[misc]
    """Tensor based PCA with L2 normalized inputs.

    The implementation supports two operation modes:
      - "linear": standard PCA on L2-normalized inputs
      - "rff":    apply a Random Fourier Feature mapping before PCA

    The user chooses the mode via the ``mode`` argument. If ``mode`` is
    ``None`` the behaviour is inferred from whether ``gamma`` or ``M`` is
    provided (those enable RFF).

    See https://arxiv.org/pdf/2505.15284 for motivation.
    """

    mu: torch.Tensor
    u: torch.Tensor
    s: torch.Tensor
    s_acc: torch.Tensor
    u_q: torch.Tensor
    u_q_dot: torch.Tensor
    q: int = 1
    exp_var: float | None = None
    n_comp: int | None = None
    gamma: float | None = None
    M: int | None = None
    mode: str = "linear"

    def __init__(
        self,
        exp_var: float | None = None,
        n_comp: int | None = None,
        gamma: float | None = None,
        M: int | None = None,
        mode: str | None = None,
    ) -> None:
        """Initialise TensorPCA.

        Parameters
        ----------
        exp_var:
            Fraction of explained variance to retain (0 < exp_var <= 1).
            If ``None``, the number of components must be supplied via
            ``n_comp``.
        n_comp:
            If provided, forces PCA to use this many components. If
            specified, takes precedence over ``exp_var``.
        gamma, M:
            Parameters for the Random Fourier Feature mapping. If either
            is provided the RFF branch is enabled (unless ``mode`` is set
            explicitly). If both are ``None`` the linear PCA branch is
            used.
        mode:
            One of "linear" or "rff". If ``None`` the mode is inferred
            from the presence of ``gamma``/``M``.
        """
        super().__init__()
        # Validate mutual exclusivity: only one of exp_var or n_comp may be set
        if (n_comp is not None) and (exp_var is not None):
            raise warnings.warn(
                "Both exp_var and n_comp are provided. n_comp will take precedence."
            )

        if exp_var is not None:
            if not (exp_var > 0.0 and exp_var <= 1.0):
                raise ValueError("exp_var must be in the interval (0, 1]")

        if n_comp is not None:
            if not (isinstance(n_comp, int) and n_comp > 0):
                raise ValueError(
                    "n_comp must be a positive integer if provided"
                )

        if mode is not None and mode not in ("linear", "rff"):
            raise ValueError("mode must be either 'linear' or 'rff'")

        # Infer mode when not explicitly provided
        inferred_mode = (
            "rff" if (gamma is not None or M is not None) else "linear"
        )
        self.mode = mode or inferred_mode

        self.exp_var = exp_var
        self.n_comp = n_comp
        self.gamma = gamma
        self.M = M

    @staticmethod
    def _l2_normalize(X: torch.Tensor) -> torch.Tensor:
        """L2 normalization of an input tensor.

        Normalises rows to unit L2 norm; zero vectors remain zero.
        """
        denom = torch.linalg.norm(X, ord=2, dim=-1, keepdims=True)
        denom = denom + 1e-10
        X = X / denom
        return X.contiguous()

    def _rff(self, X: torch.Tensor) -> torch.Tensor:
        """Apply Random Fourier Features mapping to X.

        This is a helper that respects the module's mode, device and dtype.
        """
        if self.mode != "rff" or self.gamma is None or self.M is None:
            return X
        _, D = X.shape
        if self.M <= D:
            raise ValueError("RFF dimension M must be greater than input dim D")

        device = X.device
        dtype = X.dtype
        w = math.sqrt(2.0 * float(self.gamma)) * torch.normal(
            mean=0.0, std=1.0, size=(self.M, D), device=device, dtype=dtype
        )
        u = 2.0 * math.pi * torch.rand(self.M, device=device, dtype=dtype)
        X = math.sqrt(2.0 / float(self.M)) * torch.cos(X @ w.T + u.unsqueeze(0))
        return X.contiguous()

    def _preprocess(self, X: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing pipeline according to the selected mode.

        Steps:
          - Move module to input device
          - L2 normalize rows
          - Optionally apply RFF
          - Centre by the training mean
        """
        self.to(device=X.device)
        X = self._l2_normalize(X)
        if self.mode == "rff":
            X = self._rff(X)
        X = X - self.mu
        return X.contiguous()

    def fit(self, X: torch.Tensor, Y: None = None) -> None:
        """Fit PCA on input tensor X.

        The behaviour depends on the selected mode: if ``rff`` the RFF
        mapping is applied before computing the covariance; otherwise
        standard PCA is performed on the normalized inputs.
        """
        assert X is not None
        X = self._l2_normalize(X)
        if self.mode == "rff":
            X = self._rff(X)
        self.mu = X.mean(dim=0)
        X = X - self.mu

        K = X.T @ X
        self.u, self.s, _ = torch.linalg.svd(K)
        self.s_acc = torch.cumsum(self.s, 0) / (self.s.sum() + 1e-20)

        if self.n_comp is not None:
            q = int(self.n_comp)
            q = max(1, q)
            q = min(q, int(self.s.numel()))
            self.q = q
            explained = self.s_acc[self.q - 1].item()
            logger.info(
                "Using user-specified n_comp=%s; explained variance at this "
                "dimension is %s.",
                self.q,
                f"{explained:.4f}",
            )
        else:
            q_idx = (self.s_acc >= self.exp_var).nonzero()[0][0]
            self.q = max(1, int(q_idx.item()) + 1)
            explained = self.s_acc[self.q - 1].item()
            logger.info(
                "Explained variance of %s reached at dimension %s.",
                f"{explained:.4f}",
                self.q,
            )

        self.u_q = self.u[:, : self.q]
        self.u_q_dot = self.u_q @ self.u_q.T

    def fit_transform(self, X: torch.Tensor, Y: None = None) -> torch.Tensor:
        """Fit PCA on X and return transformed principal components."""
        self.fit(X, Y)
        return self.transform(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Project input samples onto the retained principal components.

        Returns projected components of shape (N, q).
        """
        assert isinstance(X, torch.Tensor)
        X = self._preprocess(X)
        X_proj = X @ self.u_q
        return X_proj

    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        """Reconstruct samples from principal component scores.

        Returns reconstructed samples in the preprocessed space.
        """
        assert isinstance(Z, torch.Tensor)
        X_rec = (self.u_q @ Z.T).T
        return X_rec

    def reconstruct(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct an input and return the L2 reconstruction error."""
        X_p = self._preprocess(X)
        X_rec = self.inverse_transform(self.transform(X))
        error = torch.linalg.norm(X_p - X_rec, ord=2, dim=1)
        return X_rec, error
