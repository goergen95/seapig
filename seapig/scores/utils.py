"""Utilities accessed by several modules."""

import warnings
from typing import final

import torch

from seapig.utils.logging import get_logger

logger = get_logger(__name__)


@final
class TensorPCA(torch.nn.Module):  # type: ignore[misc]
    """Tensor based PCA with L2 normalized inputs.

    See https://arxiv.org/pdf/2505.15284.
    """

    # based on https://github.com/fanghenshaometeor/ood-kernel-pca/blob/main/CoP_CoRP_ImgNet.py
    mu: torch.Tensor
    u: torch.Tensor
    s: torch.Tensor
    s_acc: torch.Tensor
    u_q: torch.Tensor
    u_q_dot: torch.Tensor
    q: int = 1
    exp_var: float | None = None
    n_comp: int | None = None

    def __init__(
        self,
        exp_var: float | None = None,
        n_comp: int | None = None,
        gamma: float | None = None,
        M: int | None = None,
    ):
        """Initialise TensorPCA.

        Parameters
        ----------
        exp_var:
            Fraction of explained variance to retain (0<exp_var<=1).
            If ``None``, the number of components must be supplied via ``n_comp``.
        n_comp:
            If provided, forces PCA to use this many components. If specified,
            takes precedence over ``exp_var``.
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

        self.exp_var = exp_var
        self.n_comp = n_comp
        self.gamma = gamma
        self.M = M

    @staticmethod
    def _l2_normalize(X: torch.Tensor) -> torch.Tensor:
        """L2 normalization of an input tensor."""
        X = X / (torch.linalg.norm(X, ord=2, dim=-1, keepdims=True) + 1e-10)
        return X.contiguous()

    @staticmethod
    def _rff(
        X: torch.Tensor, gamma: float | None = 3, M: int | None = 4096
    ) -> torch.Tensor:
        if gamma is None or M is None:
            return X
        _, D = X.shape
        assert M > D
        w = torch.sqrt(torch.tensor([2 * gamma])) * torch.normal(
            mean=0, std=torch.ones(size=(M, D))
        )
        u = 2 * torch.pi * torch.rand(M)
        X = torch.sqrt(torch.tensor([2 / M])) * torch.cos(
            X @ w.T + u[torch.newaxis, :]
        )
        return X.contiguous()

    def _preprocess(self, X: torch.Tensor) -> torch.Tensor:
        self.to(device=X.device)
        X = self._l2_normalize(X)
        X = self._rff(X, gamma=self.gamma, M=self.M)
        X = X - self.mu
        return X.contiguous()

    def fit(self, X: torch.Tensor, Y: None = None) -> None:
        """Fitting the PCA based on an input tensor."""
        assert X is not None
        X = self._l2_normalize(X)
        X = self._rff(X, gamma=self.gamma, M=self.M)
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
        """Fit PCA on X and return the transformed principal components.

        This is equivalent to calling ``fit(X)`` followed by ``transform(X)``.
        """
        self.fit(X, Y)
        return self.transform(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Project input samples onto the retained principal components.

        Parameters
        ----------
        X:
            Input tensor of shape (N, D). Returns projected components of
            shape (N, q) where q is the number of retained components.
        """
        assert isinstance(X, torch.Tensor)
        X = self._preprocess(X)
        X_proj = X @ self.u_q
        return X_proj

    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        """Reconstruct samples from principal component scores.

        Parameters
        ----------
        Z:
            Component scores of shape (N, q). Returns reconstructed samples in
            the preprocessed (centered + RFF + normalized) space with shape
            (N, D'). This mirrors the reconstruction returned by
            :meth:`reconstruct` (first element).
        """
        assert isinstance(Z, torch.Tensor)
        X_rec = (self.u_q @ Z.T).T
        return X_rec

    def reconstruct(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct an input and return the error."""
        X_p = self._preprocess(X)
        X_rec = self.inverse_transform(self.transform(X))
        error = torch.linalg.norm(X_p - X_rec, ord=2, dim=1)
        return X_rec, error
