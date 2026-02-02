"""Utilities accessed by several modules."""

from typing import final

import torch


@final
class TensorPCA(torch.nn.Module):
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

    def __init__(
        self,
        exp_var: float = 0.90,
        gamma: float | None = None,
        M: int | None = None,
    ):
        super().__init__()
        assert exp_var > 0.0 and exp_var <= 1.0
        self.exp_var: float = exp_var
        self.gamma = gamma
        self.M = M

        self.register_buffer("mu", torch.tensor([]))
        self.register_buffer("u", torch.tensor([]))
        self.register_buffer("s", torch.tensor([]))
        self.register_buffer("s_acc", torch.tensor([]))
        self.register_buffer("u_q", torch.tensor([]))
        self.register_buffer("u_q_dot", torch.tensor([]))

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
        q_idx = (self.s_acc >= self.exp_var).nonzero()[0][0]
        self.q = max(1, int(q_idx.item()) + 1)
        explained = self.s_acc[self.q - 1].item()
        print(
            f"Explained variance of {explained:.4f} reached at dimension {self.q}."
        )
        self.u_q = self.u[:, : self.q]
        self.u_q_dot = self.u_q @ self.u_q.T

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Reduce an input to its principal components."""
        assert isinstance(X, torch.Tensor)
        X = self._preprocess(X)
        X = X @ self.u_q
        return X

    def reconstruct(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct an input and return the error."""
        X = self._preprocess(X)
        X_rec = (self.u_q_dot @ X.T).T
        error = torch.linalg.norm(X - X_rec, ord=2, dim=1)
        return X_rec, error
