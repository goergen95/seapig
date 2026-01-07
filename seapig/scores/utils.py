"""Utilities accessed by several modules."""

import torch


class TensorPCA:
    """Tensor based PCA with L2 normalized inputs.

    See https://arxiv.org/pdf/2505.15284.
    """

    # based on https://github.com/fanghenshaometeor/ood-kernel-pca/blob/main/CoP_CoRP_ImgNet.py

    def __init__(
        self,
        exp_var: float = 0.90,
        gamma: int | None = None,
        M: int | None = None,
    ):
        self.exp_var = exp_var
        self.gamma = gamma
        self.M = M

    def to(self, device: str = "cpu") -> None:
        """Put all tensors to the specified device."""
        self.mu = self.mu.to(device=device)
        self.u = self.u.to(device=device)
        self.s = self.s.to(device=device)
        self.s_acc = self.s_acc.to(device=device)
        self.q = self.q.to(device=device)
        self.u_q = self.u_q.to(device=device)
        self.u_q_dot = self.u_q_dot.to(device=device)

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
        X = self._l2_normalize(X)
        X = self._rff(X, gamma=self.gamma, M=self.M)
        self.mu = X.mean(dim=0)
        X = X - self.mu
        K = X.T @ X
        self.u, self.s, _ = torch.linalg.svd(K)
        self.s_acc = torch.cumsum(self.s, 0) / self.s.sum()
        self.q = (self.s_acc >= self.exp_var).nonzero()[0][0]
        print(
            f"Explained variance of {self.s_acc[self.q].item():.4f} reached at dimension {self.q.item()}."
        )
        self.u_q = self.u[:, : self.q]
        self.u_q_dot = self.u_q @ self.u_q.T

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Reduce an input to its principal components."""
        X = self._preprocess(X)
        X = X @ self.u_q
        return X

    def reconstruct(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct an input and return the error."""
        X = self._preprocess(X)
        X_rec = (self.u_q_dot @ X.T).T
        error = torch.linalg.norm(X - X_rec, ord=2, dim=1)
        return X_rec, error
