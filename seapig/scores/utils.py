"""Utilities accessed by several modules."""

import torch


class TensorPCA:
    """Tensor based PCA with L2 normalized inputs."""

    # based on https://github.com/fanghenshaometeor/ood-kernel-pca/blob/main/CoP_CoRP_ImgNet.py

    def __init__(self, exp_var: float = 0.90):
        self.exp_var = exp_var

    @staticmethod
    def _l2_normalize(X: torch.Tensor) -> torch.Tensor:
        """L2 normalization of an input tensor."""
        X = X / (torch.linalg.norm(X, ord=2, dim=-1, keepdims=True) + 1e-10)
        return X.contiguous()

    def _preprocess(self, X: torch.Tensor) -> torch.Tensor:
        X = X - self.mu
        return self._l2_normalize(X)

    def fit(self, X: torch.Tensor, Y: None = None) -> None:
        """Fitting the PCA based on an input tensor."""
        X = self._l2_normalize(X)
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
