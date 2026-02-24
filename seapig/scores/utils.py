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

        # running accumulators for partial_fit/finalize
        self._n_samples: int = 0
        self._sum_X: torch.Tensor | None = None
        self._sum_outer: torch.Tensor | None = None
        # persistent RFF params (registered as buffers when created)
        # _rff_w : Tensor shape (M, D)
        # _rff_u : Tensor shape (M,)
        self._rff_initialized: bool = False

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
        # Prefer persistent RFF parameters if they exist (registered buffers)
        w_buf = getattr(self, "_rff_w", None)
        u_buf = getattr(self, "_rff_u", None)
        if (
            (w_buf is not None)
            and (u_buf is not None)
            and self._rff_initialized
        ):
            w = w_buf.to(device=device, dtype=dtype)
            u = u_buf.to(device=device, dtype=dtype)
        else:
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
        # ensure consistent dtype with stored mean (may be float64)
        if hasattr(self, "mu") and self.mu is not None:
            X = X.to(self.mu.dtype)
        X = X - self.mu
        return X.contiguous()
        return X.contiguous()

    def fit(self, X: torch.Tensor, Y: None = None) -> None:
        """Fit PCA on the input data X.

        This is a convenience method that runs a single-batch partial fit
        followed by finalization. For large datasets or streaming data, use
        the incremental partial_fit/finalize interface instead.
        """
        # Convenience: run a single-batch partial fit then finalize
        self.reset_partial()
        self.partial_fit(X)
        self.finalize()

    def reset_partial(self) -> None:
        """Reset internal accumulators used for partial fitting."""
        self._n_samples = 0
        self._sum_X = None
        self._sum_outer = None
        # do not reset RFF params here; keep them if already initialized

    def partial_fit(self, X: torch.Tensor) -> None:
        """Process a single batch for incremental PCA.

        This accumulates sufficient statistics (sum of samples and
        sum of outer products) which are later finalised in
        :meth:`finalize` to produce the PCA decomposition.
        """
        assert X is not None
        # apply normalization and RFF (RFF params are persisted)
        X = self._l2_normalize(X)
        # Only initialize/apply RFF if both gamma and M are provided
        if self.mode == "rff" and (
            self.gamma is not None and self.M is not None
        ):
            # initialize RFF parameters on first call
            if not self._rff_initialized:
                _, D = X.shape
                if self.M <= D:
                    raise ValueError(
                        "RFF dimension M must be greater than input dim D"
                    )
                device = X.device
                dtype = X.dtype
                w = math.sqrt(2.0 * float(self.gamma)) * torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=(self.M, D),
                    device=device,
                    dtype=dtype,
                )
                u = (
                    2.0
                    * math.pi
                    * torch.rand(self.M, device=device, dtype=dtype)
                )
                # register as buffers so they move with the module
                self.register_buffer("_rff_w", w)
                self.register_buffer("_rff_u", u)
                self._rff_initialized = True
            X = self._rff(X)

        m = X.shape[0]
        # accumulate in double precision for numerical stability and to match
        # scikit-learn's float64 behaviour
        batch_sum = X.sum(dim=0).to(torch.float64)
        batch_outer = (X.T @ X).to(torch.float64)

        if self._n_samples == 0:
            self._sum_X = batch_sum.clone()
            self._sum_outer = batch_outer.clone()
            self._n_samples = m
        else:
            assert self._sum_X is not None and self._sum_outer is not None
            self._sum_X = self._sum_X + batch_sum
            self._sum_outer = self._sum_outer + batch_outer
            self._n_samples += m

    def finalize(self) -> None:
        """Finalize partial fit: compute covariance SVD and set PCA params.

        This method computes the overall mean and centred covariance from
        accumulated sums and performs SVD to extract principal components.
        """
        if self._n_samples == 0:
            raise RuntimeError("No data provided to partial_fit/finalize")
        assert self._sum_X is not None and self._sum_outer is not None

        # perform computations in float64 then cast results back to module dtype
        n = float(self._n_samples)
        mu64 = self._sum_X / n
        K64 = self._sum_outer - n * (mu64.unsqueeze(1) @ mu64.unsqueeze(0))

        # SVD in float64 for numerical agreement with sklearn
        u64, s64, _ = torch.linalg.svd(K64)
        s_acc64 = torch.cumsum(s64, 0) / (s64.sum() + 1e-20)

        # store in module attributes in float64 for numerical fidelity
        self.mu = mu64
        self.u = u64
        self.s = s64
        self.s_acc = s_acc64

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
            if self.exp_var is None:
                raise ValueError(
                    "Either n_comp or exp_var must be provided for PCA selection"
                )
            q_idx_tensor = (self.s_acc >= self.exp_var).nonzero()
            if q_idx_tensor.numel() == 0:
                # keep all components
                self.q = int(self.s.numel())
            else:
                q_idx = q_idx_tensor[0][0]
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
