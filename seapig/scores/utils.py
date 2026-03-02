"""Utilities accessed by several modules.

This module contains small helpers used by score implementations. The
primary export is :class:`TensorPCA`, a torch-compatible PCA that
supports both standard (linear) PCA on L2-normalized inputs and an
optional Random Fourier Feature (RFF) mapping prior to PCA.

Saving / loading
-----------------
- TensorPCA registers its learned state as persistent buffers (for
  example: ``mu``, ``u``, ``s``, ``s_acc``, ``u_q``, ``u_q_dot``) and
  RFF parameters (``_rff_w``, ``_rff_u``, ``_rff_initialized``). This
  makes the module safe to persist with :func:`torch.save` and
  :meth:`torch.load` via the standard state-dict API::

      torch.save(tpca.state_dict(), "tpca.pt")

      tpca2 = TensorPCA(n_components=..., gamma=..., M=..., mode=...)
      sd = torch.load("tpca.pt")
      tpca2.load_state_dict(sd)

- The class implements a custom ``_load_from_state_dict`` which accepts
  incoming tensors of arbitrary shapes (including the empty placeholders
  created at construction) and will set or register buffers to avoid
  size-mismatch errors when loading into fresh instances.

- Note: PCA internals are stored in ``float64`` for numerical fidelity;
  during preprocessing inputs are cast to match the stored mean's dtype.
"""

import math
from typing import final

import torch

from seapig.utils.logging import get_logger

logger = get_logger(__name__)


@final
class TensorPCA(torch.nn.Module):
    """Tensor-based PCA with L2-normalized inputs.

    Operation modes
    ---------------
    - ``linear``: standard PCA on L2-normalized rows
    - ``rff``: apply a Random Fourier Feature mapping before PCA

    Mode selection follows the constructor arguments: providing ``gamma``
    or ``M`` enables the RFF branch unless ``mode`` is set explicitly.

    Saving / loading
    -----------------
    Persist the module with the normal PyTorch state-dict API. The
    module registers persistent buffers for PCA and RFF state, so
    ``torch.save(instance.state_dict())`` and ``instance.load_state_dict(torch.load(path))``
    are the recommended workflow. The custom ``_load_from_state_dict``
    accepts placeholder or differently-shaped tensors and will set or
    register buffers to avoid size-mismatch errors on fresh instances.

    See https://arxiv.org/pdf/2505.15284 for motivation.
    """

    mu: torch.Tensor
    u: torch.Tensor
    s: torch.Tensor
    s_acc: torch.Tensor
    u_q: torch.Tensor
    u_q_dot: torch.Tensor
    _rff_w: torch.Tensor
    _rff_u: torch.Tensor
    _rff_initialized: torch.Tensor
    # these may be set to None during reset_partial
    _sum_X: torch.Tensor | None
    _sum_outer: torch.Tensor | None

    def __init__(
        self,
        n_components: int | float = 0.90,
        gamma: float | None = None,
        M: int | None = None,
        mode: str | None = None,
    ) -> None:
        """Initialise TensorPCA.

        Parameters
        ----------
        n_components:
            If an int, forces PCA to use this many components. If a float in
            (0, 1], the fraction of explained variance to retain. Defaults to
            ``0.90`` (retain components that explain 90% of variance) when
            not specified.
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

        # Validate n_components: must be int>0 or float in (0,1]
        if isinstance(n_components, bool):
            # bool is subclass of int — disallow
            raise ValueError(
                "n_components must be an int>0 or a float in (0,1]"
            )

        if isinstance(n_components, int):
            if n_components <= 0:
                raise ValueError(
                    "n_components must be a positive integer if provided"
                )
            self.n_components: int | float = int(n_components)
        elif isinstance(n_components, float):
            if not (n_components > 0.0 and n_components <= 1.0):
                raise ValueError(
                    "n_components as float must be in the interval (0, 1]"
                )
            self.n_components: int | float = n_components  # type: ignore[no-redef]
        else:
            raise ValueError(
                "n_components must be either an int>0 or a float in (0,1]"
            )

        if mode is not None and mode not in ("linear", "rff"):
            raise ValueError("mode must be either 'linear' or 'rff'")

        # Infer mode when not explicitly provided
        inferred_mode = (
            "rff" if (gamma is not None or M is not None) else "linear"
        )
        self.mode = mode or inferred_mode

        self.gamma = gamma
        self.M = M

        # register persistent buffers so the module can be saved/loaded
        self.register_buffer("mu", torch.tensor([], dtype=torch.float64))
        self.register_buffer("u", torch.tensor([], dtype=torch.float64))
        self.register_buffer("s", torch.tensor([], dtype=torch.float64))
        self.register_buffer("s_acc", torch.tensor([], dtype=torch.float64))
        self.register_buffer("u_q", torch.tensor([], dtype=torch.float64))
        self.register_buffer("u_q_dot", torch.tensor([], dtype=torch.float64))

        # RFF parameters (will be created during partial_fit when needed)
        self.register_buffer("_rff_w", torch.tensor([], dtype=torch.float32))
        self.register_buffer("_rff_u", torch.tensor([], dtype=torch.float32))
        self.register_buffer(
            "_rff_initialized", torch.tensor(False, dtype=torch.bool)
        )

        # running accumulators for partial_fit/finalize (float64)
        self.register_buffer("_sum_X", torch.tensor([], dtype=torch.float64))
        self.register_buffer(
            "_sum_outer", torch.tensor([], dtype=torch.float64)
        )

        # keep this as a plain python int for control flow
        self._n_samples: int = 0

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
            if not bool(
                getattr(
                    self,
                    "_rff_initialized",
                    torch.tensor(False, dtype=torch.bool),
                ).item()
            ):
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
                # set the BoolTensor buffer in-place so the buffer identity
                # is preserved (important for state_dict/save/load)
                getattr(self, "_rff_initialized").fill_(True)
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
            and (getattr(self, "_rff_initialized", None) is not None)
            and bool(getattr(self, "_rff_initialized").item())
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

        # Decide number of components q based on the unified n_components
        if isinstance(self.n_components, int):
            q = self.n_components
            q = max(1, q)
            q = min(q, int(self.s.numel()))
            self.q = q
            explained = self.s_acc[self.q - 1].item()
            logger.info(
                "Using user-specified n_components=%s; explained variance at "
                "this dimension is %s.",
                self.q,
                f"{explained:.4f}",
            )
        else:
            # n_components is a float interpreted as the explained-variance
            exp_var = float(self.n_components)
            q_idx_tensor = (self.s_acc >= exp_var).nonzero()
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

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Load state dict with custom logic for PCA parameters.

        Custom state loader that accepts tensors of arbitrary shape for
        the registered PCA buffers. This replaces the default copying logic
        which would raise size-mismatch errors when placeholder buffers
        (empty tensors) exist on a fresh instance.

        We simply set the attribute / buffer to the incoming tensor so that
        load_state_dict populates the module correctly.
        """
        # list of persistent buffers we manage
        buf_names = [
            "mu",
            "u",
            "s",
            "s_acc",
            "u_q",
            "u_q_dot",
            "_rff_w",
            "_rff_u",
            "_rff_initialized",
            "_sum_X",
            "_sum_outer",
        ]

        for name in buf_names:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                if hasattr(self, name):
                    try:
                        setattr(self, name, val)
                    except Exception:  # pragma: no cover
                        # fallback to register_buffer if direct set fails
                        self.register_buffer(name, val)
                else:
                    self.register_buffer(name, val)
