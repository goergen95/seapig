"""Utilities accessed by several modules."""

import warnings
from collections.abc import Mapping
from typing import Any

import torch

from seapig.utils.logging import get_logger

logger = get_logger(__name__)


class TensorPCA(torch.nn.Module):
    """Torch-native PCA with optional whitening, partial fitting, and reproducible SVD.

    Implements standard PCA using either a fixed number of components
    (``n_components``) or an explained-variance threshold (``exp_var``).
    Exactly one of these must be specified.

    Supports incremental fitting via :meth:`partial_fit` + :meth:`finalize`
    for memory-efficient processing of large datasets without storing all data.

    Parameters
    ----------
    n_components : int or None, optional
        Number of principal components to retain.  Mutually exclusive with
        ``exp_var``.
    exp_var : float or None, optional
        Minimum cumulative explained variance ratio (0, 1] to determine the
        number of components automatically.  Mutually exclusive with
        ``n_components``.
    whiten : bool, optional
        If ``True``, scale projected outputs to unit variance on the training
        set.  Defaults to ``False``.
    svd_solver : str, optional
        SVD algorithm.  One of ``{"auto", "full", "randomized"}``.
        ``"auto"`` uses a heuristic to pick between ``"full"`` and
        ``"randomized"``.  Defaults to ``"auto"``.
    batch_size : int or None, optional
        Batch size hint used only for the ``"auto"`` solver heuristic.
    eps : float, optional
        Small constant for numerical stability (e.g. in whitening divisors and
        explained-variance normalisation).  Defaults to ``1e-12``.
    random_state : int, torch.Generator, or None, optional
        Seed or generator for the randomized SVD draws.  Pass an ``int`` for
        fully reproducible results across repeated calls.
    device : torch.device or None, optional
        Target device for buffers.  Defaults to the device of the first
        tensor passed to :meth:`fit`.
    dtype : torch.dtype, optional
        Storage dtype for model parameters/buffers.  SVD is always performed
        in ``float64`` for numerical stability.  Defaults to ``torch.float32``.

    Attributes
    ----------
    components_ : torch.Tensor, shape (k, D)
        Principal axes (rows are components).
    mean_ : torch.Tensor, shape (D,)
        Per-feature empirical mean of the training data.
    singular_values_ : torch.Tensor, shape (k,)
        Singular values corresponding to each component.
    explained_variance_ : torch.Tensor, shape (k,)
        Variance explained by each component.
    explained_variance_ratio_ : torch.Tensor, shape (k,)
        Fraction of total variance explained by each component.
    n_samples_seen_ : torch.Tensor, scalar (int64)
        Total number of samples processed during fitting.

    Examples
    --------
    >>> import torch
    >>> from seapig.scores.utils import TensorPCA
    >>> X = torch.randn(200, 32)
    >>> pca = TensorPCA(n_components=8, whiten=True, random_state=0)
    >>> proj = pca.fit_transform(X)
    >>> recon = pca.inverse_transform(proj)
    >>> err = pca.predict(X)          # per-sample reconstruction error
    >>> X_rec, err2 = pca.reconstruct(X)   # legacy API

    Streaming / incremental fitting:

    >>> pca2 = TensorPCA(exp_var=0.90)
    >>> for batch in [X[:100], X[100:]]:
    ...     pca2.partial_fit(batch)
    >>> pca2.finalize()
    TensorPCA(exp_var=0.9)
    >>> proj2 = pca2.transform(X)
    """

    components_: torch.Tensor | None
    mean_: torch.Tensor | None
    singular_values_: torch.Tensor | None
    explained_variance_: torch.Tensor | None
    explained_variance_ratio_: torch.Tensor | None
    n_samples_seen_: torch.Tensor
    _scatter: torch.Tensor | None

    def __init__(
        self,
        n_components: int | None = None,
        exp_var: float | None = None,
        *,
        whiten: bool = False,
        svd_solver: str = "auto",
        batch_size: int | None = None,
        eps: float = 1e-12,
        random_state: int | torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.exp_var = exp_var
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.batch_size = batch_size
        self.eps = eps
        self.dtype = dtype
        self._init_device = device

        # Store seed so we can recreate a fresh generator on each fit call,
        # guaranteeing reproducibility across repeated fits.
        self._seed: int | None = None
        if isinstance(random_state, int):
            self._seed = random_state
        elif isinstance(random_state, torch.Generator):
            # Capture the current state for later replay.
            self._seed = int(
                torch.randint(0, 2**31, (1,), generator=random_state).item()
            )

        self.register_buffer("components_", None)
        self.register_buffer("mean_", None)
        self.register_buffer("singular_values_", None)
        self.register_buffer("explained_variance_", None)
        self.register_buffer("explained_variance_ratio_", None)
        self.register_buffer(
            "n_samples_seen_", torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer("_scatter", None)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_params(self) -> None:
        """Raise ValueError when n_components/exp_var are not mutually exclusive."""
        if self.n_components is None and self.exp_var is None:
            raise ValueError(
                "Exactly one of 'n_components' or 'exp_var' must be specified. "
                "Neither was provided."
            )
        if self.n_components is not None and self.exp_var is not None:
            raise ValueError(
                "Only one of 'n_components' or 'exp_var' may be specified, not both."
            )
        if self.exp_var is not None and not (0.0 < self.exp_var <= 1.0):
            raise ValueError(
                f"'exp_var' must be in (0, 1], got {self.exp_var!r}."
            )
        if self.n_components is not None and self.n_components < 1:
            raise ValueError(
                f"'n_components' must be >= 1, got {self.n_components!r}."
            )

    def _check_fitted(self) -> None:
        """Raise RuntimeError if components_ have not been computed yet."""
        if self.components_ is None:
            raise RuntimeError(
                "TensorPCA has not been fitted yet. "
                "Call fit() or call partial_fit() followed by finalize()."
            )

    # ------------------------------------------------------------------
    # Device / dtype helpers
    # ------------------------------------------------------------------

    def _get_device(self, X: torch.Tensor) -> torch.device:
        """Return the target device, preferring the user-supplied init device."""
        if self._init_device is not None:
            return torch.device(self._init_device)
        return X.device

    def _cast(self, X: torch.Tensor) -> torch.Tensor:
        """Move X to the target device and cast to self.dtype."""
        return X.to(device=self._get_device(X), dtype=self.dtype)

    def _make_rng(self, device: torch.device) -> torch.Generator | None:
        """Return a fresh Generator seeded from self._seed, or None."""
        if self._seed is None:
            return None
        g = torch.Generator(device=device)
        g.manual_seed(self._seed)
        return g

    @staticmethod
    def _dof(n_samples: int) -> int:
        """Degrees of freedom: max(n_samples - 1, 1), safe for n_samples <= 1."""
        return max(n_samples - 1, 1)

    def _reconstruction_error(
        self, X: torch.Tensor, X_rec: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample L2 reconstruction error."""
        err: torch.Tensor = torch.linalg.norm(X - X_rec, dim=1)
        return err

    # ------------------------------------------------------------------
    # SVD helpers
    # ------------------------------------------------------------------

    def _randomized_svd(
        self,
        X: torch.Tensor,
        k: int,
        n_oversampling: int = 10,
        n_power_iter: int = 4,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomized SVD — returns (singular_values, Vt) for the top-k components.

        Algorithm: random projection + power iterations + thin QR + small SVD.

        Parameters
        ----------
        X : torch.Tensor, shape (N, D), float64
        k : int
            Number of components desired.
        n_oversampling : int
            Extra sketch columns for accuracy.
        n_power_iter : int
            Number of power-iteration steps.
        generator : torch.Generator or None
            RNG for the random projection matrix.
        """
        _N, D = X.shape
        sketch_size = min(k + n_oversampling, min(X.shape[0], D))

        # Random projection
        Omega = torch.randn(
            D, sketch_size, device=X.device, dtype=X.dtype, generator=generator
        )
        Y = X @ Omega  # (N, sketch_size)

        # Power iterations
        for _ in range(n_power_iter):
            Y = X @ (X.T @ Y)

        # Thin QR
        Q, _ = torch.linalg.qr(Y)  # (N, sketch_size)

        # Project into small space and SVD
        B = Q.T @ X  # (sketch_size, D)
        _, s, Vt = torch.linalg.svd(B, full_matrices=False)

        return s[:k], Vt[:k, :]

    def _choose_solver(self, N: int, D: int, k: int) -> str:
        """Apply heuristic to select between 'full' and 'randomized'."""
        max_k = min(N, D)
        if k < int(0.8 * max_k) and max_k > 500:
            return "randomized"
        return "full"

    def _run_svd(
        self, X_centered: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run SVD and return (singular_values[:k], Vt[:k]) in self.dtype."""
        N, D = X_centered.shape
        # SVD in float64 for numerical stability
        X64 = X_centered.to(dtype=torch.float64)

        solver = self.svd_solver
        if solver == "auto":
            solver = self._choose_solver(N, D, k)

        if solver == "randomized":
            g = self._make_rng(X_centered.device)
            s, Vt = self._randomized_svd(X64, k=k, generator=g)
        else:
            # Full SVD (incremental falls back to full for now)
            _, s, Vt = torch.linalg.svd(X64, full_matrices=False)
            s = s[:k]
            Vt = Vt[:k, :]

        return s.to(self.dtype), Vt.to(self.dtype)

    # ------------------------------------------------------------------
    # Buffer finalisation
    # ------------------------------------------------------------------

    def _set_buffers(
        self, s_full: torch.Tensor, Vt_full: torch.Tensor, n_samples: int
    ) -> None:
        """Populate all registered buffers from full SVD outputs.

        If ``exp_var`` is set the smallest k satisfying the cumulative
        explained-variance threshold is determined here and buffers are
        truncated accordingly.
        """
        # Compute explained variance from all singular values (for ratio)
        ev_full = (s_full**2) / self._dof(n_samples)
        ev_total = ev_full.sum() + self.eps

        if self.exp_var is not None:
            ev_ratio_cum = torch.cumsum(ev_full / ev_total, dim=0)
            hits = (ev_ratio_cum >= self.exp_var).nonzero()
            k = int(hits[0][0].item()) + 1 if len(hits) > 0 else len(s_full)
            logger.info(
                "exp_var=%.4f reached with k=%d components "
                "(cumulative ratio=%.4f).",
                self.exp_var,
                k,
                ev_ratio_cum[k - 1].item(),
            )
        else:
            assert self.n_components is not None
            k = min(self.n_components, len(s_full))
            if k < self.n_components:
                warnings.warn(
                    f"n_components={self.n_components} exceeds the maximum "
                    f"possible rank ({k}); capping at {k}.",
                    UserWarning,
                    stacklevel=3,
                )

        s_k = s_full[:k]
        Vt_k = Vt_full[:k, :]
        ev_k = (s_k**2) / self._dof(n_samples)

        self.components_ = Vt_k
        self.singular_values_ = s_k
        self.explained_variance_ = ev_k
        self.explained_variance_ratio_ = ev_k / ev_total

        logger.debug(
            "TensorPCA fitted: k=%d components, total explained variance=%.4f.",
            k,
            self.explained_variance_ratio_.sum().item(),
        )

    # ------------------------------------------------------------------
    # Public fitting API
    # ------------------------------------------------------------------

    def fit(self, X: torch.Tensor, Y: None = None) -> "TensorPCA":
        """Fit PCA on a full dataset.

        Parameters
        ----------
        X : torch.Tensor, shape (N, D)
            Training data.
        Y : None
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        TensorPCA
            self
        """
        self._validate_params()
        X = self._cast(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}.")

        N, D = X.shape
        self.n_samples_seen_ = torch.tensor(
            N, dtype=torch.long, device=X.device
        )
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_

        # Determine how many singular vectors to compute
        max_k = min(N - 1, D)
        if self.n_components is not None:
            k = min(self.n_components, max_k)
        else:
            k = max_k  # compute all for exp_var selection

        s, Vt = self._run_svd(X_centered, k)
        self._set_buffers(s, Vt, N)
        return self

    def partial_fit(self, X: torch.Tensor) -> "TensorPCA":
        """Incrementally update running statistics with a new batch.

        Components are *not* computed until :meth:`finalize` is called.
        Use this method to stream large datasets in batches and call
        :meth:`finalize` once all batches have been processed.

        Parameters
        ----------
        X : torch.Tensor, shape (batch_N, D)
            Batch of training data.

        Returns
        -------
        TensorPCA
            self
        """
        X = self._cast(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}.")

        batch_n = X.shape[0]

        if int(self.n_samples_seen_.item()) == 0:
            # First batch — initialise accumulators.
            self.mean_ = X.mean(dim=0)
            X_centered = X - self.mean_
            self._scatter = X_centered.T @ X_centered
            self.n_samples_seen_ = torch.tensor(
                batch_n, dtype=torch.long, device=X.device
            )
        else:
            # Numerically-stable incremental mean and scatter update.
            n_old = int(self.n_samples_seen_.item())
            n_new = n_old + batch_n
            assert self.mean_ is not None
            assert self._scatter is not None

            old_mean = self.mean_
            new_mean = old_mean + (X.sum(dim=0) - batch_n * old_mean) / n_new

            X_centered_new = X - new_mean
            # Correction term: re-express the old scatter around the updated mean.
            # S_new = S_old + n_old*(old_mean - new_mean)^T*(old_mean - new_mean)
            #       + X_new_centred.T @ X_new_centred
            # This follows from expanding sum_i (x_i - new_mean)^2 across old and
            # new points and collecting the cross-term that arises from the mean shift.
            delta = old_mean - new_mean
            scatter_correction = n_old * torch.outer(delta, delta)

            self._scatter = (
                self._scatter
                + X_centered_new.T @ X_centered_new
                + scatter_correction
            )
            self.mean_ = new_mean
            self.n_samples_seen_ = torch.tensor(
                n_new, dtype=torch.long, device=X.device
            )

        return self

    def finalize(self) -> "TensorPCA":
        """Compute components from accumulated scatter statistics.

        Must be called after all :meth:`partial_fit` batches before
        :meth:`transform` or :meth:`predict` can be used.

        Returns
        -------
        TensorPCA
            self

        Raises
        ------
        RuntimeError
            If :meth:`partial_fit` has not been called yet.
        ValueError
            If neither or both of ``n_components`` and ``exp_var`` are set.
        """
        self._validate_params()
        n = int(self.n_samples_seen_.item())
        if n == 0:
            raise RuntimeError(
                "No data has been accumulated. Call partial_fit() first."
            )
        assert self._scatter is not None
        assert self.mean_ is not None

        # Eigendecomposition of the symmetric scatter matrix.
        # scatter = X_c.T @ X_c = V @ Λ @ V.T  =>  singular values = sqrt(λ).
        S64 = self._scatter.to(torch.float64)
        eigenvalues, eigenvectors = torch.linalg.eigh(S64)

        # eigh returns ascending order; reverse for descending singular values.
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx].clamp(min=0.0)
        eigenvectors = eigenvectors[:, idx]

        s = torch.sqrt(eigenvalues).to(self.dtype)
        Vt = eigenvectors.T.to(self.dtype)

        self._set_buffers(s, Vt, n)
        return self

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit PCA on X and return the projected representation.

        Parameters
        ----------
        X : torch.Tensor, shape (N, D)

        Returns
        -------
        torch.Tensor, shape (N, k)
        """
        self.fit(X)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Transform / inverse
    # ------------------------------------------------------------------

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Project X onto the principal components.

        Parameters
        ----------
        X : torch.Tensor, shape (N, D)

        Returns
        -------
        torch.Tensor, shape (N, k)
        """
        self._check_fitted()
        assert self.mean_ is not None
        assert self.components_ is not None

        X = self._cast(X)
        X_proj = (X - self.mean_) @ self.components_.T  # (N, k)

        if self.whiten:
            assert self.singular_values_ is not None
            n = int(self.n_samples_seen_.item())
            scale = self.singular_values_ / (self._dof(n) ** 0.5) + self.eps
            X_proj = X_proj / scale

        return X_proj

    def inverse_transform(self, X_proj: torch.Tensor) -> torch.Tensor:
        """Reconstruct data from a projected representation.

        Parameters
        ----------
        X_proj : torch.Tensor, shape (N, k)

        Returns
        -------
        torch.Tensor, shape (N, D)
        """
        self._check_fitted()
        assert self.mean_ is not None
        assert self.components_ is not None

        if self.whiten:
            assert self.singular_values_ is not None
            n = int(self.n_samples_seen_.item())
            scale = self.singular_values_ / (self._dof(n) ** 0.5) + self.eps
            X_proj = X_proj * scale

        return X_proj @ self.components_ + self.mean_

    # ------------------------------------------------------------------
    # Legacy / convenience API
    # ------------------------------------------------------------------

    def predict(
        self,
        X: torch.Tensor,
        return_reconstruction: bool = False,
        return_error: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return per-sample reconstruction error and/or reconstruction.

        Convenience wrapper combining :meth:`transform` and
        :meth:`inverse_transform`.  By default returns only the per-sample
        reconstruction error (lower = more confident / closer to the training
        distribution).

        Parameters
        ----------
        X : torch.Tensor, shape (N, D)
            Input samples.
        return_reconstruction : bool, optional
            Include the reconstructed tensor in the return value.
        return_error : bool, optional
            Include the per-sample L2 reconstruction error.  Defaults to True.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            * Only error (default): error tensor of shape (N,).
            * Only reconstruction: tensor of shape (N, D).
            * Both (``return_reconstruction=True``): ``(error, X_rec)``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted / finalized yet.
        """
        self._check_fitted()
        X = self._cast(X)
        X_rec = self.inverse_transform(self.transform(X))
        error: torch.Tensor = self._reconstruction_error(X, X_rec)

        if return_reconstruction and return_error:
            return error, X_rec
        elif return_reconstruction:
            return X_rec
        else:
            return error

    def reconstruct(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct X and return ``(X_reconstructed, per_sample_error)``.

        Compatibility alias providing the same two-tuple interface as the
        legacy ``TensorPCA`` API used by :class:`~seapig.scores.pca.PCAScore`.

        Parameters
        ----------
        X : torch.Tensor, shape (N, D)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(X_rec, error)`` where ``X_rec`` has shape ``(N, D)`` and
            ``error`` has shape ``(N,)``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted / finalized yet.
        """
        self._check_fitted()
        X = self._cast(X)
        X_rec = self.inverse_transform(self.transform(X))
        return X_rec, self._reconstruction_error(X, X_rec)

    # ------------------------------------------------------------------
    # Static utility methods (kept for backward compatibility)
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(X: torch.Tensor) -> torch.Tensor:
        """Row-wise L2 normalisation of a 2-D tensor."""
        result: torch.Tensor = (
            X / (torch.linalg.norm(X, ord=2, dim=-1, keepdims=True) + 1e-10)
        ).contiguous()
        return result

    @staticmethod
    def _rff(
        X: torch.Tensor, gamma: float | None = 3, M: int | None = 4096
    ) -> torch.Tensor:
        """Random Fourier Features (RFF) transformation.

        Returns ``X`` unchanged when either ``gamma`` or ``M`` is ``None``.
        """
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

    # ------------------------------------------------------------------
    # state_dict persistence
    # ------------------------------------------------------------------

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Load a ``state_dict``, correctly restoring previously-``None`` buffers.

        PyTorch omits ``None`` buffers from ``state_dict``.  After fitting,
        the buffers (``components_``, ``mean_``, …) hold tensors; a fresh
        (unfitted) ``TensorPCA`` has them as ``None``.  This override
        manually sets those buffers before delegating to the standard loader,
        so that round-tripping a fitted model works out of the box.

        Parameters
        ----------
        state_dict : Mapping
            Mapping from buffer/parameter name to tensor.
        strict : bool, optional
            Passed to ``super().load_state_dict``.  Defaults to ``True``.
        assign : bool, optional
            Passed to ``super().load_state_dict``.  Defaults to ``False``.
        """
        _nullable = {
            "components_",
            "mean_",
            "singular_values_",
            "explained_variance_",
            "explained_variance_ratio_",
            "_scatter",
        }
        # Pre-register buffers with the exact tensors from state_dict so
        # the parent's load_state_dict sees matching shapes and dtypes.
        for name in _nullable:
            if name in state_dict and state_dict[name] is not None:
                self.register_buffer(name, state_dict[name].clone())
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def __repr__(self) -> str:
        """Return a compact string representation of this TensorPCA instance."""
        if self.n_components is not None:
            spec = f"n_components={self.n_components}"
        else:
            spec = f"exp_var={self.exp_var}"
        return (
            f"TensorPCA({spec}, whiten={self.whiten}, "
            f"svd_solver={self.svd_solver!r})"
        )
