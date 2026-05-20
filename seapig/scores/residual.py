"""Residual KNN uncertainty score.

Implements ResidualScore which estimates uncertainty by bootstrapping
neighbour residuals. Delegates all KNN/index work to a provided KNNScore.
"""

from __future__ import annotations

from typing import Any

import torch
from torchmetrics import Metric, MetricCollection
from typing_extensions import override

from seapig.scores.embed import EmbeddingScore
from seapig.scores.knn import KNNScore

__all__ = ["ResidualScore"]


class ResidualScore(EmbeddingScore):
    """Residual-based uncertainty estimator built on a KNN score.

    Estimates per-query uncertainty by bootstrapping residuals of the
    K nearest neighbours obtained from a provided `scores.KNNScore` instance.

    Parameters
    ----------
    knn_score :
        A configured `scores.KNNScore` instance used for indexing and neighbour queries
        (e.g., `scores.EuclideanScore`). This argument is
        mandatory.
    n_bootstraps :
        Number of bootstrap replicates used to estimate variance (default: 200).
    bootstrap_size :
        Number of neighbour samples drawn per bootstrap replicate. If `None`,
        defaults to `knn_score.k` (default: None).
    cache_cpu :
        If `True`, batch embeddings are moved to CPU before accumulation to
        reduce GPU memory pressure (default: False).
    input_key :
        Key name used to extract input tensors from batch dictionaries (default
        `"image"`). If loader yields tuples/lists, tuple access is used
        instead.
    target_key :
        Key name used to extract target tensors from batch dictionaries
        (default `"target"`).

    Attributes
    ----------
    ref_residuals : torch.Tensor or None
        1-D tensor of per-sample residuals aligned with `ref_embeddings`.
        Stored as a non-persistent buffer.
    cal_residuals : torch.Tensor or None
        1-D tensor of per-sample residuals aligned with `cal_embeddings`.
        Stored as a non-persistent buffer.
    """

    ident: str = "residual"

    def __init__(
        self,
        knn_score: KNNScore,
        n_bootstraps: int = 200,
        bootstrap_size: int | None = None,
        cache_cpu: bool = False,
        input_key: str = "image",
        target_key: str = "target",
    ) -> None:
        super().__init__(pca=getattr(knn_score, "pca", None))
        if not isinstance(knn_score, KNNScore):
            raise TypeError(
                "knn_score must be an instance of seapig.scores.knn.KNNScore"
            )
        self.knn_provider = knn_score
        self.n_bootstraps = int(n_bootstraps)
        self.bootstrap_size = (
            int(bootstrap_size) if bootstrap_size is not None else knn_score.k
        )
        self.cache_cpu = bool(cache_cpu)
        # keys to extract inputs/targets from batch dicts
        self.input_key = str(input_key)
        self.target_key = str(target_key)

        # non-persistent residual buffers
        self.register_buffer("ref_residuals", None, persistent=False)
        self.register_buffer("cal_residuals", None, persistent=False)

    def fit(
        self,
        X: torch.Tensor | None = None,
        Y: torch.Tensor | None = None,
        ref_residuals: torch.Tensor | None = None,
        cal_residuals: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loaders: dict | None = None,
        error_metric: Metric | MetricCollection | None = None,
        q: bool | float = False,
        outdir: Any | None = None,
        prefix: str | None = None,
    ) -> None:
        """Fit the ResidualScore.

        The method supports two usage modes:

        1. Precomputed embeddings - supply `X` (training embeddings) and
           optional `Y` (calibration embeddings). In this mode the caller must
           provide `ref_residuals` (and `cal_residuals` when `Y` is
           supplied) because residuals cannot be computed without a model.

        2. Model + loaders - supply a `model` implementing `.embed()` and a
           `loaders` dictionary (containing at least the `"train"` loader).
           When `error_metric` (a `torchmetrics.Metric` or
           `torchmetrics.MetricCollection`) is provided, embeddings and
           per-sample residuals are extracted in a single pass over each loader.

        Parameters
        ----------
        X :
            Training embeddings of shape `(N, D)`. Required when not using
            `model`/`loaders`.
        Y :
            Calibration embeddings of shape `(M, D)`.
        ref_residuals :
            1-D tensor of per-training-sample residuals. Required when using
            precomputed `X` or when `error_metric` is not supplied.
        cal_residuals :
            1-D tensor of per-calibration-sample residuals. Required when
            providing `Y` with precomputed residuals.
        model :
            Model with an `.embed(x)` method used to extract embeddings when
            `X` is not provided.
        loaders :
            Dictionary of DataLoaders; must contain a `"train"` loader and
            may contain a `"val"` loader for calibration embeddings.
        error_metric :
            Metric used to compute per-sample residuals during a single-pass
            extraction. If a single `Metric` is provided it will be wrapped
            in a `MetricCollection`.
        q :
            If `False` (default), no quantile-based filtering is applied. If a
            float in `(0, 1)`, references with scores below the `q`
            quantile are kept. Behaviour depends on the KNN provider; when
            fitting with precomputed `X` or precomputed embeddings from a
            single-pass extraction, the provider is requested to return a mask
            indicating which original rows were retained.
        outdir :
            Passed to provider when delegating model+loaders fitting; used for
            optional embedding caching.
        prefix :
            Passed to provider when delegating model+loaders fitting; used for
            optional embedding caching.

        Raises
        ------
        ValueError
            If required residuals are missing or input combinations are invalid.
        KeyError
            If required loader keys (e.g., `"train"`) are missing.
        """
        if getattr(self, "knn_provider", None) is None:
            raise ValueError(
                "knn_score provider is required at construction time"
            )

        # Precomputed embeddings path
        if X is not None:
            if error_metric is not None:
                raise ValueError(
                    "error_metric cannot be used when providing precomputed embeddings X"
                )
            if ref_residuals is None:
                raise ValueError(
                    "ref_residuals must be provided when fitting with precomputed embeddings X"
                )
            if Y is not None and cal_residuals is None:
                raise ValueError(
                    "cal_residuals must be provided when fitting with calibration embeddings Y"
                )

            # Fit provider; request mask so residuals can be aligned if the provider applied q
            mask = self.knn_provider.fit(X=X, Y=Y, q=q, return_mask=True)
            # adopt provider embeddings as canonical
            self.ref_embeddings = self.knn_provider.ref_embeddings
            self.cal_embeddings = self.knn_provider.cal_embeddings

            ref_residuals = torch.as_tensor(ref_residuals, dtype=torch.float32)
            if ref_residuals.ndim == 2 and ref_residuals.shape[1] == 1:
                ref_residuals = ref_residuals.squeeze(1)
            if ref_residuals.ndim != 1:
                raise ValueError(
                    "ref_residuals must be a 1-D tensor of per-sample residuals"
                )
            if len(ref_residuals) != len(X):
                raise ValueError(
                    "Length of ref_residuals does not match number of reference embeddings"
                )

            if mask is not None:
                # mask refers to original X rows kept by provider
                self.ref_residuals = ref_residuals[mask]
            else:
                self.ref_residuals = ref_residuals

            if cal_residuals is not None:
                cal_residuals = torch.as_tensor(
                    cal_residuals, dtype=torch.float32
                )
                if cal_residuals.ndim == 2 and cal_residuals.shape[1] == 1:
                    cal_residuals = cal_residuals.squeeze(1)
                if cal_residuals.ndim != 1:
                    raise ValueError(
                        "cal_residuals must be a 1-D tensor of per-sample residuals"
                    )
                if self.cal_embeddings is None or len(cal_residuals) != len(
                    self.cal_embeddings
                ):
                    raise ValueError(
                        "Length of cal_residuals does not match number of calibration embeddings"
                    )
                # If provider returned a mask for calibration set we would need it; provider currently only
                # returns a single mask for ref embeddings. We assume cal_residuals are aligned already.
                self.cal_residuals = cal_residuals

        else:
            # model + loaders path
            if model is None or loaders is None:
                raise ValueError(
                    "Either precomputed embeddings (X) or model+loaders must be provided"
                )

            if error_metric is not None:
                if isinstance(error_metric, Metric):
                    error_metric = MetricCollection(error_metric)

                if "train" not in loaders:
                    raise KeyError(
                        "loaders must contain key 'train' when using error_metric"
                    )

                # extract train embeddings/residuals in one pass
                ref_embs, ref_res = self._extract_embs_and_res(
                    model, loaders["train"], error_metric
                )
                self.ref_embeddings = ref_embs
                self.ref_residuals = ref_res

                if "val" in loaders:
                    cal_embs, cal_res = self._extract_embs_and_res(
                        model, loaders["val"], error_metric
                    )
                    self.cal_embeddings = cal_embs
                    self.cal_residuals = cal_res
                else:
                    cal_embs = None

                # Fit provider on the precomputed embeddings and request mask
                mask = self.knn_provider.fit(
                    X=self.ref_embeddings, Y=cal_embs, q=q, return_mask=True
                )

                # If provider returned a mask, filter residuals to match provider's final ref_embeddings
                if mask is not None:
                    self.ref_residuals = self.ref_residuals[mask]

                # Update to provider's canonical embeddings (they may have been filtered)
                self.ref_embeddings = self.knn_provider.ref_embeddings
                self.cal_embeddings = self.knn_provider.cal_embeddings

            else:
                # error_metric not provided -> require ref_residuals argument
                if ref_residuals is None:
                    raise ValueError(
                        "ref_residuals must be provided when fitting from model+loaders without error_metric"
                    )

                # Fit provider using model+loaders; request mask in case provider applied q
                mask = self.knn_provider.fit(
                    model=model,
                    loaders=loaders,
                    outdir=outdir,
                    prefix=prefix,
                    q=q,
                    return_mask=True,
                )
                self.ref_embeddings = self.knn_provider.ref_embeddings
                self.cal_embeddings = self.knn_provider.cal_embeddings

                ref_residuals = torch.as_tensor(
                    ref_residuals, dtype=torch.float32
                )
                if ref_residuals.ndim == 2 and ref_residuals.shape[1] == 1:
                    ref_residuals = ref_residuals.squeeze(1)
                # ensure provider.ref_embeddings is available for length checks
                assert self.ref_embeddings is not None
                if ref_residuals.ndim != 1 or len(ref_residuals) != len(
                    self.ref_embeddings
                ):
                    # If provider returned a mask it filtered the embeddings — apply mask to original residuals instead
                    if mask is not None:
                        # mask corresponds to original training rows; validate lengths
                        if len(ref_residuals) != mask.numel():
                            raise ValueError(
                                "ref_residuals length does not match original training set length"
                            )
                        self.ref_residuals = ref_residuals[mask]
                    else:
                        raise ValueError(
                            "ref_residuals must be a 1-D tensor aligned with reference embeddings"
                        )
                else:
                    # lengths matched already
                    if mask is not None:
                        self.ref_residuals = ref_residuals[mask]
                    else:
                        self.ref_residuals = ref_residuals

        # Mark trained and optionally compute calibration scores
        self.set_trained()

        if self.cal_embeddings is not None and self.cal_residuals is not None:
            # compute scores on calibration set
            assert self.cal_embeddings is not None
            self.scores = self._score_embeddings(self.cal_embeddings)
            self.set_calibrated()
        else:
            if self.ref_embeddings is not None:
                assert self.ref_embeddings is not None
                self.scores = self._score_embeddings(self.ref_embeddings)

    def _extract_embs_and_res(
        self,
        model: torch.nn.Module,
        loader: Any,
        metric_obj: Metric | MetricCollection,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings and per-sample residuals in a single pass."""
        # Validate model has an embed method and prepare metric
        self._check_model(model)
        try:
            metric_obj.reset()
        except Exception:
            pass
        was_training = model.training
        model.eval()
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        embs_ls = []
        for batch in loader:
            if isinstance(batch, dict):
                if self.input_key not in batch or self.target_key not in batch:
                    raise KeyError(
                        f"Loader batches must contain keys '{self.input_key}' and '{self.target_key}' when using error_metric"
                    )
                x = batch[self.input_key].to(device)
                target = batch[self.target_key].to(device)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x = batch[0].to(device)
                target = batch[1].to(device)
            else:
                raise ValueError(
                    f"Unsupported batch format for error_metric; provide dicts with '{self.input_key}' and '{self.target_key}' or (x,y) tuples/lists"
                )

            assert isinstance(model, torch.nn.Module) and hasattr(
                model, "embed"
            )
            assert callable(model.embed)
            z = model.embed(x)
            if self.cache_cpu:
                embs_ls.append(z.cpu())
            else:
                embs_ls.append(z)
            preds = model(x)
            metric_obj.update(preds, target)

        residuals_raw = metric_obj.compute()
        try:
            metric_obj.reset()
        except Exception:
            pass
        if was_training:
            model.train()

        if isinstance(residuals_raw, dict):
            residuals = next(iter(residuals_raw.values()))
        else:
            residuals = residuals_raw
        if residuals.ndim == 2 and residuals.shape[1] == 1:
            residuals = residuals.squeeze(1)
        if residuals.ndim != 1:
            raise ValueError(
                "error_metric.compute() must return a 1-D tensor of per-sample residuals"
            )

        embs = torch.cat(embs_ls, dim=0)
        return embs, residuals.to(torch.float32)

    @override
    def score(
        self,
        X: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loader: Any | None = None,
        outdir: Any | None = None,
        prefix: str | None = None,
    ) -> torch.Tensor:
        """Compute residual variance scores for query samples.

        Parameters
        ----------
        X :
            Precomputed query embeddings of shape `(N, D)`. If provided, the
            score is computed directly on these embeddings.
        model :
            Model with an `.embed()` method. Required when `X` is not
            supplied and `loader` is used to extract query embeddings.
        loader :
            DataLoader used to extract query embeddings when `model` is
            provided. If `prefix`/`outdir` are given the provider may cache
            or load embeddings from disk.
        outdir :
            Optional directory passed to the provider for embedding caching.
        prefix : str, optional
            Optional prefix used for embedding cache filenames.

        Returns
        -------
        torch.Tensor
            1-D tensor of uncertainty scores with shape `(N,)`.

        Raises
        ------
        ValueError
            If neither `X` nor `model`/`loader` are provided.
        """
        if X is None:
            if model is None or loader is None:
                raise ValueError(
                    "Provide query embeddings X or model+loader to compute scores"
                )
            path = None
            embeddings = self.knn_provider._loadorembed(path, model, loader)
            return self._score_embeddings(embeddings)
        else:
            return self._score_embeddings(X)

    @override
    def select(
        self,
        X: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        loader: Any | None = None,
        outdir: Any | None = None,
        prefix: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.get_threshold() is None:
            self.set_threshold()
        assert self.threshold is not None
        scores = self.score(
            X=X, model=model, loader=loader, outdir=outdir, prefix=prefix
        )
        return {"score": scores, "selected": scores < self.threshold}

    def set_residuals(
        self,
        ref_residuals: torch.Tensor | None = None,
        cal_residuals: torch.Tensor | None = None,
    ) -> None:
        """Attach residuals to already-fitted embeddings.

        This method allows attaching externally-computed residuals to a
        previously-fitted provider. Residual tensors must be 1-D and aligned
        with the provider's corresponding embeddings.

        Parameters
        ----------
        ref_residuals :
            1-D tensor of per-reference residuals with length equal to the
            number of reference embeddings in `self.ref_embeddings`.
        cal_residuals :
            1-D tensor of per-calibration residuals with length equal to the
            number of calibration embeddings in `self.cal_embeddings`.

        Raises
        ------
        ValueError
            If embeddings are not set or residual tensors have invalid shape or
            length.
        """
        if ref_residuals is not None:
            if getattr(self, "ref_embeddings", None) is None:
                raise ValueError(
                    "Reference embeddings not set. Call fit(...) first."
                )
            ref_residuals = torch.as_tensor(ref_residuals, dtype=torch.float32)
            if ref_residuals.ndim == 2 and ref_residuals.shape[1] == 1:
                ref_residuals = ref_residuals.squeeze(1)
            assert self.ref_embeddings is not None
            if ref_residuals.ndim != 1 or len(ref_residuals) != len(
                self.ref_embeddings
            ):
                raise ValueError(
                    "ref_residuals must be a 1-D tensor matching reference embeddings length"
                )
            self.ref_residuals = ref_residuals
        if cal_residuals is not None:
            if getattr(self, "cal_embeddings", None) is None:
                raise ValueError(
                    "Calibration embeddings not set; cannot attach cal_residuals"
                )
            cal_residuals = torch.as_tensor(cal_residuals, dtype=torch.float32)
            if cal_residuals.ndim == 2 and cal_residuals.shape[1] == 1:
                cal_residuals = cal_residuals.squeeze(1)
            assert self.cal_embeddings is not None
            if cal_residuals.ndim != 1 or len(cal_residuals) != len(
                self.cal_embeddings
            ):
                raise ValueError(
                    "cal_residuals must be a 1-D tensor matching calibration embeddings length"
                )
            self.cal_residuals = cal_residuals

    def _score_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        """Compute residual variance scores for query embeddings."""
        if getattr(self, "knn_provider", None) is None:
            raise ValueError("knn_score provider is required")
        if self.ref_residuals is None:
            raise ValueError(
                "Reference residuals are required to compute uncertainty scores"
            )

        device = (
            getattr(self.knn_provider, "ref_embeddings", None).device
            if getattr(self.knn_provider, "ref_embeddings", None) is not None
            else X.device
        )
        Xq = X.to(device)

        distances, indices = self.knn_provider.knn_search(Xq, offset=0)

        ref_res = self.ref_residuals.to(indices.device)

        neigh_res = ref_res[indices]

        eps = 1e-8
        scale = distances.std(dim=1, keepdim=True)
        zero_mask = scale.abs() < eps
        if zero_mask.any():
            fallback = distances.mean(dim=1, keepdim=True)
            scale = torch.where(zero_mask, fallback + eps, scale + eps)
        else:
            scale = scale + eps
        probs = torch.softmax(-distances / scale, dim=1)

        N = distances.shape[0]
        k = distances.shape[1]
        B = int(self.n_bootstraps)
        m = (
            self.bootstrap_size
            if self.bootstrap_size is not None
            else getattr(self.knn_provider, "k", k)
        )

        draws = torch.multinomial(probs, num_samples=B * m, replacement=True)
        draws = draws.view(N, B, m)

        neigh_res_exp = neigh_res.unsqueeze(1).expand(-1, B, -1)
        sampled = torch.gather(neigh_res_exp, 2, draws)

        means = sampled.mean(dim=2)
        var = means.var(dim=1, unbiased=False)

        return var.to(device=X.device)
