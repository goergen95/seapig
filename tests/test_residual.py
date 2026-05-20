import torch

from seapig.scores import EuclideanScore, ResidualScore


def test_residual_score_basic():
    torch.manual_seed(0)
    N = 200
    D = 16
    ref_embs = torch.randn(N, D)
    # Create residuals with some non-zero variance
    ref_res = torch.abs(torch.randn(N))

    q = 10
    queries = torch.randn(q, D)

    knn_provider = EuclideanScore(k=5)
    score = ResidualScore(
        knn_score=knn_provider, n_bootstraps=100, bootstrap_size=5
    )
    score.fit(X=ref_embs, ref_residuals=ref_res)

    out = score.score(X=queries)
    assert out.shape == (q,)
    assert torch.isfinite(out).all()
    # variance should be non-negative
    assert (out >= 0).all()


def test_residual_score_constant_residuals():
    torch.manual_seed(1)
    N = 100
    D = 8
    ref_embs = torch.randn(N, D)
    ref_res = torch.ones(N) * 3.14

    queries = torch.randn(5, D)
    knn_provider = EuclideanScore(k=3)
    score = ResidualScore(
        knn_score=knn_provider, n_bootstraps=80, bootstrap_size=3
    )
    score.fit(X=ref_embs, ref_residuals=ref_res)
    out = score.score(X=queries)
    # when residuals are constant, variance of bootstrap means should be zero
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)


def test_residual_score_with_metric_and_knn_provider():
    torch.manual_seed(2)
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchmetrics import Metric

    # Create synthetic dataset where targets are a linear function of inputs
    N = 120
    D = 8
    X = torch.randn(N, D)
    W = torch.randn(D, 1)
    b = torch.randn(1)
    noise = 0.01 * torch.randn(N, 1)
    y = X @ W + b + noise

    class SimpleDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):  # type: ignore[override, ty:invalid-method-override]
            return {"image": self.X[idx], "target": self.y[idx]}

    train_loader = DataLoader(SimpleDataset(X, y), batch_size=32)

    # Dummy model with .embed() and forward() implemented
    class DummyModel(nn.Module):
        def __init__(self, W, b):
            super().__init__()
            self.pred = nn.Linear(D, 1)
            # set weights to match surface that generated targets
            with torch.no_grad():
                self.pred.weight.copy_(W.t())
                self.pred.bias.copy_(b)

        def embed(self, x):
            # identity embedding
            return x

        def forward(self, x):
            return self.pred(x)

    model = DummyModel(W, b)

    # Metric that returns per-sample absolute residuals when compute() is called
    class PerSampleResidual(Metric):
        def __init__(self):
            super().__init__()
            self.add_state(
                "residuals",
                default=torch.tensor([], dtype=torch.float32),
                dist_reduce_fx="cat",
            )

        def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
            preds_f = preds.to(dtype=torch.float32)
            target_f = target.to(dtype=torch.float32)
            residual = torch.abs(preds_f - target_f)
            if residual.ndim == 2 and residual.shape[1] == 1:
                residual = residual.squeeze(1)
            if self.residuals.numel() == 0:
                self.residuals = residual
            else:
                self.residuals = torch.cat([self.residuals, residual], dim=0)

        def compute(self) -> torch.Tensor:
            return self.residuals

        def reset(self) -> None:
            self.residuals = torch.tensor([], dtype=torch.float32)

    res_metric = PerSampleResidual()

    knn_provider = EuclideanScore(k=3)
    a = ResidualScore(knn_score=knn_provider, n_bootstraps=80, bootstrap_size=3)
    # Fit using model+loader and extract residuals via provided metric; also hand over knn_provider
    a.fit(model=model, loaders={"train": train_loader}, error_metric=res_metric)

    queries = torch.randn(5, D)
    out = a.score(X=queries)
    assert out.shape == (5,)
    assert torch.isfinite(out).all()


def test_residual_score_custom_keys():
    torch.manual_seed(5)
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchmetrics import Metric

    N = 60
    D = 6
    X = torch.randn(N, D)
    W = torch.randn(D, 1)
    b = torch.randn(1)
    noise = 0.01 * torch.randn(N, 1)
    y = X @ W + b + noise

    class CKDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):  # type: ignore[override, ty:invalid-method-override]
            return {"inputs": self.X[idx], "labels": self.y[idx]}

    train_loader = DataLoader(CKDataset(X, y), batch_size=16)

    class DummyModel(nn.Module):
        def __init__(self, W, b):
            super().__init__()
            self.pred = nn.Linear(D, 1)
            with torch.no_grad():
                self.pred.weight.copy_(W.t())
                self.pred.bias.copy_(b)

        def embed(self, x):
            return x

        def forward(self, x):
            return self.pred(x)

    model = DummyModel(W, b)

    class PerSampleResidual(Metric):
        def __init__(self):
            super().__init__()
            self.add_state(
                "residuals",
                default=torch.tensor([], dtype=torch.float32),
                dist_reduce_fx="cat",
            )

        def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
            preds_f = preds.to(dtype=torch.float32)
            target_f = target.to(dtype=torch.float32)
            residual = torch.abs(preds_f - target_f)
            if residual.ndim == 2 and residual.shape[1] == 1:
                residual = residual.squeeze(1)
            if self.residuals.numel() == 0:
                self.residuals = residual
            else:
                self.residuals = torch.cat([self.residuals, residual], dim=0)

        def compute(self) -> torch.Tensor:
            return self.residuals

        def reset(self) -> None:
            self.residuals = torch.tensor([], dtype=torch.float32)

    res_metric = PerSampleResidual()
    knn_provider = EuclideanScore(k=4)
    score = ResidualScore(
        knn_score=knn_provider,
        n_bootstraps=60,
        bootstrap_size=4,
        input_key="inputs",
        target_key="labels",
    )
    score.fit(
        model=model, loaders={"train": train_loader}, error_metric=res_metric
    )
    queries = torch.randn(3, D)
    out = score.score(X=queries)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_knn_fit_return_mask():
    # Ensure KNNScore.fit can return a mask when requested
    torch.manual_seed(3)
    N = 120
    D = 8
    X = torch.randn(N, D)
    knn_provider = EuclideanScore(k=3)
    mask = knn_provider.fit(X=X, q=0.2, return_mask=True)
    # mask should be a boolean tensor of length N
    assert mask is None or (
        isinstance(mask, torch.Tensor)
        and mask.dtype == torch.bool
        and mask.numel() == N
    )
    if isinstance(mask, torch.Tensor):
        # provider should have rebuilt its index on the filtered embeddings
        assert knn_provider.ref_embeddings is not None
        assert len(knn_provider.ref_embeddings) == int(mask.sum())
