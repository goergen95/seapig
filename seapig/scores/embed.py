"""Abstract Base Method for embeddings based confidence scores."""

import inspect
from abc import ABC
from pathlib import Path
from typing import Any, Literal, override

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seapig.scores.base import ConfidenceScore
from seapig.scores.utils import TensorPCA


class EmbeddingScore(ConfidenceScore, ABC):
    """Base class for embedding-based confidence scores.

    Parameters
    ----------
    exp_var:
        A `float` indicating the percentage of explained variance to retain
        if dimensionality reduction via PCA shall be applied. Defaults to `False`,
        indicating that dimensionality reduction is not applied.

    Attributes
    ----------
    ref_embeddings:
        A `torch.Tensor` with the embeddings of trainings samples. Defaults to `None`.
    cal_embeddings:
        A `torch.Tensor` with the embeddings of validation samples. Defaults to `None`.
    scores:
        A `torch.Tensor` with the confidence scores of the validation samples.
        Defaults to `None`.
    threshold:
        A `float` indicating the rejection threshold. Defaults to `None`.
    """

    ref_embeddings: torch.Tensor | None
    cal_embeddings: torch.Tensor | None
    train_required: bool = True
    cal_required: bool = True
    scores: torch.Tensor | None
    pca: TensorPCA | None = None

    def __init__(self, exp_var: float | bool = False) -> None:
        super().__init__()
        self.ref_embeddings = None
        self.cal_embeddings = None
        self.scores = None
        self.exp_var = exp_var

    def to(self, device: str | torch.device = "cpu") -> None:
        """Put all tensors to the specified device."""
        self.ref_embeddings = self._to(self.ref_embeddings, device=device)
        self.cal_embeddings = self._to(self.cal_embeddings, device=device)
        self.scores = self._to(self.scores, device=device)
        self.threshold = self._to(self.threshold, device=device)
        self.cal_embeddings = self._to(self.cal_embeddings, device=device)
        if self.pca is not None:
            self.pca.to(device=device)

    @staticmethod
    def _setup_path(
        outdir: Path | None = None, prefix: str | None = None
    ) -> Path | None:
        """Construct the output path for a parquet file."""
        if outdir is None or prefix is None:
            return None
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)
        return outdir / f"{prefix}.parquet"

    @staticmethod
    def _check_model(model: torch.nn.Module) -> None:
        """Check a model for compatibility with embeddings-based confidence scores."""
        assert isinstance(model, torch.nn.Module)
        if not callable(model.embed):
            raise Exception("model is required to have a `.embed()` method.")
        sig = inspect.signature(obj=model.embed)
        if "x" not in sig.parameters.keys():
            raise Exception(
                "`.embed()` method is required to except `x` as argument."
            )

    @staticmethod
    def _write_parquet(x: torch.Tensor, path: Path) -> None:
        """Write a `torch.Tensor` to parquet."""
        df = pd.DataFrame(x.cpu())
        df.to_parquet(path, index=False)

    @staticmethod
    @torch.inference_mode()
    def _load_parquet(path: Path) -> torch.Tensor:
        """Read a parquet file to a `torch.Tensor`."""
        df = pd.read_parquet(path)
        return torch.Tensor(df.values).squeeze()

    @classmethod
    def _loadorembed(
        self,
        path: Path | None,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Load from file or iterate over dataloader to extract embeddings."""
        if path is not None and path.is_file():
            print(f"Loading pre-existing embeddings from {path}.")
            v = self._load_parquet(path)
        else:
            v = self._embed_dl(model=model, loader=loader)
            if path is not None:
                self._write_parquet(v, path)
        return v

    @classmethod
    @torch.inference_mode()
    def _embed(
        self, X: torch.Tensor | dict[str, torch.Tensor], model: torch.nn.Module
    ) -> torch.Tensor:
        """Embed a batch based on a models embed method."""
        assert callable(model.embed)
        if isinstance(X, dict):
            if "image" not in X.keys():
                raise KeyError(
                    'A batch dictionary is required to contain the "image" key.'
                )
            z = model.embed(X["image"])
        else:
            z = model.embed(X)
        assert isinstance(z, torch.Tensor)
        if len(z.shape) > 2:  # we expect (B,D)
            raise ValueError(
                f"Expected embed method to return tensor of shape (B,D) but got {z.shape}"
            )
        return z

    @classmethod
    def _embed_dl(
        self,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Extract embeddings by iterating over a DataLoader."""
        assert callable(model.embed)
        pbar = tqdm(
            total=len(loader),
            desc=f"Embedding {len(loader)} batches",
            unit="batches",
        )
        embs_ls = list()
        for batch in loader:
            z = self._embed(X=batch, model=model)
            embs_ls.append(z)
            _ = pbar.update(n=1)
        embs = torch.cat(embs_ls, dim=0)
        return embs

    @classmethod
    def _embed_from_dict(
        self,
        model: torch.nn.Module,
        loaders: dict[str, DataLoader[torch.Tensor | dict[str, torch.Tensor]]],
        key: Literal["train", "val"],
        outdir: Path | None = None,
        prefix: str | None = None,
    ) -> torch.Tensor:
        """Embed a loader from a specified key in a dictionary."""
        path = None
        assert isinstance(loaders, dict)
        assert isinstance(model, torch.nn.Module)
        self._check_model(model)
        if key not in loaders.keys():
            raise KeyError(f"Missing key `{key}` in loaders dictionary.")
        loader = loaders[key]
        assert isinstance(loader, DataLoader)
        if prefix is not None:
            path = self._setup_path(outdir, prefix + f"-embeddings-{key}")
        embs = self._loadorembed(path, model, loader)
        return embs

    def _fit_pca(self) -> None:
        assert self.ref_embeddings is not None
        self.pca = TensorPCA(exp_var=self.exp_var)
        self.pca.fit(self.ref_embeddings)
        self.pca.to(device=self.ref_embeddings.device)

    @override
    def fit(
        self, X: torch.Tensor, Y: torch.Tensor | None, *args: Any, **kwargs: Any
    ) -> None:
        """Train a confidence score based on sample embeddings.

        Training embeddings are required to be supplied as a `torch.tensor` with
        parameter `X`. Calibration embeddings are supplied with the `Y` parameter.
        As an alternative, use `fit_dl()` to supply a model with an `.embed()` method
        and a dictionary with `DataLoaders` to extract embeddings on the fly.

        These are later used to retrieve confidence scores for query samples.

        ```python
        my_score = EmbeddingScore(k=2)
        my_score.fit(train_embs, val_embs)
        ```

        Parameters
        ----------
        X:
            A `torch.tensor` or an `np.Array` with samples representing training
            samples embeddings.
        Y:  A `torch.tensor` or an `np.Array` with samples representing calibration
            samples embeddings.
        """
        self.ref_embeddings = X
        self.cal_embeddings = Y
        self.to(device=self.ref_embeddings.device)

    def fit_dl(
        self,
        model: torch.nn.Module,
        loaders: dict[str, DataLoader[torch.Tensor | dict[str, torch.Tensor]]],
        outdir: Path | None = None,
        prefix: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Train a confidence score based on samples from a `DataLoader`.

        Training embeddings are extracted from the supplied models and the data
        loader with the `"train"` key in the supplied `loaders` argument.
        Calibration embeddings are extracted from the `DataLoader` object with
        the `"val"` key. The confidence score is then calibrated based on the
        extracted embeddings.

        ```python
        my_score = EmbeddingScore(k=2)
        my_score.fit_dl(model=model, loaders={"train": train_loader, "val": val_loader})
        ```

        Parameters
        ----------
        model:
            A torch.nn.Module representing a trained model instance. It is
            required to have an `.embed()` method.
        loaders:
            A `dict`ionary with dataloader objects with required keys `["train", "val"]`.
            The `DataLoaders` are expected to return `torch.Tensor`s or a `dict`
            of `torch.Tensor`s with the `"image"` key present.
        outdir:
            A `pathlib.Path` object pointing towards a directory, by default `None`.
            If specified, embeddings are read to disk, if previously written. Otherwise,
            embeddings will be written to disk.
        prefix:
            A `str`ing used as filename prefix to save embeddings, by default
            `None`. See `outdir` parameter above.
        """
        assert isinstance(loaders, dict)
        assert isinstance(model, torch.nn.Module)
        self._check_model(model)
        self.ref_embeddings = self._embed_from_dict(
            loaders=loaders,
            model=model,
            key="train",
            outdir=outdir,
            prefix=prefix,
        )
        if not self.cal_required:
            print("Confidence score does not require calibration.")
            return
        self.cal_embeddings = self._embed_from_dict(
            loaders=loaders,
            model=model,
            key="val",
            outdir=outdir,
            prefix=prefix,
        )
        self.to(device=self.ref_embeddings.device)

    @override
    def set_threshold(self, q: float = 0.99) -> None:
        """Set a threshold based on quantiles on the reference confidence scores.

        This method sets the selection threshold based on the quantile on
        the values found in the `scores` attribute. If the confidence score
        is trained, but uncalibrated, this will be based on the k-nearest-neighbor
        distances of the training samples, excluding the distance to the
        point itself. If calibrated, the distance of the calibration samples to
        the k-closest training samples are used.

        Parameters
        ----------
        q:
            A `float` indicating the quantile of confidence scores of the
            samples to set the rejection threshold to.
        """
        if self.train_required:
            assert self.is_trained  # type: ignore [truthy-function]
        if self.cal_required:
            assert self.is_calibrated  # type: ignore [truthy-function]
        assert self.scores is not None
        self.threshold = self.scores.float().quantile(q=q)

    @override
    def select(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Select samples for prediction based on their confidence score.

        Samples are selected for prediction based on their confidence score compared
        to a threshold. It is expected that the threshold was previously
        calibrated on, e.g. validation samples.

        ```python
        my_score = ConfidenceScore()
        my_score = my_score.fit(X=train_data, Y=val_data)
        scores = my_score.select(test_data)
        ```

        Parameters
        ----------
        X:
            A `torch.tensor` with samples representing testing
            embeddings to select based on a pre-calibrated threshold.
        """
        if self.train_required:
            assert self.is_trained()
        if self.cal_required:
            assert self.is_calibrated()
        if self.get_threshold() is None:
            print(
                "Threshold has not been set. Trying to set it via `set_threshold()`."
            )
            self.set_threshold()
        assert self.threshold is not None
        score = self.score(X=X)
        return {"score": score, "selected": score < self.threshold}

    def select_dl(
        self,
        model: torch.nn.Sequential,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
        outdir: Path,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        """Select samples for prediction based on their confidence score.

        Samples are selected for prediction based on their confidence score compared
        to a threshold. It is expected that the threshold was previously
        calibrated on, e.g. validation samples. This method will embed input samples
        on the fly using the supplied models `.embed()` method.

        ```python
        my_score = ConfidenceScore()
        my_score = my_score.fit(X=train_data, Y=val_data)
        scores = my_score.select(test_dl)
        ```

        Parameters
        ----------
        model:
            A torch.nn.Module representing a trained model instance. It is
            required to have an `.embed()` method, by default None.
        loader:
            A `torch.utils.data.DataLoader` object returning `torch.Tensor`s or
            a `dict` of `torch.Tensor`s with the `"image"` key available,
            by default None.
        outdir:
            A `pathlib.Path` object pointing towards a directory, by default `None`.
            If specified, embeddings are read to disk, if previously written. Otherwise,
            embeddings will be written to disk.
        prefix:
            A `str`ing used as filename prefix to save embeddings, by default
            `None`. See `outdir` parameter above.
        """
        scores = self.score_dl(
            model=model, loader=loader, outdir=outdir, prefix=prefix
        )
        assert isinstance(self.threshold, torch.Tensor)
        return {"score": scores, "selected": scores < self.threshold}

    def score_dl(
        self,
        model: torch.nn.Module,
        loader: DataLoader[torch.Tensor | dict[str, torch.Tensor]],
        outdir: Path | None,
        prefix: str | None,
    ) -> torch.Tensor:
        """Compute confidence scores for all samples in a DataLoader.

        Iterates over a dataloader, embeds samples on-the-fly using the supplied
        models `.embed()` method and returns their confidence scores.

        ```python
        my_score = KNNScore()
        scores = my_score.score_dl(model, test_dl)
        ```

        Parameters
        ----------
        model:
            A torch.nn.Module representing a trained model instance. It is
            required to have an `.embed()` method.
        loader:
            A `torch.utils.data.DataLoader` object returning `torch.Tensor`s or
            a `dict` of `torch.Tensor`s with the `"image"` key.
        outdir:
            A `pathlib.Path` object pointing towards a directory, by default `None`.
            If specified, embeddings are read to disk, if previously written. Otherwise,
            embeddings will be written to disk.
        prefix:
            A `str`ing used as filename prefix to save embeddings, by default
            `None`. See `outdir` parameter above.
        """
        path = None
        if prefix is not None:
            path = self._setup_path(outdir, prefix)
        X = self._loadorembed(path, model, loader)
        score = self.score(X)
        return score
