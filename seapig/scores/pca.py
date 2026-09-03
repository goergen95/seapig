"""PCA based dimensionality reduction and uncertainty scoring."""

import torch
from typing_extensions import override

from seapig.scores import EmbeddingScore
from seapig.scores.utils import TensorPCA


class PCAScore(EmbeddingScore):
    """Returns uncertainty scores based on PCA reconstruction errors.

    Computes reconstruction error-based uncertainty scores where low scores indicate
    samples that can be well-reconstructed from principal components (likely inliers)
    and high scores indicate samples with large reconstruction errors (likely outliers).

    See `https://arxiv.org/pdf/2402.02949v3` for the method description.

    Parameters
    ----------
    pca : `TensorPCA`, optional
        `TensorPCA` object to use. Defaults to
        `TensorPCA(n_components=0.50, gamma=3.0, M=4096)` (RFF-PCA retaining
        50% explained variance).

    Examples
    --------
    ```{python}
    import torch
    from seapig.scores import PCAScore
    from seapig.scores.utils import TensorPCA

    score = PCAScore(pca=TensorPCA(n_components=0.90))
    score.fit(X=torch.randn(200, 64), Y=torch.randn(50, 64))
    score.set_threshold(q=0.95)
    result = score.select(X=torch.randn(10, 64))
    print(result)
    ```

    See Also
    --------
    `scores.EmbeddingScore`
    `scores.utils.TensorPCA`
    """

    ident = "pca"

    def __init__(self, pca: TensorPCA) -> None:
        pca = pca or TensorPCA(n_components=0.50, gamma=3.0, M=4096)
        super().__init__(pca=pca)

    @override
    def _fit(self, q: bool | float = False) -> None:
        """Train a uncertainty score based on sample embeddings.

        Parameters
        ----------
        q:
            A `float` or `bool` indicating if outliers from the training
            distribution should be filtered before fitting. Defaults to `False`.
        """
        assert self.ref_embeddings is not None
        if q:
            assert (q >= 0.0) & (q <= 1.0)
            assert self.pca is not None
            self._fit_pca()
            _, scores = self.pca.reconstruct(self.ref_embeddings)
            threshold = torch.quantile(scores.float(), q=q)
            index = scores < threshold
            self.ref_embeddings = self.ref_embeddings[index, :]

        self._fit_pca()
        self.set_trained()
        assert self.pca is not None

        if self.cal_embeddings is None:
            _, self.scores = self.pca.reconstruct(self.ref_embeddings)
        else:
            _, self.scores = self.pca.reconstruct(self.cal_embeddings)
            self.set_calibrated()

    @override
    def _score(self, X: torch.Tensor) -> torch.Tensor:
        """Compute an uncertainty score based on sample embeddings.

        Returns reconstruction error scores where low values indicate samples that
        can be well-reconstructed (low uncertainty) and high values indicate samples
        with large reconstruction errors (high uncertainty).

        Parameters
        ----------
        X:
            A `torch.Tensor` representing sample embeddings of shape `(B, D)`.
        """
        assert self.pca is not None
        _, error = self.pca.reconstruct(X)
        return error
