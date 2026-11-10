"""PCA based dimensionality reduction and confidence scoring."""

import torch
from typing_extensions import override

from seapig.scores import EmbeddingScore
from seapig.scores.utils import TensorPCA, _tensor


class PCAScore(EmbeddingScore):
    """Returns confidence scores based on PCA reconstruction errors.

    Computes reconstruction error-based confidence scores where high scores indicate
    samples that can be well-reconstructed from principal components (high confidence)
    and low scores indicate samples with large reconstruction errors (low confidence).

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
    score.fit(ref=torch.randn(200, 64), cal=torch.randn(50, 64))
    score.set_threshold(q=0.95)
    result = score.select(query=torch.randn(10, 64))
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
        """Train a confidence score based on sample embeddings.

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
            index = scores >= threshold
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
    def _score(self, query: torch.Tensor) -> torch.Tensor:
        """Compute a confidence score based on sample embeddings.

        Returns reconstruction error scores where high values indicate samples that
        can be well-reconstructed (high confidence) and low values indicate samples
        with large reconstruction errors (low confidence).

        Parameters
        ----------
        X:
            A `torch.Tensor` representing sample embeddings of shape `(B, D)`.
        """
        assert self.pca is not None
        X = _tensor(query, "embedding")
        assert X is not None
        _, error = self.pca.reconstruct(X)
        return -error
