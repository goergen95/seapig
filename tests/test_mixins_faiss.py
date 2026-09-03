import pathlib

import faiss
import pytest
import torch

from seapig.scores.mixins import FAISSIndexMixin


class DummyKNN(FAISSIndexMixin):
    """Simple host class for FAISSIndexMixin tests."""

    def __init__(self, k: int = 5, index_path: pathlib.Path | None = None):
        self.k = k
        self.index = None
        self.indices_by_class = {}
        self.index_path = index_path

    # expose protected methods for testing convenience
    def make_index(self, embs: torch.Tensor):
        return self._make_faiss_index(embs)

    def build(self, embs: torch.Tensor):
        self._build_index(embs)
        return self.index

    def build_class(self, c: int, embs: torch.Tensor):
        self._build_index_for_class(c, embs)
        return self.indices_by_class[c]

    def query(self, query: torch.Tensor, offset: int = 0, *, index=None):
        return self._query_index(query, offset, index=index)

    def suggest_build(self, embs: torch.Tensor, k: int = 1):
        return self._suggest_build_params(embs, k)

    def suggest_query(self, embs: torch.Tensor, k: int = 1):
        return self._suggest_query_params(embs, k)


# Helper to generate deterministic embeddings
def make_embeddings(num: int, dim: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(num, dim)


def test_make_faiss_index_flat():
    embs = make_embeddings(5, 16)
    knn = DummyKNN(k=3)
    idx = knn.make_index(embs)
    assert isinstance(idx, faiss.IndexFlatL2)
    # flat index should have no hnsw attribute
    assert not hasattr(idx, "hnsw")


def test_make_faiss_index_hnsw_and_params():
    embs = make_embeddings(20_000, 128)
    knn = DummyKNN(k=7)
    idx = knn.make_index(embs)
    # With many vectors we expect HNSW index
    assert isinstance(idx, faiss.IndexHNSWFlat)
    # Check that M matches suggestion
    params = knn.suggest_build(embs, k=7)
    assert idx.hnsw.efConstruction == params["efConstruction"]


def test_suggest_build_params_various_dimensions():
    # dim <=64 -> M=16
    embs = make_embeddings(100, 32)
    params = DummyKNN().suggest_build(embs)
    assert params["M"] == 16
    # dim 200 -> M=24
    embs = make_embeddings(100, 200)
    params = DummyKNN().suggest_build(embs)
    assert params["M"] == 32


@pytest.mark.parametrize(
    "dim,expected_M",
    [(32, 16), (100, 24), (200, 32), (400, 48), (800, 48), (2000, 64)],
)
def test_suggest_build_params_dim_branches(dim, expected_M):
    embs = torch.randn(10, dim)
    params = FAISSIndexMixin._suggest_build_params(embs, k=1)
    assert params["M"] == expected_M
    # efConstruction should be >=4*M and <=1024
    ef = params["efConstruction"]
    assert ef >= 4 * expected_M and ef <= 1024


def test_build_index_writes_to_disk(tmp_path: pathlib.Path):
    embs = make_embeddings(10, 64)
    path = tmp_path / "index.bin"
    knn = DummyKNN(k=3, index_path=path)
    idx = knn.build(embs)
    assert path.exists()
    # Loading the index directly should yield same type
    loaded = faiss.read_index(str(path))
    assert isinstance(loaded, type(idx))


def test_build_index_for_class_populates_dict(tmp_path: pathlib.Path):
    embs = make_embeddings(15, 64)
    knn = DummyKNN(k=4, index_path=tmp_path)
    class_idx = knn.build_class(0, embs)
    assert 0 in knn.indices_by_class
    assert knn.indices_by_class[0] is class_idx
    # If index_path is set, a file should be written
    expected_file = tmp_path / "class0.bin"
    assert expected_file.exists()


def test_query_index_returns_correct_shape_and_offset():
    embs = make_embeddings(20, 32)
    knn = DummyKNN(k=5)
    knn.build(embs)
    # Query a batch of 3 vectors
    query = make_embeddings(3, 32)
    dists, inds = knn.query(query)
    assert dists.shape == (3, knn.k)
    assert inds.shape == (3, knn.k)
    # With offset=1 the nearest neighbor should be dropped
    dists_off, inds_off = knn.query(query, offset=1)
    assert dists_off.shape == (3, knn.k)
    assert inds_off.shape == (3, knn.k)
    # Ensure that the results differ when offset is used
    assert not torch.equal(dists, dists_off) or not torch.equal(inds, inds_off)


def test_query_index_raises_when_no_index():
    knn = DummyKNN()
    query = make_embeddings(2, 16)
    with pytest.raises(AssertionError):
        knn.query(query)


def test_suggest_query_params_consistency():
    embs = make_embeddings(5000, 64)
    knn = DummyKNN()
    build_params = knn.suggest_build(embs, k=5)
    query_params = knn.suggest_query(embs, k=5)
    # efSearch should be bounded by efConstruction
    assert query_params["efSearch"] <= build_params["efConstruction"]
    # Should be at least k
    assert query_params["efSearch"] >= 5
