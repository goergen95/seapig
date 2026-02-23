"""
Unit tests for seapig.scores.index_manager.IndexManager.

Testing matrix
--------------
* test_indexmanager_fit_and_search_brute         – exact nearest neighbours
* test_indexmanager_nmslib_hnsw_build_and_search – HNSW vs brute agreement
* test_indexmanager_nmslib_batch_add_and_search  – incremental batch population
* test_indexmanager_persistence_brute            – save/load brute-force index
* test_indexmanager_persistence_hnsw             – save/load HNSW index
* test_indexmanager_pca_reduce                   – PCA reduces dimension
* test_indexmanager_input_validation             – invalid inputs raise errors
* test_indexmanager_reset                        – reset clears all state
* test_indexmanager_get_embeddings               – getter methods
* test_indexmanager_suggest_hnsw_params          – parameter suggestion heuristic
"""

import math
from pathlib import Path

import pytest
import torch

from seapig.scores.index_manager import IndexManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def approx(t1: torch.Tensor, t2: torch.Tensor, tol: float = 1e-5) -> None:
    """Assert two tensors are element-wise close."""
    assert torch.allclose(t1.float(), t2.float(), atol=tol, rtol=0), (
        f"max diff: {(t1.float() - t2.float()).abs().max().item()}"
    )


def make_refs(n: int = 20, d: int = 4, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, d)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method must be"):
            IndexManager(method="faiss")

    def test_invalid_space_raises(self) -> None:
        with pytest.raises(ValueError, match="space must be"):
            IndexManager(space="dotproduct")

    def test_both_pca_params_raises(self) -> None:
        with pytest.raises(ValueError, match="at most one"):
            IndexManager(pca_components=4, pca_exp_var=0.9)

    def test_fit_1d_tensor_raises(self) -> None:
        mgr = IndexManager(method="brute")
        with pytest.raises(ValueError, match="2-D"):
            mgr.fit(torch.randn(10))

    def test_fit_cal_dimension_mismatch_raises(self) -> None:
        mgr = IndexManager(method="brute")
        with pytest.raises(ValueError, match="embedding dimension"):
            mgr.fit(
                ref_embeddings=torch.randn(5, 4),
                cal_embeddings=torch.randn(3, 8),
            )

    def test_add_batch_1d_raises(self) -> None:
        mgr = IndexManager(method="brute")
        with pytest.raises(ValueError, match="2-D"):
            mgr.add_batch(torch.randn(5))

    def test_add_batch_dimension_mismatch_raises(self) -> None:
        mgr = IndexManager(method="brute")
        mgr.fit(torch.randn(5, 4))
        with pytest.raises(ValueError, match="dimension mismatch"):
            mgr.add_batch(torch.randn(3, 8))

    def test_search_without_build_raises(self) -> None:
        mgr = IndexManager(method="brute")
        mgr.fit(torch.randn(5, 4))
        with pytest.raises(RuntimeError, match="build_index"):
            mgr.search(torch.randn(2, 4), k=1)

    def test_search_1d_query_raises(self) -> None:
        mgr = IndexManager(method="brute")
        mgr.fit(torch.randn(5, 4))
        mgr.build_index()
        with pytest.raises(ValueError, match="2-D"):
            mgr.search(torch.randn(4), k=1)

    def test_build_without_refs_raises(self) -> None:
        mgr = IndexManager(method="brute")
        with pytest.raises(RuntimeError, match="No reference"):
            mgr.build_index()


# ---------------------------------------------------------------------------
# Brute-force search
# ---------------------------------------------------------------------------


class TestBruteForceSearch:
    def test_exact_nearest_neighbour_l2(self) -> None:
        """Closest reference to origin is (3,4), distance² = 25."""
        refs = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
        q = torch.tensor([[0.0, 0.0]])

        mgr = IndexManager(method="brute", space="l2")
        mgr.fit(refs)
        mgr.build_index()
        indices, distances = mgr.search(q, k=1)

        assert indices.shape == (1, 1)
        assert indices[0, 0].item() == 0  # (3,4) is closer
        approx(distances, torch.tensor([[25.0]]))  # squared distance

    def test_exact_nearest_neighbour_cosinesimil(self) -> None:
        """Identical vectors have cosine distance 0."""
        refs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        q = torch.tensor([[1.0, 0.0]])

        mgr = IndexManager(method="brute", space="cosinesimil")
        mgr.fit(refs)
        mgr.build_index()
        indices, distances = mgr.search(q, k=1)

        assert indices[0, 0].item() == 0
        assert torch.isclose(distances[0, 0], torch.tensor(0.0), atol=1e-6)

    def test_returns_indices_only_when_no_distances(self) -> None:
        refs = make_refs()
        q = make_refs(n=3)

        mgr = IndexManager(method="brute", space="l2")
        mgr.fit(refs)
        mgr.build_index()
        result = mgr.search(q, k=2, return_distances=False)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 2)

    def test_topk_ordering_l2(self) -> None:
        """Verify that returned neighbours are the k closest by sq-L2."""
        torch.manual_seed(42)
        refs = torch.randn(30, 8)
        q = torch.randn(5, 8)

        mgr = IndexManager(method="brute", space="l2")
        mgr.fit(refs)
        mgr.build_index()
        indices, distances = mgr.search(q, k=3)

        # Verify manually
        for i in range(q.shape[0]):
            sq_dists = ((q[i] - refs) ** 2).sum(dim=1)
            expected_idx = sq_dists.topk(3, largest=False).indices
            assert set(indices[i].tolist()) == set(expected_idx.tolist())

    def test_zero_padding_warning_fewer_refs_than_k(self) -> None:
        """When N < k, results are zero-padded and a warning is emitted."""
        refs = torch.tensor([[0.0, 0.0]])
        q = torch.tensor([[1.0, 1.0]])

        mgr = IndexManager(method="brute", space="l2")
        mgr.fit(refs)
        mgr.build_index()

        with pytest.warns(UserWarning, match="zero padding"):
            indices, distances = mgr.search(q, k=3)

        assert indices.shape == (1, 3)
        assert distances.shape == (1, 3)
        # First distance should be 2.0 (squared), rest zero-padded
        assert not torch.isclose(distances[0, 0], torch.tensor(0.0))

    def test_multiple_queries(self) -> None:
        refs = make_refs(n=50, d=6)
        q = make_refs(n=10, d=6, seed=99)

        mgr = IndexManager(method="brute", space="l2")
        mgr.fit(refs)
        mgr.build_index()
        indices, distances = mgr.search(q, k=5)

        assert indices.shape == (10, 5)
        assert distances.shape == (10, 5)


# ---------------------------------------------------------------------------
# HNSW search
# ---------------------------------------------------------------------------


class TestHNSWSearch:
    def test_hnsw_build_and_search(self) -> None:
        """HNSW nearest-neighbour should agree with brute-force (k=1)."""
        torch.manual_seed(7)
        refs = torch.randn(100, 16)
        q = torch.randn(10, 16)

        brute = IndexManager(method="brute", space="l2")
        brute.fit(refs)
        brute.build_index()
        b_idx, _b_dist = brute.search(q, k=1)

        hnsw = IndexManager(method="hnsw", space="l2")
        hnsw.fit(refs)
        hnsw.build_index()
        h_idx, _h_dist = hnsw.search(q, k=1)

        # For well-separated data HNSW should find the same nearest neighbour
        assert b_idx.shape == h_idx.shape
        match_rate = (b_idx == h_idx).float().mean().item()
        assert match_rate >= 0.8, f"Match rate too low: {match_rate:.2f}"

    @pytest.mark.parametrize(
        "hnsw_params",
        [
            {"M": 8, "efConstruction": 100},
            {"M": 16, "efConstruction": 200},
            {"M": 32, "efConstruction": 400},
        ],
    )
    def test_hnsw_custom_params(self, hnsw_params: dict) -> None:  # type: ignore[type-arg]
        """Index builds successfully with various HNSW parameters."""
        refs = make_refs(n=80, d=8)
        q = make_refs(n=5, d=8, seed=1)

        mgr = IndexManager(method="hnsw", space="l2")
        mgr.fit(refs)
        mgr.build_index(hnsw_params=hnsw_params)
        indices, distances = mgr.search(q, k=3)

        assert indices.shape == (5, 3)
        assert distances.shape == (5, 3)

    def test_hnsw_cosinesimil(self) -> None:
        """HNSW cosine space: identical vector → distance close to 0."""
        refs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        q = torch.tensor([[1.0, 0.0]])

        mgr = IndexManager(method="hnsw", space="cosinesimil")
        mgr.fit(refs)
        mgr.build_index()
        indices, distances = mgr.search(q, k=1)

        assert indices[0, 0].item() == 0
        assert distances[0, 0].item() < 1e-4

    def test_hnsw_zero_padding_fewer_refs_than_k(self) -> None:
        """HNSW: fewer refs than k triggers zero-padding and warning."""
        refs = torch.tensor([[0.0, 0.0]])
        q = torch.tensor([[1.0, 1.0]])

        mgr = IndexManager(method="hnsw", space="l2")
        mgr.fit(refs)
        mgr.build_index()

        with pytest.warns(UserWarning, match="zero padding"):
            indices, distances = mgr.search(q, k=3)

        assert indices.shape == (1, 3)
        assert distances.shape == (1, 3)


# ---------------------------------------------------------------------------
# Batch-wise population
# ---------------------------------------------------------------------------


class TestBatchPopulation:
    def test_add_batch_accumulates_refs(self) -> None:
        mgr = IndexManager(method="brute", space="l2")
        mgr.add_batch(torch.randn(10, 4))
        mgr.add_batch(torch.randn(15, 4))
        mgr.add_batch(torch.randn(5, 4))

        refs = mgr.get_ref_embeddings()
        assert refs is not None
        assert refs.shape == (30, 4)

    def test_batch_matches_single_shot_brute(self) -> None:
        """search after add_batch should match search after fit."""
        torch.manual_seed(3)
        batch1 = torch.randn(20, 8)
        batch2 = torch.randn(15, 8)
        q = torch.randn(5, 8)
        all_refs = torch.cat([batch1, batch2], dim=0)

        single = IndexManager(method="brute", space="l2")
        single.fit(all_refs)
        single.build_index()
        s_idx, s_dist = single.search(q, k=3)

        batched = IndexManager(method="brute", space="l2")
        batched.add_batch(batch1)
        batched.add_batch(batch2)
        batched.build_index()
        b_idx, b_dist = batched.search(q, k=3)

        assert (s_idx == b_idx).all()
        approx(s_dist, b_dist)

    def test_batch_matches_single_shot_hnsw(self) -> None:
        """HNSW after add_batch: nearest neighbours agree with brute."""
        torch.manual_seed(5)
        batch1 = torch.randn(40, 8)
        batch2 = torch.randn(30, 8)
        q = torch.randn(10, 8)
        all_refs = torch.cat([batch1, batch2], dim=0)

        brute = IndexManager(method="brute", space="l2")
        brute.fit(all_refs)
        brute.build_index()
        b_idx, _ = brute.search(q, k=1)

        hnsw = IndexManager(method="hnsw", space="l2")
        hnsw.add_batch(batch1)
        hnsw.add_batch(batch2)
        hnsw.build_index()
        h_idx, _ = hnsw.search(q, k=1)

        match_rate = (b_idx == h_idx).float().mean().item()
        assert match_rate >= 0.8, f"Match rate: {match_rate:.2f}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_brute(self, tmp_path: Path) -> None:
        """Brute-force save → load produces identical search results."""
        torch.manual_seed(42)
        refs = torch.randn(30, 6)
        q = torch.randn(5, 6)

        mgr = IndexManager(method="brute", space="l2")
        mgr.fit(refs)
        mgr.build_index()
        idx_before, dist_before = mgr.search(q, k=3)

        prefix = str(tmp_path / "brute")
        saved = mgr.save(prefix)
        assert "meta" in saved
        assert "ref" in saved
        assert "brute" in saved

        mgr2 = IndexManager(method="brute", space="l2")
        mgr2.load(prefix)
        idx_after, dist_after = mgr2.search(q, k=3)

        assert (idx_before == idx_after).all()
        approx(dist_before, dist_after)

    def test_save_and_load_hnsw(self, tmp_path: Path) -> None:
        """HNSW save → load produces consistent search results."""
        torch.manual_seed(7)
        refs = torch.randn(80, 8)
        q = torch.randn(10, 8)

        mgr = IndexManager(method="hnsw", space="l2")
        mgr.fit(refs)
        mgr.build_index()
        idx_before, _ = mgr.search(q, k=3)

        prefix = str(tmp_path / "hnsw")
        mgr.save(prefix)

        mgr2 = IndexManager(method="hnsw", space="l2")
        mgr2.load(prefix)
        idx_after, _ = mgr2.search(q, k=3)

        # HNSW is approximate; at least 80% of results should match
        match_rate = (idx_before == idx_after).float().mean().item()
        assert match_rate >= 0.8, f"Match rate: {match_rate:.2f}"

    def test_save_with_cal_embeddings(self, tmp_path: Path) -> None:
        refs = torch.randn(20, 4)
        cal = torch.randn(5, 4)

        mgr = IndexManager(method="brute", space="l2")
        mgr.fit(refs, cal_embeddings=cal)
        mgr.build_index()

        prefix = str(tmp_path / "with_cal")
        saved = mgr.save(prefix)
        assert "cal" in saved

        mgr2 = IndexManager(method="brute", space="l2")
        mgr2.load(prefix)
        assert mgr2.get_val_embeddings() is not None

    def test_load_missing_meta_raises(self, tmp_path: Path) -> None:
        mgr = IndexManager(method="brute")
        with pytest.raises(FileNotFoundError):
            mgr.load(str(tmp_path / "nonexistent"))


# ---------------------------------------------------------------------------
# PCA dimensionality reduction
# ---------------------------------------------------------------------------


class TestPCA:
    def test_pca_exp_var_reduces_dimension(self) -> None:
        """PCA with exp_var should reduce embedding dimension."""
        torch.manual_seed(10)
        n, D = 60, 16
        # Data mostly on one latent dimension
        base = torch.randn(n, 1)
        direction = torch.randn(1, D)
        refs = base @ direction + 0.01 * torch.randn(n, D)
        q = torch.randn(3, D)

        mgr = IndexManager(method="brute", space="l2", pca_exp_var=0.90)
        mgr.fit(refs)
        mgr.build_index()
        indices, distances = mgr.search(q, k=2)

        assert indices.shape == (3, 2)
        # PCA should have been fitted
        assert mgr._pca is not None
        assert mgr._pca.n_components_ < D

    def test_pca_components_fixed_count(self) -> None:
        """PCA with pca_components should yield exactly that many components."""
        torch.manual_seed(11)
        refs = torch.randn(40, 12)
        q = torch.randn(5, 12)

        mgr = IndexManager(method="brute", space="l2", pca_components=4)
        mgr.fit(refs)
        mgr.build_index()
        indices, distances = mgr.search(q, k=2)

        assert mgr._pca is not None
        assert mgr._pca.n_components_ == 4
        assert indices.shape == (5, 2)

    def test_pca_consistent_with_manual_transform(self) -> None:
        """search() with PCA matches manually pre-projected brute search."""
        torch.manual_seed(12)
        n, D = 50, 10
        refs = torch.randn(n, D)
        q = torch.randn(4, D)

        # IndexManager with PCA
        mgr = IndexManager(method="brute", space="l2", pca_exp_var=0.95)
        mgr.fit(refs)
        mgr.build_index()
        idx_pca, dist_pca = mgr.search(q, k=2)

        # Manual: project refs and query, then build plain brute
        assert mgr._pca is not None
        import numpy as np

        refs_proj = torch.from_numpy(
            np.asarray(mgr._pca.transform(refs.numpy()), dtype=np.float32)
        )
        q_proj = torch.from_numpy(
            np.asarray(mgr._pca.transform(q.numpy()), dtype=np.float32)
        )

        manual = IndexManager(method="brute", space="l2")
        manual.fit(refs_proj)
        manual.build_index()
        idx_manual, dist_manual = manual.search(q_proj, k=2)

        assert (idx_pca == idx_manual).all()
        approx(dist_pca, dist_manual)

    def test_pca_save_load_preserves_transform(self, tmp_path: Path) -> None:
        """Loaded PCA manager returns the same results as the original."""
        torch.manual_seed(99)
        refs = torch.randn(40, 8)
        q = torch.randn(5, 8)

        mgr = IndexManager(method="brute", space="l2", pca_exp_var=0.90)
        mgr.fit(refs)
        mgr.build_index()
        idx_orig, dist_orig = mgr.search(q, k=2)

        prefix = str(tmp_path / "pca_mgr")
        mgr.save(prefix)

        mgr2 = IndexManager(method="brute", space="l2")
        mgr2.load(prefix)
        idx_loaded, dist_loaded = mgr2.search(q, k=2)

        assert (idx_orig == idx_loaded).all()
        approx(dist_orig, dist_loaded)


# ---------------------------------------------------------------------------
# Getter methods and reset
# ---------------------------------------------------------------------------


class TestGettersAndReset:
    def test_get_ref_embeddings(self) -> None:
        refs = torch.randn(10, 4)
        mgr = IndexManager(method="brute")
        mgr.fit(refs)
        stored = mgr.get_ref_embeddings()
        assert stored is not None
        assert stored.shape == refs.shape

    def test_get_val_embeddings(self) -> None:
        cal = torch.randn(5, 4)
        mgr = IndexManager(method="brute")
        mgr.fit(torch.randn(10, 4), cal_embeddings=cal)
        stored = mgr.get_val_embeddings()
        assert stored is not None
        assert stored.shape == cal.shape

    def test_get_embeddings_none_before_fit(self) -> None:
        mgr = IndexManager(method="brute")
        assert mgr.get_ref_embeddings() is None
        assert mgr.get_val_embeddings() is None

    def test_reset_clears_state(self) -> None:
        refs = torch.randn(10, 4)
        mgr = IndexManager(method="brute")
        mgr.fit(refs)
        mgr.build_index()
        assert mgr.get_ref_embeddings() is not None
        assert mgr._index is not None

        mgr.reset()
        assert mgr.get_ref_embeddings() is None
        assert mgr._index is None
        assert mgr._pca is None
        assert mgr._index_params == {}

    def test_reset_preserves_construction_params(self) -> None:
        mgr = IndexManager(
            method="brute", space="cosinesimil", pca_components=3
        )
        mgr.fit(torch.randn(10, 6))
        mgr.build_index()
        mgr.reset()

        # Construction-time params untouched
        assert mgr.method == "brute"
        assert mgr.space == "cosinesimil"
        assert mgr._pca_components == 3


# ---------------------------------------------------------------------------
# _suggest_hnsw_params
# ---------------------------------------------------------------------------


class TestSuggestHNSWParams:
    def test_small_n_returns_conservative_defaults(self) -> None:
        embs = torch.randn(5, 4)
        params = IndexManager._suggest_hnsw_params(embs, k=3)
        assert "build_defaults" in params
        assert "query_defaults" in params
        assert params["query_defaults"]["efSearch"] == 3

    def test_large_n_has_reasonable_m(self) -> None:
        embs = torch.randn(10_000, 64)
        params = IndexManager._suggest_hnsw_params(embs, k=10)
        M = params["build_defaults"]["M"]
        assert 8 <= M <= 64

    def test_ef_construction_within_bounds(self) -> None:
        embs = torch.randn(500, 32)
        params = IndexManager._suggest_hnsw_params(embs)
        ef = params["build_defaults"]["efConstruction"]
        assert 100 <= ef <= 2000

    def test_m_scales_with_sqrt_dimension(self) -> None:
        """M should be approximately 2 * sqrt(D), clamped to [8, 64]."""
        for D in [4, 16, 64, 512]:
            embs = torch.randn(200, D)
            params = IndexManager._suggest_hnsw_params(embs)
            expected_M = max(8, min(64, int(round(2.0 * math.sqrt(D)))))
            assert params["build_defaults"]["M"] == expected_M

    def test_1d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            IndexManager._suggest_hnsw_params(torch.randn(10))
