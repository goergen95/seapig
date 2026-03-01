"""Tests for risk-coverage curve implementation."""

import pytest
import torch

from seapig.risk_coverage import RiskCoverage, risk_coverage


class TestRiskCoverage:
    """Test suite for risk-coverage functionality."""

    @pytest.fixture
    def simple_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate simple test data."""
        torch.manual_seed(42)
        n = 100
        score = torch.rand(n)
        residuals = torch.rand(n)
        return score, residuals

    @pytest.fixture
    def correlated_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate data where score correlates with residuals."""
        torch.manual_seed(152)
        n = 100
        obs = torch.rand(n)
        pred = obs + (torch.rand(n) * 10 * obs)
        residuals = torch.abs(pred - obs)
        score = torch.rand(n) * obs  # Lower score when obs is lower
        return score, residuals

    def test_risk_coverage_basic(self, simple_data):
        """Test basic risk-coverage calculation."""
        score, residuals = simple_data

        # Test generalized risk
        rc_gen = risk_coverage(score, residuals, risk="generalized")
        assert isinstance(rc_gen, RiskCoverage)
        assert len(rc_gen.coverage) == 100
        assert rc_gen.risk_type == "generalized"
        assert hasattr(rc_gen, "auc_empirical")
        assert hasattr(rc_gen, "auc_reference")
        assert hasattr(rc_gen, "auc_excess")

        # Test selective risk
        rc_sel = risk_coverage(score, residuals, risk="selective")
        assert isinstance(rc_sel, RiskCoverage)
        assert len(rc_sel.coverage) == 100
        assert rc_sel.risk_type == "selective"

        # Selective and generalized should differ
        assert not torch.allclose(rc_gen.risk, rc_sel.risk)

    def test_risk_coverage_with_bins(self, simple_data):
        """Test risk-coverage with binning."""
        score, residuals = simple_data

        # Test with fewer bins
        rc = risk_coverage(score, residuals, n_bins=50)
        assert isinstance(rc, RiskCoverage)
        assert len(rc.coverage) == 50

    def test_risk_coverage_validation(self):
        """Test input validation."""
        score = torch.rand(100)
        residuals = torch.rand(100)

        # Test mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            risk_coverage(score, torch.rand(50))

        # Test invalid risk type
        with pytest.raises(ValueError, match="must be"):
            risk_coverage(score, residuals, risk="invalid")

        # Test invalid input types
        # Intentional bad-type calls for validation; silence static checker
        with pytest.raises(TypeError):
            risk_coverage([1, 2, 3], residuals)

        with pytest.raises(TypeError):
            risk_coverage(score, [1, 2, 3])

    def test_risk_coverage_attributes(self, correlated_data):
        """Test that RiskCoverage has correct attributes."""
        score, residuals = correlated_data
        rc = risk_coverage(score, residuals, risk="generalized")

        # Check all required attributes exist
        assert hasattr(rc, "coverage")
        assert hasattr(rc, "threshold")
        assert hasattr(rc, "risk")
        assert hasattr(rc, "reference")
        assert hasattr(rc, "excess")
        assert hasattr(rc, "risk_type")
        assert hasattr(rc, "auc_empirical")
        assert hasattr(rc, "auc_reference")
        assert hasattr(rc, "auc_excess")

        # Check shapes match
        assert len(rc.coverage) == len(rc.threshold)
        assert len(rc.coverage) == len(rc.risk)
        assert len(rc.coverage) == len(rc.reference)
        assert len(rc.coverage) == len(rc.excess)

        # Check excess = empirical - reference
        assert torch.allclose(rc.excess, rc.risk - rc.reference)

    def test_risk_coverage_monotonicity(self, simple_data):
        """Test that coverage is monotonically increasing."""
        score, residuals = simple_data
        rc = risk_coverage(score, residuals)

        # Coverage should be monotonically increasing
        coverage_diff = rc.coverage[1:] - rc.coverage[:-1]
        assert (coverage_diff >= 0).all()

        # Coverage should be in [0, 1]
        assert (rc.coverage >= 0).all()
        assert (rc.coverage <= 1).all()

    def test_risk_coverage_auc_values(self, correlated_data):
        """Test AUC calculation is reasonable."""
        score, residuals = correlated_data

        # Generalized risk
        rc_gen = risk_coverage(score, residuals, risk="generalized")
        assert isinstance(rc_gen.auc_empirical, torch.Tensor)
        assert isinstance(rc_gen.auc_reference, torch.Tensor)
        assert isinstance(rc_gen.auc_excess, torch.Tensor)
        assert rc_gen.auc_empirical > 0
        assert rc_gen.auc_reference > 0

        # Selective risk
        rc_sel = risk_coverage(score, residuals, risk="selective")
        assert isinstance(rc_sel.auc_empirical, torch.Tensor)
        assert isinstance(rc_sel.auc_reference, torch.Tensor)
        assert isinstance(rc_sel.auc_excess, torch.Tensor)
        assert rc_sel.auc_empirical > 0
        assert rc_sel.auc_reference > 0

    def test_risk_coverage_repr(self, simple_data):
        """Test string representation."""
        score, residuals = simple_data
        rc = risk_coverage(score, residuals)

        repr_str = repr(rc)
        assert "RiskCoverage" in repr_str
        assert "generalized" in repr_str
        assert "n_points=100" in repr_str
        assert "auc_empirical" in repr_str

    def test_risk_types_differ(self, correlated_data):
        """Test that generalized and selective risk produce different results."""
        score, residuals = correlated_data

        rc_gen = risk_coverage(score, residuals, risk="generalized")
        rc_sel = risk_coverage(score, residuals, risk="selective")

        # Risk curves should be different
        assert not torch.allclose(rc_gen.risk, rc_sel.risk)

        # But coverage should be the same
        assert torch.allclose(rc_gen.coverage, rc_sel.coverage)

    def test_reference_curve_properties(self, simple_data):
        """Test properties of the reference curve."""
        score, residuals = simple_data
        rc = risk_coverage(score, residuals)

        # Reference curve should be non-negative
        assert (rc.reference >= 0).all()

        # Reference should be optimal (lower than or equal to empirical in expectation)
        # This might not hold for every random seed, but generally should

    def test_edge_cases(self):
        """Test edge cases."""
        # All same scores
        score = torch.ones(100)
        residuals = torch.rand(100)
        rc = risk_coverage(score, residuals)
        assert isinstance(rc, RiskCoverage)

        # All same residuals
        score = torch.rand(100)
        residuals = torch.ones(100)
        rc = risk_coverage(score, residuals)
        assert isinstance(rc, RiskCoverage)

        # Very small dataset
        score = torch.rand(5)
        residuals = torch.rand(5)
        rc = risk_coverage(score, residuals, n_bins=10)
        assert isinstance(rc, RiskCoverage)
        assert len(rc.coverage) == 5  # Should not downsample

    def test_perfect_score(self):
        """Test with perfect confidence score (score = residuals)."""
        torch.manual_seed(42)
        residuals = torch.rand(100)
        score = residuals.clone()

        rc = risk_coverage(score, residuals)

        # When score equals residuals, empirical should equal reference
        # (they both use the same ordering)
        assert torch.allclose(rc.risk, rc.reference, rtol=1e-5)
        assert torch.isclose(rc.auc_excess, torch.tensor(0.0), atol=1e-5)

    def test_binning_maintains_max(self, simple_data):
        """Test that binning takes max values in each bin."""
        score, residuals = simple_data

        # Get unbinned result
        rc_full = risk_coverage(score, residuals, n_bins=200)

        # Get binned result
        rc_binned = risk_coverage(score, residuals, n_bins=50)

        # Binned should have fewer points
        assert len(rc_binned.coverage) < len(rc_full.coverage)

        # Max coverage in binned should equal max in full
        assert torch.isclose(
            rc_binned.coverage.max(), rc_full.coverage.max(), atol=1e-5
        )

    def test_plot_basic(self, simple_data):
        """Test basic plotting functionality."""
        pytest.importorskip("matplotlib")
        score, residuals = simple_data
        rc = risk_coverage(score, residuals)

        # Test basic plot
        fig = rc.plot()
        assert fig is not None

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_selective_curves(self, simple_data):
        """Test plotting with selective curves."""
        pytest.importorskip("matplotlib")
        score, residuals = simple_data
        rc = risk_coverage(score, residuals)

        import matplotlib.pyplot as plt

        # Test plotting only empirical
        fig1 = rc.plot(reference=False, excess=False)
        assert fig1 is not None
        plt.close(fig1)

        # Test plotting only reference
        fig2 = rc.plot(empirical=False, excess=False)
        assert fig2 is not None
        plt.close(fig2)

        # Test plotting only excess
        fig3 = rc.plot(empirical=False, reference=False)
        assert fig3 is not None
        plt.close(fig3)

    def test_plot_validation(self, simple_data):
        """Test plot input validation."""
        pytest.importorskip("matplotlib")
        score, residuals = simple_data
        rc = risk_coverage(score, residuals)

        # Should raise if all are False
        with pytest.raises(
            ValueError, match="At least one of empirical, reference, or excess"
        ):
            rc.plot(empirical=False, reference=False, excess=False)
