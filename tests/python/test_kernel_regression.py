"""Tests for biosparse.kernel.math._regression module.

Tests for regression functions:
    - loess_fit, loess_fit_sorted, loess_fit_parallel
    - weighted_polyfit_1, weighted_polyfit_2
    - tricube_weight
    - compute_vst_clip_values, compute_normalized_variance
"""

import pytest
import numpy as np

# Check if optim is available
_OPTIM_AVAILABLE = False

try:
    from biosparse.optim import disable_logging
    disable_logging()
    _OPTIM_AVAILABLE = True
except Exception:
    _OPTIM_AVAILABLE = False

if _OPTIM_AVAILABLE:
    try:
        from biosparse.kernel.math import (
            loess_fit, loess_fit_sorted, loess_fit_parallel,
            weighted_polyfit_1, weighted_polyfit_2,
            tricube_weight,
            compute_vst_clip_values, compute_normalized_variance,
        )
    except Exception:
        _OPTIM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _OPTIM_AVAILABLE,
    reason="Optim/kernel module not available"
)


# =============================================================================
# Test Tricube Weight
# =============================================================================

class TestTricubeWeight:
    """Test tricube weight function."""
    
    def test_tricube_zero(self):
        """Tricube at 0 should be 1."""
        assert tricube_weight(0.0) == 1.0
    
    def test_tricube_one(self):
        """Tricube at 1 should be 0."""
        assert tricube_weight(1.0) == 0.0
    
    def test_tricube_beyond_one(self):
        """Tricube beyond 1 should be 0."""
        assert tricube_weight(1.5) == 0.0
        assert tricube_weight(2.0) == 0.0
    
    def test_tricube_half(self):
        """Tricube at 0.5."""
        d = 0.5
        expected = (1 - d**3)**3
        result = tricube_weight(d)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_tricube_decreasing(self):
        """Tricube should be monotonically decreasing."""
        d_values = np.linspace(0.0, 1.0, 11)
        weights = [tricube_weight(d) for d in d_values]
        
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]


# =============================================================================
# Test Weighted Polynomial Fitting
# =============================================================================

class TestWeightedPolyfit:
    """Test weighted polynomial fitting."""
    
    def test_weighted_polyfit_1_linear(self):
        """Linear fit: y = 2x."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        weights = np.ones(5, dtype=np.float64)
        
        a, b = weighted_polyfit_1(x, y, weights, 5)
        
        # Should recover y = 0 + 2x
        np.testing.assert_allclose(a, 0.0, atol=1e-10)
        np.testing.assert_allclose(b, 2.0, atol=1e-10)
    
    def test_weighted_polyfit_1_with_intercept(self):
        """Linear fit: y = 3 + 2x."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 7.0, 9.0, 11.0, 13.0])  # 3 + 2x
        weights = np.ones(5, dtype=np.float64)
        
        a, b = weighted_polyfit_1(x, y, weights, 5)
        
        np.testing.assert_allclose(a, 3.0, atol=1e-10)
        np.testing.assert_allclose(b, 2.0, atol=1e-10)
    
    def test_weighted_polyfit_1_with_weights(self):
        """Linear fit with different weights."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 7.0, 9.0, 11.0, 13.0])
        # Give more weight to first points
        weights = np.array([10.0, 10.0, 1.0, 1.0, 1.0])
        
        a, b = weighted_polyfit_1(x, y, weights, 5)
        
        # Should still be reasonably close to true values
        np.testing.assert_allclose(a, 3.0, atol=0.2)
        np.testing.assert_allclose(b, 2.0, atol=0.1)
    
    def test_weighted_polyfit_2_linear(self):
        """Quadratic fit of linear data should have c ≈ 0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 7.0, 9.0, 11.0, 13.0])
        weights = np.ones(5, dtype=np.float64)
        
        a, b, c = weighted_polyfit_2(x, y, weights, 5)
        
        # c should be near zero for linear data
        np.testing.assert_allclose(c, 0.0, atol=1e-10)
    
    def test_weighted_polyfit_2_quadratic(self):
        """Quadratic fit: y = 1 + 2x + 3x^2."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 6.0, 17.0, 34.0, 57.0])  # 1 + 2x + 3x^2
        weights = np.ones(5, dtype=np.float64)
        
        a, b, c = weighted_polyfit_2(x, y, weights, 5)
        
        np.testing.assert_allclose(a, 1.0, atol=1e-8)
        np.testing.assert_allclose(b, 2.0, atol=1e-8)
        np.testing.assert_allclose(c, 3.0, atol=1e-8)


# =============================================================================
# Test LOESS Fitting
# =============================================================================

class TestLoessFit:
    """Test LOESS fitting functions."""
    
    def test_loess_fit_linear(self):
        """LOESS should fit linear data well."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.1, 50)
        
        fitted = loess_fit(x, y, span=0.5, degree=1)
        
        # Fitted values should be close to true line
        true_y = 2.0 * x + 1.0
        rmse = np.sqrt(np.mean((fitted - true_y) ** 2))
        assert rmse < 0.5
    
    def test_loess_fit_quadratic(self):
        """LOESS with degree=2 should fit quadratic data."""
        np.random.seed(42)
        x = np.linspace(-5, 5, 50)
        y = x ** 2 + np.random.normal(0, 0.5, 50)
        
        fitted = loess_fit(x, y, span=0.5, degree=2)
        
        # Fitted values should be close to parabola
        true_y = x ** 2
        rmse = np.sqrt(np.mean((fitted - true_y) ** 2))
        assert rmse < 2.0
    
    def test_loess_fit_sorted_same_as_parallel(self):
        """loess_fit_sorted should give similar results."""
        np.random.seed(42)
        x = np.sort(np.random.rand(30))  # Already sorted
        y = np.sin(x * 2 * np.pi) + np.random.normal(0, 0.1, 30)
        
        fitted1 = loess_fit_sorted(x, y, span=0.5, degree=2)
        fitted2 = loess_fit_parallel(x, y, span=0.5, degree=2)
        
        # Results should be very similar
        np.testing.assert_allclose(fitted1, fitted2, rtol=1e-6)
    
    def test_loess_fit_span_effect(self):
        """Larger span should produce smoother fit."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + np.random.normal(0, 0.3, 50)
        
        fitted_small = loess_fit(x, y, span=0.2, degree=2)
        fitted_large = loess_fit(x, y, span=0.8, degree=2)
        
        # Larger span should have smaller variance in fitted values
        var_small = np.var(fitted_small)
        var_large = np.var(fitted_large)
        
        # The large span fit should have similar or smaller variance
        # (smoother = less variation around mean)
        assert var_large <= var_small * 1.5  # Allow some tolerance
    
    def test_loess_fit_preserves_length(self):
        """LOESS output should have same length as input."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.random.randn(100)
        
        fitted = loess_fit(x, y, span=0.3, degree=2)
        
        assert len(fitted) == len(x)


# =============================================================================
# Test HVG Utility Functions
# =============================================================================

class TestVSTClipValues:
    """Test VST clip value computation."""
    
    def test_compute_vst_clip_values(self):
        """VST clip values computation."""
        means = np.array([1.0, 2.0, 3.0])
        fitted_log_var = np.array([0.0, 1.0, 2.0])  # log10 of variance
        n_cells = 100
        
        clip_vals = compute_vst_clip_values(means, fitted_log_var, n_cells)
        
        assert len(clip_vals) == 3
        # clip = reg_std * sqrt(n) + mean
        # reg_std = sqrt(10^fitted_log_var)
        sqrt_n = np.sqrt(100)
        expected = [np.sqrt(10**flv) * sqrt_n + m for m, flv in zip(means, fitted_log_var)]
        np.testing.assert_allclose(clip_vals, expected, rtol=1e-10)


class TestNormalizedVariance:
    """Test normalized variance computation."""
    
    def test_compute_normalized_variance(self):
        """Normalized variance for Seurat V3."""
        means = np.array([1.0, 2.0, 3.0])
        sum_clipped = np.array([10.0, 20.0, 30.0])
        sum_sq_clipped = np.array([120.0, 420.0, 920.0])
        reg_std = np.array([1.0, 2.0, 3.0])
        n_cells = 10
        
        norm_var = compute_normalized_variance(
            means, sum_clipped, sum_sq_clipped, reg_std, n_cells
        )
        
        assert len(norm_var) == 3
        # Values should be non-negative
        assert np.all(norm_var >= 0)
    
    def test_normalized_variance_zero_std(self):
        """Normalized variance with zero std should be zero."""
        means = np.array([1.0])
        sum_clipped = np.array([10.0])
        sum_sq_clipped = np.array([100.0])
        reg_std = np.array([0.0])  # Zero std
        n_cells = 10
        
        norm_var = compute_normalized_variance(
            means, sum_clipped, sum_sq_clipped, reg_std, n_cells
        )
        
        assert norm_var[0] == 0.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestLoessIntegration:
    """Integration tests for LOESS with realistic data."""
    
    def test_loess_with_noise(self):
        """LOESS should handle noisy data."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 4 * np.pi, n)
        y = np.sin(x) + np.random.normal(0, 0.5, n)
        
        fitted = loess_fit(x, y, span=0.3, degree=2)
        
        # Fitted should be smoother than original
        # Compare variances of consecutive differences
        diff_original = np.diff(y)
        diff_fitted = np.diff(fitted)
        
        assert np.var(diff_fitted) < np.var(diff_original)
    
    def test_loess_edge_cases(self):
        """LOESS should handle edge cases."""
        # Minimum required points (degree + 1)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        
        fitted = loess_fit(x, y, span=1.0, degree=1)  # Use all points
        
        assert len(fitted) == 3
        # Should be close to linear
        np.testing.assert_allclose(fitted, y, rtol=0.05)
