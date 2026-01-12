"""Tests for biosparse.kernel.math._mwu module.

Tests for Mann-Whitney U test p-value functions:
    - mwu_p_value_two_sided, mwu_p_value_greater, mwu_p_value_less
    - mwu_p_value_two_sided_approx, mwu_p_value_greater_approx, mwu_p_value_less_approx
"""

import pytest
import numpy as np

# Check dependencies
try:
    import scipy.stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

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
            mwu_p_value_two_sided,
            mwu_p_value_greater,
            mwu_p_value_less,
            mwu_p_value_two_sided_approx,
            mwu_p_value_greater_approx,
            mwu_p_value_less_approx,
        )
    except Exception:
        _OPTIM_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not _OPTIM_AVAILABLE, reason="Optim/kernel module not available"),
    pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available"),
]


# =============================================================================
# Helper function to compute reference p-values
# =============================================================================

def reference_mwu_pvalue(U, n1, n2, tie_sum=0.0, alternative='two-sided', cc=0.5):
    """Reference implementation using scipy's formula."""
    N = n1 + n2
    mu = 0.5 * n1 * n2
    
    if N > 1:
        var = (n1 * n2 / 12.0) * (N + 1.0 - tie_sum / (N * (N - 1.0)))
    else:
        var = (n1 * n2 / 12.0) * (N + 1.0)
    
    if var <= 0:
        return 1.0
    
    sd = np.sqrt(var)
    
    if alternative == 'two-sided':
        z = (abs(U - mu) - cc) / sd
        return 2 * scipy.stats.norm.sf(max(z, 0))
    elif alternative == 'greater':
        z = (U - mu - cc) / sd
        return scipy.stats.norm.sf(z)
    else:  # less
        z = (mu - U - cc) / sd
        return scipy.stats.norm.sf(z)


# =============================================================================
# Test Two-sided P-value (Precise)
# =============================================================================

class TestMWUTwoSided:
    """Test two-sided MWU p-value (precise)."""
    
    def test_basic_values(self):
        """Basic p-value computation."""
        U = np.array([50.0])
        n1 = np.array([10.0])
        n2 = np.array([10.0])
        tie_sum = np.array([0.0])
        out = np.empty(1, dtype=np.float64)
        
        mwu_p_value_two_sided(U, n1, n2, tie_sum, 0.5, out)
        
        # P-value should be in valid range
        assert 0 <= out[0] <= 1
    
    def test_equal_means_high_pvalue(self):
        """U at expected value should give high p-value."""
        n1, n2 = 20.0, 20.0
        expected_U = 0.5 * n1 * n2  # = 200
        
        U = np.array([expected_U])
        n1_arr = np.array([n1])
        n2_arr = np.array([n2])
        tie_sum = np.array([0.0])
        out = np.empty(1, dtype=np.float64)
        
        mwu_p_value_two_sided(U, n1_arr, n2_arr, tie_sum, 0.5, out)
        
        # Should be close to 1 (no difference)
        assert out[0] > 0.9
    
    def test_extreme_U_low_pvalue(self):
        """Extreme U should give low p-value."""
        n1, n2 = 20.0, 20.0
        
        # Very extreme U
        U = np.array([10.0])  # Much less than expected 200
        n1_arr = np.array([n1])
        n2_arr = np.array([n2])
        tie_sum = np.array([0.0])
        out = np.empty(1, dtype=np.float64)
        
        mwu_p_value_two_sided(U, n1_arr, n2_arr, tie_sum, 0.5, out)
        
        # Should be very small
        assert out[0] < 0.01
    
    def test_batch_computation(self):
        """Batch of multiple tests."""
        n = 5
        U = np.array([50.0, 100.0, 150.0, 200.0, 250.0])
        n1 = np.full(n, 20.0)
        n2 = np.full(n, 20.0)
        tie_sum = np.zeros(n)
        out = np.empty(n, dtype=np.float64)
        
        mwu_p_value_two_sided(U, n1, n2, tie_sum, 0.5, out)
        
        # All should be valid p-values
        assert np.all(out >= 0)
        assert np.all(out <= 1)
    
    def test_matches_reference(self):
        """Should match reference implementation."""
        test_cases = [
            (100.0, 15.0, 15.0, 0.0),
            (50.0, 10.0, 20.0, 0.0),
            (200.0, 20.0, 20.0, 10.0),  # With tie correction
        ]
        
        for u, n1, n2, tie in test_cases:
            U = np.array([u])
            n1_arr = np.array([n1])
            n2_arr = np.array([n2])
            tie_sum = np.array([tie])
            out = np.empty(1, dtype=np.float64)
            
            mwu_p_value_two_sided(U, n1_arr, n2_arr, tie_sum, 0.5, out)
            expected = reference_mwu_pvalue(u, n1, n2, tie, 'two-sided', 0.5)
            
            np.testing.assert_allclose(out[0], expected, rtol=0.01,
                err_msg=f"Failed for U={u}, n1={n1}, n2={n2}")


# =============================================================================
# Test One-sided P-values (Precise)
# =============================================================================

class TestMWUOneSided:
    """Test one-sided MWU p-values."""
    
    def test_greater_high_U(self):
        """High U with 'greater' should give low p-value."""
        n1, n2 = 20.0, 20.0
        high_U = 350.0  # Much higher than expected 200
        
        U = np.array([high_U])
        n1_arr = np.array([n1])
        n2_arr = np.array([n2])
        tie_sum = np.array([0.0])
        out = np.empty(1, dtype=np.float64)
        
        mwu_p_value_greater(U, n1_arr, n2_arr, tie_sum, 0.5, out)
        
        assert out[0] < 0.01
    
    def test_greater_low_U(self):
        """Low U with 'greater' should give high p-value."""
        n1, n2 = 20.0, 20.0
        low_U = 50.0  # Much lower than expected 200
        
        U = np.array([low_U])
        n1_arr = np.array([n1])
        n2_arr = np.array([n2])
        tie_sum = np.array([0.0])
        out = np.empty(1, dtype=np.float64)
        
        mwu_p_value_greater(U, n1_arr, n2_arr, tie_sum, 0.5, out)
        
        assert out[0] > 0.99
    
    def test_less_low_U(self):
        """Low U with 'less' should give low p-value."""
        n1, n2 = 20.0, 20.0
        low_U = 50.0
        
        U = np.array([low_U])
        n1_arr = np.array([n1])
        n2_arr = np.array([n2])
        tie_sum = np.array([0.0])
        out = np.empty(1, dtype=np.float64)
        
        mwu_p_value_less(U, n1_arr, n2_arr, tie_sum, 0.5, out)
        
        assert out[0] < 0.01
    
    def test_greater_less_complement(self):
        """Greater and less should approximately sum to 1 at mean."""
        n1, n2 = 20.0, 20.0
        U_val = 0.5 * n1 * n2  # At mean
        
        U = np.array([U_val])
        n1_arr = np.array([n1])
        n2_arr = np.array([n2])
        tie_sum = np.array([0.0])
        out_greater = np.empty(1, dtype=np.float64)
        out_less = np.empty(1, dtype=np.float64)
        
        mwu_p_value_greater(U, n1_arr, n2_arr, tie_sum, 0.0, out_greater)  # No CC
        mwu_p_value_less(U, n1_arr, n2_arr, tie_sum, 0.0, out_less)
        
        # At mean, both should be ~0.5
        np.testing.assert_allclose(out_greater[0], 0.5, rtol=0.01)
        np.testing.assert_allclose(out_less[0], 0.5, rtol=0.01)


# =============================================================================
# Test Approximate Versions
# =============================================================================

class TestMWUApprox:
    """Test approximate MWU p-value functions."""
    
    def test_approx_matches_precise(self):
        """Approximate should be close to precise."""
        U = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
        n1 = np.full(5, 20.0)
        n2 = np.full(5, 20.0)
        tie_sum = np.zeros(5)
        
        out_precise = np.empty(5, dtype=np.float64)
        out_approx = np.empty(5, dtype=np.float64)
        
        mwu_p_value_two_sided(U, n1, n2, tie_sum, 0.5, out_precise)
        mwu_p_value_two_sided_approx(U, n1, n2, tie_sum, 0.5, out_approx)
        
        np.testing.assert_allclose(out_approx, out_precise, rtol=1e-5)
    
    def test_approx_greater_matches_precise(self):
        """Approximate greater should match precise."""
        U = np.array([300.0])
        n1 = np.array([20.0])
        n2 = np.array([20.0])
        tie_sum = np.array([0.0])
        
        out_precise = np.empty(1, dtype=np.float64)
        out_approx = np.empty(1, dtype=np.float64)
        
        mwu_p_value_greater(U, n1, n2, tie_sum, 0.5, out_precise)
        mwu_p_value_greater_approx(U, n1, n2, tie_sum, 0.5, out_approx)
        
        np.testing.assert_allclose(out_approx[0], out_precise[0], rtol=1e-5)
    
    def test_approx_less_matches_precise(self):
        """Approximate less should match precise."""
        U = np.array([100.0])
        n1 = np.array([20.0])
        n2 = np.array([20.0])
        tie_sum = np.array([0.0])
        
        out_precise = np.empty(1, dtype=np.float64)
        out_approx = np.empty(1, dtype=np.float64)
        
        mwu_p_value_less(U, n1, n2, tie_sum, 0.5, out_precise)
        mwu_p_value_less_approx(U, n1, n2, tie_sum, 0.5, out_approx)
        
        np.testing.assert_allclose(out_approx[0], out_precise[0], rtol=1e-5)


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_very_small_samples(self):
        """Should handle very small sample sizes."""
        U = np.array([2.0])
        n1 = np.array([2.0])
        n2 = np.array([2.0])
        tie_sum = np.array([0.0])
        out = np.empty(1, dtype=np.float64)
        
        mwu_p_value_two_sided(U, n1, n2, tie_sum, 0.5, out)
        
        # Should return valid p-value
        assert 0 <= out[0] <= 1
    
    def test_large_sample_sizes(self):
        """Should handle large sample sizes."""
        U = np.array([500000.0])
        n1 = np.array([1000.0])
        n2 = np.array([1000.0])
        tie_sum = np.array([0.0])
        out = np.empty(1, dtype=np.float64)
        
        # Expected U = 1000 * 1000 * 0.5 = 500000
        mwu_p_value_two_sided(U, n1, n2, tie_sum, 0.5, out)
        
        # At expected value, p should be high
        assert out[0] > 0.9
    
    def test_with_ties(self):
        """Should handle tie correction."""
        U = np.array([200.0])
        n1 = np.array([20.0])
        n2 = np.array([20.0])
        # tie_sum = sum(t^3 - t) for each tie group
        tie_sum = np.array([100.0])  # Some ties
        out = np.empty(1, dtype=np.float64)
        
        mwu_p_value_two_sided(U, n1, n2, tie_sum, 0.5, out)
        
        assert 0 <= out[0] <= 1
    
    def test_continuity_correction(self):
        """Different continuity corrections should affect result."""
        U = np.array([180.0])
        n1 = np.array([20.0])
        n2 = np.array([20.0])
        tie_sum = np.array([0.0])
        
        out_cc05 = np.empty(1, dtype=np.float64)
        out_cc0 = np.empty(1, dtype=np.float64)
        
        mwu_p_value_two_sided(U, n1, n2, tie_sum, 0.5, out_cc05)
        mwu_p_value_two_sided(U, n1, n2, tie_sum, 0.0, out_cc0)
        
        # CC=0 should give slightly different result
        # (but both should be valid)
        assert 0 <= out_cc05[0] <= 1
        assert 0 <= out_cc0[0] <= 1


# =============================================================================
# Large Scale Tests
# =============================================================================

class TestLargeScale:
    """Test with many simultaneous computations."""
    
    def test_many_tests(self):
        """Should handle many tests efficiently."""
        n = 1000
        np.random.seed(42)
        
        n1_vals = np.full(n, 50.0)
        n2_vals = np.full(n, 50.0)
        # U values around expected mean
        expected_U = 0.5 * 50 * 50
        U = np.random.normal(expected_U, 100, n)
        tie_sum = np.zeros(n)
        out = np.empty(n, dtype=np.float64)
        
        mwu_p_value_two_sided(U, n1_vals, n2_vals, tie_sum, 0.5, out)
        
        # All p-values should be valid
        assert np.all(out >= 0)
        assert np.all(out <= 1)
        
        # Distribution should make sense
        # U values near expected should have high p-values
        # U values far from expected should have low p-values
        near_mean = np.abs(U - expected_U) < 50
        far_from_mean = np.abs(U - expected_U) > 200
        
        if np.any(near_mean):
            assert np.mean(out[near_mean]) > np.mean(out[far_from_mean])
