"""Tests for biosparse.kernel.math._ttest module.

Tests for t-test utility functions:
    - welch_test, student_test: Complete t-test p-value computation
    - welch_se, welch_df, pooled_se: Standard error and degrees of freedom
    - welch_test_approx, student_test_approx: Approximate versions
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
            welch_test, student_test,
            welch_se, welch_df, pooled_se,
        )
        # Try to import approx versions
        try:
            from biosparse.kernel.math._ttest import (
                welch_test_approx, student_test_approx
            )
            APPROX_AVAILABLE = True
        except ImportError:
            APPROX_AVAILABLE = False
    except Exception:
        _OPTIM_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not _OPTIM_AVAILABLE, reason="Optim/kernel module not available"),
    pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available"),
]


# =============================================================================
# Test Standard Error Functions
# =============================================================================

class TestWelchSE:
    """Test Welch's standard error computation."""
    
    def test_basic_computation(self):
        """Basic SE computation."""
        var1 = np.array([4.0])
        n1 = np.array([10.0])
        var2 = np.array([9.0])
        n2 = np.array([10.0])
        out = np.empty(1, dtype=np.float64)
        
        welch_se(var1, n1, var2, n2, out)
        
        # SE = sqrt(4/10 + 9/10) = sqrt(1.3)
        expected = np.sqrt(4.0/10.0 + 9.0/10.0)
        np.testing.assert_allclose(out[0], expected, rtol=1e-10)
    
    def test_equal_variances(self):
        """SE with equal variances."""
        var1 = np.array([4.0])
        n1 = np.array([20.0])
        var2 = np.array([4.0])
        n2 = np.array([20.0])
        out = np.empty(1, dtype=np.float64)
        
        welch_se(var1, n1, var2, n2, out)
        
        expected = np.sqrt(4.0/20.0 + 4.0/20.0)
        np.testing.assert_allclose(out[0], expected, rtol=1e-10)
    
    def test_batch(self):
        """Batch SE computation."""
        n = 5
        var1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n1 = np.full(n, 10.0)
        var2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n2 = np.full(n, 10.0)
        out = np.empty(n, dtype=np.float64)
        
        welch_se(var1, n1, var2, n2, out)
        
        expected = np.sqrt(var1/n1 + var2/n2)
        np.testing.assert_allclose(out, expected, rtol=1e-10)


class TestWelchDF:
    """Test Welch-Satterthwaite degrees of freedom."""
    
    def test_equal_variances_equal_n(self):
        """DF with equal variances and sample sizes."""
        var1 = np.array([4.0])
        n1 = np.array([10.0])
        var2 = np.array([4.0])
        n2 = np.array([10.0])
        out = np.empty(1, dtype=np.float64)
        
        welch_df(var1, n1, var2, n2, out)
        
        # With equal var and n, df should be n1 + n2 - 2 = 18
        np.testing.assert_allclose(out[0], 18.0, rtol=1e-5)
    
    def test_unequal_variances(self):
        """DF with unequal variances."""
        var1 = np.array([1.0])
        n1 = np.array([20.0])
        var2 = np.array([9.0])
        n2 = np.array([20.0])
        out = np.empty(1, dtype=np.float64)
        
        welch_df(var1, n1, var2, n2, out)
        
        # DF should be positive and less than n1 + n2 - 2
        assert out[0] > 0
        assert out[0] < 38
    
    def test_matches_scipy(self):
        """DF should match scipy's computation."""
        # Generate some random data
        np.random.seed(42)
        x = np.random.normal(0, 1, 30)
        y = np.random.normal(0, 2, 25)
        
        var1 = np.array([np.var(x, ddof=1)])
        n1 = np.array([float(len(x))])
        var2 = np.array([np.var(y, ddof=1)])
        n2 = np.array([float(len(y))])
        out = np.empty(1, dtype=np.float64)
        
        welch_df(var1, n1, var2, n2, out)
        
        # Compute expected DF using Welch-Satterthwaite formula
        v1_n1 = var1[0] / n1[0]
        v2_n2 = var2[0] / n2[0]
        sum_v = v1_n1 + v2_n2
        denom = (v1_n1**2) / (n1[0] - 1) + (v2_n2**2) / (n2[0] - 1)
        expected_df = (sum_v**2) / denom
        
        np.testing.assert_allclose(out[0], expected_df, rtol=1e-10)


class TestPooledSE:
    """Test pooled standard error for Student's t-test."""
    
    def test_basic_computation(self):
        """Basic pooled SE computation."""
        var1 = np.array([4.0])
        n1 = np.array([10.0])
        var2 = np.array([4.0])
        n2 = np.array([10.0])
        out = np.empty(1, dtype=np.float64)
        
        pooled_se(var1, n1, var2, n2, out)
        
        # pooled_var = ((10-1)*4 + (10-1)*4) / (10+10-2) = 72/18 = 4
        # SE = sqrt(4 * (1/10 + 1/10)) = sqrt(0.8)
        expected = np.sqrt(4.0 * (1/10 + 1/10))
        np.testing.assert_allclose(out[0], expected, rtol=1e-10)
    
    def test_unequal_sample_sizes(self):
        """Pooled SE with unequal sample sizes."""
        var1 = np.array([4.0])
        n1 = np.array([30.0])
        var2 = np.array([4.0])
        n2 = np.array([10.0])
        out = np.empty(1, dtype=np.float64)
        
        pooled_se(var1, n1, var2, n2, out)
        
        df = 30 + 10 - 2
        pooled_var = ((30-1)*4 + (10-1)*4) / df
        expected = np.sqrt(pooled_var * (1/30 + 1/10))
        np.testing.assert_allclose(out[0], expected, rtol=1e-10)


# =============================================================================
# Test Complete T-test Functions
# =============================================================================

class TestWelchTest:
    """Test Welch's t-test."""
    
    def test_no_difference(self):
        """Same means should give high p-value."""
        mean1 = np.array([5.0])
        var1 = np.array([1.0])
        n1 = np.array([30.0])
        mean2 = np.array([5.0])
        var2 = np.array([1.0])
        n2 = np.array([30.0])
        out = np.empty(1, dtype=np.float64)
        
        welch_test(mean1, var1, n1, mean2, var2, n2, out)
        
        # Same means = no difference
        np.testing.assert_allclose(out[0], 1.0, rtol=1e-5)
    
    def test_large_difference(self):
        """Large mean difference should give low p-value."""
        mean1 = np.array([0.0])
        var1 = np.array([1.0])
        n1 = np.array([50.0])
        mean2 = np.array([5.0])
        var2 = np.array([1.0])
        n2 = np.array([50.0])
        out = np.empty(1, dtype=np.float64)
        
        welch_test(mean1, var1, n1, mean2, var2, n2, out)
        
        # Large difference
        assert out[0] < 0.001
    
    def test_batch(self):
        """Batch t-test computation."""
        n = 5
        mean1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        var1 = np.ones(n)
        n1 = np.full(n, 30.0)
        mean2 = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        var2 = np.ones(n)
        n2 = np.full(n, 30.0)
        out = np.empty(n, dtype=np.float64)
        
        welch_test(mean1, var1, n1, mean2, var2, n2, out)
        
        # P-values should decrease as difference increases
        for i in range(n - 1):
            assert out[i] >= out[i + 1] - 0.01  # Allow small tolerance
    
    def test_matches_scipy(self):
        """Should match scipy.stats.ttest_ind."""
        np.random.seed(42)
        
        for _ in range(5):
            # Generate random data
            x = np.random.normal(0, 1, 30)
            y = np.random.normal(0.5, 1.5, 25)
            
            mean1 = np.array([np.mean(x)])
            var1 = np.array([np.var(x, ddof=1)])
            n1 = np.array([float(len(x))])
            mean2 = np.array([np.mean(y)])
            var2 = np.array([np.var(y, ddof=1)])
            n2 = np.array([float(len(y))])
            out = np.empty(1, dtype=np.float64)
            
            welch_test(mean1, var1, n1, mean2, var2, n2, out)
            
            # Compare with scipy
            _, expected_p = scipy.stats.ttest_ind(x, y, equal_var=False)
            
            np.testing.assert_allclose(out[0], expected_p, rtol=1e-6, atol=1e-10)


class TestStudentTest:
    """Test Student's t-test."""
    
    def test_no_difference(self):
        """Same means should give high p-value."""
        mean1 = np.array([5.0])
        var1 = np.array([1.0])
        n1 = np.array([30.0])
        mean2 = np.array([5.0])
        var2 = np.array([1.0])
        n2 = np.array([30.0])
        out = np.empty(1, dtype=np.float64)
        
        student_test(mean1, var1, n1, mean2, var2, n2, out)
        
        np.testing.assert_allclose(out[0], 1.0, rtol=1e-5)
    
    def test_large_difference(self):
        """Large mean difference should give low p-value."""
        mean1 = np.array([0.0])
        var1 = np.array([1.0])
        n1 = np.array([50.0])
        mean2 = np.array([5.0])
        var2 = np.array([1.0])
        n2 = np.array([50.0])
        out = np.empty(1, dtype=np.float64)
        
        student_test(mean1, var1, n1, mean2, var2, n2, out)
        
        assert out[0] < 0.001
    
    def test_matches_scipy(self):
        """Should match scipy.stats.ttest_ind with equal_var=True."""
        np.random.seed(42)
        
        for _ in range(5):
            # Generate random data with equal variance
            x = np.random.normal(0, 1, 30)
            y = np.random.normal(0.5, 1, 25)
            
            mean1 = np.array([np.mean(x)])
            var1 = np.array([np.var(x, ddof=1)])
            n1 = np.array([float(len(x))])
            mean2 = np.array([np.mean(y)])
            var2 = np.array([np.var(y, ddof=1)])
            n2 = np.array([float(len(y))])
            out = np.empty(1, dtype=np.float64)
            
            student_test(mean1, var1, n1, mean2, var2, n2, out)
            
            # Compare with scipy
            _, expected_p = scipy.stats.ttest_ind(x, y, equal_var=True)
            
            np.testing.assert_allclose(out[0], expected_p, rtol=1e-6, atol=1e-10)


# =============================================================================
# Test Approximate Versions
# =============================================================================

@pytest.mark.skipif(not (APPROX_AVAILABLE if _OPTIM_AVAILABLE else False), 
                    reason="Approx functions not available")
class TestApproxVersions:
    """Test approximate t-test functions.
    
    Note: Approximate versions use normal approximation for speed.
    They are designed for cases where speed matters more than precision.
    We only verify they produce valid p-values and have correct ordering.
    """
    
    def test_welch_approx_valid_pvalues(self):
        """Approximate should return valid p-values in [0, 1]."""
        mean1 = np.array([0.0, 0.0, 0.0, 0.0])
        var1 = np.array([1.0, 1.0, 1.0, 1.0])
        n1 = np.array([50.0, 50.0, 50.0, 50.0])
        mean2 = np.array([0.0, 0.5, 1.0, 2.0])  # Increasing difference
        var2 = np.array([1.0, 1.0, 1.0, 1.0])
        n2 = np.array([50.0, 50.0, 50.0, 50.0])
        
        out_approx = np.empty(4, dtype=np.float64)
        welch_test_approx(mean1, var1, n1, mean2, var2, n2, out_approx)
        
        # All p-values should be valid
        assert np.all(out_approx >= 0)
        assert np.all(out_approx <= 1)
        
        # P-values should decrease as difference increases
        assert out_approx[0] > out_approx[1] > out_approx[2] > out_approx[3]
        
        # No difference should give high p-value
        np.testing.assert_allclose(out_approx[0], 1.0, rtol=1e-5)
    
    def test_student_approx_valid_pvalues(self):
        """Approximate should return valid p-values in [0, 1]."""
        mean1 = np.array([0.0, 0.0, 0.0, 0.0])
        var1 = np.array([1.0, 1.0, 1.0, 1.0])
        n1 = np.array([50.0, 50.0, 50.0, 50.0])
        mean2 = np.array([0.0, 0.5, 1.0, 2.0])  # Increasing difference
        var2 = np.array([1.0, 1.0, 1.0, 1.0])
        n2 = np.array([50.0, 50.0, 50.0, 50.0])
        
        out_approx = np.empty(4, dtype=np.float64)
        student_test_approx(mean1, var1, n1, mean2, var2, n2, out_approx)
        
        # All p-values should be valid
        assert np.all(out_approx >= 0)
        assert np.all(out_approx <= 1)
        
        # P-values should decrease as difference increases
        assert out_approx[0] > out_approx[1] > out_approx[2] > out_approx[3]
        
        # No difference should give high p-value
        np.testing.assert_allclose(out_approx[0], 1.0, rtol=1e-5)


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_very_small_variance(self):
        """Should handle very small variance."""
        mean1 = np.array([1.0])
        var1 = np.array([1e-15])
        n1 = np.array([10.0])
        mean2 = np.array([1.0])
        var2 = np.array([1e-15])
        n2 = np.array([10.0])
        out = np.empty(1, dtype=np.float64)
        
        welch_test(mean1, var1, n1, mean2, var2, n2, out)
        
        # Should return valid p-value (likely 1.0)
        assert 0 <= out[0] <= 1
    
    def test_zero_variance(self):
        """Should handle zero variance gracefully."""
        mean1 = np.array([1.0])
        var1 = np.array([0.0])
        n1 = np.array([10.0])
        mean2 = np.array([1.0])
        var2 = np.array([0.0])
        n2 = np.array([10.0])
        out = np.empty(1, dtype=np.float64)
        
        welch_test(mean1, var1, n1, mean2, var2, n2, out)
        
        # Should return 1.0 (no meaningful test possible)
        assert out[0] == 1.0
    
    def test_small_sample_size(self):
        """Should handle small sample sizes."""
        mean1 = np.array([1.0])
        var1 = np.array([1.0])
        n1 = np.array([3.0])
        mean2 = np.array([2.0])
        var2 = np.array([1.0])
        n2 = np.array([3.0])
        out = np.empty(1, dtype=np.float64)
        
        welch_test(mean1, var1, n1, mean2, var2, n2, out)
        
        # Should return valid p-value
        assert 0 <= out[0] <= 1
    
    def test_very_large_sample_size(self):
        """Should handle large sample sizes."""
        mean1 = np.array([0.0])
        var1 = np.array([1.0])
        n1 = np.array([10000.0])
        mean2 = np.array([0.1])
        var2 = np.array([1.0])
        n2 = np.array([10000.0])
        out = np.empty(1, dtype=np.float64)
        
        welch_test(mean1, var1, n1, mean2, var2, n2, out)
        
        # With large n, even small difference should be significant
        assert out[0] < 0.001


# =============================================================================
# Large Scale Tests
# =============================================================================

class TestLargeScale:
    """Test with many simultaneous computations."""
    
    def test_many_tests(self):
        """Should handle many tests efficiently."""
        n = 1000
        np.random.seed(42)
        
        mean1 = np.zeros(n)
        var1 = np.ones(n)
        n1 = np.full(n, 50.0)
        mean2 = np.random.normal(0, 1, n)  # Random differences
        var2 = np.ones(n)
        n2 = np.full(n, 50.0)
        out = np.empty(n, dtype=np.float64)
        
        welch_test(mean1, var1, n1, mean2, var2, n2, out)
        
        # All p-values should be valid
        assert np.all(out >= 0)
        assert np.all(out <= 1)
        
        # Larger differences should have smaller p-values
        large_diff = np.abs(mean2) > 1.0
        small_diff = np.abs(mean2) < 0.1
        
        if np.any(large_diff) and np.any(small_diff):
            assert np.mean(out[large_diff]) < np.mean(out[small_diff])
