"""Tests for biosparse.kernel.math._tdist module.

Tests for Student's t-distribution functions:
    - stdtr, stdtr_sf: t-distribution CDF and survival function
    - t_test_pvalue, t_test_pvalue_batch: p-value computation
    - t_cdf_two_sided: two-sided CDF
    - betainc: regularized incomplete beta function
"""

import pytest
import numpy as np

# Check dependencies
try:
    import scipy.stats
    import scipy.special
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
            stdtr, stdtr_sf, t_test_pvalue, t_test_pvalue_batch,
            t_cdf_two_sided, betainc,
        )
    except Exception:
        _OPTIM_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not _OPTIM_AVAILABLE, reason="Optim/kernel module not available"),
    pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available"),
]


# =============================================================================
# Test Incomplete Beta Function
# =============================================================================

class TestBetainc:
    """Test regularized incomplete beta function."""
    
    def test_betainc_edge_cases(self):
        """Test edge cases for betainc."""
        # betainc(a, b, 0) = 0
        assert betainc(1.0, 1.0, 0.0) == 0.0
        
        # betainc(a, b, 1) = 1
        assert betainc(1.0, 1.0, 1.0) == 1.0
    
    def test_betainc_symmetric(self):
        """Test symmetric case: betainc(1, 1, x) = x."""
        for x in [0.1, 0.25, 0.5, 0.75, 0.9]:
            result = betainc(1.0, 1.0, x)
            np.testing.assert_allclose(result, x, rtol=1e-10)
    
    def test_betainc_matches_scipy(self):
        """betainc should match scipy.special.betainc."""
        test_cases = [
            (0.5, 0.5, 0.5),
            (1.0, 2.0, 0.3),
            (2.0, 1.0, 0.7),
            (5.0, 5.0, 0.5),
            (10.0, 5.0, 0.3),
        ]
        
        for a, b, x in test_cases:
            result = betainc(a, b, x)
            expected = scipy.special.betainc(a, b, x)
            np.testing.assert_allclose(result, expected, rtol=1e-6,
                err_msg=f"betainc({a}, {b}, {x}) failed")


# =============================================================================
# Test Student's t-distribution CDF
# =============================================================================

class TestStdtr:
    """Test Student's t-distribution CDF."""
    
    def test_stdtr_at_zero(self):
        """stdtr(df, 0) should be 0.5."""
        for df in [1.0, 5.0, 10.0, 30.0, 100.0]:
            result = stdtr(df, 0.0)
            np.testing.assert_allclose(result, 0.5, rtol=1e-10)
    
    def test_stdtr_symmetry(self):
        """stdtr(df, t) + stdtr(df, -t) = 1."""
        for df in [5.0, 10.0, 30.0]:
            for t in [0.5, 1.0, 2.0]:
                cdf_pos = stdtr(df, t)
                cdf_neg = stdtr(df, -t)
                np.testing.assert_allclose(cdf_pos + cdf_neg, 1.0, rtol=1e-10)
    
    def test_stdtr_increasing(self):
        """stdtr should be monotonically increasing in t."""
        df = 10.0
        t_values = np.linspace(-3, 3, 20)
        cdf_values = [stdtr(df, t) for t in t_values]
        
        for i in range(len(cdf_values) - 1):
            assert cdf_values[i] <= cdf_values[i + 1] + 1e-10
    
    def test_stdtr_matches_scipy(self):
        """stdtr should match scipy.stats.t.cdf."""
        test_cases = [
            (5.0, 0.0),
            (5.0, 1.0),
            (5.0, -1.0),
            (10.0, 2.0),
            (30.0, 1.5),
            (100.0, -2.5),
        ]
        
        for df, t in test_cases:
            result = stdtr(df, t)
            expected = scipy.stats.t.cdf(t, df)
            np.testing.assert_allclose(result, expected, rtol=1e-4,
                err_msg=f"stdtr({df}, {t}) failed")


class TestStdtrSF:
    """Test Student's t-distribution survival function."""
    
    def test_stdtr_sf_at_zero(self):
        """stdtr_sf(df, 0) should be 0.5."""
        for df in [5.0, 10.0, 30.0]:
            result = stdtr_sf(df, 0.0)
            np.testing.assert_allclose(result, 0.5, rtol=1e-10)
    
    def test_stdtr_sf_complement(self):
        """stdtr(df, t) + stdtr_sf(df, t) = 1."""
        for df in [5.0, 10.0, 30.0]:
            for t in [-2.0, 0.0, 2.0]:
                cdf = stdtr(df, t)
                sf = stdtr_sf(df, t)
                np.testing.assert_allclose(cdf + sf, 1.0, rtol=1e-10)


# =============================================================================
# Test P-value Computation
# =============================================================================

class TestTTestPvalue:
    """Test t-test p-value computation."""
    
    def test_pvalue_two_sided_at_zero(self):
        """Two-sided p-value at t=0 should be 1."""
        for df in [5.0, 10.0, 30.0]:
            result = t_test_pvalue(0.0, df, alternative=0)
            np.testing.assert_allclose(result, 1.0, rtol=1e-10)
    
    def test_pvalue_two_sided_extreme(self):
        """Large |t| should give small p-value."""
        result = t_test_pvalue(10.0, 100.0, alternative=0)
        assert result < 0.01
    
    def test_pvalue_one_sided_less(self):
        """One-sided less: P(T < t)."""
        df = 10.0
        
        # Negative t should give small p-value for "less"
        result_neg = t_test_pvalue(-3.0, df, alternative=-1)
        assert result_neg < 0.01
        
        # Positive t should give large p-value for "less"
        result_pos = t_test_pvalue(3.0, df, alternative=-1)
        assert result_pos > 0.9
    
    def test_pvalue_one_sided_greater(self):
        """One-sided greater: P(T > t)."""
        df = 10.0
        
        # Positive t should give small p-value for "greater"
        result_pos = t_test_pvalue(3.0, df, alternative=1)
        assert result_pos < 0.01
        
        # Negative t should give large p-value for "greater"
        result_neg = t_test_pvalue(-3.0, df, alternative=1)
        assert result_neg > 0.9
    
    def test_pvalue_matches_scipy(self):
        """P-values should match scipy.stats.t.sf * 2."""
        test_cases = [
            (0.0, 10.0),
            (1.0, 10.0),
            (2.0, 30.0),
            (-1.5, 50.0),
        ]
        
        for t_stat, df in test_cases:
            result = t_test_pvalue(t_stat, df, alternative=0)
            expected = 2 * scipy.stats.t.sf(abs(t_stat), df)
            np.testing.assert_allclose(result, expected, rtol=1e-3,
                err_msg=f"t_test_pvalue({t_stat}, {df}) failed")


class TestTTestPvalueBatch:
    """Test batch p-value computation."""
    
    def test_batch_matches_single(self):
        """Batch should match individual calls."""
        t_stats = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
        dfs = np.array([10.0, 10.0, 10.0, 30.0, 30.0])
        
        batch_result = t_test_pvalue_batch(t_stats, dfs, alternative=0)
        
        for i in range(len(t_stats)):
            single_result = t_test_pvalue(t_stats[i], dfs[i], alternative=0)
            np.testing.assert_allclose(batch_result[i], single_result, rtol=1e-10)
    
    def test_batch_large(self):
        """Batch with many values."""
        n = 1000
        np.random.seed(42)
        t_stats = np.random.randn(n) * 2
        dfs = np.full(n, 30.0)
        
        batch_result = t_test_pvalue_batch(t_stats, dfs, alternative=0)
        
        # All p-values should be in [0, 1]
        assert np.all(batch_result >= 0)
        assert np.all(batch_result <= 1)


class TestTCdfTwoSided:
    """Test two-sided t CDF (p-value)."""
    
    def test_two_sided_at_zero(self):
        """t_cdf_two_sided(0, df) = 1."""
        for df in [5.0, 10.0, 30.0]:
            result = t_cdf_two_sided(0.0, df)
            np.testing.assert_allclose(result, 1.0, rtol=1e-10)
    
    def test_two_sided_symmetric(self):
        """t_cdf_two_sided(t, df) = t_cdf_two_sided(-t, df)."""
        for df in [10.0, 30.0]:
            for t in [0.5, 1.0, 2.0]:
                result_pos = t_cdf_two_sided(t, df)
                result_neg = t_cdf_two_sided(-t, df)
                np.testing.assert_allclose(result_pos, result_neg, rtol=1e-10)
    
    def test_two_sided_equals_t_test_pvalue(self):
        """t_cdf_two_sided should equal t_test_pvalue with alternative=0."""
        for df in [10.0, 30.0, 100.0]:
            for t in [0.0, 1.0, -2.0, 3.0]:
                result1 = t_cdf_two_sided(t, df)
                result2 = t_test_pvalue(t, df, alternative=0)
                np.testing.assert_allclose(result1, result2, rtol=1e-10)


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_small_df(self):
        """Functions should work with small degrees of freedom."""
        result = stdtr(1.0, 0.0)  # Cauchy distribution
        np.testing.assert_allclose(result, 0.5, rtol=1e-6)
    
    def test_large_df(self):
        """Large df should approximate normal distribution."""
        df = 1000.0
        t = 1.96
        
        result = stdtr(df, t)
        # Should be close to normal CDF at 1.96 ≈ 0.975
        np.testing.assert_allclose(result, 0.975, rtol=0.01)
    
    def test_large_t(self):
        """Large |t| should give CDF close to 0 or 1."""
        df = 10.0
        
        result_pos = stdtr(df, 10.0)
        assert result_pos > 0.9999
        
        result_neg = stdtr(df, -10.0)
        assert result_neg < 0.0001
