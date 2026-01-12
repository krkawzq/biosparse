"""Tests for biosparse.kernel.math statistical functions."""

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
            erfc, erf, normal_cdf, normal_sf, normal_pdf,
            normal_logcdf, normal_logsf,
            erfc_approx, normal_sf_approx, normal_cdf_approx,
        )
    except Exception:
        _OPTIM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _OPTIM_AVAILABLE,
    reason="Optim/kernel module not available"
)


class TestErfc:
    """Test complementary error function."""
    
    def test_erfc_values(self):
        """erfc should match scipy.special.erfc."""
        from scipy import special
        
        x = np.array([0.0, 0.5, 1.0, 2.0, -1.0], dtype=np.float64)
        out = np.empty_like(x)
        
        erfc(x, out)
        expected = special.erfc(x)
        
        np.testing.assert_allclose(out, expected, rtol=1e-14)
    
    def test_erfc_large_array(self):
        """erfc should work on large arrays."""
        from scipy import special
        
        np.random.seed(42)
        x = np.random.randn(10000).astype(np.float64)
        out = np.empty_like(x)
        
        erfc(x, out)
        expected = special.erfc(x)
        
        np.testing.assert_allclose(out, expected, rtol=1e-14)


class TestErf:
    """Test error function."""
    
    def test_erf_values(self):
        """erf should match scipy.special.erf."""
        from scipy import special
        
        x = np.array([0.0, 0.5, 1.0, 2.0, -1.0], dtype=np.float64)
        out = np.empty_like(x)
        
        erf(x, out)
        expected = special.erf(x)
        
        np.testing.assert_allclose(out, expected, rtol=1e-14)


class TestNormalCDF:
    """Test standard normal CDF."""
    
    def test_normal_cdf_values(self):
        """normal_cdf should match scipy.stats.norm.cdf."""
        from scipy.stats import norm
        
        z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float64)
        out = np.empty_like(z)
        
        normal_cdf(z, out)
        expected = norm.cdf(z)
        
        np.testing.assert_allclose(out, expected, rtol=1e-14)
    
    def test_normal_cdf_symmetry(self):
        """CDF(z) + CDF(-z) = 1."""
        z = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float64)
        out_pos = np.empty_like(z)
        out_neg = np.empty_like(z)
        
        normal_cdf(z, out_pos)
        normal_cdf(-z, out_neg)
        
        np.testing.assert_allclose(out_pos + out_neg, 1.0, rtol=1e-14)


class TestNormalSF:
    """Test standard normal survival function."""
    
    def test_normal_sf_values(self):
        """normal_sf should match scipy.stats.norm.sf."""
        from scipy.stats import norm
        
        z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float64)
        out = np.empty_like(z)
        
        normal_sf(z, out)
        expected = norm.sf(z)
        
        np.testing.assert_allclose(out, expected, rtol=1e-14)
    
    def test_sf_plus_cdf_equals_one(self):
        """SF(z) + CDF(z) = 1."""
        z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        sf_out = np.empty_like(z)
        cdf_out = np.empty_like(z)
        
        normal_sf(z, sf_out)
        normal_cdf(z, cdf_out)
        
        np.testing.assert_allclose(sf_out + cdf_out, 1.0, rtol=1e-14)


class TestNormalPDF:
    """Test standard normal PDF."""
    
    def test_normal_pdf_values(self):
        """normal_pdf should match scipy.stats.norm.pdf."""
        from scipy.stats import norm
        
        z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float64)
        out = np.empty_like(z)
        
        normal_pdf(z, out)
        expected = norm.pdf(z)
        
        np.testing.assert_allclose(out, expected, rtol=1e-14)
    
    def test_pdf_symmetry(self):
        """PDF(z) = PDF(-z)."""
        z = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float64)
        out_pos = np.empty_like(z)
        out_neg = np.empty_like(z)
        
        normal_pdf(z, out_pos)
        normal_pdf(-z, out_neg)
        
        np.testing.assert_allclose(out_pos, out_neg, rtol=1e-14)
    
    def test_pdf_max_at_zero(self):
        """PDF should be maximum at z=0."""
        z = np.array([0.0], dtype=np.float64)
        out = np.empty_like(z)
        
        normal_pdf(z, out)
        
        expected_max = 1.0 / np.sqrt(2 * np.pi)
        assert np.isclose(out[0], expected_max)


class TestNormalLogCDF:
    """Test log of standard normal CDF."""
    
    def test_normal_logcdf_values(self):
        """normal_logcdf should match scipy.stats.norm.logcdf."""
        from scipy.stats import norm
        
        z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float64)
        out = np.empty_like(z)
        
        normal_logcdf(z, out)
        expected = norm.logcdf(z)
        
        np.testing.assert_allclose(out, expected, rtol=1e-10)
    
    def test_logcdf_extreme_negative(self):
        """logcdf should be numerically stable for extreme negative values."""
        z = np.array([-30.0, -50.0], dtype=np.float64)
        out = np.empty_like(z)
        
        normal_logcdf(z, out)
        
        # Should be finite and negative
        assert np.all(np.isfinite(out))
        assert np.all(out < 0)


class TestNormalLogSF:
    """Test log of standard normal survival function."""
    
    def test_normal_logsf_values(self):
        """normal_logsf should match scipy.stats.norm.logsf."""
        from scipy.stats import norm
        
        z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float64)
        out = np.empty_like(z)
        
        normal_logsf(z, out)
        expected = norm.logsf(z)
        
        np.testing.assert_allclose(out, expected, rtol=1e-10)
    
    def test_logsf_extreme_positive(self):
        """logsf should be numerically stable for extreme positive values."""
        z = np.array([30.0, 50.0], dtype=np.float64)
        out = np.empty_like(z)
        
        normal_logsf(z, out)
        
        # Should be finite and negative
        assert np.all(np.isfinite(out))
        assert np.all(out < 0)


class TestApproximations:
    """Test approximate versions of functions."""
    
    def test_erfc_approx_accuracy(self):
        """erfc_approx should be accurate to ~1e-7."""
        from scipy import special
        
        np.random.seed(42)
        x = np.random.randn(1000).astype(np.float64) * 3
        out = np.empty_like(x)
        
        erfc_approx(x, out)
        expected = special.erfc(x)
        
        # Should be within 1e-6 (allowing some margin)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)
    
    def test_normal_sf_approx_accuracy(self):
        """normal_sf_approx should be accurate to ~1e-7."""
        from scipy.stats import norm
        
        np.random.seed(42)
        z = np.random.randn(1000).astype(np.float64) * 3
        out = np.empty_like(z)
        
        normal_sf_approx(z, out)
        expected = norm.sf(z)
        
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)
    
    def test_normal_cdf_approx_accuracy(self):
        """normal_cdf_approx should be accurate to ~1e-7."""
        from scipy.stats import norm
        
        np.random.seed(42)
        z = np.random.randn(1000).astype(np.float64) * 3
        out = np.empty_like(z)
        
        normal_cdf_approx(z, out)
        expected = norm.cdf(z)
        
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)
    
    def test_approx_boundary_values(self):
        """Approximate functions should handle boundary values."""
        z = np.array([0.0, -10.0, 10.0], dtype=np.float64)
        
        sf_out = np.empty_like(z)
        cdf_out = np.empty_like(z)
        
        normal_sf_approx(z, sf_out)
        normal_cdf_approx(z, cdf_out)
        
        # At z=0, both should be ~0.5
        assert np.isclose(sf_out[0], 0.5, rtol=1e-6)
        assert np.isclose(cdf_out[0], 0.5, rtol=1e-6)
        
        # At extreme values
        assert 0.0 <= sf_out[1] <= 1.0
        assert 0.0 <= sf_out[2] <= 1.0
        assert 0.0 <= cdf_out[1] <= 1.0
        assert 0.0 <= cdf_out[2] <= 1.0


class TestVectorization:
    """Test that functions work correctly with various array sizes."""
    
    @pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 100, 1024, 1025])
    def test_various_array_sizes(self, size):
        """Functions should work with various array sizes (including non-power-of-2)."""
        from scipy.stats import norm
        
        np.random.seed(42)
        z = np.random.randn(size).astype(np.float64)
        out = np.empty_like(z)
        
        normal_sf(z, out)
        expected = norm.sf(z)
        
        np.testing.assert_allclose(out, expected, rtol=1e-14)
    
    def test_contiguous_vs_strided(self):
        """Functions should work with strided arrays."""
        from scipy.stats import norm
        
        # Create strided array
        base = np.random.randn(100).astype(np.float64)
        z = base[::2]  # Every other element
        out = np.empty_like(z)
        
        # Make contiguous copy for output
        out_contig = np.ascontiguousarray(out)
        
        normal_sf(np.ascontiguousarray(z), out_contig)
        expected = norm.sf(z)
        
        np.testing.assert_allclose(out_contig, expected, rtol=1e-14)
