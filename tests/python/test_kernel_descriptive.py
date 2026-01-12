"""Tests for biosparse.kernel.math._descriptive module.

Tests for descriptive statistics functions:
    - median, mad, quantile, percentile, quantiles_batch
    - argsort_full, argpartition
    - assign_bins_equal_width, assign_bins_by_quantiles, compute_bin_edges_quantile
    - group_mean_std, group_median_mad
    - zscore_by_bin, zscore_by_bin_mad
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
            median, mad, quantile, percentile, quantiles_batch,
            argsort_full, argpartition,
            assign_bins_equal_width, assign_bins_by_quantiles, compute_bin_edges_quantile,
            group_mean_std, group_median_mad,
            zscore_by_bin, zscore_by_bin_mad,
        )
    except Exception:
        _OPTIM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _OPTIM_AVAILABLE,
    reason="Optim/kernel module not available"
)


# =============================================================================
# Test Core Statistics
# =============================================================================

class TestMedian:
    """Test median function."""
    
    def test_median_odd_length(self):
        """Median of odd-length array."""
        arr = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        assert median(arr) == 3.0
    
    def test_median_even_length(self):
        """Median of even-length array."""
        arr = np.array([1.0, 5.0, 3.0, 7.0])
        assert median(arr) == 4.0  # (3 + 5) / 2
    
    def test_median_single_element(self):
        """Median of single element."""
        arr = np.array([42.0])
        assert median(arr) == 42.0
    
    def test_median_sorted(self):
        """Median of already sorted array."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert median(arr) == 3.0
    
    def test_median_matches_numpy(self):
        """Median should match numpy.median."""
        np.random.seed(42)
        for _ in range(5):
            arr = np.random.randn(100).astype(np.float64)
            np.testing.assert_allclose(median(arr), np.median(arr), rtol=1e-10)


class TestMAD:
    """Test Median Absolute Deviation."""
    
    def test_mad_basic(self):
        """Basic MAD computation."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # median = 3, deviations = [2, 1, 0, 1, 2], median of devs = 1
        assert mad(arr) == 1.0
    
    def test_mad_with_center(self):
        """MAD with provided center."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Using center=2: deviations = [1, 0, 1, 2, 3], median = 1
        result = mad(arr, center=2.0)
        assert result == 1.0
    
    def test_mad_constant_array(self):
        """MAD of constant array should be 0."""
        arr = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        assert mad(arr) == 0.0


class TestQuantile:
    """Test quantile function."""
    
    def test_quantile_extremes(self):
        """Quantile at 0 and 1."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert quantile(arr, 0.0) == 1.0
        assert quantile(arr, 1.0) == 5.0
    
    def test_quantile_median(self):
        """Quantile at 0.5 should be median."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert quantile(arr, 0.5) == 3.0
    
    def test_quantile_interpolation(self):
        """Quantile with linear interpolation."""
        arr = np.array([0.0, 10.0])
        assert quantile(arr, 0.5) == 5.0
        assert quantile(arr, 0.25) == 2.5
        assert quantile(arr, 0.75) == 7.5
    
    def test_quantile_matches_numpy(self):
        """Quantile should be close to numpy.quantile."""
        np.random.seed(42)
        arr = np.random.randn(100).astype(np.float64)
        
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            result = quantile(arr, q)
            expected = np.quantile(arr, q)
            np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestPercentile:
    """Test percentile function."""
    
    def test_percentile_extremes(self):
        """Percentile at 0 and 100."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert percentile(arr, 0.0) == 1.0
        assert percentile(arr, 100.0) == 5.0
    
    def test_percentile_median(self):
        """Percentile at 50 should be median."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert percentile(arr, 50.0) == 3.0


class TestQuantilesBatch:
    """Test batch quantile computation."""
    
    def test_quantiles_batch(self):
        """Batch quantiles should match individual calls."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        qs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        
        result = quantiles_batch(arr, qs)
        
        for i, q in enumerate(qs):
            expected = quantile(arr, q)
            np.testing.assert_allclose(result[i], expected, rtol=1e-10)


# =============================================================================
# Test Sorting Utilities
# =============================================================================

class TestArgsort:
    """Test argsort_full function."""
    
    def test_argsort_basic(self):
        """Basic argsort."""
        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
        idx = argsort_full(arr)
        
        # Check that sorted array is actually sorted
        sorted_arr = arr[idx]
        assert np.all(sorted_arr[:-1] <= sorted_arr[1:])
    
    def test_argsort_matches_numpy(self):
        """Argsort should produce same ordering as numpy."""
        np.random.seed(42)
        arr = np.random.randn(100).astype(np.float64)
        
        our_idx = argsort_full(arr)
        np_idx = np.argsort(arr)
        
        # The sorted arrays should be the same (indices might differ for ties)
        np.testing.assert_array_equal(arr[our_idx], arr[np_idx])
    
    def test_argsort_single_element(self):
        """Argsort of single element."""
        arr = np.array([42.0])
        idx = argsort_full(arr)
        assert idx[0] == 0


class TestArgpartition:
    """Test argpartition function."""
    
    def test_argpartition_basic(self):
        """Basic argpartition."""
        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
        k = 3
        idx = argpartition(arr, k)
        
        # Elements at indices 0..k-1 should all be <= element at index k
        pivot = arr[idx[k]]
        for i in range(k):
            assert arr[idx[i]] <= pivot
    
    def test_argpartition_finds_k_smallest(self):
        """Argpartition first k should contain k smallest."""
        arr = np.array([5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0])
        k = 3
        idx = argpartition(arr, k)
        
        # The k smallest values are 1, 2, 3
        smallest_k = set(arr[idx[:k+1]])  # First k+1 includes pivot
        true_smallest = set(sorted(arr)[:k+1])
        
        # Should have significant overlap
        assert len(smallest_k.intersection(true_smallest)) >= k


# =============================================================================
# Test Binning Utilities
# =============================================================================

class TestBinning:
    """Test binning functions."""
    
    def test_assign_bins_equal_width(self):
        """Equal-width binning."""
        values = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        n_bins = 4
        
        idx, edges = assign_bins_equal_width(values, n_bins)
        
        # Check bin indices are valid
        assert np.all(idx >= 0)
        assert np.all(idx < n_bins)
        
        # Check edges
        assert len(edges) == n_bins + 1
        assert edges[0] == 0.0
        assert edges[-1] == 10.0
    
    def test_assign_bins_constant_values(self):
        """Equal-width binning with constant values."""
        values = np.array([5.0, 5.0, 5.0, 5.0])
        n_bins = 4
        
        idx, edges = assign_bins_equal_width(values, n_bins)
        
        # All should be in same bin
        assert np.all(idx == idx[0])
    
    def test_compute_bin_edges_quantile(self):
        """Quantile-based bin edges."""
        values = np.arange(100, dtype=np.float64)
        percentiles = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        
        edges = compute_bin_edges_quantile(values, percentiles)
        
        assert len(edges) == 5
        assert edges[0] == 0.0
        assert edges[-1] == 99.0
    
    def test_assign_bins_by_quantiles(self):
        """Quantile-based binning."""
        values = np.arange(100, dtype=np.float64)
        percentiles = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        
        idx, edges = assign_bins_by_quantiles(values, percentiles)
        
        # Check bin indices are valid
        assert np.all(idx >= 0)
        assert np.all(idx < 4)  # 4 bins from 5 edges


# =============================================================================
# Test Group Statistics
# =============================================================================

class TestGroupStatistics:
    """Test per-group statistics."""
    
    def test_group_mean_std(self):
        """Group mean and std."""
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        bin_indices = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        n_bins = 2
        
        means, stds, counts = group_mean_std(values, bin_indices, n_bins)
        
        # Bin 0: [1, 2, 3], mean=2, std=1
        np.testing.assert_allclose(means[0], 2.0)
        np.testing.assert_allclose(stds[0], 1.0, rtol=1e-10)
        assert counts[0] == 3
        
        # Bin 1: [10, 20, 30], mean=20, std=10
        np.testing.assert_allclose(means[1], 20.0)
        np.testing.assert_allclose(stds[1], 10.0, rtol=1e-10)
        assert counts[1] == 3
    
    def test_group_median_mad(self):
        """Group median and MAD."""
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        bin_indices = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        n_bins = 2
        
        medians, mads, counts = group_median_mad(values, bin_indices, n_bins)
        
        # Bin 0: [1, 2, 3], median=2
        np.testing.assert_allclose(medians[0], 2.0)
        assert counts[0] == 3
        
        # Bin 1: [10, 20, 30], median=20
        np.testing.assert_allclose(medians[1], 20.0)
        assert counts[1] == 3
    
    def test_group_stats_empty_bins(self):
        """Group stats with empty bins."""
        values = np.array([1.0, 2.0, 3.0])
        bin_indices = np.array([0, 0, 0], dtype=np.int64)
        n_bins = 3  # Bins 1 and 2 are empty
        
        means, stds, counts = group_mean_std(values, bin_indices, n_bins)
        
        assert counts[0] == 3
        assert counts[1] == 0
        assert counts[2] == 0


# =============================================================================
# Test Z-score Normalization
# =============================================================================

class TestZScore:
    """Test z-score normalization."""
    
    def test_zscore_by_bin(self):
        """Z-score using per-bin mean/std."""
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        bin_indices = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        bin_means = np.array([2.0, 20.0])
        bin_stds = np.array([1.0, 10.0])
        out = np.empty(6, dtype=np.float64)
        
        zscore_by_bin(values, bin_indices, bin_means, bin_stds, out)
        
        # Bin 0: (1-2)/1 = -1, (2-2)/1 = 0, (3-2)/1 = 1
        np.testing.assert_allclose(out[0], -1.0)
        np.testing.assert_allclose(out[1], 0.0)
        np.testing.assert_allclose(out[2], 1.0)
        
        # Bin 1: (10-20)/10 = -1, (20-20)/10 = 0, (30-20)/10 = 1
        np.testing.assert_allclose(out[3], -1.0)
        np.testing.assert_allclose(out[4], 0.0)
        np.testing.assert_allclose(out[5], 1.0)
    
    def test_zscore_by_bin_mad(self):
        """Z-score using per-bin median/MAD."""
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        bin_indices = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        bin_medians = np.array([2.0, 20.0])
        bin_mads = np.array([1.0, 10.0])
        out = np.empty(6, dtype=np.float64)
        
        zscore_by_bin_mad(values, bin_indices, bin_medians, bin_mads, out)
        
        # Similar to zscore_by_bin but with median/MAD
        np.testing.assert_allclose(out[0], -1.0)
        np.testing.assert_allclose(out[1], 0.0)
        np.testing.assert_allclose(out[2], 1.0)
    
    def test_zscore_zero_std(self):
        """Z-score with zero std should give zero."""
        values = np.array([5.0, 5.0, 5.0])
        bin_indices = np.array([0, 0, 0], dtype=np.int64)
        bin_means = np.array([5.0])
        bin_stds = np.array([0.0])  # Zero std
        out = np.empty(3, dtype=np.float64)
        
        zscore_by_bin(values, bin_indices, bin_means, bin_stds, out)
        
        # All should be zero (no meaningful z-score)
        np.testing.assert_array_equal(out, 0.0)


# =============================================================================
# Test Large Arrays (Performance)
# =============================================================================

class TestLargeArrays:
    """Test with larger arrays for correctness."""
    
    def test_median_large(self):
        """Median of large array."""
        np.random.seed(42)
        arr = np.random.randn(10000).astype(np.float64)
        
        result = median(arr)
        expected = np.median(arr)
        
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_argsort_large(self):
        """Argsort of large array."""
        np.random.seed(42)
        arr = np.random.randn(1000).astype(np.float64)
        
        idx = argsort_full(arr)
        sorted_arr = arr[idx]
        
        # Check sorted
        assert np.all(sorted_arr[:-1] <= sorted_arr[1:])
    
    def test_binning_large(self):
        """Binning of large array."""
        np.random.seed(42)
        values = np.random.randn(10000).astype(np.float64)
        n_bins = 20
        
        idx, edges = assign_bins_equal_width(values, n_bins)
        
        # All indices should be valid
        assert np.all(idx >= 0)
        assert np.all(idx < n_bins)
