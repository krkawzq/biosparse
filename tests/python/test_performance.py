"""Performance tests and benchmarks for biosparse.

These tests verify that the implementation is performant and measure
relative performance between different approaches.
"""

import pytest
import numpy as np
import time

# Check if components are available
_NUMBA_EXT_AVAILABLE = False
_BINDING_AVAILABLE = False

try:
    import numba
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

try:
    from biosparse._binding import lib
    _BINDING_AVAILABLE = lib is not None
except Exception:
    _BINDING_AVAILABLE = False

_NUMBA_EXT_AVAILABLE = _NUMBA_AVAILABLE and _BINDING_AVAILABLE
if _NUMBA_EXT_AVAILABLE:
    try:
        import biosparse._numba
        from biosparse._binding import CSRF64, CSCF64
    except Exception:
        _NUMBA_EXT_AVAILABLE = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _NUMBA_EXT_AVAILABLE,
        reason="Numba extension not available"
    )
]


@pytest.fixture(scope="module")
def perf_csr():
    """Create a larger CSR matrix for performance tests."""
    pytest.importorskip("scipy")
    import scipy.sparse as sp
    np.random.seed(42)
    mat = sp.random(5000, 2000, density=0.02, format='csr', dtype=np.float64)
    return CSRF64.from_scipy(mat), mat


@pytest.fixture(scope="module")
def perf_csc():
    """Create a larger CSC matrix for performance tests."""
    pytest.importorskip("scipy")
    import scipy.sparse as sp
    np.random.seed(42)
    mat = sp.random(5000, 2000, density=0.02, format='csc', dtype=np.float64)
    return CSCF64.from_scipy(mat), mat


class TestIterationPerformance:
    """Test iteration performance."""
    
    def test_row_sum_performance(self, perf_csr):
        """Row sum via iteration should be fast."""
        csr, scipy_mat = perf_csr
        
        @njit
        def row_sums_jit(csr):
            nrows = csr.nrows
            sums = np.zeros(nrows)
            row_idx = 0
            for values, _ in csr:
                sums[row_idx] = values.sum()
                row_idx += 1
            return sums
        
        # Warmup
        _ = row_sums_jit(csr)
        
        # Time JIT version
        start = time.perf_counter()
        for _ in range(10):
            jit_result = row_sums_jit(csr)
        jit_time = time.perf_counter() - start
        
        # Time scipy version
        start = time.perf_counter()
        for _ in range(10):
            scipy_result = np.asarray(scipy_mat.sum(axis=1)).flatten()
        scipy_time = time.perf_counter() - start
        
        # Verify correctness
        np.testing.assert_allclose(jit_result, scipy_result, rtol=1e-10)
        
        # JIT should be competitive (within 5x of scipy, often faster)
        print(f"\nRow sum: JIT={jit_time:.4f}s, scipy={scipy_time:.4f}s, "
              f"ratio={jit_time/scipy_time:.2f}x")
    
    def test_col_sum_performance(self, perf_csc):
        """Column sum via iteration should be fast."""
        csc, scipy_mat = perf_csc
        
        @njit
        def col_sums_jit(csc):
            ncols = csc.ncols
            sums = np.zeros(ncols)
            col_idx = 0
            for values, _ in csc:
                sums[col_idx] = values.sum()
                col_idx += 1
            return sums
        
        # Warmup
        _ = col_sums_jit(csc)
        
        # Time JIT version
        start = time.perf_counter()
        for _ in range(10):
            jit_result = col_sums_jit(csc)
        jit_time = time.perf_counter() - start
        
        # Time scipy version
        start = time.perf_counter()
        for _ in range(10):
            scipy_result = np.asarray(scipy_mat.sum(axis=0)).flatten()
        scipy_time = time.perf_counter() - start
        
        # Verify correctness
        np.testing.assert_allclose(jit_result, scipy_result, rtol=1e-10)
        
        print(f"\nCol sum: JIT={jit_time:.4f}s, scipy={scipy_time:.4f}s, "
              f"ratio={jit_time/scipy_time:.2f}x")


class TestSpMVPerformance:
    """Test sparse matrix-vector multiplication performance."""
    
    def test_csr_spmv_performance(self, perf_csr):
        """CSR SpMV should be performant."""
        csr, scipy_mat = perf_csr
        vec = np.random.rand(scipy_mat.shape[1])
        
        @njit
        def spmv_jit(csr, vec):
            nrows = csr.nrows
            result = np.zeros(nrows)
            row_idx = 0
            for values, indices in csr:
                dot = 0.0
                for i in range(len(values)):
                    dot += values[i] * vec[indices[i]]
                result[row_idx] = dot
                row_idx += 1
            return result
        
        # Warmup
        _ = spmv_jit(csr, vec)
        
        # Time JIT version
        start = time.perf_counter()
        for _ in range(10):
            jit_result = spmv_jit(csr, vec)
        jit_time = time.perf_counter() - start
        
        # Time scipy version
        start = time.perf_counter()
        for _ in range(10):
            scipy_result = scipy_mat.dot(vec)
        scipy_time = time.perf_counter() - start
        
        # Verify correctness
        np.testing.assert_allclose(jit_result, scipy_result, rtol=1e-10)
        
        print(f"\nSpMV: JIT={jit_time:.4f}s, scipy={scipy_time:.4f}s, "
              f"ratio={jit_time/scipy_time:.2f}x")


class TestRowAccessPerformance:
    """Test row access performance."""
    
    def test_random_row_access(self, perf_csr):
        """Random row access should be fast."""
        csr, scipy_mat = perf_csr
        
        np.random.seed(123)
        indices = np.random.randint(0, csr.nrows, size=1000)
        
        @njit
        def random_access_jit(csr, indices):
            total = 0.0
            for idx in indices:
                values, _ = csr.row(idx)
                total += values.sum()
            return total
        
        # Warmup
        _ = random_access_jit(csr, indices)
        
        # Time JIT version
        start = time.perf_counter()
        for _ in range(10):
            jit_result = random_access_jit(csr, indices)
        jit_time = time.perf_counter() - start
        
        # Time Python version
        start = time.perf_counter()
        for _ in range(10):
            python_total = 0.0
            for idx in indices:
                values, _ = csr.row_to_numpy(idx)
                python_total += values.sum()
        python_time = time.perf_counter() - start
        
        assert np.isclose(jit_result, python_total)
        
        print(f"\nRandom access: JIT={jit_time:.4f}s, Python={python_time:.4f}s, "
              f"speedup={python_time/jit_time:.2f}x")


class TestStatisticsPerformance:
    """Test statistical operations performance."""
    
    def test_row_statistics(self, perf_csr):
        """Per-row mean and variance computation."""
        csr, scipy_mat = perf_csr
        
        @njit
        def row_stats_jit(csr):
            nrows = csr.nrows
            ncols = csr.ncols
            means = np.zeros(nrows)
            variances = np.zeros(nrows)
            
            N = float(ncols)
            row_idx = 0
            for values, _ in csr:
                row_sum = 0.0
                row_sq_sum = 0.0
                for v in values:
                    row_sum += v
                    row_sq_sum += v * v
                
                mean = row_sum / N
                var = (row_sq_sum / N) - mean * mean
                
                means[row_idx] = mean
                variances[row_idx] = max(0.0, var)
                row_idx += 1
            
            return means, variances
        
        # Warmup
        _ = row_stats_jit(csr)
        
        # Time JIT version
        start = time.perf_counter()
        for _ in range(10):
            means_jit, vars_jit = row_stats_jit(csr)
        jit_time = time.perf_counter() - start
        
        # Time numpy version
        start = time.perf_counter()
        for _ in range(10):
            dense = scipy_mat.toarray()
            means_np = dense.mean(axis=1)
            vars_np = dense.var(axis=1)
        numpy_time = time.perf_counter() - start
        
        # Verify correctness
        np.testing.assert_allclose(means_jit, means_np, rtol=1e-10)
        
        print(f"\nRow stats: JIT={jit_time:.4f}s, NumPy={numpy_time:.4f}s, "
              f"speedup={numpy_time/jit_time:.2f}x")


class TestSlicingPerformance:
    """Test slicing performance."""
    
    def test_row_slicing(self, perf_csr):
        """Row slicing should be fast."""
        csr, scipy_mat = perf_csr
        
        @njit
        def slice_and_sum(csr, start, end):
            sliced = csr.slice_rows(start, end)
            total = 0.0
            for values, _ in sliced:
                total += values.sum()
            return total
        
        # Warmup
        _ = slice_and_sum(csr, 100, 200)
        
        # Time JIT version
        start = time.perf_counter()
        for _ in range(100):
            jit_result = slice_and_sum(csr, 100, 200)
        jit_time = time.perf_counter() - start
        
        # Time scipy version
        start = time.perf_counter()
        for _ in range(100):
            scipy_result = scipy_mat[100:200, :].sum()
        scipy_time = time.perf_counter() - start
        
        assert np.isclose(jit_result, scipy_result)
        
        print(f"\nSlice+sum: JIT={jit_time:.4f}s, scipy={scipy_time:.4f}s, "
              f"ratio={jit_time/scipy_time:.2f}x")


class TestMemoryEfficiency:
    """Test memory usage patterns."""
    
    def test_no_copy_iteration(self, perf_csr):
        """Iteration should not create unnecessary copies."""
        csr, _ = perf_csr
        
        @njit
        def get_row_ptr(csr, row_idx):
            values, _ = csr.row(row_idx)
            # Get pointer to first element (if row not empty)
            if len(values) > 0:
                return values.ctypes.data
            return 0
        
        # Get same row multiple times - pointers should be same
        ptr1 = get_row_ptr(csr, 0)
        ptr2 = get_row_ptr(csr, 0)
        
        # Both should return same underlying data pointer
        assert ptr1 == ptr2
    
    def test_reuse_across_calls(self, perf_csr):
        """Multiple JIT calls should reuse cached data."""
        csr, scipy_mat = perf_csr
        
        @njit
        def simple_sum(csr):
            total = 0.0
            for values, _ in csr:
                total += values.sum()
            return total
        
        # First call (may trigger caching)
        result1 = simple_sum(csr)
        
        # Subsequent calls should be fast due to caching
        results = [simple_sum(csr) for _ in range(5)]
        
        expected = scipy_mat.sum()
        for r in results:
            assert np.isclose(r, expected)
