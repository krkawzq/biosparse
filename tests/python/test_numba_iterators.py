"""Tests for CSR/CSC Numba iterator support.

Tests _numba/_iterators.py: for values, indices in csr/csc
"""

import pytest
import numpy as np

# Check if numba extension is available
_NUMBA_EXT_AVAILABLE = False
_BINDING_AVAILABLE = False

try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

try:
    from _binding import lib
    _BINDING_AVAILABLE = lib is not None
except Exception:
    _BINDING_AVAILABLE = False

_NUMBA_EXT_AVAILABLE = _NUMBA_AVAILABLE and _BINDING_AVAILABLE
if _NUMBA_EXT_AVAILABLE:
    try:
        import _numba
    except Exception:
        _NUMBA_EXT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _NUMBA_EXT_AVAILABLE,
    reason="Numba extension not available (requires Rust FFI + Numba)"
)


@pytest.fixture
def csr_f64():
    """Create a small CSR matrix for testing."""
    pytest.importorskip("scipy")
    if not _BINDING_AVAILABLE:
        pytest.skip("Rust FFI bindings not available")
    
    import scipy.sparse as sp
    from _binding import CSRF64
    
    np.random.seed(42)
    mat = sp.random(20, 15, density=0.2, format='csr', dtype=np.float64)
    return CSRF64.from_scipy(mat), mat


@pytest.fixture
def csc_f64():
    """Create a small CSC matrix for testing."""
    pytest.importorskip("scipy")
    if not _BINDING_AVAILABLE:
        pytest.skip("Rust FFI bindings not available")
    
    import scipy.sparse as sp
    from _binding import CSCF64
    
    np.random.seed(42)
    mat = sp.random(20, 15, density=0.2, format='csc', dtype=np.float64)
    return CSCF64.from_scipy(mat), mat


class TestCSRIterator:
    """Test CSR iteration in JIT."""
    
    def test_iterate_rows(self, csr_f64):
        """Should iterate over all rows of CSR matrix."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def sum_all_values(csr):
            total = 0.0
            for values, indices in csr:
                for v in values:
                    total += v
            return total
        
        result = sum_all_values(csr)
        expected = scipy_mat.data.sum()
        assert np.isclose(result, expected)
    
    def test_row_count(self, csr_f64):
        """Should iterate over correct number of rows."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def count_rows(csr):
            count = 0
            for values, indices in csr:
                count += 1
            return count
        
        result = count_rows(csr)
        assert result == scipy_mat.shape[0]
    
    def test_access_indices(self, csr_f64):
        """Should be able to access column indices in iteration."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def sum_col_indices(csr):
            total = 0
            for values, indices in csr:
                for idx in indices:
                    total += idx
            return total
        
        result = sum_col_indices(csr)
        expected = scipy_mat.indices.sum()
        assert result == expected
    
    def test_row_dot_product(self, csr_f64):
        """Should be able to compute row-wise operations."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        vec = np.random.rand(scipy_mat.shape[1])
        
        @njit
        def spmv(csr, vec):
            nrows = len(csr)
            result = np.zeros(nrows)
            row_idx = 0
            for values, indices in csr:
                dot = 0.0
                for i in range(len(values)):
                    dot += values[i] * vec[indices[i]]
                result[row_idx] = dot
                row_idx += 1
            return result
        
        result = spmv(csr, vec)
        expected = scipy_mat.dot(vec)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestCSCIterator:
    """Test CSC iteration in JIT."""
    
    def test_iterate_columns(self, csc_f64):
        """Should iterate over all columns of CSC matrix."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def sum_all_values(csc):
            total = 0.0
            for values, indices in csc:
                for v in values:
                    total += v
            return total
        
        result = sum_all_values(csc)
        expected = scipy_mat.data.sum()
        assert np.isclose(result, expected)
    
    def test_col_count(self, csc_f64):
        """Should iterate over correct number of columns."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def count_cols(csc):
            count = 0
            for values, indices in csc:
                count += 1
            return count
        
        result = count_cols(csc)
        assert result == scipy_mat.shape[1]
    
    def test_access_row_indices(self, csc_f64):
        """Should be able to access row indices in iteration."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def sum_row_indices(csc):
            total = 0
            for values, indices in csc:
                for idx in indices:
                    total += idx
            return total
        
        result = sum_row_indices(csc)
        expected = scipy_mat.indices.sum()
        assert result == expected
    
    def test_column_sum(self, csc_f64):
        """Should be able to compute column sums."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def col_sums(csc):
            ncols = len(csc)
            result = np.zeros(ncols)
            col_idx = 0
            for values, indices in csc:
                for v in values:
                    result[col_idx] += v
                col_idx += 1
            return result
        
        result = col_sums(csc)
        expected = np.asarray(scipy_mat.sum(axis=0)).ravel()
        np.testing.assert_allclose(result, expected, rtol=1e-10)
