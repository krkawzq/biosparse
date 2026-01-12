"""Tests for CSR/CSC Numba row/column access methods.

Tests _numba/_overloads.py: row_len, row_to_numpy, row, col, get
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
    mat = sp.random(100, 50, density=0.1, format='csr', dtype=np.float64)
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
    mat = sp.random(100, 50, density=0.1, format='csc', dtype=np.float64)
    return CSCF64.from_scipy(mat), mat


class TestCSRRowAccess:
    """Test CSR row access methods in JIT."""
    
    def test_row_len(self, csr_f64):
        """csr.row_len(i) should return number of non-zeros in row i."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def get_row_len(csr, row_idx):
            return csr.row_len(row_idx)
        
        # Test several rows
        for row in [0, 10, 50, 99]:
            expected = scipy_mat.indptr[row + 1] - scipy_mat.indptr[row]
            assert get_row_len(csr, row) == expected
    
    def test_row_to_numpy(self, csr_f64):
        """csr.row_to_numpy(i) should return (values, indices) for row i."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def get_row_data(csr, row_idx):
            values, indices = csr.row_to_numpy(row_idx)
            return values.copy(), indices.copy()
        
        for row in [0, 10, 50]:
            values, indices = get_row_data(csr, row)
            
            start = scipy_mat.indptr[row]
            end = scipy_mat.indptr[row + 1]
            expected_vals = scipy_mat.data[start:end]
            expected_idx = scipy_mat.indices[start:end]
            
            np.testing.assert_allclose(values, expected_vals)
            np.testing.assert_array_equal(indices, expected_idx)
    
    def test_row(self, csr_f64):
        """csr.row(i) should be equivalent to row_to_numpy(i)."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def get_row(csr, row_idx):
            values, indices = csr.row(row_idx)
            return values.copy(), indices.copy()
        
        values, indices = get_row(csr, 10)
        
        start = scipy_mat.indptr[10]
        end = scipy_mat.indptr[11]
        np.testing.assert_allclose(values, scipy_mat.data[start:end])
        np.testing.assert_array_equal(indices, scipy_mat.indices[start:end])
    
    def test_get_existing_element(self, csr_f64):
        """csr.get(i, j) should return the value at (i, j)."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def get_element(csr, i, j):
            return csr.get(i, j)
        
        # Find a non-zero element
        dense = scipy_mat.toarray()
        nonzero_rows, nonzero_cols = np.where(dense != 0)
        if len(nonzero_rows) > 0:
            i, j = int(nonzero_rows[0]), int(nonzero_cols[0])
            expected = dense[i, j]
            result = get_element(csr, i, j)
            assert np.isclose(result, expected)
    
    def test_get_zero_element(self, csr_f64):
        """csr.get(i, j) should return 0.0 for zero elements."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def get_element(csr, i, j):
            return csr.get(i, j)
        
        # Find a zero element
        dense = scipy_mat.toarray()
        zero_rows, zero_cols = np.where(dense == 0)
        if len(zero_rows) > 0:
            i, j = int(zero_rows[0]), int(zero_cols[0])
            result = get_element(csr, i, j)
            assert result == 0.0
    
    def test_get_with_default(self, csr_f64):
        """csr.get(i, j, default) should return default for out-of-bounds."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, _ = csr_f64
        
        @njit
        def get_element_with_default(csr, i, j, default):
            return csr.get(i, j, default)
        
        # Out of bounds should return default
        result = get_element_with_default(csr, 1000, 0, -999.0)
        assert result == -999.0


class TestCSCColumnAccess:
    """Test CSC column access methods in JIT."""
    
    def test_col_len(self, csc_f64):
        """csc.col_len(j) should return number of non-zeros in column j."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def get_col_len(csc, col_idx):
            return csc.col_len(col_idx)
        
        for col in [0, 10, 25, 49]:
            expected = scipy_mat.indptr[col + 1] - scipy_mat.indptr[col]
            assert get_col_len(csc, col) == expected
    
    def test_col_to_numpy(self, csc_f64):
        """csc.col_to_numpy(j) should return (values, indices) for column j."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def get_col_data(csc, col_idx):
            values, indices = csc.col_to_numpy(col_idx)
            return values.copy(), indices.copy()
        
        for col in [0, 10, 25]:
            values, indices = get_col_data(csc, col)
            
            start = scipy_mat.indptr[col]
            end = scipy_mat.indptr[col + 1]
            expected_vals = scipy_mat.data[start:end]
            expected_idx = scipy_mat.indices[start:end]
            
            np.testing.assert_allclose(values, expected_vals)
            np.testing.assert_array_equal(indices, expected_idx)
    
    def test_col(self, csc_f64):
        """csc.col(j) should be equivalent to col_to_numpy(j)."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def get_col(csc, col_idx):
            values, indices = csc.col(col_idx)
            return values.copy(), indices.copy()
        
        values, indices = get_col(csc, 10)
        
        start = scipy_mat.indptr[10]
        end = scipy_mat.indptr[11]
        np.testing.assert_allclose(values, scipy_mat.data[start:end])
        np.testing.assert_array_equal(indices, scipy_mat.indices[start:end])
    
    def test_get_element(self, csc_f64):
        """csc.get(i, j) should return the value at (i, j)."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def get_element(csc, i, j):
            return csc.get(i, j)
        
        # Find a non-zero element
        dense = scipy_mat.toarray()
        nonzero_rows, nonzero_cols = np.where(dense != 0)
        if len(nonzero_rows) > 0:
            i, j = int(nonzero_rows[0]), int(nonzero_cols[0])
            expected = dense[i, j]
            result = get_element(csc, i, j)
            assert np.isclose(result, expected)
