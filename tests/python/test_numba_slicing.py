"""Tests for CSR/CSC Numba slicing operations.

Tests _numba/_operators.py: slice_rows, slice_cols, __getitem__
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
    mat = sp.random(50, 40, density=0.15, format='csr', dtype=np.float64)
    return CSRF64.from_scipy(mat), mat


@pytest.fixture
def csc_f64():
    """Create a small CSC matrix for testing."""
    pytest.importorskip("scipy")
    if not BINDING_AVAILABLE:
        pytest.skip("Rust FFI bindings not available")
    
    import scipy.sparse as sp
    from _binding import CSCF64
    
    np.random.seed(42)
    mat = sp.random(50, 40, density=0.15, format='csc', dtype=np.float64)
    return CSCF64.from_scipy(mat), mat


class TestCSRSlicing:
    """Test CSR slicing in JIT."""
    
    def test_slice_rows(self, csr_f64):
        """csr.slice_rows(start, end) should return sub-matrix."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def slice_rows_jit(csr, start, end):
            sliced = csr.slice_rows(start, end)
            return sliced.shape, sliced.nnz
        
        shape, nnz = slice_rows_jit(csr, 10, 30)
        
        expected = scipy_mat[10:30, :]
        assert shape == expected.shape
        assert nnz == expected.nnz
    
    def test_slice_cols(self, csr_f64):
        """csr.slice_cols(start, end) should return sub-matrix."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def slice_cols_jit(csr, start, end):
            sliced = csr.slice_cols(start, end)
            return sliced.shape, sliced.nnz
        
        shape, nnz = slice_cols_jit(csr, 5, 25)
        
        expected = scipy_mat[:, 5:25]
        assert shape == expected.shape
        assert nnz == expected.nnz
    
    def test_getitem_row_slice(self, csr_f64):
        """csr[10:20] should slice rows."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def getitem_row(csr):
            sliced = csr[10:20]
            return sliced.shape
        
        shape = getitem_row(csr)
        expected = scipy_mat[10:20, :]
        assert shape == expected.shape
    
    def test_getitem_element(self, csr_f64):
        """csr[i, j] should return element value."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def getitem_element(csr, i, j):
            return csr[i, j]
        
        dense = scipy_mat.toarray()
        # Test a few elements
        for i, j in [(0, 0), (10, 5), (25, 20)]:
            result = getitem_element(csr, i, j)
            expected = dense[i, j]
            assert np.isclose(result, expected)
    
    def test_getitem_both_slices(self, csr_f64):
        """csr[10:20, 5:15] should slice both dimensions."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def getitem_both(csr):
            sliced = csr[10:20, 5:15]
            return sliced.shape, sliced.nnz
        
        shape, nnz = getitem_both(csr)
        expected = scipy_mat[10:20, 5:15]
        assert shape == expected.shape
        assert nnz == expected.nnz
    
    def test_slice_and_sum(self, csr_f64):
        """Sliced matrix should contain correct values."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def slice_and_sum(csr):
            sliced = csr.slice_rows(10, 20)
            total = 0.0
            for values, indices in sliced:
                for v in values:
                    total += v
            return total
        
        result = slice_and_sum(csr)
        expected = scipy_mat[10:20, :].sum()
        assert np.isclose(result, expected)


class TestCSCSlicing:
    """Test CSC slicing in JIT."""
    
    def test_slice_rows(self, csc_f64):
        """csc.slice_rows(start, end) should return sub-matrix."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def slice_rows_jit(csc, start, end):
            sliced = csc.slice_rows(start, end)
            return sliced.shape, sliced.nnz
        
        shape, nnz = slice_rows_jit(csc, 10, 30)
        
        expected = scipy_mat[10:30, :]
        assert shape == expected.shape
        assert nnz == expected.nnz
    
    def test_slice_cols(self, csc_f64):
        """csc.slice_cols(start, end) should return sub-matrix."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def slice_cols_jit(csc, start, end):
            sliced = csc.slice_cols(start, end)
            return sliced.shape, sliced.nnz
        
        shape, nnz = slice_cols_jit(csc, 5, 25)
        
        expected = scipy_mat[:, 5:25]
        assert shape == expected.shape
        assert nnz == expected.nnz
    
    def test_getitem_element(self, csc_f64):
        """csc[i, j] should return element value."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def getitem_element(csc, i, j):
            return csc[i, j]
        
        dense = scipy_mat.toarray()
        for i, j in [(0, 0), (10, 5), (25, 20)]:
            result = getitem_element(csc, i, j)
            expected = dense[i, j]
            assert np.isclose(result, expected)
    
    def test_slice_and_iterate(self, csc_f64):
        """Sliced CSC should be iterable."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def slice_and_sum(csc):
            sliced = csc.slice_cols(5, 20)
            total = 0.0
            for values, indices in sliced:
                for v in values:
                    total += v
            return total
        
        result = slice_and_sum(csc)
        expected = scipy_mat[:, 5:20].sum()
        assert np.isclose(result, expected)
