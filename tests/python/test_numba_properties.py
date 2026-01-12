"""Tests for CSR/CSC Numba property overloads.

Tests _numba/_overloads.py: shape, density, sparsity, is_empty, is_zero, len()
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


class TestCSRProperties:
    """Test CSR property overloads in JIT."""
    
    def test_shape(self, csr_f64):
        """csr.shape should return (nrows, ncols)."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def get_shape(csr):
            return csr.shape
        
        shape = get_shape(csr)
        assert shape == scipy_mat.shape
    
    def test_density(self, csr_f64):
        """csr.density should return nnz / (nrows * ncols)."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def get_density(csr):
            return csr.density
        
        density = get_density(csr)
        expected = scipy_mat.nnz / (scipy_mat.shape[0] * scipy_mat.shape[1])
        assert np.isclose(density, expected)
    
    def test_sparsity(self, csr_f64):
        """csr.sparsity should return 1 - density."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def get_sparsity(csr):
            return csr.sparsity
        
        sparsity = get_sparsity(csr)
        expected = 1.0 - scipy_mat.nnz / (scipy_mat.shape[0] * scipy_mat.shape[1])
        assert np.isclose(sparsity, expected)
    
    def test_is_empty(self, csr_f64):
        """csr.is_empty should return False for non-empty matrix."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, _ = csr_f64
        
        @njit
        def get_is_empty(csr):
            return csr.is_empty
        
        assert get_is_empty(csr) == False
    
    def test_is_zero(self, csr_f64):
        """csr.is_zero should return False for matrix with non-zeros."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, _ = csr_f64
        
        @njit
        def get_is_zero(csr):
            return csr.is_zero
        
        assert get_is_zero(csr) == False
    
    def test_len(self, csr_f64):
        """len(csr) should return nrows."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def get_len(csr):
            return len(csr)
        
        assert get_len(csr) == scipy_mat.shape[0]


class TestCSCProperties:
    """Test CSC property overloads in JIT."""
    
    def test_shape(self, csc_f64):
        """csc.shape should return (nrows, ncols)."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def get_shape(csc):
            return csc.shape
        
        shape = get_shape(csc)
        assert shape == scipy_mat.shape
    
    def test_density(self, csc_f64):
        """csc.density should return nnz / (nrows * ncols)."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def get_density(csc):
            return csc.density
        
        density = get_density(csc)
        expected = scipy_mat.nnz / (scipy_mat.shape[0] * scipy_mat.shape[1])
        assert np.isclose(density, expected)
    
    def test_sparsity(self, csc_f64):
        """csc.sparsity should return 1 - density."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def get_sparsity(csc):
            return csc.sparsity
        
        sparsity = get_sparsity(csc)
        expected = 1.0 - scipy_mat.nnz / (scipy_mat.shape[0] * scipy_mat.shape[1])
        assert np.isclose(sparsity, expected)
    
    def test_len(self, csc_f64):
        """len(csc) should return ncols."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def get_len(csc):
            return len(csc)
        
        assert get_len(csc) == scipy_mat.shape[1]
