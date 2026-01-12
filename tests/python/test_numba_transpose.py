"""Tests for transpose functionality in numba JIT context."""

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


@pytest.mark.skipif(not _NUMBA_EXT_AVAILABLE, reason="Numba extension not available")
class TestTransposeJIT:
    """Tests for transpose in JIT context."""
    
    def test_csr_transpose_basic_jit(self):
        """Test CSR.T() in JIT returns correct dimensions."""
        from numba import njit
        import scipy.sparse as sp
        from _binding._sparse import CSRF64, CSCF64
        
        @njit
        def transpose_csr(csr):
            csc = csr.T()
            return csc.nrows, csc.ncols, csc.nnz
        
        # Create test matrix
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        indices = np.array([0, 2, 1, 3, 0], dtype=np.int64)
        indptr = np.array([0, 2, 4, 5], dtype=np.int64)
        scipy_csr = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        csr = CSRF64.from_scipy(scipy_csr)
        
        nrows, ncols, nnz = transpose_csr(csr)
        
        assert nrows == 4, f"Expected 4 rows, got {nrows}"
        assert ncols == 3, f"Expected 3 cols, got {ncols}"
        assert nnz == 5, f"Expected nnz=5, got {nnz}"
    
    def test_csc_transpose_basic_jit(self):
        """Test CSC.T() in JIT returns correct dimensions."""
        from numba import njit
        import scipy.sparse as sp
        from _binding._sparse import CSCF64, CSRF64
        
        @njit
        def transpose_csc(csc):
            csr = csc.T()
            return csr.nrows, csr.ncols, csr.nnz
        
        # Create test matrix
        data = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        indices = np.array([0, 2, 1, 0, 1], dtype=np.int64)
        indptr = np.array([0, 2, 3, 5], dtype=np.int64)
        scipy_csc = sp.csc_matrix((data, indices, indptr), shape=(4, 3))
        csc = CSCF64.from_scipy(scipy_csc)
        
        nrows, ncols, nnz = transpose_csc(csc)
        
        assert nrows == 3, f"Expected 3 rows, got {nrows}"
        assert ncols == 4, f"Expected 4 cols, got {ncols}"
        assert nnz == 5, f"Expected nnz=5, got {nnz}"
    
    def test_double_transpose_jit(self):
        """Test CSR.T().T() in JIT returns to CSR."""
        from numba import njit
        import scipy.sparse as sp
        from _binding._sparse import CSRF64
        
        @njit
        def double_transpose(csr):
            csc = csr.T()
            csr2 = csc.T()
            return csr2.nrows, csr2.ncols, csr2.nnz
        
        data = np.array([1.0, 2.0, 3.0, 4.0])
        indices = np.array([0, 2, 1, 3], dtype=np.int64)
        indptr = np.array([0, 2, 4], dtype=np.int64)
        scipy_csr = sp.csr_matrix((data, indices, indptr), shape=(2, 4))
        csr = CSRF64.from_scipy(scipy_csr)
        
        nrows, ncols, nnz = double_transpose(csr)
        
        assert nrows == 2, f"Expected 2 rows, got {nrows}"
        assert ncols == 4, f"Expected 4 cols, got {ncols}"
        assert nnz == 4, f"Expected nnz=4, got {nnz}"
    
    def test_transpose_f32_jit(self):
        """Test float32 transpose in JIT."""
        from numba import njit
        import scipy.sparse as sp
        from _binding._sparse import CSRF32, CSCF32
        
        @njit
        def transpose_f32(csr):
            csc = csr.T()
            return csc.nrows, csc.ncols
        
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.int64)
        indptr = np.array([0, 1, 2, 3], dtype=np.int64)
        scipy_csr = sp.csr_matrix((data, indices, indptr), shape=(3, 3))
        csr = CSRF32.from_scipy(scipy_csr)
        
        nrows, ncols = transpose_f32(csr)
        
        assert nrows == 3
        assert ncols == 3
    
    def test_transpose_return_value_jit(self):
        """Test that transpose returns proper typed object in JIT."""
        from numba import njit
        import scipy.sparse as sp
        from _binding._sparse import CSRF64, CSCF64
        
        @njit
        def get_transposed(csr):
            csc = csr.T()
            return csc
        
        data = np.array([1.0, 2.0, 3.0])
        indices = np.array([0, 1, 2], dtype=np.int64)
        indptr = np.array([0, 1, 2, 3], dtype=np.int64)
        scipy_csr = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        csr = CSRF64.from_scipy(scipy_csr)
        
        csc = get_transposed(csr)
        
        assert isinstance(csc, CSCF64), f"Expected CSCF64, got {type(csc)}"
        assert csc.shape == (4, 3)
        assert csc.nnz == 3


@pytest.mark.skipif(not _BINDING_AVAILABLE, reason="Binding not available")
class TestTransposePython:
    """Tests for transpose in pure Python context."""
    
    def test_csr_transpose_correctness(self):
        """Test CSR.T() produces correct transposed matrix."""
        import scipy.sparse as sp
        from _binding._sparse import CSRF64, CSCF64
        
        np.random.seed(42)
        scipy_csr = sp.random(50, 80, density=0.1, format='csr', dtype=np.float64)
        
        csr = CSRF64.from_scipy(scipy_csr)
        csc = csr.T()
        
        expected = scipy_csr.toarray().T
        actual = csc.to_dense()
        
        np.testing.assert_array_almost_equal(actual, expected, decimal=10)
    
    def test_csc_transpose_correctness(self):
        """Test CSC.T() produces correct transposed matrix."""
        import scipy.sparse as sp
        from _binding._sparse import CSCF64, CSRF64
        
        np.random.seed(123)
        scipy_csc = sp.random(60, 40, density=0.08, format='csc', dtype=np.float64)
        
        csc = CSCF64.from_scipy(scipy_csc)
        csr = csc.T()
        
        expected = scipy_csc.toarray().T
        actual = csr.to_dense()
        
        np.testing.assert_array_almost_equal(actual, expected, decimal=10)
    
    def test_transpose_preserves_values(self):
        """Test that transpose preserves all non-zero values."""
        import scipy.sparse as sp
        from _binding._sparse import CSRF64
        
        # Create matrix with specific values
        data = np.array([1.5, 2.7, 3.14, 4.0, 5.5])
        indices = np.array([0, 2, 1, 3, 0], dtype=np.int64)
        indptr = np.array([0, 2, 4, 5], dtype=np.int64)
        scipy_csr = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
        
        csr = CSRF64.from_scipy(scipy_csr)
        csc = csr.T()
        csr2 = csc.T()
        
        # After double transpose, values should be preserved
        original_dense = csr.to_dense()
        restored_dense = csr2.to_dense()
        
        np.testing.assert_array_almost_equal(original_dense, restored_dense, decimal=10)
    
    def test_transpose_empty_matrix(self):
        """Test transpose of empty matrix."""
        import scipy.sparse as sp
        from _binding._sparse import CSRF64
        
        scipy_csr = sp.csr_matrix((10, 20), dtype=np.float64)
        
        csr = CSRF64.from_scipy(scipy_csr)
        csc = csr.T()
        
        assert csc.shape == (20, 10)
        assert csc.nnz == 0
