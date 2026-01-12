"""Tests for Numba boxing/unboxing of CSR/CSC types.

These tests verify that sparse matrices can be correctly:
1. Passed into JIT functions (unboxing)
2. Returned from JIT functions (boxing)
3. Maintain data integrity through the round-trip
"""

import pytest
import numpy as np

# Check if numba extension is available
_NUMBA_EXT_AVAILABLE = False
_BINDING_AVAILABLE = False

try:
    import numba
    from numba import njit
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
        from biosparse._binding import CSRF64, CSRF32, CSCF64, CSCF32
    except Exception:
        _NUMBA_EXT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _NUMBA_EXT_AVAILABLE,
    reason="Numba extension not available (requires Rust FFI + Numba)"
)


@pytest.fixture
def small_csr_f64():
    """Create a small CSR matrix."""
    pytest.importorskip("scipy")
    import scipy.sparse as sp
    np.random.seed(42)
    mat = sp.random(50, 40, density=0.15, format='csr', dtype=np.float64)
    return CSRF64.from_scipy(mat), mat


@pytest.fixture
def small_csr_f32():
    """Create a small CSR matrix with float32."""
    pytest.importorskip("scipy")
    import scipy.sparse as sp
    np.random.seed(42)
    mat = sp.random(50, 40, density=0.15, format='csr', dtype=np.float32)
    return CSRF32.from_scipy(mat), mat


@pytest.fixture
def small_csc_f64():
    """Create a small CSC matrix."""
    pytest.importorskip("scipy")
    import scipy.sparse as sp
    np.random.seed(42)
    mat = sp.random(50, 40, density=0.15, format='csc', dtype=np.float64)
    return CSCF64.from_scipy(mat), mat


@pytest.fixture
def small_csc_f32():
    """Create a small CSC matrix with float32."""
    pytest.importorskip("scipy")
    import scipy.sparse as sp
    np.random.seed(42)
    mat = sp.random(50, 40, density=0.15, format='csc', dtype=np.float32)
    return CSCF32.from_scipy(mat), mat


class TestCSRUnboxing:
    """Test unboxing CSR matrices into JIT functions."""
    
    def test_csr_f64_unbox_basic(self, small_csr_f64):
        """Basic unboxing should work for CSRF64."""
        csr, scipy_mat = small_csr_f64
        
        @njit
        def get_shape(csr):
            return (csr.nrows, csr.ncols)
        
        result = get_shape(csr)
        assert result == scipy_mat.shape
    
    def test_csr_f32_unbox_basic(self, small_csr_f32):
        """Basic unboxing should work for CSRF32."""
        csr, scipy_mat = small_csr_f32
        
        @njit
        def get_nnz(csr):
            return csr.nnz
        
        result = get_nnz(csr)
        assert result == scipy_mat.nnz
    
    def test_csr_iteration_after_unbox(self, small_csr_f64):
        """Iteration should work after unboxing."""
        csr, scipy_mat = small_csr_f64
        
        @njit
        def sum_all(csr):
            total = 0.0
            for values, indices in csr:
                for v in values:
                    total += v
            return total
        
        result = sum_all(csr)
        expected = scipy_mat.sum()
        assert np.isclose(result, expected)
    
    def test_csr_row_access_after_unbox(self, small_csr_f64):
        """Row access should work after unboxing."""
        csr, scipy_mat = small_csr_f64
        
        @njit
        def sum_first_row(csr):
            values, indices = csr.row(0)
            return values.sum()
        
        result = sum_first_row(csr)
        expected = scipy_mat.getrow(0).sum()
        assert np.isclose(result, expected)


class TestCSCUnboxing:
    """Test unboxing CSC matrices into JIT functions."""
    
    def test_csc_f64_unbox_basic(self, small_csc_f64):
        """Basic unboxing should work for CSCF64."""
        csc, scipy_mat = small_csc_f64
        
        @njit
        def get_shape(csc):
            return (csc.nrows, csc.ncols)
        
        result = get_shape(csc)
        assert result == scipy_mat.shape
    
    def test_csc_f32_unbox_basic(self, small_csc_f32):
        """Basic unboxing should work for CSCF32."""
        csc, scipy_mat = small_csc_f32
        
        @njit
        def get_nnz(csc):
            return csc.nnz
        
        result = get_nnz(csc)
        assert result == scipy_mat.nnz
    
    def test_csc_iteration_after_unbox(self, small_csc_f64):
        """Iteration should work after unboxing."""
        csc, scipy_mat = small_csc_f64
        
        @njit
        def sum_all(csc):
            total = 0.0
            for values, indices in csc:
                for v in values:
                    total += v
            return total
        
        result = sum_all(csc)
        expected = scipy_mat.sum()
        assert np.isclose(result, expected)


class TestCSRBoxing:
    """Test boxing CSR matrices returned from JIT functions."""
    
    def test_csr_clone_boxing(self, small_csr_f64):
        """Cloned CSR should be boxed correctly."""
        csr, scipy_mat = small_csr_f64
        
        @njit
        def clone_csr(csr):
            return csr.clone()
        
        result = clone_csr(csr)
        
        # Verify it's a proper CSR object
        assert isinstance(result, CSRF64)
        assert result.shape == csr.shape
        assert result.nnz == csr.nnz
        
        # Verify data integrity
        np.testing.assert_allclose(result.to_dense(), csr.to_dense())
    
    def test_csr_slice_boxing(self, small_csr_f64):
        """Sliced CSR should be boxed correctly."""
        csr, scipy_mat = small_csr_f64
        
        @njit
        def slice_csr(csr):
            return csr.slice_rows(10, 30)
        
        result = slice_csr(csr)
        
        assert isinstance(result, CSRF64)
        assert result.nrows == 20
        assert result.ncols == csr.ncols
        
        # Verify data
        expected = scipy_mat[10:30, :].toarray()
        np.testing.assert_allclose(result.to_dense(), expected)
    
    def test_csr_to_csc_boxing(self, small_csr_f64):
        """CSR to CSC conversion should box correctly."""
        csr, scipy_mat = small_csr_f64
        
        @njit
        def convert_to_csc(csr):
            return csr.to_csc()
        
        result = convert_to_csc(csr)
        
        assert isinstance(result, CSCF64)
        assert result.shape == csr.shape
        assert result.nnz == csr.nnz


class TestCSCBoxing:
    """Test boxing CSC matrices returned from JIT functions."""
    
    def test_csc_clone_boxing(self, small_csc_f64):
        """Cloned CSC should be boxed correctly."""
        csc, scipy_mat = small_csc_f64
        
        @njit
        def clone_csc(csc):
            return csc.clone()
        
        result = clone_csc(csc)
        
        assert isinstance(result, CSCF64)
        assert result.shape == csc.shape
        assert result.nnz == csc.nnz
    
    def test_csc_to_csr_boxing(self, small_csc_f64):
        """CSC to CSR conversion should box correctly."""
        csc, scipy_mat = small_csc_f64
        
        @njit
        def convert_to_csr(csc):
            return csc.to_csr()
        
        result = convert_to_csr(csc)
        
        assert isinstance(result, CSRF64)
        assert result.shape == csc.shape
        assert result.nnz == csc.nnz


class TestRoundTrip:
    """Test complete round-trip through JIT."""
    
    def test_csr_roundtrip_sum(self, small_csr_f64):
        """Sum computed in JIT should match Python."""
        csr, scipy_mat = small_csr_f64
        
        @njit
        def compute_sum(csr):
            total = 0.0
            for values, _ in csr:
                total += values.sum()
            return total
        
        jit_sum = compute_sum(csr)
        python_sum = sum(csr.row_to_numpy(i)[0].sum() for i in range(csr.nrows))
        scipy_sum = scipy_mat.sum()
        
        assert np.isclose(jit_sum, python_sum)
        assert np.isclose(jit_sum, scipy_sum)
    
    def test_csr_multiple_calls(self, small_csr_f64):
        """Multiple calls should work correctly."""
        csr, scipy_mat = small_csr_f64
        
        @njit
        def get_nnz(csr):
            return csr.nnz
        
        # Call multiple times
        for _ in range(5):
            result = get_nnz(csr)
            assert result == scipy_mat.nnz
    
    def test_different_matrices_same_function(self, small_csr_f64, small_csc_f64):
        """Same function should work with different matrix types."""
        csr, scipy_csr = small_csr_f64
        csc, scipy_csc = small_csc_f64
        
        @njit
        def get_density(mat):
            return mat.density
        
        csr_density = get_density(csr)
        csc_density = get_density(csc)
        
        assert np.isclose(csr_density, scipy_csr.nnz / (scipy_csr.shape[0] * scipy_csr.shape[1]))
        assert np.isclose(csc_density, scipy_csc.nnz / (scipy_csc.shape[0] * scipy_csc.shape[1]))


class TestEdgeCases:
    """Test edge cases in boxing/unboxing."""
    
    def test_empty_rows(self):
        """Handle matrices with empty rows."""
        pytest.importorskip("scipy")
        import scipy.sparse as sp
        
        # Create matrix with some empty rows
        row_idx = np.array([0, 0, 5, 5, 5], dtype=np.int64)
        col_idx = np.array([0, 3, 1, 2, 4], dtype=np.int64)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mat = sp.csr_matrix((data, (row_idx, col_idx)), shape=(10, 5))
        
        csr = CSRF64.from_scipy(mat)
        
        @njit
        def count_empty_rows(csr):
            count = 0
            for values, _ in csr:
                if len(values) == 0:
                    count += 1
            return count
        
        # Rows 1, 2, 3, 4, 6, 7, 8, 9 are empty (8 total)
        result = count_empty_rows(csr)
        assert result == 8
    
    def test_single_element(self):
        """Handle 1x1 matrix."""
        pytest.importorskip("scipy")
        import scipy.sparse as sp
        
        mat = sp.csr_matrix([[5.0]])
        csr = CSRF64.from_scipy(mat)
        
        @njit
        def get_single(csr):
            values, indices = csr.row(0)
            return values[0], indices[0]
        
        val, idx = get_single(csr)
        assert val == 5.0
        assert idx == 0
