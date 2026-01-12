"""Tests for CSR/CSC Numba conversion methods.

Tests _numba/_conversions.py: to_dense, to_coo, to_csc, to_csr, clone
"""

import pytest
import numpy as np

from conftest import NUMBA_EXT_AVAILABLE, BINDING_AVAILABLE

pytestmark = pytest.mark.skipif(
    not NUMBA_EXT_AVAILABLE,
    reason="Numba extension not available (requires Rust FFI + Numba)"
)


@pytest.fixture
def csr_f64():
    """Create a small CSR matrix for testing."""
    pytest.importorskip("scipy")
    if not BINDING_AVAILABLE:
        pytest.skip("Rust FFI bindings not available")
    
    import scipy.sparse as sp
    from _binding import CSRF64
    
    np.random.seed(42)
    mat = sp.random(30, 25, density=0.2, format='csr', dtype=np.float64)
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
    mat = sp.random(30, 25, density=0.2, format='csc', dtype=np.float64)
    return CSCF64.from_scipy(mat), mat


class TestCSRConversions:
    """Test CSR conversion methods in JIT."""
    
    def test_to_dense(self, csr_f64):
        """csr.to_dense() should return correct dense array."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def convert_to_dense(csr):
            return csr.to_dense()
        
        result = convert_to_dense(csr)
        expected = scipy_mat.toarray()
        np.testing.assert_allclose(result, expected)
    
    def test_to_coo(self, csr_f64):
        """csr.to_coo() should return (rows, cols, data) arrays."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def convert_to_coo(csr):
            rows, cols, data = csr.to_coo()
            return rows.copy(), cols.copy(), data.copy()
        
        rows, cols, data = convert_to_coo(csr)
        
        coo = scipy_mat.tocoo()
        # COO format may have different ordering, so compare via dense
        result_dense = np.zeros(scipy_mat.shape)
        for i in range(len(rows)):
            result_dense[rows[i], cols[i]] = data[i]
        
        expected_dense = scipy_mat.toarray()
        np.testing.assert_allclose(result_dense, expected_dense)
    
    def test_to_csc(self, csr_f64):
        """csr.to_csc() should return equivalent CSC matrix."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def convert_to_csc(csr):
            csc = csr.to_csc()
            return csc.shape, csc.nnz
        
        shape, nnz = convert_to_csc(csr)
        
        assert shape == scipy_mat.shape
        assert nnz == scipy_mat.nnz
    
    def test_to_csc_values(self, csr_f64):
        """csr.to_csc() should preserve values."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def convert_and_sum(csr):
            csc = csr.to_csc()
            total = 0.0
            for values, indices in csc:
                for v in values:
                    total += v
            return total
        
        result = convert_and_sum(csr)
        expected = scipy_mat.sum()
        assert np.isclose(result, expected)
    
    def test_clone(self, csr_f64):
        """csr.clone() should create independent copy."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def clone_and_check(csr):
            cloned = csr.clone()
            return cloned.shape, cloned.nnz
        
        shape, nnz = clone_and_check(csr)
        
        assert shape == scipy_mat.shape
        assert nnz == scipy_mat.nnz
    
    def test_clone_values(self, csr_f64):
        """csr.clone() should preserve all values."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def clone_and_sum(csr):
            cloned = csr.clone()
            total = 0.0
            for values, indices in cloned:
                for v in values:
                    total += v
            return total
        
        result = clone_and_sum(csr)
        expected = scipy_mat.sum()
        assert np.isclose(result, expected)


class TestCSCConversions:
    """Test CSC conversion methods in JIT."""
    
    def test_to_dense(self, csc_f64):
        """csc.to_dense() should return correct dense array."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def convert_to_dense(csc):
            return csc.to_dense()
        
        result = convert_to_dense(csc)
        expected = scipy_mat.toarray()
        np.testing.assert_allclose(result, expected)
    
    def test_to_coo(self, csc_f64):
        """csc.to_coo() should return (rows, cols, data) arrays."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def convert_to_coo(csc):
            rows, cols, data = csc.to_coo()
            return rows.copy(), cols.copy(), data.copy()
        
        rows, cols, data = convert_to_coo(csc)
        
        # Compare via dense
        result_dense = np.zeros(scipy_mat.shape)
        for i in range(len(rows)):
            result_dense[rows[i], cols[i]] = data[i]
        
        expected_dense = scipy_mat.toarray()
        np.testing.assert_allclose(result_dense, expected_dense)
    
    def test_to_csr(self, csc_f64):
        """csc.to_csr() should return equivalent CSR matrix."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def convert_to_csr(csc):
            csr = csc.to_csr()
            return csr.shape, csr.nnz
        
        shape, nnz = convert_to_csr(csc)
        
        assert shape == scipy_mat.shape
        assert nnz == scipy_mat.nnz
    
    def test_to_csr_values(self, csc_f64):
        """csc.to_csr() should preserve values."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def convert_and_sum(csc):
            csr = csc.to_csr()
            total = 0.0
            for values, indices in csr:
                for v in values:
                    total += v
            return total
        
        result = convert_and_sum(csc)
        expected = scipy_mat.sum()
        assert np.isclose(result, expected)
    
    def test_clone(self, csc_f64):
        """csc.clone() should create independent copy."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def clone_and_check(csc):
            cloned = csc.clone()
            return cloned.shape, cloned.nnz
        
        shape, nnz = clone_and_check(csc)
        
        assert shape == scipy_mat.shape
        assert nnz == scipy_mat.nnz


class TestRoundTrip:
    """Test round-trip conversions."""
    
    def test_csr_to_csc_to_csr(self, csr_f64):
        """CSR -> CSC -> CSR should preserve data."""
        from numba import njit
        import _numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def roundtrip(csr):
            csc = csr.to_csc()
            back = csc.to_csr()
            return back.to_dense()
        
        result = roundtrip(csr)
        expected = scipy_mat.toarray()
        np.testing.assert_allclose(result, expected)
    
    def test_csc_to_csr_to_csc(self, csc_f64):
        """CSC -> CSR -> CSC should preserve data."""
        from numba import njit
        import _numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        
        @njit
        def roundtrip(csc):
            csr = csc.to_csr()
            back = csr.to_csc()
            return back.to_dense()
        
        result = roundtrip(csc)
        expected = scipy_mat.toarray()
        np.testing.assert_allclose(result, expected)
