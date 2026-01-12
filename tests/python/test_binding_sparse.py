"""Tests for biosparse CSR/CSC sparse matrices (requires Rust FFI)."""

import pytest
import numpy as np

# Skip all tests if bindings not available
pytestmark = pytest.mark.binding

try:
    from _binding import CSRF64, CSCF64, lib
    if lib is None:
        pytest.skip("Rust FFI bindings not loaded", allow_module_level=True)
except (ImportError, OSError):
    pytest.skip("Rust FFI bindings not available", allow_module_level=True)


class TestCSRF64Creation:
    """Test CSR matrix creation."""
    
    def test_from_scipy(self, scipy_csr):
        """Create CSR from scipy matrix."""
        csr = CSRF64.from_scipy(scipy_csr)
        assert csr.shape == scipy_csr.shape
        assert csr.nnz == scipy_csr.nnz
    
    def test_from_coo(self):
        """Create CSR from COO format."""
        row_idx = np.array([0, 0, 1, 2], dtype=np.int64)
        col_idx = np.array([0, 2, 1, 0], dtype=np.int64)
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        
        csr = CSRF64.from_coo(3, 3, row_idx, col_idx, data)
        
        assert csr.shape == (3, 3)
        assert csr.nnz == 4


class TestCSRF64Properties:
    """Test CSR matrix properties."""
    
    def test_shape(self, csr_matrix):
        """Shape should match original."""
        assert len(csr_matrix.shape) == 2
        assert csr_matrix.shape[0] == csr_matrix.nrows
        assert csr_matrix.shape[1] == csr_matrix.ncols
    
    def test_nnz(self, csr_matrix):
        """NNZ should be positive."""
        assert csr_matrix.nnz > 0
    
    def test_density(self, csr_matrix):
        """Density should be in [0, 1]."""
        assert 0 <= csr_matrix.density <= 1
    
    def test_sparsity(self, csr_matrix):
        """Sparsity should be 1 - density."""
        assert np.isclose(csr_matrix.sparsity + csr_matrix.density, 1.0)
    
    def test_is_valid(self, csr_matrix):
        """Matrix should be valid."""
        assert csr_matrix.is_valid
    
    def test_is_sorted(self, csr_matrix):
        """Matrix should be sorted."""
        assert csr_matrix.is_sorted


class TestCSRF64RowAccess:
    """Test CSR row access methods."""
    
    def test_row_to_numpy(self, csr_matrix):
        """Should return values and indices arrays."""
        values, indices = csr_matrix.row_to_numpy(0)
        assert len(values) == len(indices)
        assert values.dtype == np.float64
        assert indices.dtype == np.int64
    
    def test_row_len(self, csr_matrix):
        """Row length should match array length."""
        for i in range(min(5, csr_matrix.nrows)):
            values, _ = csr_matrix.row_to_numpy(i)
            assert len(values) == csr_matrix.row_len(i)


class TestCSRF64Slicing:
    """Test CSR slicing operations."""
    
    def test_slice_rows(self, csr_matrix):
        """Row slicing should work."""
        sliced = csr_matrix.slice_rows(10, 20)
        assert sliced.nrows == 10
        assert sliced.ncols == csr_matrix.ncols
    
    def test_slice_cols(self, csr_matrix):
        """Column slicing should work."""
        sliced = csr_matrix.slice_cols(5, 15)
        assert sliced.nrows == csr_matrix.nrows
        assert sliced.ncols == 10
    
    def test_getitem_rows(self, csr_matrix):
        """Slicing via __getitem__ should work."""
        sliced = csr_matrix[10:20]
        assert sliced.nrows == 10
    
    def test_getitem_both(self, csr_matrix):
        """2D slicing should work."""
        sliced = csr_matrix[10:20, 5:15]
        assert sliced.shape == (10, 10)
    
    def test_getitem_element(self, csr_matrix, scipy_csr):
        """Element access should return correct value."""
        csr = CSRF64.from_scipy(scipy_csr)
        dense = scipy_csr.toarray()
        
        # Test a few elements
        for i in range(min(5, csr.nrows)):
            for j in range(min(5, csr.ncols)):
                assert np.isclose(csr[i, j], dense[i, j])


class TestCSRF64Conversion:
    """Test CSR conversion methods."""
    
    def test_to_dense(self, csr_matrix, scipy_csr):
        """Conversion to dense should match scipy."""
        csr = CSRF64.from_scipy(scipy_csr)
        dense = csr.to_dense()
        expected = scipy_csr.toarray()
        
        assert np.allclose(dense, expected)
    
    def test_to_coo(self, csr_matrix, scipy_csr):
        """Conversion to COO should preserve data."""
        csr = CSRF64.from_scipy(scipy_csr)
        row_idx, col_idx, data = csr.to_coo()
        
        assert len(row_idx) == csr.nnz
        assert len(col_idx) == csr.nnz
        assert len(data) == csr.nnz
    
    def test_to_scipy(self, scipy_csr):
        """Round-trip to scipy should preserve data."""
        csr = CSRF64.from_scipy(scipy_csr)
        back = csr.to_scipy()
        
        assert np.allclose(back.toarray(), scipy_csr.toarray())
    
    def test_to_csc(self, scipy_csr):
        """Conversion to CSC should work."""
        csr = CSRF64.from_scipy(scipy_csr)
        csc = csr.to_csc()
        
        assert csc.shape == csr.shape
        assert csc.nnz == csr.nnz


class TestCSRF64Clone:
    """Test CSR cloning."""
    
    def test_clone(self, csr_matrix):
        """Clone should create independent copy."""
        cloned = csr_matrix.clone()
        
        assert cloned.shape == csr_matrix.shape
        assert cloned.nnz == csr_matrix.nnz
        # Should be different objects
        assert cloned.handle != csr_matrix.handle


class TestCSRF64Stack:
    """Test CSR stacking operations."""
    
    def test_vstack(self, scipy_csr):
        """Vertical stacking should work."""
        csr1 = CSRF64.from_scipy(scipy_csr)
        csr2 = CSRF64.from_scipy(scipy_csr)
        
        stacked = CSRF64.vstack([csr1, csr2])
        
        assert stacked.nrows == 2 * csr1.nrows
        assert stacked.ncols == csr1.ncols
    
    def test_hstack(self, scipy_csr):
        """Horizontal stacking should work."""
        csr1 = CSRF64.from_scipy(scipy_csr)
        csr2 = CSRF64.from_scipy(scipy_csr)
        
        stacked = CSRF64.hstack([csr1, csr2])
        
        assert stacked.nrows == csr1.nrows
        assert stacked.ncols == 2 * csr1.ncols


class TestCSCF64:
    """Test CSC matrix (basic tests)."""
    
    def test_from_scipy(self, scipy_csc):
        """Create CSC from scipy matrix."""
        csc = CSCF64.from_scipy(scipy_csc)
        assert csc.shape == scipy_csc.shape
        assert csc.nnz == scipy_csc.nnz
    
    def test_col_access(self, scipy_csc):
        """Column access should work."""
        csc = CSCF64.from_scipy(scipy_csc)
        values, indices = csc.col_to_numpy(0)
        
        assert len(values) == len(indices)
        assert values.dtype == np.float64
    
    def test_to_csr(self, scipy_csc):
        """Conversion to CSR should work."""
        csc = CSCF64.from_scipy(scipy_csc)
        csr = csc.to_csr()
        
        assert csr.shape == csc.shape
        assert csr.nnz == csc.nnz
