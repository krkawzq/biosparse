"""Integration tests for the complete _numba extension.

Tests combining multiple features: iteration, slicing, conversion, and access.
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
    from biosparse._binding import lib
    _BINDING_AVAILABLE = lib is not None
except Exception:
    _BINDING_AVAILABLE = False

_NUMBA_EXT_AVAILABLE = _NUMBA_AVAILABLE and _BINDING_AVAILABLE
if _NUMBA_EXT_AVAILABLE:
    try:
        import biosparse._numba
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
    from biosparse._binding import CSRF64
    
    np.random.seed(42)
    mat = sp.random(100, 80, density=0.1, format='csr', dtype=np.float64)
    return CSRF64.from_scipy(mat), mat


@pytest.fixture
def csc_f64():
    """Create a small CSC matrix for testing."""
    pytest.importorskip("scipy")
    if not _BINDING_AVAILABLE:
        pytest.skip("Rust FFI bindings not available")
    
    import scipy.sparse as sp
    from biosparse._binding import CSCF64
    
    np.random.seed(42)
    mat = sp.random(100, 80, density=0.1, format='csc', dtype=np.float64)
    return CSCF64.from_scipy(mat), mat


class TestSpMV:
    """Test sparse matrix-vector multiplication."""
    
    def test_csr_spmv(self, csr_f64):
        """CSR SpMV should match scipy result."""
        from numba import njit
        import biosparse._numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        vec = np.random.rand(scipy_mat.shape[1])
        
        @njit
        def spmv(csr, vec):
            nrows = csr.shape[0]
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
    
    def test_csc_spmv(self, csc_f64):
        """CSC SpMV (A^T * v) should match scipy result."""
        from numba import njit
        import biosparse._numba  # noqa: F401
        
        csc, scipy_mat = csc_f64
        vec = np.random.rand(scipy_mat.shape[0])
        
        @njit
        def spmv_transpose(csc, vec):
            ncols = csc.shape[1]
            result = np.zeros(ncols)
            col_idx = 0
            for values, indices in csc:
                dot = 0.0
                for i in range(len(values)):
                    dot += values[i] * vec[indices[i]]
                result[col_idx] = dot
                col_idx += 1
            return result
        
        result = spmv_transpose(csc, vec)
        expected = scipy_mat.T.dot(vec)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestBlockOperations:
    """Test operations on matrix blocks."""
    
    def test_block_sum(self, csr_f64):
        """Sum of sliced block should match scipy."""
        from numba import njit
        import biosparse._numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def block_sum(csr, row_start, row_end, col_start, col_end):
            block = csr[row_start:row_end, col_start:col_end]
            total = 0.0
            for values, indices in block:
                for v in values:
                    total += v
            return total
        
        result = block_sum(csr, 20, 50, 10, 40)
        expected = scipy_mat[20:50, 10:40].sum()
        assert np.isclose(result, expected)
    
    def test_diagonal_extraction(self, csr_f64):
        """Extract diagonal elements using get()."""
        from numba import njit
        import biosparse._numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def extract_diagonal(csr):
            n = min(csr.shape[0], csr.shape[1])
            diag = np.zeros(n)
            for i in range(n):
                diag[i] = csr.get(i, i)
            return diag
        
        result = extract_diagonal(csr)
        expected = scipy_mat.diagonal()
        np.testing.assert_allclose(result, expected)


class TestChainingOperations:
    """Test chaining multiple operations."""
    
    def test_slice_convert_iterate(self, csr_f64):
        """Slice -> Convert -> Iterate chain should work."""
        from numba import njit
        import biosparse._numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def chain_operations(csr):
            # Slice rows
            sliced = csr.slice_rows(10, 50)
            # Convert to CSC
            csc = sliced.to_csc()
            # Sum via iteration
            total = 0.0
            for values, indices in csc:
                for v in values:
                    total += v
            return total
        
        result = chain_operations(csr)
        expected = scipy_mat[10:50, :].sum()
        assert np.isclose(result, expected)
    
    def test_clone_modify_original_unchanged(self, csr_f64):
        """Clone should create independent copy."""
        from numba import njit
        import biosparse._numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def test_independence(csr):
            original_sum = 0.0
            for values, indices in csr:
                for v in values:
                    original_sum += v
            
            cloned = csr.clone()
            cloned_sum = 0.0
            for values, indices in cloned:
                for v in values:
                    cloned_sum += v
            
            return original_sum, cloned_sum
        
        orig, cloned = test_independence(csr)
        assert np.isclose(orig, cloned)
        assert np.isclose(orig, scipy_mat.sum())


class TestComplexAlgorithms:
    """Test more complex sparse algorithms."""
    
    def test_row_normalize(self, csr_f64):
        """Row normalization should produce correct results."""
        from numba import njit
        import biosparse._numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def compute_row_norms(csr):
            norms = np.zeros(csr.shape[0])
            row_idx = 0
            for values, indices in csr:
                norm_sq = 0.0
                for v in values:
                    norm_sq += v * v
                norms[row_idx] = np.sqrt(norm_sq)
                row_idx += 1
            return norms
        
        result = compute_row_norms(csr)
        expected = np.sqrt((scipy_mat.toarray() ** 2).sum(axis=1))
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_nonzero_pattern(self, csr_f64):
        """Count non-zeros per row should be correct."""
        from numba import njit
        import biosparse._numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def count_nnz_per_row(csr):
            counts = np.zeros(csr.shape[0], dtype=np.int64)
            row_idx = 0
            for values, indices in csr:
                counts[row_idx] = len(values)
                row_idx += 1
            return counts
        
        result = count_nnz_per_row(csr)
        expected = np.diff(scipy_mat.indptr)
        np.testing.assert_array_equal(result, expected)
    
    def test_find_max_per_row(self, csr_f64):
        """Find maximum value per row."""
        from numba import njit
        import biosparse._numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def max_per_row(csr):
            maxes = np.zeros(csr.shape[0])
            row_idx = 0
            for values, indices in csr:
                if len(values) > 0:
                    max_val = values[0]
                    for v in values:
                        if v > max_val:
                            max_val = v
                    maxes[row_idx] = max_val
                row_idx += 1
            return maxes
        
        result = max_per_row(csr)
        # For rows with no non-zeros, scipy returns 0
        dense = scipy_mat.toarray()
        for i in range(len(result)):
            row = dense[i, :]
            nonzero = row[row != 0]
            if len(nonzero) > 0:
                assert np.isclose(result[i], nonzero.max())
            else:
                assert result[i] == 0.0


class TestPerformancePatterns:
    """Test common performance-critical patterns."""
    
    @pytest.mark.slow
    def test_repeated_access(self, csr_f64):
        """Repeated row access should be efficient."""
        from numba import njit
        import biosparse._numba  # noqa: F401
        
        csr, scipy_mat = csr_f64
        
        @njit
        def repeated_access(csr, iterations):
            total = 0.0
            for _ in range(iterations):
                for i in range(csr.shape[0]):
                    values, indices = csr.row(i)
                    total += values.sum()
            return total
        
        result = repeated_access(csr, 10)
        expected = scipy_mat.sum() * 10
        assert np.isclose(result, expected)
    
    def test_combined_csr_csc_operations(self, csr_f64, csc_f64):
        """Operations using both CSR and CSC should work."""
        from numba import njit
        import biosparse._numba  # noqa: F401
        
        csr, scipy_csr = csr_f64
        csc, scipy_csc = csc_f64
        
        @njit
        def combined_sum(csr, csc):
            csr_sum = 0.0
            for values, indices in csr:
                csr_sum += values.sum()
            
            csc_sum = 0.0
            for values, indices in csc:
                csc_sum += values.sum()
            
            return csr_sum + csc_sum
        
        result = combined_sum(csr, csc)
        expected = scipy_csr.sum() + scipy_csc.sum()
        assert np.isclose(result, expected)
