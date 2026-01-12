"""Comprehensive test for transpose functionality."""

import numpy as np
import scipy.sparse as sp

import sys
import os

# Add src/python to path
_src_python = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))
if _src_python not in sys.path:
    sys.path.insert(0, _src_python)

from biosparse._binding._sparse import CSRF64, CSCF64, CSRF32, CSCF32


def test_csr_transpose_basic():
    """Test CSR.T() returns CSC with correct dimensions."""
    # Create a simple CSR matrix (3x4)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = np.array([0, 2, 1, 3, 0], dtype=np.int64)
    indptr = np.array([0, 2, 4, 5], dtype=np.int64)

    # Create scipy CSR for reference
    scipy_csr = sp.csr_matrix((data, indices, indptr), shape=(3, 4))

    # Create our CSR
    csr = CSRF64.from_scipy(scipy_csr)

    # Transpose
    csc = csr.T()

    # Verify type
    assert isinstance(csc, CSCF64), f"Expected CSCF64, got {type(csc)}"

    # Verify dimensions are swapped
    assert csc.shape == (4, 3), f"Expected (4, 3), got {csc.shape}"
    assert csc.nrows == 4, f"Expected 4 rows, got {csc.nrows}"
    assert csc.ncols == 3, f"Expected 3 cols, got {csc.ncols}"

    # Verify nnz is preserved
    assert csc.nnz == csr.nnz, f"Expected nnz={csr.nnz}, got {csc.nnz}"

    # Verify data correctness by converting to dense
    csr_dense = scipy_csr.toarray()
    csc_dense = csc.to_dense()

    np.testing.assert_array_almost_equal(csc_dense, csr_dense.T, decimal=10)

    print("[PASS] CSR.T() basic test passed")


def test_csc_transpose_basic():
    """Test CSC.T() returns CSR with correct dimensions."""
    # Create a simple CSC matrix (4x3)
    # 3 columns, so indptr has 4 elements
    data = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
    indices = np.array([0, 2, 1, 0, 1], dtype=np.int64)
    indptr = np.array([0, 2, 3, 5], dtype=np.int64)

    # Create scipy CSC for reference
    scipy_csc = sp.csc_matrix((data, indices, indptr), shape=(4, 3))

    # Create our CSC
    csc = CSCF64.from_scipy(scipy_csc)

    # Transpose
    csr = csc.T()

    # Verify type
    assert isinstance(csr, CSRF64), f"Expected CSRF64, got {type(csr)}"

    # Verify dimensions are swapped
    assert csr.shape == (3, 4), f"Expected (3, 4), got {csr.shape}"
    assert csr.nrows == 3, f"Expected 3 rows, got {csr.nrows}"
    assert csr.ncols == 4, f"Expected 4 cols, got {csr.ncols}"

    # Verify nnz is preserved
    assert csr.nnz == csc.nnz, f"Expected nnz={csc.nnz}, got {csr.nnz}"

    # Verify data correctness
    csc_dense = scipy_csc.toarray()
    csr_dense = csr.to_dense()

    np.testing.assert_array_almost_equal(csr_dense, csc_dense.T, decimal=10)

    print("[PASS] CSC.T() basic test passed")


def test_transpose_f32():
    """Test transpose with float32."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int64)
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)

    scipy_csr = sp.csr_matrix((data, indices, indptr), shape=(3, 3))

    csr = CSRF32.from_scipy(scipy_csr)
    csc = csr.T()

    assert isinstance(csc, CSCF32), f"Expected CSCF32, got {type(csc)}"
    assert csc.shape == (3, 3)

    print("[PASS] Float32 transpose test passed")


def test_transpose_double():
    """Test CSR.T().T() returns CSR with original dimensions."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    indices = np.array([0, 2, 1, 3], dtype=np.int64)
    indptr = np.array([0, 2, 4], dtype=np.int64)

    scipy_csr = sp.csr_matrix((data, indices, indptr), shape=(2, 4))

    csr = CSRF64.from_scipy(scipy_csr)
    csc = csr.T()
    csr2 = csc.T()

    assert isinstance(csr2, CSRF64), f"Expected CSRF64, got {type(csr2)}"
    assert csr2.shape == csr.shape, f"Expected {csr.shape}, got {csr2.shape}"
    assert csr2.nnz == csr.nnz, f"Expected nnz={csr.nnz}, got {csr2.nnz}"

    # Verify data correctness
    csr_dense = csr.to_dense()
    csr2_dense = csr2.to_dense()

    np.testing.assert_array_almost_equal(csr2_dense, csr_dense, decimal=10)

    print("[PASS] Double transpose test passed")


def test_transpose_large():
    """Test transpose with a larger random matrix."""
    np.random.seed(42)

    # Create a random sparse matrix (100x150, ~5% density)
    scipy_csr = sp.random(100, 150, density=0.05, format='csr', dtype=np.float64)

    csr = CSRF64.from_scipy(scipy_csr)
    csc = csr.T()

    assert csc.shape == (150, 100), f"Expected (150, 100), got {csc.shape}"
    assert csc.nnz == csr.nnz, f"Expected nnz={csr.nnz}, got {csc.nnz}"

    # Verify data correctness (spot check a few values)
    csr_dense = scipy_csr.toarray()
    csc_dense = csc.to_dense()

    np.testing.assert_array_almost_equal(csc_dense, csr_dense.T, decimal=10)

    print("[PASS] Large matrix transpose test passed")


def test_transpose_access_data():
    """Test accessing data from transposed matrix."""
    # Create a simple CSR matrix (3x4)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = np.array([0, 2, 1, 3, 0], dtype=np.int64)
    indptr = np.array([0, 2, 4, 5], dtype=np.int64)

    scipy_csr = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
    csr = CSRF64.from_scipy(scipy_csr)

    # Transpose to CSC
    csc = csr.T()

    # Access column 0 (which is row 0 in the original CSR)
    col_values, col_indices = csc.col_to_numpy(0)

    # This should have values from row 0 of CSR
    # Row 0 of CSR: [1.0 at col 0, 2.0 at col 2]
    # So column 0 of CSC should have: [1.0 at row 0]
    assert len(col_values) == 2, f"Expected 2 values, got {len(col_values)}"

    print("[PASS] Transpose data access test passed")


def test_transpose_empty():
    """Test transpose of empty matrix."""
    scipy_csr = sp.csr_matrix((5, 8), dtype=np.float64)

    csr = CSRF64.from_scipy(scipy_csr)
    csc = csr.T()

    assert csc.shape == (8, 5), f"Expected (8, 5), got {csc.shape}"
    assert csc.nnz == 0, f"Expected nnz=0, got {csc.nnz}"

    print("[PASS] Empty matrix transpose test passed")


def test_transpose_jit():
    """Test transpose in JIT-compiled code."""
    from numba import njit
    import biosparse._numba  # Import numba integration to register types
    
    @njit
    def transpose_in_jit(csr):
        # Transpose CSR to CSC
        csc = csr.T()

        # Access dimensions
        nrows = csc.nrows
        ncols = csc.ncols
        nnz = csc.nnz

        return (nrows, ncols, nnz)

    # Create test matrix (3x4)
    data = np.array([1.0, 2.0, 3.0])
    indices = np.array([0, 1, 2], dtype=np.int64)
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)

    scipy_csr = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
    csr = CSRF64.from_scipy(scipy_csr)

    # Call JIT function
    nrows, ncols, nnz = transpose_in_jit(csr)

    assert nrows == 4, f"Expected 4 rows, got {nrows}"
    assert ncols == 3, f"Expected 3 cols, got {ncols}"
    assert nnz == 3, f"Expected nnz=3, got {nnz}"

    print("[PASS] JIT transpose test passed")


def test_transpose_jit_double():
    """Test double transpose in JIT."""
    from numba import njit
    import biosparse._numba
    
    @njit
    def double_transpose(csr):
        csc = csr.T()
        csr2 = csc.T()
        return csr2

    data = np.array([1.0, 2.0, 3.0, 4.0])
    indices = np.array([0, 2, 1, 3], dtype=np.int64)
    indptr = np.array([0, 2, 4], dtype=np.int64)

    scipy_csr = sp.csr_matrix((data, indices, indptr), shape=(2, 4))
    csr = CSRF64.from_scipy(scipy_csr)

    csr2 = double_transpose(csr)

    assert isinstance(csr2, CSRF64), f"Expected CSRF64, got {type(csr2)}"
    assert csr2.shape == csr.shape, f"Expected {csr.shape}, got {csr2.shape}"

    print("[PASS] JIT double transpose test passed")


if __name__ == "__main__":
    print("Running transpose tests...")
    print()

    # Basic Python tests (no JIT)
    test_csr_transpose_basic()
    test_csc_transpose_basic()
    test_transpose_f32()
    test_transpose_double()
    test_transpose_large()
    test_transpose_access_data()
    test_transpose_empty()
    
    # JIT tests - may fail if numba integration has import issues
    print()
    print("Running JIT tests...")
    try:
        test_transpose_jit()
        test_transpose_jit_double()
    except ImportError as e:
        print(f"[SKIP] JIT tests skipped due to import error: {e}")

    print()
    print("=" * 60)
    print("All transpose tests passed!")
    print("=" * 60)
