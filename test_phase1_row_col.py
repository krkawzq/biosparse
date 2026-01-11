"""Test Phase 1: row()/col() simplified methods"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import scipy.sparse as sp
from python._binding._sparse import CSRF64, CSCF64
from numba import njit

# Create test matrix
mat_scipy = sp.random(10, 8, density=0.3, format='csr', dtype=np.float64)
print(f"Test matrix: {mat_scipy.shape}, nnz={mat_scipy.nnz}")

# Convert to CSR
csr = CSRF64.from_scipy(mat_scipy)

# Convert to CSC
csc_scipy = mat_scipy.tocsc()
csc = CSCF64.from_scipy(csc_scipy)

# =============================================================================
# Test 1: Python - CSR.row() and CSR.col()
# =============================================================================

print("\n" + "="*70)
print("Test 1: Python - CSR.row() and CSR.col()")
print("="*70)

# Test CSR.row() - should be equivalent to row_to_numpy()
row_idx = 3
vals1, idxs1 = csr.row_to_numpy(row_idx)
vals2, idxs2 = csr.row(row_idx)

assert np.array_equal(vals1, vals2), "CSR.row() values mismatch!"
assert np.array_equal(idxs1, idxs2), "CSR.row() indices mismatch!"
print(f"[PASS] CSR.row({row_idx}) matches row_to_numpy({row_idx})")

# Test CSR.col() - non-contiguous dimension
col_idx = 5
vals_col, rows_col = csr.col(col_idx)

# Verify against scipy
expected = mat_scipy[:, col_idx].toarray().flatten()
result = np.zeros(csr.nrows)
result[rows_col] = vals_col

assert np.allclose(expected, result), "CSR.col() result mismatch!"
print(f"[PASS] CSR.col({col_idx}) correctly extracts column")
print(f"  Found {len(vals_col)} nonzeros in column {col_idx}")

# =============================================================================
# Test 2: Python - CSC.col() and CSC.row()
# =============================================================================

print("\n" + "="*70)
print("Test 2: Python - CSC.col() and CSC.row()")
print("="*70)

# Test CSC.col() - should be equivalent to col_to_numpy()
col_idx = 2
vals1, idxs1 = csc.col_to_numpy(col_idx)
vals2, idxs2 = csc.col(col_idx)

assert np.array_equal(vals1, vals2), "CSC.col() values mismatch!"
assert np.array_equal(idxs1, idxs2), "CSC.col() indices mismatch!"
print(f"[PASS] CSC.col({col_idx}) matches col_to_numpy({col_idx})")

# Test CSC.row() - non-contiguous dimension
row_idx = 4
vals_row, cols_row = csc.row(row_idx)

# Verify against scipy
expected = csc_scipy[row_idx, :].toarray().flatten()
result = np.zeros(csc.ncols)
result[cols_row] = vals_row

assert np.allclose(expected, result), "CSC.row() result mismatch!"
print(f"[PASS] CSC.row({row_idx}) correctly extracts row")
print(f"  Found {len(vals_row)} nonzeros in row {row_idx}")

# =============================================================================
# Test 3: Numba JIT - CSR.row() and CSR.col()
# =============================================================================

print("\n" + "="*70)
print("Test 3: Numba JIT - CSR.row() and CSR.col()")
print("="*70)

@njit
def test_csr_row_jit(csr, row_idx):
    """Test CSR.row() in JIT"""
    values, indices = csr.row(row_idx)
    return values.sum()

@njit
def test_csr_col_jit(csr, col_idx):
    """Test CSR.col() in JIT"""
    values, row_indices = csr.col(col_idx)
    return values.sum(), len(values)

# Test CSR.row() in JIT
row_idx = 3
jit_sum = test_csr_row_jit(csr, row_idx)
py_sum = csr.row(row_idx)[0].sum()
assert np.isclose(jit_sum, py_sum), "JIT CSR.row() mismatch!"
print(f"[PASS] JIT CSR.row({row_idx}) works correctly")

# Test CSR.col() in JIT
col_idx = 5
jit_sum, jit_count = test_csr_col_jit(csr, col_idx)
py_vals, _ = csr.col(col_idx)
assert np.isclose(jit_sum, py_vals.sum()), "JIT CSR.col() sum mismatch!"
assert jit_count == len(py_vals), "JIT CSR.col() count mismatch!"
print(f"[PASS] JIT CSR.col({col_idx}) works correctly")

# =============================================================================
# Test 4: Numba JIT - CSC.col() and CSC.row()
# =============================================================================

print("\n" + "="*70)
print("Test 4: Numba JIT - CSC.col() and CSC.row()")
print("="*70)

@njit
def test_csc_col_jit(csc, col_idx):
    """Test CSC.col() in JIT"""
    values, indices = csc.col(col_idx)
    return values.sum()

@njit
def test_csc_row_jit(csc, row_idx):
    """Test CSC.row() in JIT"""
    values, col_indices = csc.row(row_idx)
    return values.sum(), len(values)

# Test CSC.col() in JIT
col_idx = 2
jit_sum = test_csc_col_jit(csc, col_idx)
py_sum = csc.col(col_idx)[0].sum()
assert np.isclose(jit_sum, py_sum), "JIT CSC.col() mismatch!"
print(f"[PASS] JIT CSC.col({col_idx}) works correctly")

# Test CSC.row() in JIT
row_idx = 4
jit_sum, jit_count = test_csc_row_jit(csc, row_idx)
py_vals, _ = csc.row(row_idx)
assert np.isclose(jit_sum, py_vals.sum()), "JIT CSC.row() sum mismatch!"
assert jit_count == len(py_vals), "JIT CSC.row() count mismatch!"
print(f"[PASS] JIT CSC.row({row_idx}) works correctly")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("Phase 1 Test Results: ALL PASSED [PASS]")
print("="*70)
print("""
Implemented and tested:
  [PASS] CSR.row(i) - Python and JIT
  [PASS] CSR.col(j) - Python and JIT (non-contiguous, uses binary search)
  [PASS] CSC.col(j) - Python and JIT
  [PASS] CSC.row(i) - Python and JIT (non-contiguous, uses binary search)

All operations work correctly in both Python and Numba JIT mode!
""")
