"""Test zero-cost transpose operation."""

import sys
sys.path.insert(0, 'src')

import numpy as np
import scipy.sparse as sp
from python._binding._sparse import CSRF64
from numba import njit

print("="*70)
print("Testing Zero-Cost Transpose (.T)")
print("="*70)

# Create test matrix
mat_scipy = sp.random(5, 8, density=0.3, format='csr', dtype=np.float64)
csr = CSRF64.from_scipy(mat_scipy)
dense = mat_scipy.toarray()

print(f"\nOriginal CSR matrix: {csr.shape}, nnz={csr.nnz}")
print(f"Dense representation:\n{dense}")

# =============================================================================
# Test 1: Python - Zero-cost transpose
# =============================================================================

print("\n" + "="*70)
print("Test 1: Python - Zero-cost CSR.T() → CSC")
print("="*70)

csc = csr.T()
print(f"[PASS] CSR({csr.shape[0]}×{csr.shape[1]}) → CSC({csc.shape[0]}×{csc.shape[1]})")
print(f"[INFO] CSR handle: {csr.handle_as_int}")
print(f"[INFO] CSC handle: {csc.handle_as_int} (should be same!)")
assert csr.handle_as_int == csc.handle_as_int, "Handles should be identical for zero-cost transpose"
print(f"[PASS] Handles are identical - zero-cost transpose confirmed!")

# Verify dimensions are swapped
assert csc.shape == (csr.shape[1], csr.shape[0]), "Dimensions should be swapped"
print(f"[PASS] Dimensions correctly swapped")

# Verify the transpose is correct
csc_dense = csc.to_dense()
expected_dense = dense.T
assert np.allclose(csc_dense, expected_dense), "Transpose values should match"
print(f"[PASS] Transpose values are correct")

# =============================================================================
# Test 2: Python - Double transpose
# =============================================================================

print("\n" + "="*70)
print("Test 2: Python - Double transpose CSR.T().T() → CSR")
print("="*70)

csr2 = csc.T()
print(f"[PASS] CSC.T() → CSR")
print(f"[INFO] Original CSR handle: {csr.handle_as_int}")
print(f"[INFO] Double transposed CSR handle: {csr2.handle_as_int}")
assert csr2.shape == csr.shape, "Double transpose should restore original shape"
print(f"[PASS] Double transpose restores original shape")

csr2_dense = csr2.to_dense()
assert np.allclose(csr2_dense, dense), "Double transpose should equal original"
print(f"[PASS] Double transpose equals original matrix")

# =============================================================================
# Test 3: Numba JIT - Zero-cost transpose
# =============================================================================

print("\n" + "="*70)
print("Test 3: Numba JIT - Zero-cost transpose in compiled code")
print("="*70)

@njit
def test_transpose_jit(csr):
    """Test transpose in JIT-compiled code."""
    # CSR to CSC transpose
    csc = csr.T()

    # Get shapes
    csr_rows, csr_cols = csr.nrows, csr.ncols
    csc_rows, csc_cols = csc.nrows, csc.ncols

    # Double transpose
    csr2 = csc.T()
    csr2_rows, csr2_cols = csr2.nrows, csr2.ncols

    return (csr_rows, csr_cols, csc_rows, csc_cols, csr2_rows, csr2_cols)

result = test_transpose_jit(csr)
print(f"[INFO] Original CSR shape: ({result[0]}, {result[1]})")
print(f"[INFO] Transposed CSC shape: ({result[2]}, {result[3]})")
print(f"[INFO] Double transposed CSR shape: ({result[4]}, {result[5]})")

assert (result[2], result[3]) == (result[1], result[0]), "JIT: Dimensions should be swapped"
assert (result[4], result[5]) == (result[0], result[1]), "JIT: Double transpose should restore shape"
print(f"[PASS] JIT transpose works correctly")

# =============================================================================
# Test 4: Numba JIT - Element access through transpose
# =============================================================================

print("\n" + "="*70)
print("Test 4: Numba JIT - Element access through transposed view")
print("="*70)

@njit
def test_transpose_access_jit(csr):
    """Test accessing elements through transposed view."""
    csc = csr.T()

    # Access element [i,j] in CSR and [j,i] in CSC
    # They should be the same value
    val1 = csr[2, 3]
    val2 = csc[3, 2]

    return val1, val2

val1, val2 = test_transpose_access_jit(csr)
expected = dense[2, 3]
print(f"[INFO] CSR[2,3] = {val1:.6f}")
print(f"[INFO] CSC.T[2,3] = CSC[3,2] = {val2:.6f}")
print(f"[INFO] Expected = {expected:.6f}")
assert np.isclose(val1, expected), "CSR element should match dense"
assert np.isclose(val2, expected), "Transposed CSC element should match"
assert np.isclose(val1, val2), "Values should be equal"
print(f"[PASS] Element access through transpose works correctly")

# =============================================================================
# Test 5: Row/Col access through transpose
# =============================================================================

print("\n" + "="*70)
print("Test 5: Numba JIT - Row/Col access through transpose")
print("="*70)

@njit
def test_transpose_row_col_jit(csr):
    """Test row/col access through transposed view."""
    csc = csr.T()

    # Get row 2 from CSR
    csr_vals, csr_idxs = csr.row(2)

    # Get col 2 from CSC (which is row 2 of the original matrix transposed)
    csc_vals, csc_idxs = csc.col(2)

    return csr_vals.sum(), csc_vals.sum()

csr_sum, csc_sum = test_transpose_row_col_jit(csr)
expected_sum = dense[2, :].sum()
print(f"[INFO] CSR.row(2) sum = {csr_sum:.6f}")
print(f"[INFO] CSC.col(2) sum (= CSR.row(2)) = {csc_sum:.6f}")
print(f"[INFO] Expected sum = {expected_sum:.6f}")
assert np.isclose(csr_sum, expected_sum), "CSR row sum should match"
assert np.isclose(csc_sum, expected_sum), "CSC col sum should match"
print(f"[PASS] Row/Col access through transpose works correctly")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("ALL TRANSPOSE TESTS PASSED!")
print("="*70)
print("""
Zero-Cost Transpose Features:

1. Handle Sharing:
   - CSR.T() returns CSC with the SAME handle
   - CSC.T() returns CSR with the SAME handle
   - No memory allocation or data copying

2. Dimension Swapping:
   - CSR(m×n).T() → CSC(n×m)
   - CSC(m×n).T() → CSR(n×m)
   - Double transpose restores original

3. Format Reinterpretation:
   - CSR row data = CSC column data of transpose
   - CSC column data = CSR row data of transpose
   - Completely zero-cost operation

4. Full Numba JIT Support:
   - Works in @njit compiled functions
   - All element access methods work correctly
   - Row/Col access works through transposed view

Performance: O(1) time, O(1) space - truly zero-cost!
""")
