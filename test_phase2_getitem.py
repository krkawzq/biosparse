"""Test Phase 2: __getitem__ single element access"""

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
# Test 1: Python - CSR[i, j] element access
# =============================================================================

print("\n" + "="*70)
print("Test 1: Python - CSR[i, j] element access")
print("="*70)

# Test accessing an existing nonzero element
dense = mat_scipy.toarray()
for i in range(csr.nrows):
    for j in range(csr.ncols):
        val_sparse = csr[i, j]
        val_dense = dense[i, j]
        assert np.isclose(val_sparse, val_dense), f"Mismatch at [{i}, {j}]: {val_sparse} != {val_dense}"

print("[PASS] All CSR[i, j] accesses match dense matrix")

# Test bounds checking
try:
    _ = csr[100, 0]
    assert False, "Should have raised IndexError"
except IndexError as e:
    print(f"[PASS] Row out of bounds correctly raises IndexError: {e}")

try:
    _ = csr[0, 100]
    assert False, "Should have raised IndexError"
except IndexError as e:
    print(f"[PASS] Column out of bounds correctly raises IndexError: {e}")

# =============================================================================
# Test 2: Python - CSC[i, j] element access
# =============================================================================

print("\n" + "="*70)
print("Test 2: Python - CSC[i, j] element access")
print("="*70)

# Test accessing elements
for i in range(csc.nrows):
    for j in range(csc.ncols):
        val_sparse = csc[i, j]
        val_dense = dense[i, j]
        assert np.isclose(val_sparse, val_dense), f"Mismatch at [{i}, {j}]: {val_sparse} != {val_dense}"

print("[PASS] All CSC[i, j] accesses match dense matrix")

# =============================================================================
# Test 3: Numba JIT - CSR[i, j]
# =============================================================================

print("\n" + "="*70)
print("Test 3: Numba JIT - CSR[i, j]")
print("="*70)

@njit
def test_csr_getitem_jit(csr, i, j):
    """Test CSR[i, j] in JIT"""
    return csr[i, j]

# Test all elements
for i in range(csr.nrows):
    for j in range(csr.ncols):
        val_jit = test_csr_getitem_jit(csr, i, j)
        val_expected = dense[i, j]
        assert np.isclose(val_jit, val_expected), f"JIT mismatch at [{i}, {j}]: {val_jit} != {val_expected}"

print("[PASS] All JIT CSR[i, j] accesses match dense matrix")

# Test bounds checking in JIT
@njit
def test_bounds_check(csr):
    try:
        _ = csr[100, 0]
        return False
    except:
        return True

assert test_bounds_check(csr), "JIT should raise error for out of bounds"
print("[PASS] JIT bounds checking works")

# =============================================================================
# Test 4: Numba JIT - CSC[i, j]
# =============================================================================

print("\n" + "="*70)
print("Test 4: Numba JIT - CSC[i, j]")
print("="*70)

@njit
def test_csc_getitem_jit(csc, i, j):
    """Test CSC[i, j] in JIT"""
    return csc[i, j]

# Test all elements
for i in range(csc.nrows):
    for j in range(csc.ncols):
        val_jit = test_csc_getitem_jit(csc, i, j)
        val_expected = dense[i, j]
        assert np.isclose(val_jit, val_expected), f"JIT mismatch at [{i}, {j}]: {val_jit} != {val_expected}"

print("[PASS] All JIT CSC[i, j] accesses match dense matrix")

# =============================================================================
# Test 5: Mixed slicing still works
# =============================================================================

print("\n" + "="*70)
print("Test 5: Slicing still works after adding element access")
print("="*70)

# Test row slicing
sub_csr = csr[2:5, :]
assert sub_csr.nrows == 3, "Row slicing failed"
print("[PASS] Row slicing works")

# Test column slicing
sub_csr2 = csr[:, 1:4]
assert sub_csr2.ncols == 3, "Column slicing failed"
print("[PASS] Column slicing works")

# Test both
sub_csr3 = csr[2:5, 1:4]
assert sub_csr3.shape == (3, 3), "Row+col slicing failed"
print("[PASS] Row and column slicing works")

# Test in JIT
@njit
def test_slice_jit(csr):
    sub = csr[2:5, 1:4]
    return sub.shape

shape = test_slice_jit(csr)
assert shape == (3, 3), "JIT slicing failed"
print("[PASS] JIT slicing works")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("Phase 2 Test Results: ALL PASSED")
print("="*70)
print("""
Implemented and tested:
  [PASS] CSR[i, j] - Python element access with binary search
  [PASS] CSC[i, j] - Python element access with binary search
  [PASS] CSR[i, j] - JIT element access
  [PASS] CSC[i, j] - JIT element access
  [PASS] Bounds checking works in both Python and JIT
  [PASS] Slicing still works after adding element access
  [PASS] Returns 0.0 for sparse (nonexistent) elements

All single element access operations work correctly!
""")
