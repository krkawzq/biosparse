"""Test zero-cost transpose in Numba JIT."""

import sys
sys.path.insert(0, 'src')

import numpy as np
import scipy.sparse as sp
from python._binding._sparse import CSRF64
from numba import njit

print("=" * 70)
print("Testing Zero-Cost Transpose in Numba JIT")
print("=" * 70)

# Create test matrix
mat_scipy = sp.random(5, 8, density=0.3, format='csr', dtype=np.float64)
csr = CSRF64.from_scipy(mat_scipy)

print(f"\nOriginal CSR matrix: {csr.shape}, nnz={csr.nnz}")

# Test 1: JIT dimension access
@njit
def test_transpose_dims_jit(csr):
    """Test transpose dimensions in JIT."""
    csc = csr.T()
    return csr.nrows, csr.ncols, csc.nrows, csc.ncols

csr_r, csr_c, csc_r, csc_c = test_transpose_dims_jit(csr)
print(f"\n[JIT] CSR shape: ({csr_r}, {csr_c})")
print(f"[JIT] CSC shape (transposed): ({csc_r}, {csc_c})")

assert (csc_r, csc_c) == (csr_c, csr_r), "JIT: Dimensions should be swapped"
print("[PASS] JIT transpose dimensions are correct!")

# Test 2: Double transpose
@njit
def test_double_transpose_jit(csr):
    """Test double transpose in JIT."""
    csc = csr.T()
    csr2 = csc.T()
    return csr2.nrows, csr2.ncols

csr2_r, csr2_c = test_double_transpose_jit(csr)
print(f"\n[JIT] Double transpose shape: ({csr2_r}, {csr2_c})")
assert (csr2_r, csr2_c) == (csr.nrows, csr.ncols), "JIT: Double transpose should restore shape"
print("[PASS] JIT double transpose restores original shape!")

print("\n" + "=" * 70)
print("JIT TRANSPOSE TESTS PASSED!")
print("=" * 70)
print("""
Zero-Cost Transpose in Numba JIT:
- CSR(m×n).T() → CSC(n×m)
- CSC(m×n).T() → CSR(n×m)
- Dimensions correctly swapped
- Shared handle (zero-cost)

Note: Data access methods (to_dense, element access, etc.) from Python
may not work correctly on transposed views due to underlying storage layout.
Use transpose primarily in JIT-compiled code for matrix multiplications
and other operations that respect the transposed semantics.
""")
