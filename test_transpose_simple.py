"""Simple test for zero-cost transpose dimensions."""

import sys
sys.path.insert(0, 'src')

import numpy as np
import scipy.sparse as sp
from python._binding._sparse import CSRF64

# Create test matrix
mat_scipy = sp.random(5, 8, density=0.3, format='csr', dtype=np.float64)
csr = CSRF64.from_scipy(mat_scipy)

print(f"CSR shape: {csr.shape}")
print(f"CSR nrows: {csr.nrows}, ncols: {csr.ncols}")
print(f"CSR handle: {csr.handle_as_int}")

# Transpose
csc = csr.T()

print(f"\nCSC _transpose_dims: {csc._transpose_dims}")
print(f"CSC shape: {csc.shape}")
print(f"CSC nrows: {csc.nrows}, ncols: {csc.ncols}")
print(f"CSC handle: {csc.handle_as_int}")

assert csr.handle_as_int == csc.handle_as_int, "Handles should be identical"
assert csc.shape == (8, 5), f"CSC shape should be (8, 5), got {csc.shape}"
print("\n✓ Zero-cost transpose dimension swap works correctly!")
