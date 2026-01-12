"""Simple debug test for transpose."""

import numpy as np
import scipy.sparse as sp

import sys
import os
# Add src to path so biosparse package can be imported
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src not in sys.path:
    sys.path.insert(0, _src)

from biosparse._binding._sparse import CSRF64, CSCF64


# Create a simple 3x4 CSR matrix:
# Row 0: [1, 0, 2, 0]  -> values=[1, 2] at cols=[0, 2]
# Row 1: [0, 3, 0, 4]  -> values=[3, 4] at cols=[1, 3]
# Row 2: [5, 0, 0, 0]  -> values=[5] at col=[0]

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
indices = np.array([0, 2, 1, 3, 0], dtype=np.int64)
indptr = np.array([0, 2, 4, 5], dtype=np.int64)

scipy_csr = sp.csr_matrix((data, indices, indptr), shape=(3, 4))
print("Original CSR matrix (3x4):")
print(scipy_csr.toarray())
print()

# Expected transpose (4x3):
# Col 0 becomes Row 0: [1, 0, 5]  -> values=[1, 5] at cols=[0, 2]
# Col 1 becomes Row 1: [0, 3, 0]  -> values=[3] at col=[1]
# Col 2 becomes Row 2: [2, 0, 0]  -> values=[2] at col=[0]
# Col 3 becomes Row 3: [0, 4, 0]  -> values=[4] at col=[1]

print("Expected transpose (scipy):")
scipy_transpose = scipy_csr.T.toarray()
print(scipy_transpose)
print()

# Create our CSR and transpose
csr = CSRF64.from_scipy(scipy_csr)
csc = csr.T()

print("Our transpose shape:", csc.shape)
print("Our transpose nnz:", csc.nnz)
print()

# Check each column
for col in range(csc.ncols):
    vals, idxs = csc.col_to_numpy(col)
    print(f"Column {col}: values={vals}, row_indices={idxs}")

print()
print("Our transpose as dense:")
csc_dense = csc.to_dense()
print(csc_dense)
print()

print("Difference:")
print(csc_dense - scipy_transpose)
