"""SCL Core Python Bindings.

This module provides Python bindings for the scl-core Rust library.

Main Classes:
    - CSRF32, CSRF64: CSR sparse matrices (single/double precision)
    - CSCF32, CSCF64: CSC sparse matrices (single/double precision)
    - SpanF32, SpanF64, SpanI32, SpanI64: Typed memory views

Example Usage:

    import numpy as np
    import scipy.sparse as sp
    from scl._binding import CSRF64
    
    # Create from scipy
    scipy_mat = sp.random(1000, 1000, density=0.01, format='csr')
    csr = CSRF64.from_scipy(scipy_mat)
    
    # Basic properties
    print(csr.shape)    # (1000, 1000)
    print(csr.nnz)       # Number of non-zero elements
    print(csr.density)   # Density
    
    # Slicing operations
    sub = csr[100:200, 50:150]  # Row and column slicing
    
    # Stacking operations
    stacked = CSRF64.vstack([csr, csr])
    
    # Format conversion
    csc = csr.to_csc()
    dense = csr.to_dense()
    scipy_back = csr.to_scipy()
"""

from ._cffi import ffi, lib, FfiResult, ABI_VERSION, SPAN_FLAG_VIEW, SPAN_FLAG_ALIGNED, SPAN_FLAG_MUTABLE
from ._span import Span, SpanF32, SpanF64, SpanI32, SpanI64
from ._sparse import CSR, CSRF32, CSRF64, CSC, CSCF32, CSCF64

__all__ = [
    # FFI low-level
    "ffi",
    "lib", 
    "FfiResult",
    "ABI_VERSION",
    
    # Constants
    "SPAN_FLAG_VIEW",
    "SPAN_FLAG_ALIGNED",
    "SPAN_FLAG_MUTABLE",
    
    # Span classes
    "Span",
    "SpanF32",
    "SpanF64",
    "SpanI32",
    "SpanI64",
    
    # Sparse matrix classes
    "CSR",
    "CSRF32",
    "CSRF64",
    "CSC",
    "CSCF32",
    "CSCF64",
]

__version__ = "0.1.0"
