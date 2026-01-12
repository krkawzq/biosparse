"""BioSparse - High-performance sparse matrix computation library.

BioSparse is a high-performance sparse matrix computation
library for bioinformatics. The core is implemented in Rust and provides Python
bindings through CFFI.

Features:
    - High-performance CSR/CSC sparse matrices
    - Zero-copy memory views
    - Seamless interoperability with scipy.sparse
    - Multi-threaded parallel computation

Quick Start:

    import scipy.sparse as sp
    from biosparse import CSRF64
    
    # Create CSR matrix from scipy
    scipy_mat = sp.random(1000, 500, density=0.01, format='csr', dtype='float64')
    csr = CSRF64.from_scipy(scipy_mat)
    
    # Slicing operations
    sliced = csr[100:200, 50:150]
    
    # Convert back to scipy
    back = sliced.to_scipy()
"""

from ._binding import (
    # FFI
    ffi,
    lib,
    FfiResult,
    ABI_VERSION,
    
    # Constants
    SPAN_FLAG_VIEW,
    SPAN_FLAG_ALIGNED, 
    SPAN_FLAG_MUTABLE,
    
    # Span
    Span,
    SpanF32,
    SpanF64,
    SpanI32,
    SpanI64,
    
    # CSR
    CSR,
    CSRF32,
    CSRF64,
    
    # CSC
    CSC,
    CSCF32,
    CSCF64,
)

__all__ = [
    # FFI
    "ffi",
    "lib",
    "FfiResult",
    "ABI_VERSION",
    
    # Constants
    "SPAN_FLAG_VIEW",
    "SPAN_FLAG_ALIGNED",
    "SPAN_FLAG_MUTABLE",
    
    # Span
    "Span",
    "SpanF32",
    "SpanF64",
    "SpanI32",
    "SpanI64",
    
    # CSR
    "CSR",
    "CSRF32",
    "CSRF64",
    
    # CSC
    "CSC",
    "CSCF32",
    "CSCF64",
]

__version__ = "0.1.0"
