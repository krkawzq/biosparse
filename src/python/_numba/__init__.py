"""Numba extensions for SCL sparse matrices.

This module provides Numba JIT support for CSR/CSC sparse matrices.
It enables efficient compilation of sparse matrix operations.

Features:
    - Type registration for CSR/CSC matrices
    - Automatic unbox/box for JIT function arguments/returns
    - Method overloads for data access, slicing, stacking
    - Iterator support for `for row in csr:` syntax
    - NRT-based lifetime management

Usage:
    # Simply import this module to enable Numba support
    import scl._numba
    
    # Then use CSR/CSC in JIT functions
    from numba import njit
    from scl import CSRF64
    
    @njit
    def process(csr):
        for values, indices in csr:
            # values and indices are typed arrays
            pass

Note:
    This module requires numba to be installed.
    If numba is not available, importing this module will raise ImportError.
"""

# Import types first (registers typeof_impl)
from ._types import (
    CSRType,
    CSCType,
    CSRFloat32Type,
    CSRFloat64Type,
    CSCFloat32Type,
    CSCFloat64Type,
)

# Import boxing (registers unbox/box)
from . import _boxing

# Import overloads (registers method overloads and iterators)
from . import _overloads

__all__ = [
    "CSRType",
    "CSCType",
    "CSRFloat32Type",
    "CSRFloat64Type",
    "CSCFloat32Type",
    "CSCFloat64Type",
]
