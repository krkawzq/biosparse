"""Numba extensions for BioSparse sparse matrices.

This module provides complete Numba JIT support for CSR/CSC sparse matrices,
enabling high-performance sparse matrix operations in nopython mode.

Features:
    - Type registration for CSR/CSC matrices (float32 and float64)
    - Automatic unbox/box for seamless Python <-> JIT conversion
    - Full method overloads for all matrix operations
    - Iterator support: `for values, indices in csr:`
    - Slice syntax: `csr[10:20, :]`
    - Stack operations: `hstack`, `vstack`
    - Conversion: `to_dense`, `to_coo`, `to_csc`/`to_csr`
    - NRT-based lifetime management for objects created in JIT

Usage:
    ```python
    from biosparse import CSRF64
    from numba import njit
    import scipy.sparse as sp
    
    # Create a sparse matrix
    mat = sp.random(1000, 1000, density=0.01, format='csr')
    csr = CSRF64.from_scipy(mat)
    
    # Use in JIT function
    @njit
    def process(csr):
        total = 0.0
        for values, indices in csr:
            total += values.sum()
        return total
    
    result = process(csr)
    ```

Note:
    This module requires numba to be installed.
    If numba is not available, importing this module will raise ImportError.
"""

# =============================================================================
# Register FFI symbols with LLVM JIT engine (non-Windows only)
# On Linux/macOS, numba's LLVM JIT cannot automatically resolve symbols from
# dynamically loaded CFFI libraries. We manually register them here.
# On Windows, symbol resolution works differently and this is not needed.
# =============================================================================
import sys

if sys.platform != 'win32':
    def _register_ffi_symbols():
        """Register all FFI function symbols with llvmlite's symbol table.
        
        Automatically discovers all function symbols from the CFFI lib module,
        excluding constants (uppercase names) and private symbols.
        """
        try:
            from llvmlite import binding as llvm
            from .._binding._cffi import ffi, lib
            
            # Auto-discover all FFI functions from lib
            # Filter out: private attrs (_), constants (UPPERCASE), and non-functions
            all_symbols = dir(lib)
            ffi_functions = [
                name for name in all_symbols
                if not name.startswith('_') and not name.isupper()
            ]
            
            for func_name in ffi_functions:
                try:
                    # Get function pointer from CFFI lib
                    func_ptr = ffi.addressof(lib, func_name)
                    addr = int(ffi.cast("uintptr_t", func_ptr))
                    # Register with LLVM
                    llvm.add_symbol(func_name, addr)
                except (TypeError, ffi.error):
                    # Skip non-function symbols (e.g., struct types)
                    pass
                    
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to register FFI symbols with LLVM: {e}")

    _register_ffi_symbols()

# Import types first (registers typeof_impl)
from ._types import (
    CSRType,
    CSCType,
    CSRFloat32Type,
    CSRFloat64Type,
    CSCFloat32Type,
    CSCFloat64Type,
    CSRIteratorType,
    CSCIteratorType,
)

# Import models (registers data layouts and attribute wrappers)
from . import _models

# Import FFI support (registers CFFI module and intrinsic functions)
from . import _ffi

# Import boxing (registers unbox/box functions)
from . import _boxing

# Import overloads (registers basic method and attribute overloads)
from . import _overloads

# Import iterators (registers getiter and iternext)
from . import _iterators

# Import operators (registers slice, stack operations)
from . import _operators

# Import conversions (registers to_dense, to_coo, to_csc/csr, clone)
from . import _conversions

# Import validation (registers validation and sorting methods)
from . import _validation


__all__ = [
    "CSRType",
    "CSCType",
    "CSRFloat32Type",
    "CSRFloat64Type",
    "CSCFloat32Type",
    "CSCFloat64Type",
    "CSRIteratorType",
    "CSCIteratorType",
]
