"""Numba type definitions for CSR/CSC sparse matrices.

This module defines the Numba type system representations for sparse matrices.
"""

from numba import types
from numba.core import cgutils
from numba.extending import typeof_impl, register_model, models, make_attribute_wrapper

# Import the Python classes for type registration
from .._binding._sparse import CSR, CSC, CSRF32, CSRF64, CSCF32, CSCF64


# =============================================================================
# CSR Type
# =============================================================================

class CSRType(types.Type):
    """Numba type for CSR sparse matrix.
    
    This type represents a CSR matrix in Numba's type system.
    The actual data layout is defined in CSRModel.
    """
    
    def __init__(self, dtype):
        self.dtype = dtype
        super().__init__(name=f'CSR[{dtype}]')
    
    @property
    def key(self):
        return self.dtype
    
    def __hash__(self):
        return hash(('CSRType', self.dtype))
    
    def __eq__(self, other):
        if isinstance(other, CSRType):
            return self.dtype == other.dtype
        return False


class CSCType(types.Type):
    """Numba type for CSC sparse matrix.
    
    This type represents a CSC matrix in Numba's type system.
    The actual data layout is defined in CSCModel.
    """
    
    def __init__(self, dtype):
        self.dtype = dtype
        super().__init__(name=f'CSC[{dtype}]')
    
    @property
    def key(self):
        return self.dtype
    
    def __hash__(self):
        return hash(('CSCType', self.dtype))
    
    def __eq__(self, other):
        if isinstance(other, CSCType):
            return self.dtype == other.dtype
        return False


# Concrete type instances
CSRFloat32Type = CSRType(types.float32)
CSRFloat64Type = CSRType(types.float64)
CSCFloat32Type = CSCType(types.float32)
CSCFloat64Type = CSCType(types.float64)


# =============================================================================
# Type Inference: Python object -> Numba type
# =============================================================================

@typeof_impl.register(CSRF32)
def typeof_csrf32(val, c):
    """Infer Numba type for CSRF32 Python object."""
    return CSRFloat32Type


@typeof_impl.register(CSRF64)
def typeof_csrf64(val, c):
    """Infer Numba type for CSRF64 Python object."""
    return CSRFloat64Type


@typeof_impl.register(CSCF32)
def typeof_cscf32(val, c):
    """Infer Numba type for CSCF32 Python object."""
    return CSCFloat32Type


@typeof_impl.register(CSCF64)
def typeof_cscf64(val, c):
    """Infer Numba type for CSCF64 Python object."""
    return CSCFloat64Type


# =============================================================================
# Data Models: Memory layout in compiled code
# =============================================================================

@register_model(CSRType)
class CSRModel(models.StructModel):
    """Memory layout for CSR in compiled code.
    
    This struct is what Numba actually works with during JIT compilation.
    It contains:
        - handle: The FFI handle (for calling FFI functions)
        - meminfo: NRT memory info (for reference counting and destruction)
        - nrows, ncols: Matrix dimensions
        - values_ptrs: Array of pointers to row value arrays
        - indices_ptrs: Array of pointers to row index arrays  
        - row_lens: Array of row lengths (number of non-zeros per row)
    """
    
    def __init__(self, dmm, fe_type):
        members = [
            # FFI handle (needed for FFI calls like hstack, slice, etc.)
            ('handle', types.voidptr),
            # NRT memory info for lifetime management
            ('meminfo', types.MemInfoPointer(types.voidptr)),
            # Dimensions
            ('nrows', types.int64),
            ('ncols', types.int64),
            ('nnz', types.int64),
            # Pointer arrays for fast row access
            # values_ptrs[i] points to the values of row i
            ('values_ptrs', types.CPointer(types.CPointer(fe_type.dtype))),
            # indices_ptrs[i] points to the column indices of row i
            ('indices_ptrs', types.CPointer(types.CPointer(types.int64))),
            # row_lens[i] is the number of non-zeros in row i
            ('row_lens', types.CPointer(types.intp)),
        ]
        super().__init__(dmm, fe_type, members)


@register_model(CSCType)
class CSCModel(models.StructModel):
    """Memory layout for CSC in compiled code.
    
    Similar to CSRModel but organized by columns.
    """
    
    def __init__(self, dmm, fe_type):
        members = [
            ('handle', types.voidptr),
            ('meminfo', types.MemInfoPointer(types.voidptr)),
            ('nrows', types.int64),
            ('ncols', types.int64),
            ('nnz', types.int64),
            # col_ptrs[j] points to the values of column j
            ('values_ptrs', types.CPointer(types.CPointer(fe_type.dtype))),
            # indices_ptrs[j] points to the row indices of column j
            ('indices_ptrs', types.CPointer(types.CPointer(types.int64))),
            # col_lens[j] is the number of non-zeros in column j
            ('col_lens', types.CPointer(types.intp)),
        ]
        super().__init__(dmm, fe_type, members)


# =============================================================================
# Attribute Wrappers: Allow accessing struct fields as attributes
# =============================================================================

# CSR attributes
make_attribute_wrapper(CSRType, 'handle', 'handle')
make_attribute_wrapper(CSRType, 'nrows', 'nrows')
make_attribute_wrapper(CSRType, 'ncols', 'ncols')
make_attribute_wrapper(CSRType, 'nnz', 'nnz')
make_attribute_wrapper(CSRType, 'values_ptrs', 'values_ptrs')
make_attribute_wrapper(CSRType, 'indices_ptrs', 'indices_ptrs')
make_attribute_wrapper(CSRType, 'row_lens', 'row_lens')

# CSC attributes
make_attribute_wrapper(CSCType, 'handle', 'handle')
make_attribute_wrapper(CSCType, 'nrows', 'nrows')
make_attribute_wrapper(CSCType, 'ncols', 'ncols')
make_attribute_wrapper(CSCType, 'nnz', 'nnz')
make_attribute_wrapper(CSCType, 'values_ptrs', 'values_ptrs')
make_attribute_wrapper(CSCType, 'indices_ptrs', 'indices_ptrs')
make_attribute_wrapper(CSCType, 'col_lens', 'col_lens')
