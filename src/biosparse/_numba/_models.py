"""Numba data models for CSR/CSC sparse matrices.

This module defines how sparse matrix types are laid out in memory during
JIT compilation.
"""

from numba import types
from numba.core import cgutils
from numba.extending import register_model, models, make_attribute_wrapper

from ._types import CSRType, CSCType, CSRIteratorType, CSCIteratorType


# =============================================================================
# CSR Data Model
# =============================================================================

@register_model(CSRType)
class CSRModel(models.StructModel):
    """Memory layout for CSR in compiled code.
    
    This struct is what Numba actually works with during JIT compilation.
    
    Fields:
        handle: FFI handle to the Rust CSR object (for FFI calls)
        meminfo: NRT memory info (for objects created in JIT)
        nrows: Number of rows (cached for fast access)
        ncols: Number of columns (cached for fast access)
        nnz: Number of non-zero elements (cached for fast access)
        values_ptrs: Pointer array where values_ptrs[i] points to row i's values
        indices_ptrs: Pointer array where indices_ptrs[i] points to row i's indices
        row_lens: Length array where row_lens[i] is the nnz count of row i
        owns_data: Whether this object owns the handle (for cleanup)
    """
    
    def __init__(self, dmm, fe_type):
        members = [
            # FFI handle - opaque pointer to Rust CSR<V, I>
            ('handle', types.voidptr),
            
            # NRT memory management - used when object created in JIT
            # NULL when object came from Python
            ('meminfo', types.MemInfoPointer(types.voidptr)),
            
            # Cached dimensions - avoid repeated FFI calls
            ('nrows', types.int64),
            ('ncols', types.int64),
            ('nnz', types.int64),
            
            # Pointer arrays for fast row access (prepared during unbox)
            # values_ptrs[i] = pointer to f32/f64 array of row i's values
            ('values_ptrs', types.CPointer(types.CPointer(fe_type.dtype))),
            # indices_ptrs[i] = pointer to i64 array of row i's column indices
            ('indices_ptrs', types.CPointer(types.CPointer(types.int64))),
            # row_lens[i] = number of non-zeros in row i
            ('row_lens', types.CPointer(types.intp)),
            
            # Ownership flag - determines if handle should be freed
            ('owns_data', types.boolean),
        ]
        super().__init__(dmm, fe_type, members)


@register_model(CSCType)
class CSCModel(models.StructModel):
    """Memory layout for CSC in compiled code.
    
    Similar to CSRModel but organized by columns.
    
    Fields:
        handle: FFI handle to the Rust CSC object
        meminfo: NRT memory info (for objects created in JIT)
        nrows: Number of rows (cached for fast access)
        ncols: Number of columns (cached for fast access)
        nnz: Number of non-zero elements (cached for fast access)
        values_ptrs: Pointer array where values_ptrs[j] points to column j's values
        indices_ptrs: Pointer array where indices_ptrs[j] points to column j's indices
        col_lens: Length array where col_lens[j] is the nnz count of column j
        owns_data: Whether this object owns the handle (for cleanup)
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
            ('owns_data', types.boolean),
        ]
        super().__init__(dmm, fe_type, members)


# =============================================================================
# Iterator Data Models
# =============================================================================

@register_model(CSRIteratorType)
class CSRIteratorModel(models.StructModel):
    """Memory layout for CSR iterator.

    The iterator maintains pointers to the parent CSR's data and tracks
    the current position via a pointer to allow mutation across iterations.

    Fields:
        values_ptrs: Reference to parent's values pointer array
        indices_ptrs: Reference to parent's indices pointer array
        row_lens: Reference to parent's row lengths array
        nrows: Total number of rows (from parent)
        index_ptr: Pointer to current iteration position (allows mutation)
    """

    def __init__(self, dmm, fe_type):
        members = [
            ('values_ptrs', types.CPointer(types.CPointer(fe_type.csr_type.dtype))),
            ('indices_ptrs', types.CPointer(types.CPointer(types.int64))),
            ('row_lens', types.CPointer(types.intp)),
            ('nrows', types.int64),
            # Use a pointer so we can mutate the index across iterations
            ('index_ptr', types.CPointer(types.int64)),
        ]
        super().__init__(dmm, fe_type, members)


@register_model(CSCIteratorType)
class CSCIteratorModel(models.StructModel):
    """Memory layout for CSC iterator.

    Similar to CSRIteratorModel but for columns.

    Fields:
        values_ptrs: Reference to parent's values pointer array
        indices_ptrs: Reference to parent's indices pointer array
        col_lens: Reference to parent's column lengths array
        ncols: Total number of columns (from parent)
        index_ptr: Pointer to current iteration position (allows mutation)
    """

    def __init__(self, dmm, fe_type):
        members = [
            ('values_ptrs', types.CPointer(types.CPointer(fe_type.csc_type.dtype))),
            ('indices_ptrs', types.CPointer(types.CPointer(types.int64))),
            ('col_lens', types.CPointer(types.intp)),
            ('ncols', types.int64),
            # Use a pointer so we can mutate the index across iterations
            ('index_ptr', types.CPointer(types.int64)),
        ]
        super().__init__(dmm, fe_type, members)


# =============================================================================
# Attribute Wrappers: Direct field access
# =============================================================================

# CSR attributes - allow csr.nrows, csr.ncols, etc.
make_attribute_wrapper(CSRType, 'handle', 'handle')
make_attribute_wrapper(CSRType, 'meminfo', 'meminfo')
make_attribute_wrapper(CSRType, 'nrows', 'nrows')
make_attribute_wrapper(CSRType, 'ncols', 'ncols')
make_attribute_wrapper(CSRType, 'nnz', 'nnz')
make_attribute_wrapper(CSRType, 'values_ptrs', 'values_ptrs')
make_attribute_wrapper(CSRType, 'indices_ptrs', 'indices_ptrs')
make_attribute_wrapper(CSRType, 'row_lens', 'row_lens')
make_attribute_wrapper(CSRType, 'owns_data', 'owns_data')

# CSC attributes - allow csc.nrows, csc.ncols, etc.
make_attribute_wrapper(CSCType, 'handle', 'handle')
make_attribute_wrapper(CSCType, 'meminfo', 'meminfo')
make_attribute_wrapper(CSCType, 'nrows', 'nrows')
make_attribute_wrapper(CSCType, 'ncols', 'ncols')
make_attribute_wrapper(CSCType, 'nnz', 'nnz')
make_attribute_wrapper(CSCType, 'values_ptrs', 'values_ptrs')
make_attribute_wrapper(CSCType, 'indices_ptrs', 'indices_ptrs')
make_attribute_wrapper(CSCType, 'col_lens', 'col_lens')
make_attribute_wrapper(CSCType, 'owns_data', 'owns_data')

# Iterator attributes - internal use only
make_attribute_wrapper(CSRIteratorType, 'values_ptrs', 'values_ptrs')
make_attribute_wrapper(CSRIteratorType, 'indices_ptrs', 'indices_ptrs')
make_attribute_wrapper(CSRIteratorType, 'row_lens', 'row_lens')
make_attribute_wrapper(CSRIteratorType, 'nrows', 'nrows')
make_attribute_wrapper(CSRIteratorType, 'index_ptr', 'index_ptr')

make_attribute_wrapper(CSCIteratorType, 'values_ptrs', 'values_ptrs')
make_attribute_wrapper(CSCIteratorType, 'indices_ptrs', 'indices_ptrs')
make_attribute_wrapper(CSCIteratorType, 'col_lens', 'col_lens')
make_attribute_wrapper(CSCIteratorType, 'ncols', 'ncols')
make_attribute_wrapper(CSCIteratorType, 'index_ptr', 'index_ptr')
