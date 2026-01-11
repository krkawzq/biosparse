"""Method and attribute overloads for CSR/CSC types.

This module provides Numba implementations for all CSR/CSC methods and
properties, enabling efficient JIT compilation.
"""

import numpy as np
from numba import types
from numba.core import cgutils
from numba.extending import overload_method, overload_attribute, overload

from ._types import CSRType, CSCType
from ._ffi import _make_array_from_ptr_f64, _make_array_from_ptr_f32, _make_array_from_ptr_i64


# =============================================================================
# CSR: Basic Properties
# =============================================================================

@overload_attribute(CSRType, 'shape')
def csr_shape_get(csr):
    """Get matrix shape as (nrows, ncols) tuple."""
    def getter(csr):
        return (csr.nrows, csr.ncols)
    return getter


@overload_attribute(CSRType, 'density')
def csr_density_get(csr):
    """Calculate density (proportion of non-zero elements)."""
    def getter(csr):
        total = csr.nrows * csr.ncols
        if total == 0:
            return 0.0
        return float(csr.nnz) / float(total)
    return getter


@overload_attribute(CSRType, 'sparsity')
def csr_sparsity_get(csr):
    """Calculate sparsity (proportion of zero elements)."""
    def getter(csr):
        total = csr.nrows * csr.ncols
        if total == 0:
            return 1.0
        return 1.0 - (float(csr.nnz) / float(total))
    return getter


@overload_attribute(CSRType, 'is_empty')
def csr_is_empty_get(csr):
    """Check if matrix has zero rows or columns."""
    def getter(csr):
        return csr.nrows == 0 or csr.ncols == 0
    return getter


@overload_attribute(CSRType, 'is_zero')
def csr_is_zero_get(csr):
    """Check if matrix has no non-zero elements."""
    def getter(csr):
        return csr.nnz == 0
    return getter


# =============================================================================
# CSC: Basic Properties
# =============================================================================

@overload_attribute(CSCType, 'shape')
def csc_shape_get(csc):
    """Get matrix shape as (nrows, ncols) tuple."""
    def getter(csc):
        return (csc.nrows, csc.ncols)
    return getter


@overload_attribute(CSCType, 'density')
def csc_density_get(csc):
    """Calculate density."""
    def getter(csc):
        total = csc.nrows * csc.ncols
        if total == 0:
            return 0.0
        return float(csc.nnz) / float(total)
    return getter


@overload_attribute(CSCType, 'sparsity')
def csc_sparsity_get(csc):
    """Calculate sparsity."""
    def getter(csc):
        total = csc.nrows * csc.ncols
        if total == 0:
            return 1.0
        return 1.0 - (float(csc.nnz) / float(total))
    return getter


@overload_attribute(CSCType, 'is_empty')
def csc_is_empty_get(csc):
    """Check if matrix is empty."""
    def getter(csc):
        return csc.nrows == 0 or csc.ncols == 0
    return getter


@overload_attribute(CSCType, 'is_zero')
def csc_is_zero_get(csc):
    """Check if matrix has no non-zeros."""
    def getter(csc):
        return csc.nnz == 0
    return getter


# =============================================================================
# len() builtin support
# =============================================================================

@overload(len)
def csr_len_overload(csr):
    """len(csr) returns number of rows."""
    if isinstance(csr, CSRType):
        def impl(csr):
            return csr.nrows
        return impl


@overload(len)
def csc_len_overload(csc):
    """len(csc) returns number of columns."""
    if isinstance(csc, CSCType):
        def impl(csc):
            return csc.ncols
        return impl


# =============================================================================
# CSR: Row Access Methods
# =============================================================================

@overload_method(CSRType, 'row_len')
def csr_row_len_impl(csr, row_idx):
    """Get the number of non-zeros in a row."""
    def impl(csr, row_idx):
        if row_idx < 0 or row_idx >= csr.nrows:
            raise IndexError("row index out of bounds")
        return csr.row_lens[row_idx]
    return impl


@overload_method(CSRType, 'row_to_numpy')
def csr_row_to_numpy_impl(csr, row_idx, copy=False):
    """Get row data as numpy arrays.
    
    Note: copy parameter is ignored in Numba (always returns views).
    """
    # Check dtype at overload time and use appropriate intrinsic
    if csr.dtype == types.float64:
        def impl(csr, row_idx, copy=False):
            if row_idx < 0 or row_idx >= csr.nrows:
                raise IndexError("row index out of bounds")
            val_ptr = csr.values_ptrs[row_idx]
            idx_ptr = csr.indices_ptrs[row_idx]
            length = csr.row_lens[row_idx]
            values = _make_array_from_ptr_f64(val_ptr, length)
            indices = _make_array_from_ptr_i64(idx_ptr, length)
            return (values, indices)
        return impl
    else:  # float32
        def impl(csr, row_idx, copy=False):
            if row_idx < 0 or row_idx >= csr.nrows:
                raise IndexError("row index out of bounds")
            val_ptr = csr.values_ptrs[row_idx]
            idx_ptr = csr.indices_ptrs[row_idx]
            length = csr.row_lens[row_idx]
            values = _make_array_from_ptr_f32(val_ptr, length)
            indices = _make_array_from_ptr_i64(idx_ptr, length)
            return (values, indices)
        return impl


# =============================================================================
# CSC: Column Access Methods
# =============================================================================

@overload_method(CSCType, 'col_len')
def csc_col_len_impl(csc, col_idx):
    """Get the number of non-zeros in a column."""
    def impl(csc, col_idx):
        if col_idx < 0 or col_idx >= csc.ncols:
            raise IndexError("column index out of bounds")
        return csc.col_lens[col_idx]
    return impl


@overload_method(CSCType, 'col_to_numpy')
def csc_col_to_numpy_impl(csc, col_idx, copy=False):
    """Get column data as numpy arrays.

    Note: copy parameter is ignored in Numba (always returns views).
    """
    if csc.dtype == types.float64:
        def impl(csc, col_idx, copy=False):
            if col_idx < 0 or col_idx >= csc.ncols:
                raise IndexError("column index out of bounds")
            val_ptr = csc.values_ptrs[col_idx]
            idx_ptr = csc.indices_ptrs[col_idx]
            length = csc.col_lens[col_idx]
            values = _make_array_from_ptr_f64(val_ptr, length)
            indices = _make_array_from_ptr_i64(idx_ptr, length)
            return (values, indices)
        return impl
    else:  # float32
        def impl(csc, col_idx, copy=False):
            if col_idx < 0 or col_idx >= csc.ncols:
                raise IndexError("column index out of bounds")
            val_ptr = csc.values_ptrs[col_idx]
            idx_ptr = csc.indices_ptrs[col_idx]
            length = csc.col_lens[col_idx]
            values = _make_array_from_ptr_f32(val_ptr, length)
            indices = _make_array_from_ptr_i64(idx_ptr, length)
            return (values, indices)
        return impl


# =============================================================================
# CSR/CSC: Simplified row()/col() methods
# =============================================================================

@overload_method(CSRType, 'row')
def csr_row_impl(csr, row_idx, copy=False):
    """CSR.row() - simplified method, equivalent to row_to_numpy()."""
    # Directly reuse the row_to_numpy implementation
    return csr_row_to_numpy_impl(csr, row_idx, copy)


@overload_method(CSRType, 'col')
def csr_col_impl(csr, col_idx):
    """CSR.col() - non-contiguous dimension, requires traversing all rows."""
    if csr.dtype == types.float64:
        def impl(csr, col_idx):
            # Bounds check
            if col_idx < 0 or col_idx >= csr.ncols:
                raise IndexError("column index out of bounds")

            # Pre-allocate maximum possible size
            temp_values = np.empty(csr.nrows, dtype=np.float64)
            temp_indices = np.empty(csr.nrows, dtype=np.int64)
            count = 0

            # Traverse each row and use binary search
            for i in range(csr.nrows):
                values, indices = csr.row_to_numpy(i)
                # Binary search for the column
                pos = np.searchsorted(indices, col_idx)
                if pos < len(indices) and indices[pos] == col_idx:
                    temp_values[count] = values[pos]
                    temp_indices[count] = i
                    count += 1

            # Return exact-size arrays
            return (temp_values[:count].copy(), temp_indices[:count].copy())
        return impl
    else:  # float32
        def impl(csr, col_idx):
            if col_idx < 0 or col_idx >= csr.ncols:
                raise IndexError("column index out of bounds")

            temp_values = np.empty(csr.nrows, dtype=np.float32)
            temp_indices = np.empty(csr.nrows, dtype=np.int64)
            count = 0

            for i in range(csr.nrows):
                values, indices = csr.row_to_numpy(i)
                pos = np.searchsorted(indices, col_idx)
                if pos < len(indices) and indices[pos] == col_idx:
                    temp_values[count] = values[pos]
                    temp_indices[count] = i
                    count += 1

            return (temp_values[:count].copy(), temp_indices[:count].copy())
        return impl


@overload_method(CSCType, 'col')
def csc_col_impl(csc, col_idx, copy=False):
    """CSC.col() - simplified method, equivalent to col_to_numpy()."""
    # Directly reuse the col_to_numpy implementation
    return csc_col_to_numpy_impl(csc, col_idx, copy)


@overload_method(CSCType, 'row')
def csc_row_impl(csc, row_idx):
    """CSC.row() - non-contiguous dimension, requires traversing all columns."""
    if csc.dtype == types.float64:
        def impl(csc, row_idx):
            # Bounds check
            if row_idx < 0 or row_idx >= csc.nrows:
                raise IndexError("row index out of bounds")

            # Pre-allocate maximum possible size
            temp_values = np.empty(csc.ncols, dtype=np.float64)
            temp_indices = np.empty(csc.ncols, dtype=np.int64)
            count = 0

            # Traverse each column and use binary search
            for j in range(csc.ncols):
                values, indices = csc.col_to_numpy(j)
                # Binary search for the row
                pos = np.searchsorted(indices, row_idx)
                if pos < len(indices) and indices[pos] == row_idx:
                    temp_values[count] = values[pos]
                    temp_indices[count] = j
                    count += 1

            # Return exact-size arrays
            return (temp_values[:count].copy(), temp_indices[:count].copy())
        return impl
    else:  # float32
        def impl(csc, row_idx):
            if row_idx < 0 or row_idx >= csc.nrows:
                raise IndexError("row index out of bounds")

            temp_values = np.empty(csc.ncols, dtype=np.float32)
            temp_indices = np.empty(csc.ncols, dtype=np.int64)
            count = 0

            for j in range(csc.ncols):
                values, indices = csc.col_to_numpy(j)
                pos = np.searchsorted(indices, row_idx)
                if pos < len(indices) and indices[pos] == row_idx:
                    temp_values[count] = values[pos]
                    temp_indices[count] = j
                    count += 1

            return (temp_values[:count].copy(), temp_indices[:count].copy())
        return impl


# =============================================================================
# CSR/CSC: Safe get() method
# =============================================================================

@overload_method(CSRType, 'get')
def csr_get_impl(csr, row_idx, col_idx, default=0.0):
    """CSR.get() - safe element access with default value."""
    if csr.dtype == types.float64:
        def impl(csr, row_idx, col_idx, default=0.0):
            # Bounds check - return default instead of raising
            if row_idx < 0 or row_idx >= csr.nrows:
                return default
            if col_idx < 0 or col_idx >= csr.ncols:
                return default

            # Get row data
            values, indices = csr.row_to_numpy(row_idx)

            # Binary search
            pos = np.searchsorted(indices, col_idx)
            if pos < len(indices) and indices[pos] == col_idx:
                return values[pos]
            return default
        return impl
    else:  # float32
        def impl(csr, row_idx, col_idx, default=0.0):
            if row_idx < 0 or row_idx >= csr.nrows:
                return default
            if col_idx < 0 or col_idx >= csr.ncols:
                return default

            values, indices = csr.row_to_numpy(row_idx)
            pos = np.searchsorted(indices, col_idx)
            if pos < len(indices) and indices[pos] == col_idx:
                return values[pos]
            return default
        return impl


@overload_method(CSCType, 'get')
def csc_get_impl(csc, row_idx, col_idx, default=0.0):
    """CSC.get() - safe element access with default value."""
    if csc.dtype == types.float64:
        def impl(csc, row_idx, col_idx, default=0.0):
            # Bounds check - return default instead of raising
            if row_idx < 0 or row_idx >= csc.nrows:
                return default
            if col_idx < 0 or col_idx >= csc.ncols:
                return default

            # Get column data (contiguous dimension for CSC)
            values, indices = csc.col_to_numpy(col_idx)

            # Binary search for row
            pos = np.searchsorted(indices, row_idx)
            if pos < len(indices) and indices[pos] == row_idx:
                return values[pos]
            return default
        return impl
    else:  # float32
        def impl(csc, row_idx, col_idx, default=0.0):
            if row_idx < 0 or row_idx >= csc.nrows:
                return default
            if col_idx < 0 or col_idx >= csc.ncols:
                return default

            values, indices = csc.col_to_numpy(col_idx)
            pos = np.searchsorted(indices, row_idx)
            if pos < len(indices) and indices[pos] == row_idx:
                return values[pos]
            return default
        return impl

