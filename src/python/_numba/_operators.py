"""Operators and complex methods for CSR/CSC types.

This module implements slicing, stacking, and other structural operations.
"""

import numpy as np
import operator
from numba import types
from numba.core import cgutils
from numba.extending import overload_method, overload, intrinsic
import llvmlite.ir as lir

from ._types import CSRType, CSCType
from ._ffi import (
    _alloca_voidptr,
    _load_voidptr,
    _alloca_voidptr_array,
    ffi_csr_f64_slice_rows,
    ffi_csr_f64_slice_cols,
    ffi_csr_f32_slice_rows,
    ffi_csr_f32_slice_cols,
    ffi_csc_f64_slice_rows,
    ffi_csc_f64_slice_cols,
    ffi_csc_f32_slice_rows,
    ffi_csc_f32_slice_cols,
    ffi_csr_f64_hstack,
    ffi_csr_f64_vstack,
    ffi_csr_f32_hstack,
    ffi_csr_f32_vstack,
    ffi_csc_f64_hstack,
    ffi_csc_f64_vstack,
    ffi_csc_f32_hstack,
    ffi_csc_f32_vstack,
)

# Import optimization intrinsics
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from optim import assume, unlikely


# =============================================================================
# Helper: Create CSR/CSC from handle in JIT
# =============================================================================

@intrinsic
def _create_csr_from_handle_f64(typingctx, handle_ty, owns_ty):
    """Create a CSRF64 object from a handle in JIT code.

    Note: This creates a minimal CSR struct with just the handle.
    Dimensions will be fetched from FFI when accessed via shape/nrows/etc.
    The pointer arrays are set to NULL - row_to_numpy uses FFI directly.
    """
    from ._types import CSRFloat64Type
    sig = CSRFloat64Type(types.voidptr, types.boolean)

    def codegen(context, builder, sig, args):
        [handle, owns] = args

        # Allocate CSR struct
        csr_type = sig.return_type
        csr = cgutils.create_struct_proxy(csr_type)(context, builder)

        # Set handle
        csr.handle = handle

        # Set ownership
        csr.owns_data = owns

        # Call FFI to get dimensions
        # Function signature: int64 csr_f64_rows(void* handle)
        fnty_i64 = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()])

        fn_rows = cgutils.get_or_insert_function(builder.module, fnty_i64, "csr_f64_rows")
        fn_cols = cgutils.get_or_insert_function(builder.module, fnty_i64, "csr_f64_cols")
        fn_nnz = cgutils.get_or_insert_function(builder.module, fnty_i64, "csr_f64_nnz")

        csr.nrows = builder.call(fn_rows, [handle])
        csr.ncols = builder.call(fn_cols, [handle])
        csr.nnz = builder.call(fn_nnz, [handle])

        # Set pointers to NULL - we don't prepare them for JIT-created objects
        # The row_to_numpy method should use FFI directly for these
        null_ptr = context.get_constant_null(types.voidptr)
        csr.values_ptrs = builder.bitcast(null_ptr, csr.values_ptrs.type)
        csr.indices_ptrs = builder.bitcast(null_ptr, csr.indices_ptrs.type)
        csr.row_lens = builder.bitcast(null_ptr, csr.row_lens.type)

        # Set meminfo to NULL
        csr.meminfo = context.get_constant_null(types.MemInfoPointer(types.voidptr))

        return csr._getvalue()

    return sig, codegen


@intrinsic
def _create_csr_from_handle_f32(typingctx, handle_ty, owns_ty):
    """Create a CSRF32 object from a handle in JIT code."""
    from ._types import CSRFloat32Type
    sig = CSRFloat32Type(types.voidptr, types.boolean)

    def codegen(context, builder, sig, args):
        [handle, owns] = args
        csr_type = sig.return_type
        csr = cgutils.create_struct_proxy(csr_type)(context, builder)
        csr.handle = handle
        csr.owns_data = owns

        # Call FFI to get dimensions
        fnty_i64 = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()])
        fn_rows = cgutils.get_or_insert_function(builder.module, fnty_i64, "csr_f32_rows")
        fn_cols = cgutils.get_or_insert_function(builder.module, fnty_i64, "csr_f32_cols")
        fn_nnz = cgutils.get_or_insert_function(builder.module, fnty_i64, "csr_f32_nnz")

        csr.nrows = builder.call(fn_rows, [handle])
        csr.ncols = builder.call(fn_cols, [handle])
        csr.nnz = builder.call(fn_nnz, [handle])

        null_ptr = context.get_constant_null(types.voidptr)
        csr.values_ptrs = builder.bitcast(null_ptr, csr.values_ptrs.type)
        csr.indices_ptrs = builder.bitcast(null_ptr, csr.indices_ptrs.type)
        csr.row_lens = builder.bitcast(null_ptr, csr.row_lens.type)
        csr.meminfo = context.get_constant_null(types.MemInfoPointer(types.voidptr))
        return csr._getvalue()

    return sig, codegen


@intrinsic
def _create_csc_from_handle_f64(typingctx, handle_ty, owns_ty):
    """Create a CSCF64 object from a handle in JIT code."""
    from ._types import CSCFloat64Type
    sig = CSCFloat64Type(types.voidptr, types.boolean)

    def codegen(context, builder, sig, args):
        [handle, owns] = args
        csc_type = sig.return_type
        csc = cgutils.create_struct_proxy(csc_type)(context, builder)
        csc.handle = handle
        csc.owns_data = owns

        # Call FFI to get dimensions
        fnty_i64 = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()])
        fn_rows = cgutils.get_or_insert_function(builder.module, fnty_i64, "csc_f64_rows")
        fn_cols = cgutils.get_or_insert_function(builder.module, fnty_i64, "csc_f64_cols")
        fn_nnz = cgutils.get_or_insert_function(builder.module, fnty_i64, "csc_f64_nnz")

        csc.nrows = builder.call(fn_rows, [handle])
        csc.ncols = builder.call(fn_cols, [handle])
        csc.nnz = builder.call(fn_nnz, [handle])

        null_ptr = context.get_constant_null(types.voidptr)
        csc.values_ptrs = builder.bitcast(null_ptr, csc.values_ptrs.type)
        csc.indices_ptrs = builder.bitcast(null_ptr, csc.indices_ptrs.type)
        csc.col_lens = builder.bitcast(null_ptr, csc.col_lens.type)
        csc.meminfo = context.get_constant_null(types.MemInfoPointer(types.voidptr))
        return csc._getvalue()

    return sig, codegen


@intrinsic
def _create_csc_from_handle_f32(typingctx, handle_ty, owns_ty):
    """Create a CSCF32 object from a handle in JIT code."""
    from ._types import CSCFloat32Type
    sig = CSCFloat32Type(types.voidptr, types.boolean)

    def codegen(context, builder, sig, args):
        [handle, owns] = args
        csc_type = sig.return_type
        csc = cgutils.create_struct_proxy(csc_type)(context, builder)
        csc.handle = handle
        csc.owns_data = owns

        # Call FFI to get dimensions
        fnty_i64 = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()])
        fn_rows = cgutils.get_or_insert_function(builder.module, fnty_i64, "csc_f32_rows")
        fn_cols = cgutils.get_or_insert_function(builder.module, fnty_i64, "csc_f32_cols")
        fn_nnz = cgutils.get_or_insert_function(builder.module, fnty_i64, "csc_f32_nnz")

        csc.nrows = builder.call(fn_rows, [handle])
        csc.ncols = builder.call(fn_cols, [handle])
        csc.nnz = builder.call(fn_nnz, [handle])

        null_ptr = context.get_constant_null(types.voidptr)
        csc.values_ptrs = builder.bitcast(null_ptr, csc.values_ptrs.type)
        csc.indices_ptrs = builder.bitcast(null_ptr, csc.indices_ptrs.type)
        csc.col_lens = builder.bitcast(null_ptr, csc.col_lens.type)
        csc.meminfo = context.get_constant_null(types.MemInfoPointer(types.voidptr))
        return csc._getvalue()

    return sig, codegen


# =============================================================================
# CSR: Slice Rows
# =============================================================================

@overload_method(CSRType, 'slice_rows')
def csr_slice_rows_overload(csr, start, end):
    """Slice rows [start:end) and return new CSR matrix."""
    if csr.dtype == types.float64:
        def impl(csr, start, end):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csr_f64_slice_rows(csr.handle, start, end, out_handle_ptr)
            if result != 0:
                raise RuntimeError("slice_rows failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csr_from_handle_f64(new_handle, True)
        return impl
    else:  # float32
        def impl(csr, start, end):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csr_f32_slice_rows(csr.handle, start, end, out_handle_ptr)
            if result != 0:
                raise RuntimeError("slice_rows failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csr_from_handle_f32(new_handle, True)
        return impl


# =============================================================================
# CSR: Slice Cols
# =============================================================================

@overload_method(CSRType, 'slice_cols')
def csr_slice_cols_overload(csr, start, end):
    """Slice columns [start:end) and return new CSR matrix."""
    if csr.dtype == types.float64:
        def impl(csr, start, end):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csr_f64_slice_cols(csr.handle, start, end, out_handle_ptr)
            if result != 0:
                raise RuntimeError("slice_cols failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csr_from_handle_f64(new_handle, True)
        return impl
    else:  # float32
        def impl(csr, start, end):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csr_f32_slice_cols(csr.handle, start, end, out_handle_ptr)
            if result != 0:
                raise RuntimeError("slice_cols failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csr_from_handle_f32(new_handle, True)
        return impl


# =============================================================================
# CSC: Slice Rows
# =============================================================================

@overload_method(CSCType, 'slice_rows')
def csc_slice_rows_overload(csc, start, end):
    """Slice rows [start:end) and return new CSC matrix."""
    if csc.dtype == types.float64:
        def impl(csc, start, end):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csc_f64_slice_rows(csc.handle, start, end, out_handle_ptr)
            if result != 0:
                raise RuntimeError("slice_rows failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csc_from_handle_f64(new_handle, True)
        return impl
    else:  # float32
        def impl(csc, start, end):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csc_f32_slice_rows(csc.handle, start, end, out_handle_ptr)
            if result != 0:
                raise RuntimeError("slice_rows failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csc_from_handle_f32(new_handle, True)
        return impl


# =============================================================================
# CSC: Slice Cols
# =============================================================================

@overload_method(CSCType, 'slice_cols')
def csc_slice_cols_overload(csc, start, end):
    """Slice columns [start:end) and return new CSC matrix."""
    if csc.dtype == types.float64:
        def impl(csc, start, end):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csc_f64_slice_cols(csc.handle, start, end, out_handle_ptr)
            if result != 0:
                raise RuntimeError("slice_cols failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csc_from_handle_f64(new_handle, True)
        return impl
    else:  # float32
        def impl(csc, start, end):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csc_f32_slice_cols(csc.handle, start, end, out_handle_ptr)
            if result != 0:
                raise RuntimeError("slice_cols failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csc_from_handle_f32(new_handle, True)
        return impl


# =============================================================================
# CSR: __getitem__ (slice syntax)
# =============================================================================

@overload(operator.getitem)
def csr_getitem_overload(csr, key):
    """Support csr[i:j], csr[i:j, k:l] slice syntax, and csr[i, j] element access."""
    if not isinstance(csr, CSRType):
        return None

    # Handle single slice: csr[10:20]
    if isinstance(key, types.SliceType):
        def impl(csr, key):
            start = 0 if key.start is None else key.start
            stop = csr.nrows if key.stop is None else key.stop
            return csr.slice_rows(start, stop)
        return impl

    # Handle tuple of slices or integers: csr[10:20, 5:15] or csr[i, j]
    elif isinstance(key, (types.Tuple, types.UniTuple)):
        if len(key) == 2:
            row_key_type, col_key_type = key.types if hasattr(key, 'types') else (key[0], key[1])

            # Single element access: both are integers
            if isinstance(row_key_type, types.Integer) and isinstance(col_key_type, types.Integer):
                if csr.dtype == types.float64:
                    def impl(csr, key):
                        row_idx, col_idx = key

                        # Bounds check
                        if row_idx < 0 or row_idx >= csr.nrows:
                            raise IndexError("row index out of bounds")
                        if col_idx < 0 or col_idx >= csr.ncols:
                            raise IndexError("column index out of bounds")

                        # Get row data
                        values, indices = csr.row_to_numpy(row_idx)
                        row_len = len(indices)

                        # Assume valid row data
                        assume(row_len >= 0)
                        assume(len(values) == row_len)

                        # Binary search for the column
                        pos = np.searchsorted(indices, col_idx)

                        # Sparse position is more common than dense
                        if unlikely(pos < row_len and indices[pos] == col_idx):
                            return values[pos]
                        else:
                            return 0.0
                    return impl
                else:  # float32
                    def impl(csr, key):
                        row_idx, col_idx = key

                        if row_idx < 0 or row_idx >= csr.nrows:
                            raise IndexError("row index out of bounds")
                        if col_idx < 0 or col_idx >= csr.ncols:
                            raise IndexError("column index out of bounds")

                        values, indices = csr.row_to_numpy(row_idx)
                        row_len = len(indices)

                        assume(row_len >= 0)
                        assume(len(values) == row_len)

                        pos = np.searchsorted(indices, col_idx)

                        if unlikely(pos < row_len and indices[pos] == col_idx):
                            return values[pos]
                        else:
                            return 0.0
                    return impl

            # Both are slices
            elif isinstance(row_key_type, types.SliceType) and isinstance(col_key_type, types.SliceType):
                def impl(csr, key):
                    row_slice, col_slice = key

                    # Apply row slice
                    row_start = 0 if row_slice.start is None else row_slice.start
                    row_stop = csr.nrows if row_slice.stop is None else row_slice.stop
                    result = csr.slice_rows(row_start, row_stop)

                    # Apply column slice
                    col_start = 0 if col_slice.start is None else col_slice.start
                    col_stop = result.ncols if col_slice.stop is None else col_slice.stop
                    result = result.slice_cols(col_start, col_stop)

                    return result
                return impl

            # Row is slice, col is colon (keep all columns)
            elif isinstance(row_key_type, types.SliceType):
                def impl(csr, key):
                    row_slice = key[0]
                    start = 0 if row_slice.start is None else row_slice.start
                    stop = csr.nrows if row_slice.stop is None else row_slice.stop
                    return csr.slice_rows(start, stop)
                return impl

            # Col is slice, row is colon (keep all rows)
            elif isinstance(col_key_type, types.SliceType):
                def impl(csr, key):
                    col_slice = key[1]
                    start = 0 if col_slice.start is None else col_slice.start
                    stop = csr.ncols if col_slice.stop is None else col_slice.stop
                    return csr.slice_cols(start, stop)
                return impl


# =============================================================================
# CSC: __getitem__ (slice syntax)
# =============================================================================

@overload(operator.getitem)
def csc_getitem_overload(csc, key):
    """Support csc[i:j], csc[i:j, k:l] slice syntax, and csc[i, j] element access."""
    if not isinstance(csc, CSCType):
        return None

    # Handle single slice: csc[10:20] (columns)
    if isinstance(key, types.SliceType):
        def impl(csc, key):
            start = 0 if key.start is None else key.start
            stop = csc.ncols if key.stop is None else key.stop
            return csc.slice_cols(start, stop)
        return impl

    # Handle tuple of slices or integers
    elif isinstance(key, (types.Tuple, types.UniTuple)):
        if len(key) == 2:
            row_key_type, col_key_type = key.types if hasattr(key, 'types') else (key[0], key[1])

            # Single element access: both are integers
            if isinstance(row_key_type, types.Integer) and isinstance(col_key_type, types.Integer):
                if csc.dtype == types.float64:
                    def impl(csc, key):
                        row_idx, col_idx = key

                        # Bounds check
                        if row_idx < 0 or row_idx >= csc.nrows:
                            raise IndexError("row index out of bounds")
                        if col_idx < 0 or col_idx >= csc.ncols:
                            raise IndexError("column index out of bounds")

                        # Get column data (contiguous dimension for CSC)
                        values, indices = csc.col_to_numpy(col_idx)
                        col_len = len(indices)

                        # Assume valid column data
                        assume(col_len >= 0)
                        assume(len(values) == col_len)

                        # Binary search for the row
                        pos = np.searchsorted(indices, row_idx)

                        # Sparse position is more common
                        if unlikely(pos < col_len and indices[pos] == row_idx):
                            return values[pos]
                        else:
                            return 0.0
                    return impl
                else:  # float32
                    def impl(csc, key):
                        row_idx, col_idx = key

                        if row_idx < 0 or row_idx >= csc.nrows:
                            raise IndexError("row index out of bounds")
                        if col_idx < 0 or col_idx >= csc.ncols:
                            raise IndexError("column index out of bounds")

                        values, indices = csc.col_to_numpy(col_idx)
                        col_len = len(indices)

                        assume(col_len >= 0)
                        assume(len(values) == col_len)

                        pos = np.searchsorted(indices, row_idx)

                        if unlikely(pos < col_len and indices[pos] == row_idx):
                            return values[pos]
                        else:
                            return 0.0
                    return impl

            # Both are slices
            elif isinstance(row_key_type, types.SliceType) and isinstance(col_key_type, types.SliceType):
                def impl(csc, key):
                    row_slice, col_slice = key

                    # Apply row slice
                    row_start = 0 if row_slice.start is None else row_slice.start
                    row_stop = csc.nrows if row_slice.stop is None else row_slice.stop
                    result = csc.slice_rows(row_start, row_stop)

                    # Apply column slice
                    col_start = 0 if col_slice.start is None else col_slice.start
                    col_stop = result.ncols if col_slice.stop is None else col_slice.stop
                    result = result.slice_cols(col_start, col_stop)

                    return result
                return impl


# =============================================================================
# CSR: hstack (horizontal stacking)
# =============================================================================

@overload_method(CSRType, 'hstack')
def csr_hstack_overload(matrices):
    """Horizontally stack multiple CSR matrices."""
    # This is called as a class method, but in overload it's like a static method
    # We need to check the first element's type
    if hasattr(matrices, 'dtype'):
        elem_type = matrices.dtype
        if isinstance(elem_type, CSRType):
            if elem_type.dtype == types.float64:
                def impl(matrices):
                    n = len(matrices)
                    if n == 0:
                        raise ValueError("Cannot hstack empty list")
                    
                    # Allocate handle array
                    handles = _alloca_voidptr_array(n)
                    for i in range(n):
                        handles[i] = matrices[i].handle
                    
                    # Call FFI
                    out_handle_ptr = _alloca_voidptr()
                    result = ffi_csr_f64_hstack(handles, n, out_handle_ptr)
                    if result != 0:
                        raise RuntimeError("hstack failed")
                    
                    new_handle = _load_voidptr(out_handle_ptr)
                    return _create_csr_from_handle_f64(new_handle, True)
                return impl
            else:  # float32
                def impl(matrices):
                    n = len(matrices)
                    if n == 0:
                        raise ValueError("Cannot hstack empty list")
                    handles = _alloca_voidptr_array(n)
                    for i in range(n):
                        handles[i] = matrices[i].handle
                    out_handle_ptr = _alloca_voidptr()
                    result = ffi_csr_f32_hstack(handles, n, out_handle_ptr)
                    if result != 0:
                        raise RuntimeError("hstack failed")
                    new_handle = _load_voidptr(out_handle_ptr)
                    return _create_csr_from_handle_f32(new_handle, True)
                return impl


# =============================================================================
# CSR: vstack (vertical stacking)
# =============================================================================

@overload_method(CSRType, 'vstack')
def csr_vstack_overload(matrices):
    """Vertically stack multiple CSR matrices."""
    if hasattr(matrices, 'dtype'):
        elem_type = matrices.dtype
        if isinstance(elem_type, CSRType):
            if elem_type.dtype == types.float64:
                def impl(matrices):
                    n = len(matrices)
                    if n == 0:
                        raise ValueError("Cannot vstack empty list")
                    handles = _alloca_voidptr_array(n)
                    for i in range(n):
                        handles[i] = matrices[i].handle
                    out_handle_ptr = _alloca_voidptr()
                    result = ffi_csr_f64_vstack(handles, n, out_handle_ptr)
                    if result != 0:
                        raise RuntimeError("vstack failed")
                    new_handle = _load_voidptr(out_handle_ptr)
                    return _create_csr_from_handle_f64(new_handle, True)
                return impl
            else:  # float32
                def impl(matrices):
                    n = len(matrices)
                    if n == 0:
                        raise ValueError("Cannot vstack empty list")
                    handles = _alloca_voidptr_array(n)
                    for i in range(n):
                        handles[i] = matrices[i].handle
                    out_handle_ptr = _alloca_voidptr()
                    result = ffi_csr_f32_vstack(handles, n, out_handle_ptr)
                    if result != 0:
                        raise RuntimeError("vstack failed")
                    new_handle = _load_voidptr(out_handle_ptr)
                    return _create_csr_from_handle_f32(new_handle, True)
                return impl


# =============================================================================
# CSC: hstack
# =============================================================================

@overload_method(CSCType, 'hstack')
def csc_hstack_overload(matrices):
    """Horizontally stack multiple CSC matrices."""
    if hasattr(matrices, 'dtype'):
        elem_type = matrices.dtype
        if isinstance(elem_type, CSCType):
            if elem_type.dtype == types.float64:
                def impl(matrices):
                    n = len(matrices)
                    if n == 0:
                        raise ValueError("Cannot hstack empty list")
                    handles = _alloca_voidptr_array(n)
                    for i in range(n):
                        handles[i] = matrices[i].handle
                    out_handle_ptr = _alloca_voidptr()
                    result = ffi_csc_f64_hstack(handles, n, out_handle_ptr)
                    if result != 0:
                        raise RuntimeError("hstack failed")
                    new_handle = _load_voidptr(out_handle_ptr)
                    return _create_csc_from_handle_f64(new_handle, True)
                return impl
            else:  # float32
                def impl(matrices):
                    n = len(matrices)
                    if n == 0:
                        raise ValueError("Cannot hstack empty list")
                    handles = _alloca_voidptr_array(n)
                    for i in range(n):
                        handles[i] = matrices[i].handle
                    out_handle_ptr = _alloca_voidptr()
                    result = ffi_csc_f32_hstack(handles, n, out_handle_ptr)
                    if result != 0:
                        raise RuntimeError("hstack failed")
                    new_handle = _load_voidptr(out_handle_ptr)
                    return _create_csc_from_handle_f32(new_handle, True)
                return impl


# =============================================================================
# CSC: vstack
# =============================================================================

@overload_method(CSCType, 'vstack')
def csc_vstack_overload(matrices):
    """Vertically stack multiple CSC matrices."""
    if hasattr(matrices, 'dtype'):
        elem_type = matrices.dtype
        if isinstance(elem_type, CSCType):
            if elem_type.dtype == types.float64:
                def impl(matrices):
                    n = len(matrices)
                    if n == 0:
                        raise ValueError("Cannot vstack empty list")
                    handles = _alloca_voidptr_array(n)
                    for i in range(n):
                        handles[i] = matrices[i].handle
                    out_handle_ptr = _alloca_voidptr()
                    result = ffi_csc_f64_vstack(handles, n, out_handle_ptr)
                    if result != 0:
                        raise RuntimeError("vstack failed")
                    new_handle = _load_voidptr(out_handle_ptr)
                    return _create_csc_from_handle_f64(new_handle, True)
                return impl
            else:  # float32
                def impl(matrices):
                    n = len(matrices)
                    if n == 0:
                        raise ValueError("Cannot vstack empty list")
                    handles = _alloca_voidptr_array(n)
                    for i in range(n):
                        handles[i] = matrices[i].handle
                    out_handle_ptr = _alloca_voidptr()
                    result = ffi_csc_f32_vstack(handles, n, out_handle_ptr)
                    if result != 0:
                        raise RuntimeError("vstack failed")
                    new_handle = _load_voidptr(out_handle_ptr)
                    return _create_csc_from_handle_f32(new_handle, True)
                return impl
