"""Conversion methods for CSR/CSC types.

This module implements conversion operations: to_dense, to_coo, to_csc/to_csr, clone.
"""

import numpy as np
from numba import types, literally
from numba.core import cgutils
from numba.extending import overload_method, intrinsic
import llvmlite.ir as lir

from ._types import CSRType, CSCType
from ._ffi import (
    _alloca_voidptr,
    _load_voidptr,
    ffi_csr_f64_clone,
    ffi_csr_f32_clone,
    ffi_csc_f64_clone,
    ffi_csc_f32_clone,
    ffi_csc_f64_from_csr,
    ffi_csc_f32_from_csr,
    ffi_csr_f64_from_csc,
    ffi_csr_f32_from_csc,
    ffi_csc_f64_transpose_from_csr,
    ffi_csc_f32_transpose_from_csr,
    ffi_csr_f64_transpose_from_csc,
    ffi_csr_f32_transpose_from_csc,
)


# =============================================================================
# FFI Intrinsics for to_dense and to_coo
# =============================================================================

@intrinsic
def _ffi_csr_f64_to_dense(typingctx, handle_ty, ptr_ty, size_ty, col_major_ty):
    """Call csr_f64_to_dense FFI function."""
    sig = types.int32(types.voidptr, types.voidptr, types.intp, types.boolean)
    
    def codegen(context, builder, sig, args):
        [handle, ptr, size, col_major] = args
        
        # int32_t csr_f64_to_dense(void* handle, double* out, size_t size, bool col_major)
        fnty = lir.FunctionType(
            lir.IntType(32),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(1)
            ]
        )
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f64_to_dense")
        return builder.call(fn, [handle, ptr, size, col_major])
    
    return sig, codegen


@intrinsic
def _ffi_csr_f32_to_dense(typingctx, handle_ty, ptr_ty, size_ty, col_major_ty):
    """Call csr_f32_to_dense FFI function."""
    sig = types.int32(types.voidptr, types.voidptr, types.intp, types.boolean)
    
    def codegen(context, builder, sig, args):
        [handle, ptr, size, col_major] = args
        fnty = lir.FunctionType(
            lir.IntType(32),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(1)
            ]
        )
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f32_to_dense")
        return builder.call(fn, [handle, ptr, size, col_major])
    
    return sig, codegen


@intrinsic
def _ffi_csc_f64_to_dense(typingctx, handle_ty, ptr_ty, size_ty, col_major_ty):
    """Call csc_f64_to_dense FFI function."""
    sig = types.int32(types.voidptr, types.voidptr, types.intp, types.boolean)
    
    def codegen(context, builder, sig, args):
        [handle, ptr, size, col_major] = args
        fnty = lir.FunctionType(
            lir.IntType(32),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(1)
            ]
        )
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f64_to_dense")
        return builder.call(fn, [handle, ptr, size, col_major])
    
    return sig, codegen


@intrinsic
def _ffi_csc_f32_to_dense(typingctx, handle_ty, ptr_ty, size_ty, col_major_ty):
    """Call csc_f32_to_dense FFI function."""
    sig = types.int32(types.voidptr, types.voidptr, types.intp, types.boolean)
    
    def codegen(context, builder, sig, args):
        [handle, ptr, size, col_major] = args
        fnty = lir.FunctionType(
            lir.IntType(32),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(1)
            ]
        )
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f32_to_dense")
        return builder.call(fn, [handle, ptr, size, col_major])
    
    return sig, codegen


@intrinsic
def _ffi_csr_f64_to_coo(typingctx, handle_ty, row_ptr_ty, col_ptr_ty, data_ptr_ty, nnz_ty):
    """Call csr_f64_to_coo FFI function."""
    sig = types.int32(types.voidptr, types.voidptr, types.voidptr, types.voidptr, types.intp)
    
    def codegen(context, builder, sig, args):
        [handle, row_ptr, col_ptr, data_ptr, nnz] = args
        
        # int32_t csr_f64_to_coo(void* handle, int64_t* rows, int64_t* cols, double* data, size_t nnz)
        fnty = lir.FunctionType(
            lir.IntType(32),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64)
            ]
        )
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f64_to_coo")
        return builder.call(fn, [handle, row_ptr, col_ptr, data_ptr, nnz])
    
    return sig, codegen


@intrinsic
def _ffi_csr_f32_to_coo(typingctx, handle_ty, row_ptr_ty, col_ptr_ty, data_ptr_ty, nnz_ty):
    """Call csr_f32_to_coo FFI function."""
    sig = types.int32(types.voidptr, types.voidptr, types.voidptr, types.voidptr, types.intp)
    
    def codegen(context, builder, sig, args):
        [handle, row_ptr, col_ptr, data_ptr, nnz] = args
        fnty = lir.FunctionType(
            lir.IntType(32),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64)
            ]
        )
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f32_to_coo")
        return builder.call(fn, [handle, row_ptr, col_ptr, data_ptr, nnz])
    
    return sig, codegen


@intrinsic
def _ffi_csc_f64_to_coo(typingctx, handle_ty, row_ptr_ty, col_ptr_ty, data_ptr_ty, nnz_ty):
    """Call csc_f64_to_coo FFI function."""
    sig = types.int32(types.voidptr, types.voidptr, types.voidptr, types.voidptr, types.intp)
    
    def codegen(context, builder, sig, args):
        [handle, row_ptr, col_ptr, data_ptr, nnz] = args
        fnty = lir.FunctionType(
            lir.IntType(32),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64)
            ]
        )
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f64_to_coo")
        return builder.call(fn, [handle, row_ptr, col_ptr, data_ptr, nnz])
    
    return sig, codegen


@intrinsic
def _ffi_csc_f32_to_coo(typingctx, handle_ty, row_ptr_ty, col_ptr_ty, data_ptr_ty, nnz_ty):
    """Call csc_f32_to_coo FFI function."""
    sig = types.int32(types.voidptr, types.voidptr, types.voidptr, types.voidptr, types.intp)
    
    def codegen(context, builder, sig, args):
        [handle, row_ptr, col_ptr, data_ptr, nnz] = args
        fnty = lir.FunctionType(
            lir.IntType(32),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64)
            ]
        )
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f32_to_coo")
        return builder.call(fn, [handle, row_ptr, col_ptr, data_ptr, nnz])
    
    return sig, codegen


# =============================================================================
# CSR: to_dense
# =============================================================================

@overload_method(CSRType, 'to_dense')
def csr_to_dense_overload(csr, order='C'):
    """Convert CSR to dense numpy array."""
    if csr.dtype == types.float64:
        def impl(csr, order='C'):
            # Allocate output array
            out = np.zeros((csr.nrows, csr.ncols), dtype=np.float64)
            
            # Determine if column-major
            col_major = (order == 'F')
            
            # Get data pointer
            out_ptr = out.ctypes.data
            
            # Call FFI to fill array
            result = _ffi_csr_f64_to_dense(csr.handle, out_ptr, out.size, col_major)
            if result != 0:
                raise RuntimeError("to_dense failed")
            
            return out
        return impl
    else:  # float32
        def impl(csr, order='C'):
            out = np.zeros((csr.nrows, csr.ncols), dtype=np.float32)
            col_major = (order == 'F')
            out_ptr = out.ctypes.data
            result = _ffi_csr_f32_to_dense(csr.handle, out_ptr, out.size, col_major)
            if result != 0:
                raise RuntimeError("to_dense failed")
            return out
        return impl


# =============================================================================
# CSR: to_coo
# =============================================================================

@overload_method(CSRType, 'to_coo')
def csr_to_coo_overload(csr):
    """Convert CSR to COO format."""
    if csr.dtype == types.float64:
        def impl(csr):
            nnz = csr.nnz
            
            # Allocate output arrays
            row_indices = np.empty(nnz, dtype=np.int64)
            col_indices = np.empty(nnz, dtype=np.int64)
            data = np.empty(nnz, dtype=np.float64)
            
            # Get data pointers
            row_ptr = row_indices.ctypes.data
            col_ptr = col_indices.ctypes.data
            data_ptr = data.ctypes.data
            
            # Call FFI
            result = _ffi_csr_f64_to_coo(csr.handle, row_ptr, col_ptr, data_ptr, nnz)
            if result != 0:
                raise RuntimeError("to_coo failed")
            
            return (row_indices, col_indices, data)
        return impl
    else:  # float32
        def impl(csr):
            nnz = csr.nnz
            row_indices = np.empty(nnz, dtype=np.int64)
            col_indices = np.empty(nnz, dtype=np.int64)
            data = np.empty(nnz, dtype=np.float32)
            row_ptr = row_indices.ctypes.data
            col_ptr = col_indices.ctypes.data
            data_ptr = data.ctypes.data
            result = _ffi_csr_f32_to_coo(csr.handle, row_ptr, col_ptr, data_ptr, nnz)
            if result != 0:
                raise RuntimeError("to_coo failed")
            return (row_indices, col_indices, data)
        return impl


# =============================================================================
# CSC: to_dense
# =============================================================================

@overload_method(CSCType, 'to_dense')
def csc_to_dense_overload(csc, order='C'):
    """Convert CSC to dense numpy array."""
    if csc.dtype == types.float64:
        def impl(csc, order='C'):
            out = np.zeros((csc.nrows, csc.ncols), dtype=np.float64)
            col_major = (order == 'F')
            out_ptr = out.ctypes.data
            result = _ffi_csc_f64_to_dense(csc.handle, out_ptr, out.size, col_major)
            if result != 0:
                raise RuntimeError("to_dense failed")
            return out
        return impl
    else:  # float32
        def impl(csc, order='C'):
            out = np.zeros((csc.nrows, csc.ncols), dtype=np.float32)
            col_major = (order == 'F')
            out_ptr = out.ctypes.data
            result = _ffi_csc_f32_to_dense(csc.handle, out_ptr, out.size, col_major)
            if result != 0:
                raise RuntimeError("to_dense failed")
            return out
        return impl


# =============================================================================
# CSC: to_coo
# =============================================================================

@overload_method(CSCType, 'to_coo')
def csc_to_coo_overload(csc):
    """Convert CSC to COO format."""
    if csc.dtype == types.float64:
        def impl(csc):
            nnz = csc.nnz
            row_indices = np.empty(nnz, dtype=np.int64)
            col_indices = np.empty(nnz, dtype=np.int64)
            data = np.empty(nnz, dtype=np.float64)
            row_ptr = row_indices.ctypes.data
            col_ptr = col_indices.ctypes.data
            data_ptr = data.ctypes.data
            result = _ffi_csc_f64_to_coo(csc.handle, row_ptr, col_ptr, data_ptr, nnz)
            if result != 0:
                raise RuntimeError("to_coo failed")
            return (row_indices, col_indices, data)
        return impl
    else:  # float32
        def impl(csc):
            nnz = csc.nnz
            row_indices = np.empty(nnz, dtype=np.int64)
            col_indices = np.empty(nnz, dtype=np.int64)
            data = np.empty(nnz, dtype=np.float32)
            row_ptr = row_indices.ctypes.data
            col_ptr = col_indices.ctypes.data
            data_ptr = data.ctypes.data
            result = _ffi_csc_f32_to_coo(csc.handle, row_ptr, col_ptr, data_ptr, nnz)
            if result != 0:
                raise RuntimeError("to_coo failed")
            return (row_indices, col_indices, data)
        return impl


# =============================================================================
# CSR: to_csc
# =============================================================================

@overload_method(CSRType, 'to_csc')
def csr_to_csc_overload(csr):
    """Convert CSR to CSC format."""
    from ._operators import _create_csc_from_handle_f64, _create_csc_from_handle_f32
    
    if csr.dtype == types.float64:
        def impl(csr):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csc_f64_from_csr(csr.handle, out_handle_ptr)
            if result != 0:
                raise RuntimeError("to_csc failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csc_from_handle_f64(new_handle, True)
        return impl
    else:  # float32
        def impl(csr):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csc_f32_from_csr(csr.handle, out_handle_ptr)
            if result != 0:
                raise RuntimeError("to_csc failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csc_from_handle_f32(new_handle, True)
        return impl


# =============================================================================
# CSC: to_csr
# =============================================================================

@overload_method(CSCType, 'to_csr')
def csc_to_csr_overload(csc):
    """Convert CSC to CSR format."""
    from ._operators import _create_csr_from_handle_f64, _create_csr_from_handle_f32
    
    if csc.dtype == types.float64:
        def impl(csc):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csr_f64_from_csc(csc.handle, out_handle_ptr)
            if result != 0:
                raise RuntimeError("to_csr failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csr_from_handle_f64(new_handle, True)
        return impl
    else:  # float32
        def impl(csc):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csr_f32_from_csc(csc.handle, out_handle_ptr)
            if result != 0:
                raise RuntimeError("to_csr failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csr_from_handle_f32(new_handle, True)
        return impl


# =============================================================================
# CSR: clone
# =============================================================================

@overload_method(CSRType, 'clone')
def csr_clone_overload(csr):
    """Create a deep copy of the CSR matrix."""
    from ._operators import _create_csr_from_handle_f64, _create_csr_from_handle_f32

    if csr.dtype == types.float64:
        def impl(csr):
            # ffi_csr_f64_clone returns the new handle directly
            new_handle = ffi_csr_f64_clone(csr.handle)
            return _create_csr_from_handle_f64(new_handle, True)
        return impl
    else:  # float32
        def impl(csr):
            new_handle = ffi_csr_f32_clone(csr.handle)
            return _create_csr_from_handle_f32(new_handle, True)
        return impl


# =============================================================================
# CSC: clone
# =============================================================================

@overload_method(CSCType, 'clone')
def csc_clone_overload(csc):
    """Create a deep copy of the CSC matrix."""
    from ._operators import _create_csc_from_handle_f64, _create_csc_from_handle_f32

    if csc.dtype == types.float64:
        def impl(csc):
            new_handle = ffi_csc_f64_clone(csc.handle)
            return _create_csc_from_handle_f64(new_handle, True)
        return impl
    else:  # float32
        def impl(csc):
            new_handle = ffi_csc_f32_clone(csc.handle)
            return _create_csc_from_handle_f32(new_handle, True)
        return impl


# =============================================================================
# CSR: T (transpose)
# =============================================================================

@overload_method(CSRType, 'T')
def csr_T_overload(csr):
    """Transpose CSR to CSC format."""
    from ._operators import _create_csc_from_handle_f64, _create_csc_from_handle_f32

    if csr.dtype == types.float64:
        def impl(csr):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csc_f64_transpose_from_csr(csr.handle, out_handle_ptr)
            if result != 0:
                raise RuntimeError("transpose failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csc_from_handle_f64(new_handle, True)
        return impl
    else:  # float32
        def impl(csr):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csc_f32_transpose_from_csr(csr.handle, out_handle_ptr)
            if result != 0:
                raise RuntimeError("transpose failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csc_from_handle_f32(new_handle, True)
        return impl


# =============================================================================
# CSC: T (transpose)
# =============================================================================

@overload_method(CSCType, 'T')
def csc_T_overload(csc):
    """Transpose CSC to CSR format."""
    from ._operators import _create_csr_from_handle_f64, _create_csr_from_handle_f32

    if csc.dtype == types.float64:
        def impl(csc):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csr_f64_transpose_from_csc(csc.handle, out_handle_ptr)
            if result != 0:
                raise RuntimeError("transpose failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csr_from_handle_f64(new_handle, True)
        return impl
    else:  # float32
        def impl(csc):
            out_handle_ptr = _alloca_voidptr()
            result = ffi_csr_f32_transpose_from_csc(csc.handle, out_handle_ptr)
            if result != 0:
                raise RuntimeError("transpose failed")
            new_handle = _load_voidptr(out_handle_ptr)
            return _create_csr_from_handle_f32(new_handle, True)
        return impl


# End of file
