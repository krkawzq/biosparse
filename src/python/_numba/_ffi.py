"""FFI integration for Numba.

This module provides intrinsic functions to call FFI functions from 
JIT-compiled code.

Note: We use direct LLVM function declarations instead of CFFI registration,
as our dynamic library is loaded via cffi.dlopen() and functions are resolved
by symbol name at compile time.
"""

from numba import types
from numba.core import cgutils
from numba.extending import intrinsic
import llvmlite.ir as lir


# =============================================================================
# FFI Library Access
# =============================================================================

# The CFFI library is available for fallback but we don't register it with Numba
# Instead, we declare external functions directly in LLVM IR
_FFI_AVAILABLE = False

try:
    from .._binding._cffi import ffi, lib
    _FFI_AVAILABLE = True
except ImportError:
    try:
        from _binding._cffi import ffi, lib
        _FFI_AVAILABLE = True
    except ImportError:
        pass


# =============================================================================
# Helper Intrinsics
# =============================================================================

@intrinsic
def _alloca_voidptr(typingctx):
    """Allocate space for a void* on the stack.
    
    Returns a pointer to void* (i.e., void**).
    """
    sig = types.CPointer(types.voidptr)()
    
    def codegen(context, builder, sig, args):
        ptr = cgutils.alloca_once(builder, cgutils.voidptr_t)
        return ptr
    
    return sig, codegen


@intrinsic
def _load_voidptr(typingctx, ptr_ty):
    """Load a void* from a void** pointer."""
    sig = types.voidptr(types.CPointer(types.voidptr))
    
    def codegen(context, builder, sig, args):
        [ptr] = args
        return builder.load(ptr)
    
    return sig, codegen


@intrinsic
def _store_voidptr(typingctx, ptr_ty, val_ty):
    """Store a void* value into a void** pointer."""
    sig = types.void(types.CPointer(types.voidptr), types.voidptr)
    
    def codegen(context, builder, sig, args):
        [ptr, val] = args
        builder.store(val, ptr)
        return context.get_dummy_value()
    
    return sig, codegen


@intrinsic
def _alloca_voidptr_array(typingctx, n_ty):
    """Allocate an array of void* pointers on the stack.
    
    Args:
        n: Number of pointers to allocate
        
    Returns:
        Pointer to void*[] array
    """
    sig = types.CPointer(types.voidptr)(types.intp)
    
    def codegen(context, builder, sig, args):
        [n] = args
        # Allocate n * sizeof(void*)
        arr = builder.alloca(cgutils.voidptr_t, size=n)
        return arr
    
    return sig, codegen


# =============================================================================
# Array Creation Helpers
# =============================================================================

@intrinsic
def _make_array_from_ptr_f64(typingctx, ptr_ty, length_ty):
    """Create a 1D float64 array from a pointer and length."""
    dtype = types.float64
    array_type = types.Array(dtype, 1, 'C')
    sig = array_type(ptr_ty, length_ty)
    
    def codegen(context, builder, signature, args):
        ptr, length = args
        ary = context.make_array(array_type)(context, builder)
        llvm_dtype = context.get_data_type(dtype)
        itemsize = context.get_constant(types.intp, context.get_abi_sizeof(llvm_dtype))
        shape = cgutils.pack_array(builder, [length])
        strides = cgutils.pack_array(builder, [itemsize])
        context.populate_array(ary, data=builder.bitcast(ptr, ary.data.type),
                              shape=shape, strides=strides, itemsize=itemsize, meminfo=None)
        return ary._getvalue()
    return sig, codegen


@intrinsic
def _make_array_from_ptr_f32(typingctx, ptr_ty, length_ty):
    """Create a 1D float32 array from a pointer and length."""
    dtype = types.float32
    array_type = types.Array(dtype, 1, 'C')
    sig = array_type(ptr_ty, length_ty)
    
    def codegen(context, builder, signature, args):
        ptr, length = args
        ary = context.make_array(array_type)(context, builder)
        llvm_dtype = context.get_data_type(dtype)
        itemsize = context.get_constant(types.intp, context.get_abi_sizeof(llvm_dtype))
        shape = cgutils.pack_array(builder, [length])
        strides = cgutils.pack_array(builder, [itemsize])
        context.populate_array(ary, data=builder.bitcast(ptr, ary.data.type),
                              shape=shape, strides=strides, itemsize=itemsize, meminfo=None)
        return ary._getvalue()
    return sig, codegen


@intrinsic
def _make_array_from_ptr_i64(typingctx, ptr_ty, length_ty):
    """Create a 1D int64 array from a pointer and length."""
    dtype = types.int64
    array_type = types.Array(dtype, 1, 'C')
    sig = array_type(ptr_ty, length_ty)
    
    def codegen(context, builder, signature, args):
        ptr, length = args
        ary = context.make_array(array_type)(context, builder)
        llvm_dtype = context.get_data_type(dtype)
        itemsize = context.get_constant(types.intp, context.get_abi_sizeof(llvm_dtype))
        shape = cgutils.pack_array(builder, [length])
        strides = cgutils.pack_array(builder, [itemsize])
        context.populate_array(ary, data=builder.bitcast(ptr, ary.data.type),
                              shape=shape, strides=strides, itemsize=itemsize, meminfo=None)
        return ary._getvalue()
    return sig, codegen


# =============================================================================
# FFI Function Wrappers - Scalar Returns
# =============================================================================

def _make_ffi_scalar_getter(fname, ret_type):
    """Factory for creating FFI scalar getter intrinsics.
    
    Args:
        fname: FFI function name (e.g., "csr_f64_rows")
        ret_type: Return Numba type (e.g., types.int64)
    """
    @intrinsic
    def _ffi_call(typingctx, handle_ty):
        sig = ret_type(types.voidptr)
        
        def codegen(context, builder, sig, args):
            [handle] = args
            
            # Determine LLVM return type
            if ret_type == types.int64:
                llvm_ret = lir.IntType(64)
            elif ret_type == types.int32:
                llvm_ret = lir.IntType(32)
            elif ret_type == types.boolean:
                llvm_ret = lir.IntType(1)
            elif ret_type == types.float64:
                llvm_ret = lir.DoubleType()
            elif ret_type == types.float32:
                llvm_ret = lir.FloatType()
            else:
                raise ValueError(f"Unsupported return type: {ret_type}")
            
            # Function signature: ret_type function_name(void* handle)
            fnty = lir.FunctionType(llvm_ret, [lir.IntType(8).as_pointer()])
            fn = cgutils.get_or_insert_function(builder.module, fnty, fname)
            
            return builder.call(fn, [handle])
        
        return sig, codegen
    
    return _ffi_call


# =============================================================================
# FFI Function Wrappers - Pointer Returns
# =============================================================================

def _make_ffi_ptr_getter(fname):
    """Factory for creating FFI pointer getter intrinsics.
    
    Args:
        fname: FFI function name (e.g., "csr_f64_row_values_ptr")
    """
    @intrinsic
    def _ffi_call(typingctx, handle_ty, index_ty):
        sig = types.voidptr(types.voidptr, types.int64)
        
        def codegen(context, builder, sig, args):
            [handle, index] = args
            
            # Function signature: void* function_name(void* handle, int64_t index)
            fnty = lir.FunctionType(
                lir.IntType(8).as_pointer(),
                [lir.IntType(8).as_pointer(), lir.IntType(64)]
            )
            fn = cgutils.get_or_insert_function(builder.module, fnty, fname)
            
            return builder.call(fn, [handle, index])
        
        return sig, codegen
    
    return _ffi_call


# =============================================================================
# FFI Function Wrappers - Object Creation (returns via out parameter)
# =============================================================================

def _make_ffi_creator_2args(fname):
    """Factory for FFI functions that create objects with 2 arguments.
    
    Pattern: int32_t func(void* handle, int64_t arg1, int64_t arg2, void** out)
    """
    @intrinsic
    def _ffi_call(typingctx, handle_ty, arg1_ty, arg2_ty, out_ty):
        sig = types.int32(types.voidptr, types.int64, types.int64, types.CPointer(types.voidptr))
        
        def codegen(context, builder, sig, args):
            [handle, arg1, arg2, out_ptr] = args
            
            # int32_t func(void* handle, int64_t arg1, int64_t arg2, void** out)
            fnty = lir.FunctionType(
                lir.IntType(32),
                [
                    lir.IntType(8).as_pointer(),
                    lir.IntType(64),
                    lir.IntType(64),
                    lir.IntType(8).as_pointer().as_pointer()
                ]
            )
            fn = cgutils.get_or_insert_function(builder.module, fnty, fname)
            
            return builder.call(fn, [handle, arg1, arg2, out_ptr])
        
        return sig, codegen
    
    return _ffi_call


def _make_ffi_creator_array(fname):
    """Factory for FFI functions that create objects from array of handles.
    
    Pattern: int32_t func(void** handles, size_t n, void** out)
    """
    @intrinsic
    def _ffi_call(typingctx, handles_ty, n_ty, out_ty):
        sig = types.int32(types.CPointer(types.voidptr), types.intp, types.CPointer(types.voidptr))
        
        def codegen(context, builder, sig, args):
            [handles, n, out_ptr] = args
            
            # int32_t func(void** handles, size_t n, void** out)
            fnty = lir.FunctionType(
                lir.IntType(32),
                [
                    lir.IntType(8).as_pointer().as_pointer(),
                    lir.IntType(64),
                    lir.IntType(8).as_pointer().as_pointer()
                ]
            )
            fn = cgutils.get_or_insert_function(builder.module, fnty, fname)
            
            return builder.call(fn, [handles, n, out_ptr])
        
        return sig, codegen
    
    return _ffi_call


# =============================================================================
# FFI Function Wrappers - Conversion Functions
# =============================================================================

def _make_ffi_converter_1arg(fname):
    """Factory for FFI conversion functions with 1 argument.
    
    Pattern: int32_t func(void* handle, void** out)
    """
    @intrinsic
    def _ffi_call(typingctx, handle_ty, out_ty):
        sig = types.int32(types.voidptr, types.CPointer(types.voidptr))
        
        def codegen(context, builder, sig, args):
            [handle, out_ptr] = args
            
            # int32_t func(void* handle, void** out)
            fnty = lir.FunctionType(
                lir.IntType(32),
                [
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer().as_pointer()
                ]
            )
            fn = cgutils.get_or_insert_function(builder.module, fnty, fname)
            
            return builder.call(fn, [handle, out_ptr])
        
        return sig, codegen

    return _ffi_call


def _make_ffi_clone(fname):
    """Factory for FFI clone functions that return a handle directly.

    Pattern: void* func(void* handle)
    """
    @intrinsic
    def _ffi_call(typingctx, handle_ty):
        sig = types.voidptr(types.voidptr)

        def codegen(context, builder, sig, args):
            [handle] = args

            # void* func(void* handle)
            fnty = lir.FunctionType(
                lir.IntType(8).as_pointer(),
                [lir.IntType(8).as_pointer()]
            )
            fn = cgutils.get_or_insert_function(builder.module, fnty, fname)

            return builder.call(fn, [handle])

        return sig, codegen

    return _ffi_call


# =============================================================================
# Exported FFI Function Intrinsics
# =============================================================================

# CSR F64 - Scalar getters
ffi_csr_f64_rows = _make_ffi_scalar_getter("csr_f64_rows", types.int64)
ffi_csr_f64_cols = _make_ffi_scalar_getter("csr_f64_cols", types.int64)
ffi_csr_f64_nnz = _make_ffi_scalar_getter("csr_f64_nnz", types.int64)
ffi_csr_f64_is_valid = _make_ffi_scalar_getter("csr_f64_is_valid", types.boolean)
ffi_csr_f64_is_sorted = _make_ffi_scalar_getter("csr_f64_is_sorted", types.boolean)
ffi_csr_f64_density = _make_ffi_scalar_getter("csr_f64_density", types.float64)
ffi_csr_f64_sparsity = _make_ffi_scalar_getter("csr_f64_sparsity", types.float64)

# CSR F64 - Pointer getters
ffi_csr_f64_row_values_ptr = _make_ffi_ptr_getter("csr_f64_row_values_ptr")
ffi_csr_f64_row_indices_ptr = _make_ffi_ptr_getter("csr_f64_row_indices_ptr")
ffi_csr_f64_row_len = _make_ffi_scalar_getter("csr_f64_row_len", types.int64)

# CSR F64 - Object creators
ffi_csr_f64_slice_rows = _make_ffi_creator_2args("csr_f64_slice_rows")
ffi_csr_f64_slice_cols = _make_ffi_creator_2args("csr_f64_slice_cols")
ffi_csr_f64_hstack = _make_ffi_creator_array("csr_f64_hstack")
ffi_csr_f64_vstack = _make_ffi_creator_array("csr_f64_vstack")
ffi_csr_f64_clone = _make_ffi_clone("csr_f64_clone")

# CSR F64 - Converters
ffi_csc_f64_from_csr = _make_ffi_converter_1arg("csc_f64_from_csr")

# CSR F32 - Scalar getters
ffi_csr_f32_rows = _make_ffi_scalar_getter("csr_f32_rows", types.int64)
ffi_csr_f32_cols = _make_ffi_scalar_getter("csr_f32_cols", types.int64)
ffi_csr_f32_nnz = _make_ffi_scalar_getter("csr_f32_nnz", types.int64)
ffi_csr_f32_is_valid = _make_ffi_scalar_getter("csr_f32_is_valid", types.boolean)
ffi_csr_f32_is_sorted = _make_ffi_scalar_getter("csr_f32_is_sorted", types.boolean)
ffi_csr_f32_density = _make_ffi_scalar_getter("csr_f32_density", types.float64)
ffi_csr_f32_sparsity = _make_ffi_scalar_getter("csr_f32_sparsity", types.float64)

# CSR F32 - Pointer getters
ffi_csr_f32_row_values_ptr = _make_ffi_ptr_getter("csr_f32_row_values_ptr")
ffi_csr_f32_row_indices_ptr = _make_ffi_ptr_getter("csr_f32_row_indices_ptr")
ffi_csr_f32_row_len = _make_ffi_scalar_getter("csr_f32_row_len", types.int64)

# CSR F32 - Object creators
ffi_csr_f32_slice_rows = _make_ffi_creator_2args("csr_f32_slice_rows")
ffi_csr_f32_slice_cols = _make_ffi_creator_2args("csr_f32_slice_cols")
ffi_csr_f32_hstack = _make_ffi_creator_array("csr_f32_hstack")
ffi_csr_f32_vstack = _make_ffi_creator_array("csr_f32_vstack")
ffi_csr_f32_clone = _make_ffi_clone("csr_f32_clone")

# CSR F32 - Converters
ffi_csc_f32_from_csr = _make_ffi_converter_1arg("csc_f32_from_csr")

# CSC F64 - Scalar getters
ffi_csc_f64_rows = _make_ffi_scalar_getter("csc_f64_rows", types.int64)
ffi_csc_f64_cols = _make_ffi_scalar_getter("csc_f64_cols", types.int64)
ffi_csc_f64_nnz = _make_ffi_scalar_getter("csc_f64_nnz", types.int64)
ffi_csc_f64_is_valid = _make_ffi_scalar_getter("csc_f64_is_valid", types.boolean)
ffi_csc_f64_is_sorted = _make_ffi_scalar_getter("csc_f64_is_sorted", types.boolean)
ffi_csc_f64_density = _make_ffi_scalar_getter("csc_f64_density", types.float64)
ffi_csc_f64_sparsity = _make_ffi_scalar_getter("csc_f64_sparsity", types.float64)

# CSC F64 - Pointer getters
ffi_csc_f64_col_values_ptr = _make_ffi_ptr_getter("csc_f64_col_values_ptr")
ffi_csc_f64_col_indices_ptr = _make_ffi_ptr_getter("csc_f64_col_indices_ptr")
ffi_csc_f64_col_len = _make_ffi_scalar_getter("csc_f64_col_len", types.int64)

# CSC F64 - Object creators
ffi_csc_f64_slice_rows = _make_ffi_creator_2args("csc_f64_slice_rows")
ffi_csc_f64_slice_cols = _make_ffi_creator_2args("csc_f64_slice_cols")
ffi_csc_f64_hstack = _make_ffi_creator_array("csc_f64_hstack")
ffi_csc_f64_vstack = _make_ffi_creator_array("csc_f64_vstack")
ffi_csc_f64_clone = _make_ffi_clone("csc_f64_clone")

# CSC F64 - Converters
ffi_csr_f64_from_csc = _make_ffi_converter_1arg("csr_f64_from_csc")

# CSC F32 - Scalar getters
ffi_csc_f32_rows = _make_ffi_scalar_getter("csc_f32_rows", types.int64)
ffi_csc_f32_cols = _make_ffi_scalar_getter("csc_f32_cols", types.int64)
ffi_csc_f32_nnz = _make_ffi_scalar_getter("csc_f32_nnz", types.int64)
ffi_csc_f32_is_valid = _make_ffi_scalar_getter("csc_f32_is_valid", types.boolean)
ffi_csc_f32_is_sorted = _make_ffi_scalar_getter("csc_f32_is_sorted", types.boolean)
ffi_csc_f32_density = _make_ffi_scalar_getter("csc_f32_density", types.float64)
ffi_csc_f32_sparsity = _make_ffi_scalar_getter("csc_f32_sparsity", types.float64)

# CSC F32 - Pointer getters
ffi_csc_f32_col_values_ptr = _make_ffi_ptr_getter("csc_f32_col_values_ptr")
ffi_csc_f32_col_indices_ptr = _make_ffi_ptr_getter("csc_f32_col_indices_ptr")
ffi_csc_f32_col_len = _make_ffi_scalar_getter("csc_f32_col_len", types.int64)

# CSC F32 - Object creators
ffi_csc_f32_slice_rows = _make_ffi_creator_2args("csc_f32_slice_rows")
ffi_csc_f32_slice_cols = _make_ffi_creator_2args("csc_f32_slice_cols")
ffi_csc_f32_hstack = _make_ffi_creator_array("csc_f32_hstack")
ffi_csc_f32_vstack = _make_ffi_creator_array("csc_f32_vstack")
ffi_csc_f32_clone = _make_ffi_clone("csc_f32_clone")

# CSC F32 - Converters
ffi_csr_f32_from_csc = _make_ffi_converter_1arg("csr_f32_from_csc")

# Transpose functions
ffi_csc_f64_transpose_from_csr = _make_ffi_converter_1arg("csc_f64_transpose_from_csr")
ffi_csc_f32_transpose_from_csr = _make_ffi_converter_1arg("csc_f32_transpose_from_csr")
ffi_csr_f64_transpose_from_csc = _make_ffi_converter_1arg("csr_f64_transpose_from_csc")
ffi_csr_f32_transpose_from_csc = _make_ffi_converter_1arg("csr_f32_transpose_from_csc")
