"""Validation and sorting methods for CSR/CSC types.

This module implements validation, sorting, and integrity checking operations.
"""

from numba import types
from numba.core import cgutils
from numba.extending import overload_method, overload_attribute, intrinsic
import llvmlite.ir as lir

from ._types import CSRType, CSCType


# =============================================================================
# FFI Intrinsics for validation
# =============================================================================

@intrinsic
def _ffi_csr_f64_is_valid(typingctx, handle_ty):
    """Call csr_f64_is_valid FFI function."""
    sig = types.boolean(types.voidptr)
    
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f64_is_valid")
        return builder.call(fn, [handle])
    
    return sig, codegen


@intrinsic
def _ffi_csr_f32_is_valid(typingctx, handle_ty):
    """Call csr_f32_is_valid FFI function."""
    sig = types.boolean(types.voidptr)
    
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f32_is_valid")
        return builder.call(fn, [handle])
    
    return sig, codegen


@intrinsic
def _ffi_csr_f64_is_sorted(typingctx, handle_ty):
    """Call csr_f64_is_sorted FFI function."""
    sig = types.boolean(types.voidptr)
    
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f64_is_sorted")
        return builder.call(fn, [handle])
    
    return sig, codegen


@intrinsic
def _ffi_csr_f32_is_sorted(typingctx, handle_ty):
    """Call csr_f32_is_sorted FFI function."""
    sig = types.boolean(types.voidptr)
    
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f32_is_sorted")
        return builder.call(fn, [handle])
    
    return sig, codegen


@intrinsic
def _ffi_csr_f64_validate(typingctx, handle_ty):
    """Call csr_f64_validate FFI function."""
    sig = types.boolean(types.voidptr)
    
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f64_validate")
        return builder.call(fn, [handle])
    
    return sig, codegen


@intrinsic
def _ffi_csr_f32_validate(typingctx, handle_ty):
    """Call csr_f32_validate FFI function."""
    sig = types.boolean(types.voidptr)
    
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f32_validate")
        return builder.call(fn, [handle])
    
    return sig, codegen


@intrinsic
def _ffi_csr_f64_indices_in_bounds(typingctx, handle_ty):
    """Call csr_f64_indices_in_bounds FFI function."""
    sig = types.boolean(types.voidptr)
    
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f64_indices_in_bounds")
        return builder.call(fn, [handle])
    
    return sig, codegen


@intrinsic
def _ffi_csr_f32_indices_in_bounds(typingctx, handle_ty):
    """Call csr_f32_indices_in_bounds FFI function."""
    sig = types.boolean(types.voidptr)
    
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f32_indices_in_bounds")
        return builder.call(fn, [handle])
    
    return sig, codegen


# Similar intrinsics for CSC
@intrinsic
def _ffi_csc_f64_is_valid(typingctx, handle_ty):
    sig = types.boolean(types.voidptr)
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f64_is_valid")
        return builder.call(fn, [handle])
    return sig, codegen


@intrinsic
def _ffi_csc_f32_is_valid(typingctx, handle_ty):
    sig = types.boolean(types.voidptr)
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f32_is_valid")
        return builder.call(fn, [handle])
    return sig, codegen


@intrinsic
def _ffi_csc_f64_is_sorted(typingctx, handle_ty):
    sig = types.boolean(types.voidptr)
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f64_is_sorted")
        return builder.call(fn, [handle])
    return sig, codegen


@intrinsic
def _ffi_csc_f32_is_sorted(typingctx, handle_ty):
    sig = types.boolean(types.voidptr)
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f32_is_sorted")
        return builder.call(fn, [handle])
    return sig, codegen


@intrinsic
def _ffi_csc_f64_validate(typingctx, handle_ty):
    sig = types.boolean(types.voidptr)
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f64_validate")
        return builder.call(fn, [handle])
    return sig, codegen


@intrinsic
def _ffi_csc_f32_validate(typingctx, handle_ty):
    sig = types.boolean(types.voidptr)
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f32_validate")
        return builder.call(fn, [handle])
    return sig, codegen


@intrinsic
def _ffi_csc_f64_indices_in_bounds(typingctx, handle_ty):
    sig = types.boolean(types.voidptr)
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f64_indices_in_bounds")
        return builder.call(fn, [handle])
    return sig, codegen


@intrinsic
def _ffi_csc_f32_indices_in_bounds(typingctx, handle_ty):
    sig = types.boolean(types.voidptr)
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f32_indices_in_bounds")
        return builder.call(fn, [handle])
    return sig, codegen


# =============================================================================
# CSR: Validation Properties
# =============================================================================

@overload_attribute(CSRType, 'is_valid')
def csr_is_valid_get(csr):
    """Check if CSR structure is valid."""
    if csr.dtype == types.float64:
        def getter(csr):
            return _ffi_csr_f64_is_valid(csr.handle)
        return getter
    else:
        def getter(csr):
            return _ffi_csr_f32_is_valid(csr.handle)
        return getter


@overload_attribute(CSRType, 'is_sorted')
def csr_is_sorted_get(csr):
    """Check if all row indices are sorted."""
    if csr.dtype == types.float64:
        def getter(csr):
            return _ffi_csr_f64_is_sorted(csr.handle)
        return getter
    else:
        def getter(csr):
            return _ffi_csr_f32_is_sorted(csr.handle)
        return getter


@overload_attribute(CSRType, 'indices_in_bounds')
def csr_indices_in_bounds_get(csr):
    """Check if all indices are within column bounds."""
    if csr.dtype == types.float64:
        def getter(csr):
            return _ffi_csr_f64_indices_in_bounds(csr.handle)
        return getter
    else:
        def getter(csr):
            return _ffi_csr_f32_indices_in_bounds(csr.handle)
        return getter


@overload_method(CSRType, 'validate')
def csr_validate_impl(csr):
    """Full validation check."""
    if csr.dtype == types.float64:
        def impl(csr):
            return _ffi_csr_f64_validate(csr.handle)
        return impl
    else:
        def impl(csr):
            return _ffi_csr_f32_validate(csr.handle)
        return impl


# =============================================================================
# CSC: Validation Properties
# =============================================================================

@overload_attribute(CSCType, 'is_valid')
def csc_is_valid_get(csc):
    """Check if CSC structure is valid."""
    if csc.dtype == types.float64:
        def getter(csc):
            return _ffi_csc_f64_is_valid(csc.handle)
        return getter
    else:
        def getter(csc):
            return _ffi_csc_f32_is_valid(csc.handle)
        return getter


@overload_attribute(CSCType, 'is_sorted')
def csc_is_sorted_get(csc):
    """Check if all column indices are sorted."""
    if csc.dtype == types.float64:
        def getter(csc):
            return _ffi_csc_f64_is_sorted(csc.handle)
        return getter
    else:
        def getter(csc):
            return _ffi_csc_f32_is_sorted(csc.handle)
        return getter


@overload_attribute(CSCType, 'indices_in_bounds')
def csc_indices_in_bounds_get(csc):
    """Check if all indices are within row bounds."""
    if csc.dtype == types.float64:
        def getter(csc):
            return _ffi_csc_f64_indices_in_bounds(csc.handle)
        return getter
    else:
        def getter(csc):
            return _ffi_csc_f32_indices_in_bounds(csc.handle)
        return getter


@overload_method(CSCType, 'validate')
def csc_validate_impl(csc):
    """Full validation check."""
    if csc.dtype == types.float64:
        def impl(csc):
            return _ffi_csc_f64_validate(csc.handle)
        return impl
    else:
        def impl(csc):
            return _ffi_csc_f32_validate(csc.handle)
        return impl


# =============================================================================
# FFI Intrinsics for sorting
# =============================================================================

@intrinsic
def _ffi_csr_f64_ensure_sorted(typingctx, handle_ty):
    """Call csr_f64_ensure_sorted FFI function."""
    sig = types.void(types.voidptr)
    
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f64_ensure_sorted")
        builder.call(fn, [handle])
        return context.get_dummy_value()
    
    return sig, codegen


@intrinsic
def _ffi_csr_f32_ensure_sorted(typingctx, handle_ty):
    """Call csr_f32_ensure_sorted FFI function."""
    sig = types.void(types.voidptr)
    
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f32_ensure_sorted")
        builder.call(fn, [handle])
        return context.get_dummy_value()
    
    return sig, codegen


@intrinsic
def _ffi_csr_f64_ensure_sorted_checked(typingctx, handle_ty):
    """Call csr_f64_ensure_sorted_checked FFI function."""
    sig = types.boolean(types.voidptr)
    
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f64_ensure_sorted_checked")
        return builder.call(fn, [handle])
    
    return sig, codegen


@intrinsic
def _ffi_csr_f32_ensure_sorted_checked(typingctx, handle_ty):
    """Call csr_f32_ensure_sorted_checked FFI function."""
    sig = types.boolean(types.voidptr)
    
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f32_ensure_sorted_checked")
        return builder.call(fn, [handle])
    
    return sig, codegen


@intrinsic
def _ffi_csc_f64_ensure_sorted(typingctx, handle_ty):
    sig = types.void(types.voidptr)
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f64_ensure_sorted")
        builder.call(fn, [handle])
        return context.get_dummy_value()
    return sig, codegen


@intrinsic
def _ffi_csc_f32_ensure_sorted(typingctx, handle_ty):
    sig = types.void(types.voidptr)
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f32_ensure_sorted")
        builder.call(fn, [handle])
        return context.get_dummy_value()
    return sig, codegen


@intrinsic
def _ffi_csc_f64_ensure_sorted_checked(typingctx, handle_ty):
    sig = types.boolean(types.voidptr)
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f64_ensure_sorted_checked")
        return builder.call(fn, [handle])
    return sig, codegen


@intrinsic
def _ffi_csc_f32_ensure_sorted_checked(typingctx, handle_ty):
    sig = types.boolean(types.voidptr)
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csc_f32_ensure_sorted_checked")
        return builder.call(fn, [handle])
    return sig, codegen


# =============================================================================
# CSR: Sorting Methods
# =============================================================================

@overload_method(CSRType, 'ensure_sorted')
def csr_ensure_sorted_impl(csr):
    """Sort all row indices in-place."""
    if csr.dtype == types.float64:
        def impl(csr):
            _ffi_csr_f64_ensure_sorted(csr.handle)
            # Invalidate pointer cache if it exists
            # (In JIT, we don't have the cache, so this is a no-op)
        return impl
    else:
        def impl(csr):
            _ffi_csr_f32_ensure_sorted(csr.handle)
        return impl


@overload_method(CSRType, 'ensure_sorted_checked')
def csr_ensure_sorted_checked_impl(csr):
    """Sort row indices if needed, return True if sorting was performed."""
    if csr.dtype == types.float64:
        def impl(csr):
            return _ffi_csr_f64_ensure_sorted_checked(csr.handle)
        return impl
    else:
        def impl(csr):
            return _ffi_csr_f32_ensure_sorted_checked(csr.handle)
        return impl


# =============================================================================
# CSC: Sorting Methods
# =============================================================================

@overload_method(CSCType, 'ensure_sorted')
def csc_ensure_sorted_impl(csc):
    """Sort all column indices in-place."""
    if csc.dtype == types.float64:
        def impl(csc):
            _ffi_csc_f64_ensure_sorted(csc.handle)
        return impl
    else:
        def impl(csc):
            _ffi_csc_f32_ensure_sorted(csc.handle)
        return impl


@overload_method(CSCType, 'ensure_sorted_checked')
def csc_ensure_sorted_checked_impl(csc):
    """Sort column indices if needed, return True if sorting was performed."""
    if csc.dtype == types.float64:
        def impl(csc):
            return _ffi_csc_f64_ensure_sorted_checked(csc.handle)
        return impl
    else:
        def impl(csc):
            return _ffi_csc_f32_ensure_sorted_checked(csc.handle)
        return impl
