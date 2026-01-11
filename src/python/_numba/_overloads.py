"""Numba method overloads for CSR/CSC sparse matrices.

This module provides Numba implementations for CSR/CSC methods.
When a JIT function calls a method like `csr.get_row(i)`, Numba uses
the overloaded implementation instead of calling Python.
"""

import operator
from numba import types
from numba.core import cgutils
from numba.extending import overload_method, overload_attribute, overload, intrinsic
from numba.core.imputils import lower_builtin

from ._types import CSRType, CSCType


# =============================================================================
# Helper: Create array from pointer (carray-like)
# =============================================================================

@intrinsic
def _carray_from_ptr(typingctx, ptr_ty, length_ty, dtype_ty):
    """Create a 1D array view from a pointer and length.
    
    This is similar to numba.carray but works with our pointer types.
    """
    if not isinstance(dtype_ty, types.DTypeSpec):
        return None
    
    dtype = dtype_ty.dtype
    ret_type = types.Array(dtype, 1, 'C')
    sig = ret_type(ptr_ty, length_ty, dtype_ty)
    
    def codegen(context, builder, signature, args):
        ptr, length, _ = args
        
        # Create array struct
        aryty = signature.return_type
        ary = context.make_array(aryty)(context, builder)
        
        # Set up the array
        itemsize = context.get_constant(types.intp, context.get_abi_sizeof(aryty.dtype))
        
        # Shape is (length,)
        shape = cgutils.pack_array(builder, [length])
        strides = cgutils.pack_array(builder, [itemsize])
        
        # Populate array struct
        context.populate_array(ary,
                              data=builder.bitcast(ptr, ary.data.type),
                              shape=shape,
                              strides=strides,
                              itemsize=itemsize,
                              meminfo=None)
        
        return ary._getvalue()
    
    return sig, codegen


# =============================================================================
# CSR: get_row method
# =============================================================================

@overload_method(CSRType, 'get_row')
def csr_get_row_overload(csr, row_idx):
    """Get values and indices for a single row.
    
    In Python: returns (np.ndarray, np.ndarray)
    In Numba: returns (typed array view, typed array view)
    """
    dtype = csr.dtype
    
    def impl(csr, row_idx):
        # Get pointers for this row
        val_ptr = csr.values_ptrs[row_idx]
        idx_ptr = csr.indices_ptrs[row_idx]
        length = csr.row_lens[row_idx]
        
        # Create array views (no copy!)
        values = _carray_from_ptr(val_ptr, length, dtype)
        indices = _carray_from_ptr(idx_ptr, length, types.int64)
        
        return values, indices
    
    return impl


# =============================================================================
# CSR: shape property
# =============================================================================

@overload_attribute(CSRType, 'shape')
def csr_shape_overload(csr):
    """Get matrix shape as (nrows, ncols) tuple."""
    def getter(csr):
        return (csr.nrows, csr.ncols)
    return getter


# =============================================================================
# CSC: get_col method
# =============================================================================

@overload_method(CSCType, 'get_col')
def csc_get_col_overload(csc, col_idx):
    """Get values and indices for a single column."""
    dtype = csc.dtype
    
    def impl(csc, col_idx):
        val_ptr = csc.values_ptrs[col_idx]
        idx_ptr = csc.indices_ptrs[col_idx]
        length = csc.col_lens[col_idx]
        
        values = _carray_from_ptr(val_ptr, length, dtype)
        indices = _carray_from_ptr(idx_ptr, length, types.int64)
        
        return values, indices
    
    return impl


# =============================================================================
# CSC: shape property
# =============================================================================

@overload_attribute(CSCType, 'shape')
def csc_shape_overload(csc):
    """Get matrix shape as (nrows, ncols) tuple."""
    def getter(csc):
        return (csc.nrows, csc.ncols)
    return getter


# =============================================================================
# len() overload
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
# CSR Iterator Support
# =============================================================================

class CSRIteratorType(types.SimpleIteratorType):
    """Iterator type for CSR row iteration."""
    
    def __init__(self, csr_type):
        self.csr_type = csr_type
        # Each iteration yields (values_array, indices_array)
        yield_type = types.Tuple([
            types.Array(csr_type.dtype, 1, 'C'),
            types.Array(types.int64, 1, 'C'),
        ])
        super().__init__(name=f'iter(CSR[{csr_type.dtype}])', yield_type=yield_type)


class CSCIteratorType(types.SimpleIteratorType):
    """Iterator type for CSC column iteration."""
    
    def __init__(self, csc_type):
        self.csc_type = csc_type
        yield_type = types.Tuple([
            types.Array(csc_type.dtype, 1, 'C'),
            types.Array(types.int64, 1, 'C'),
        ])
        super().__init__(name=f'iter(CSC[{csc_type.dtype}])', yield_type=yield_type)


# Register iterator models
from numba.extending import register_model, models

@register_model(CSRIteratorType)
class CSRIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('values_ptrs', types.CPointer(types.CPointer(fe_type.csr_type.dtype))),
            ('indices_ptrs', types.CPointer(types.CPointer(types.int64))),
            ('row_lens', types.CPointer(types.intp)),
            ('nrows', types.int64),
            ('index', types.int64),
        ]
        super().__init__(dmm, fe_type, members)


@register_model(CSCIteratorType)
class CSCIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('values_ptrs', types.CPointer(types.CPointer(fe_type.csc_type.dtype))),
            ('indices_ptrs', types.CPointer(types.CPointer(types.int64))),
            ('col_lens', types.CPointer(types.intp)),
            ('ncols', types.int64),
            ('index', types.int64),
        ]
        super().__init__(dmm, fe_type, members)


# =============================================================================
# getiter / iternext for CSR
# =============================================================================

from numba.core.imputils import iternext_impl, RefType
from numba.core.typing import signature

@lower_builtin('getiter', CSRType)
def csr_getiter_impl(context, builder, sig, args):
    """Create an iterator for CSR (for row in csr:)"""
    [csr_val] = args
    csr_type = sig.args[0]
    iter_type = CSRIteratorType(csr_type)
    
    csr = cgutils.create_struct_proxy(csr_type)(context, builder, value=csr_val)
    it = cgutils.create_struct_proxy(iter_type)(context, builder)
    
    it.values_ptrs = csr.values_ptrs
    it.indices_ptrs = csr.indices_ptrs
    it.row_lens = csr.row_lens
    it.nrows = csr.nrows
    it.index = context.get_constant(types.int64, 0)
    
    return it._getvalue()


@lower_builtin('iternext', CSRIteratorType)
@iternext_impl(RefType.UNTRACKED)
def csr_iternext_impl(context, builder, sig, args, result):
    """Get next row from CSR iterator."""
    [iter_type] = sig.args
    [iter_val] = args
    
    it = cgutils.create_struct_proxy(iter_type)(context, builder, value=iter_val)
    
    # Check if we have more rows
    nrows = it.nrows
    index = it.index
    is_valid = builder.icmp_signed('<', index, nrows)
    
    with builder.if_then(is_valid):
        # Get pointers for current row
        val_ptr_ptr = builder.gep(it.values_ptrs, [index])
        val_ptr = builder.load(val_ptr_ptr)
        
        idx_ptr_ptr = builder.gep(it.indices_ptrs, [index])
        idx_ptr = builder.load(idx_ptr_ptr)
        
        len_ptr = builder.gep(it.row_lens, [index])
        length = builder.load(len_ptr)
        
        # Create arrays for values and indices
        dtype = iter_type.csr_type.dtype
        values_arrty = types.Array(dtype, 1, 'C')
        indices_arrty = types.Array(types.int64, 1, 'C')
        
        # Build value array
        val_ary = context.make_array(values_arrty)(context, builder)
        val_itemsize = context.get_constant(types.intp, context.get_abi_sizeof(dtype))
        val_shape = cgutils.pack_array(builder, [length])
        val_strides = cgutils.pack_array(builder, [val_itemsize])
        context.populate_array(val_ary,
                              data=builder.bitcast(val_ptr, val_ary.data.type),
                              shape=val_shape,
                              strides=val_strides,
                              itemsize=val_itemsize,
                              meminfo=None)
        
        # Build indices array
        idx_ary = context.make_array(indices_arrty)(context, builder)
        idx_itemsize = context.get_constant(types.intp, 8)  # int64 = 8 bytes
        idx_shape = cgutils.pack_array(builder, [length])
        idx_strides = cgutils.pack_array(builder, [idx_itemsize])
        context.populate_array(idx_ary,
                              data=builder.bitcast(idx_ptr, idx_ary.data.type),
                              shape=idx_shape,
                              strides=idx_strides,
                              itemsize=idx_itemsize,
                              meminfo=None)
        
        # Create tuple (values, indices)
        tuple_type = iter_type.yield_type
        row_tuple = context.make_tuple(builder, tuple_type, 
                                       [val_ary._getvalue(), idx_ary._getvalue()])
        result.yield_(row_tuple)
        
        # Increment index
        next_index = builder.add(index, context.get_constant(types.int64, 1))
        it.index = next_index
    
    result.set_valid(is_valid)


# =============================================================================
# getiter / iternext for CSC
# =============================================================================

@lower_builtin('getiter', CSCType)
def csc_getiter_impl(context, builder, sig, args):
    """Create an iterator for CSC (for col in csc:)"""
    [csc_val] = args
    csc_type = sig.args[0]
    iter_type = CSCIteratorType(csc_type)
    
    csc = cgutils.create_struct_proxy(csc_type)(context, builder, value=csc_val)
    it = cgutils.create_struct_proxy(iter_type)(context, builder)
    
    it.values_ptrs = csc.values_ptrs
    it.indices_ptrs = csc.indices_ptrs
    it.col_lens = csc.col_lens
    it.ncols = csc.ncols
    it.index = context.get_constant(types.int64, 0)
    
    return it._getvalue()


@lower_builtin('iternext', CSCIteratorType)
@iternext_impl(RefType.UNTRACKED)
def csc_iternext_impl(context, builder, sig, args, result):
    """Get next column from CSC iterator."""
    [iter_type] = sig.args
    [iter_val] = args
    
    it = cgutils.create_struct_proxy(iter_type)(context, builder, value=iter_val)
    
    ncols = it.ncols
    index = it.index
    is_valid = builder.icmp_signed('<', index, ncols)
    
    with builder.if_then(is_valid):
        val_ptr_ptr = builder.gep(it.values_ptrs, [index])
        val_ptr = builder.load(val_ptr_ptr)
        
        idx_ptr_ptr = builder.gep(it.indices_ptrs, [index])
        idx_ptr = builder.load(idx_ptr_ptr)
        
        len_ptr = builder.gep(it.col_lens, [index])
        length = builder.load(len_ptr)
        
        dtype = iter_type.csc_type.dtype
        values_arrty = types.Array(dtype, 1, 'C')
        indices_arrty = types.Array(types.int64, 1, 'C')
        
        val_ary = context.make_array(values_arrty)(context, builder)
        val_itemsize = context.get_constant(types.intp, context.get_abi_sizeof(dtype))
        val_shape = cgutils.pack_array(builder, [length])
        val_strides = cgutils.pack_array(builder, [val_itemsize])
        context.populate_array(val_ary,
                              data=builder.bitcast(val_ptr, val_ary.data.type),
                              shape=val_shape,
                              strides=val_strides,
                              itemsize=val_itemsize,
                              meminfo=None)
        
        idx_ary = context.make_array(indices_arrty)(context, builder)
        idx_itemsize = context.get_constant(types.intp, 8)
        idx_shape = cgutils.pack_array(builder, [length])
        idx_strides = cgutils.pack_array(builder, [idx_itemsize])
        context.populate_array(idx_ary,
                              data=builder.bitcast(idx_ptr, idx_ary.data.type),
                              shape=idx_shape,
                              strides=idx_strides,
                              itemsize=idx_itemsize,
                              meminfo=None)
        
        tuple_type = iter_type.yield_type
        col_tuple = context.make_tuple(builder, tuple_type,
                                       [val_ary._getvalue(), idx_ary._getvalue()])
        result.yield_(col_tuple)
        
        next_index = builder.add(index, context.get_constant(types.int64, 1))
        it.index = next_index
    
    result.set_valid(is_valid)


# =============================================================================
# Type inference for iterators
# =============================================================================

from numba.core.typing.templates import AbstractTemplate, signature, infer_global
from numba.core.typing import typeof

@infer_global(iter)
class CSRIterType(AbstractTemplate):
    def generic(self, args, kws):
        if len(args) == 1 and isinstance(args[0], CSRType):
            return signature(CSRIteratorType(args[0]), args[0])


@infer_global(iter)
class CSCIterType(AbstractTemplate):
    def generic(self, args, kws):
        if len(args) == 1 and isinstance(args[0], CSCType):
            return signature(CSCIteratorType(args[0]), args[0])
