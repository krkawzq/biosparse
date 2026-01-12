"""Iterator implementations for CSR/CSC types.

This module provides full iterator support for sparse matrices, enabling
`for values, indices in csr:` syntax in JIT-compiled code.

Key design: The iterator uses a pointer (index_ptr) to store the current index,
allowing the index to be mutated across iterations even though Numba passes
the iterator struct by value.
"""

from numba import types
from numba.core import cgutils
from numba.core.imputils import lower_builtin, iternext_impl, RefType
import llvmlite.ir as lir

from ._types import CSRType, CSCType, CSRIteratorType, CSCIteratorType


# =============================================================================
# CSR Iterator: getiter
# =============================================================================

@lower_builtin('getiter', CSRType)
def csr_getiter_impl(context, builder, sig, args):
    """Create an iterator for CSR matrix.

    This is called when entering a `for` loop with a CSR object.
    We create an iterator struct that references the parent's data.
    """
    [csr_val] = args
    csr_type = sig.args[0]
    iter_type = CSRIteratorType(csr_type)

    # Extract data from CSR struct
    csr = cgutils.create_struct_proxy(csr_type)(context, builder, value=csr_val)

    # Allocate iterator struct on stack
    iter_alloca = cgutils.alloca_once(builder, context.get_value_type(iter_type))
    it = cgutils.create_struct_proxy(iter_type)(context, builder, ref=iter_alloca)

    # Allocate index on stack - this persists across iterations
    index_alloca = cgutils.alloca_once_value(builder, context.get_constant(types.int64, 0))

    # Initialize iterator fields
    it.values_ptrs = csr.values_ptrs
    it.indices_ptrs = csr.indices_ptrs
    it.row_lens = csr.row_lens
    it.nrows = csr.nrows
    it.index_ptr = index_alloca  # Store pointer to the mutable index

    # Return the iterator value
    return builder.load(iter_alloca)


# =============================================================================
# CSR Iterator: iternext
# =============================================================================

@lower_builtin('iternext', CSRIteratorType)
@iternext_impl(RefType.BORROWED)
def csr_iternext_impl(context, builder, sig, args, result):
    """Get the next row from CSR iterator.

    This is called for each iteration of the loop. We yield (values, indices)
    tuples and increment the current index via the pointer.
    """
    [iter_type] = sig.args
    [iter_val] = args

    # Create iterator proxy from value
    it = cgutils.create_struct_proxy(iter_type)(context, builder, value=iter_val)

    # Load current index from pointer
    index = builder.load(it.index_ptr)
    nrows = it.nrows

    # Check if we have more rows
    is_valid = builder.icmp_signed('<', index, nrows)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        # Get pointers for current row
        val_ptr_addr = builder.gep(it.values_ptrs, [index])
        val_ptr = builder.load(val_ptr_addr)

        idx_ptr_addr = builder.gep(it.indices_ptrs, [index])
        idx_ptr = builder.load(idx_ptr_addr)

        len_addr = builder.gep(it.row_lens, [index])
        length = builder.load(len_addr)

        # Create array views for values and indices
        dtype = iter_type.csr_type.dtype
        val_array_type = types.Array(dtype, 1, 'C')
        idx_array_type = types.Array(types.int64, 1, 'C')

        # Build value array
        val_ary = context.make_array(val_array_type)(context, builder)
        val_llvm_dtype = context.get_data_type(dtype)
        val_itemsize = context.get_constant(types.intp, context.get_abi_sizeof(val_llvm_dtype))
        val_shape = cgutils.pack_array(builder, [length])
        val_strides = cgutils.pack_array(builder, [val_itemsize])
        context.populate_array(
            val_ary,
            data=builder.bitcast(val_ptr, val_ary.data.type),
            shape=val_shape,
            strides=val_strides,
            itemsize=val_itemsize,
            meminfo=None
        )

        # Build indices array
        idx_ary = context.make_array(idx_array_type)(context, builder)
        idx_llvm_dtype = context.get_data_type(types.int64)
        idx_itemsize = context.get_constant(types.intp, context.get_abi_sizeof(idx_llvm_dtype))
        idx_shape = cgutils.pack_array(builder, [length])
        idx_strides = cgutils.pack_array(builder, [idx_itemsize])
        context.populate_array(
            idx_ary,
            data=builder.bitcast(idx_ptr, idx_ary.data.type),
            shape=idx_shape,
            strides=idx_strides,
            itemsize=idx_itemsize,
            meminfo=None
        )

        # Create tuple (values, indices)
        row_tuple = context.make_tuple(
            builder,
            iter_type.yield_type,
            [val_ary._getvalue(), idx_ary._getvalue()]
        )
        result.yield_(row_tuple)

        # Increment index - write to pointer so it persists
        next_index = builder.add(index, context.get_constant(types.int64, 1))
        builder.store(next_index, it.index_ptr)


# =============================================================================
# CSC Iterator: getiter
# =============================================================================

@lower_builtin('getiter', CSCType)
def csc_getiter_impl(context, builder, sig, args):
    """Create an iterator for CSC matrix.

    This is called when entering a `for` loop with a CSC object.
    """
    [csc_val] = args
    csc_type = sig.args[0]
    iter_type = CSCIteratorType(csc_type)

    # Extract data from CSC struct
    csc = cgutils.create_struct_proxy(csc_type)(context, builder, value=csc_val)

    # Allocate iterator struct on stack
    iter_alloca = cgutils.alloca_once(builder, context.get_value_type(iter_type))
    it = cgutils.create_struct_proxy(iter_type)(context, builder, ref=iter_alloca)

    # Allocate index on stack - this persists across iterations
    index_alloca = cgutils.alloca_once_value(builder, context.get_constant(types.int64, 0))

    # Initialize iterator fields
    it.values_ptrs = csc.values_ptrs
    it.indices_ptrs = csc.indices_ptrs
    it.col_lens = csc.col_lens
    it.ncols = csc.ncols
    it.index_ptr = index_alloca

    # Return the iterator value
    return builder.load(iter_alloca)


# =============================================================================
# CSC Iterator: iternext
# =============================================================================

@lower_builtin('iternext', CSCIteratorType)
@iternext_impl(RefType.BORROWED)
def csc_iternext_impl(context, builder, sig, args, result):
    """Get the next column from CSC iterator."""
    [iter_type] = sig.args
    [iter_val] = args

    # Create iterator proxy from value
    it = cgutils.create_struct_proxy(iter_type)(context, builder, value=iter_val)

    # Load current index from pointer
    index = builder.load(it.index_ptr)
    ncols = it.ncols

    # Check if we have more columns
    is_valid = builder.icmp_signed('<', index, ncols)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        # Get pointers for current column
        val_ptr_addr = builder.gep(it.values_ptrs, [index])
        val_ptr = builder.load(val_ptr_addr)

        idx_ptr_addr = builder.gep(it.indices_ptrs, [index])
        idx_ptr = builder.load(idx_ptr_addr)

        len_addr = builder.gep(it.col_lens, [index])
        length = builder.load(len_addr)

        # Create array views
        dtype = iter_type.csc_type.dtype
        val_array_type = types.Array(dtype, 1, 'C')
        idx_array_type = types.Array(types.int64, 1, 'C')

        # Build value array
        val_ary = context.make_array(val_array_type)(context, builder)
        val_llvm_dtype = context.get_data_type(dtype)
        val_itemsize = context.get_constant(types.intp, context.get_abi_sizeof(val_llvm_dtype))
        val_shape = cgutils.pack_array(builder, [length])
        val_strides = cgutils.pack_array(builder, [val_itemsize])
        context.populate_array(
            val_ary,
            data=builder.bitcast(val_ptr, val_ary.data.type),
            shape=val_shape,
            strides=val_strides,
            itemsize=val_itemsize,
            meminfo=None
        )

        # Build indices array
        idx_ary = context.make_array(idx_array_type)(context, builder)
        idx_llvm_dtype = context.get_data_type(types.int64)
        idx_itemsize = context.get_constant(types.intp, context.get_abi_sizeof(idx_llvm_dtype))
        idx_shape = cgutils.pack_array(builder, [length])
        idx_strides = cgutils.pack_array(builder, [idx_itemsize])
        context.populate_array(
            idx_ary,
            data=builder.bitcast(idx_ptr, idx_ary.data.type),
            shape=idx_shape,
            strides=idx_strides,
            itemsize=idx_itemsize,
            meminfo=None
        )

        # Create tuple (values, indices)
        col_tuple = context.make_tuple(
            builder,
            iter_type.yield_type,
            [val_ary._getvalue(), idx_ary._getvalue()]
        )
        result.yield_(col_tuple)

        # Increment index - write to pointer so it persists
        next_index = builder.add(index, context.get_constant(types.int64, 1))
        builder.store(next_index, it.index_ptr)
