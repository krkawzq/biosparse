"""Iterator implementations for CSR/CSC types.

This module provides full iterator support for sparse matrices, enabling
`for values, indices in csr:` syntax in JIT-compiled code.

Key design: The iterator stores the FFI handle and fetches row/column data
on-demand using FFI calls. This ensures correct behavior for both:
- Python-created objects (passed to JIT)
- JIT-created objects (e.g., from clone(), to_csc(), slice_rows())
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
    We create an iterator struct that stores the handle for FFI calls.
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
    it.handle = csr.handle
    it.nrows = csr.nrows
    it.index_ptr = index_alloca
    # Store dtype info for FFI dispatch
    it.dtype_is_f64 = context.get_constant(types.boolean, csr_type.dtype == types.float64)

    # Return the iterator value
    return builder.load(iter_alloca)


# =============================================================================
# CSR Iterator: iternext
# =============================================================================

@lower_builtin('iternext', CSRIteratorType)
@iternext_impl(RefType.BORROWED)
def csr_iternext_impl(context, builder, sig, args, result):
    """Get the next row from CSR iterator.

    This is called for each iteration of the loop. We use FFI to fetch
    the row data (values, indices, length) and yield them as numpy arrays.
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
        handle = it.handle
        dtype = iter_type.csr_type.dtype
        
        # Determine FFI function suffix based on dtype
        if dtype == types.float64:
            val_ptr_fn_name = "csr_f64_row_values_ptr"
            idx_ptr_fn_name = "csr_f64_row_indices_ptr"
            row_len_fn_name = "csr_f64_row_len"
        else:
            val_ptr_fn_name = "csr_f32_row_values_ptr"
            idx_ptr_fn_name = "csr_f32_row_indices_ptr"
            row_len_fn_name = "csr_f32_row_len"
        
        # FFI call to get row values pointer
        # Signature: void* csr_f64_row_values_ptr(void* handle, size_t row)
        fnty_ptr = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer(), lir.IntType(64)]
        )
        fn_val_ptr = cgutils.get_or_insert_function(builder.module, fnty_ptr, val_ptr_fn_name)
        val_ptr = builder.call(fn_val_ptr, [handle, index])
        
        # FFI call to get row indices pointer
        fn_idx_ptr = cgutils.get_or_insert_function(builder.module, fnty_ptr, idx_ptr_fn_name)
        idx_ptr = builder.call(fn_idx_ptr, [handle, index])
        
        # FFI call to get row length
        # Signature: int64 csr_f64_row_len(void* handle, size_t row)
        fnty_len = lir.FunctionType(
            lir.IntType(64),
            [lir.IntType(8).as_pointer(), lir.IntType(64)]
        )
        fn_row_len = cgutils.get_or_insert_function(builder.module, fnty_len, row_len_fn_name)
        length = builder.call(fn_row_len, [handle, index])

        # Create array views for values and indices
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
    it.handle = csc.handle
    it.ncols = csc.ncols
    it.index_ptr = index_alloca
    it.dtype_is_f64 = context.get_constant(types.boolean, csc_type.dtype == types.float64)

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
        handle = it.handle
        dtype = iter_type.csc_type.dtype
        
        # Determine FFI function suffix based on dtype
        if dtype == types.float64:
            val_ptr_fn_name = "csc_f64_col_values_ptr"
            idx_ptr_fn_name = "csc_f64_col_indices_ptr"
            col_len_fn_name = "csc_f64_col_len"
        else:
            val_ptr_fn_name = "csc_f32_col_values_ptr"
            idx_ptr_fn_name = "csc_f32_col_indices_ptr"
            col_len_fn_name = "csc_f32_col_len"
        
        # FFI call to get column values pointer
        fnty_ptr = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer(), lir.IntType(64)]
        )
        fn_val_ptr = cgutils.get_or_insert_function(builder.module, fnty_ptr, val_ptr_fn_name)
        val_ptr = builder.call(fn_val_ptr, [handle, index])
        
        # FFI call to get column indices pointer
        fn_idx_ptr = cgutils.get_or_insert_function(builder.module, fnty_ptr, idx_ptr_fn_name)
        idx_ptr = builder.call(fn_idx_ptr, [handle, index])
        
        # FFI call to get column length
        fnty_len = lir.FunctionType(
            lir.IntType(64),
            [lir.IntType(8).as_pointer(), lir.IntType(64)]
        )
        fn_col_len = cgutils.get_or_insert_function(builder.module, fnty_len, col_len_fn_name)
        length = builder.call(fn_col_len, [handle, index])

        # Create array views
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
