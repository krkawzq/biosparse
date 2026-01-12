"""Boxing and unboxing for CSR/CSC types.

This module implements the conversion between Python objects and Numba's
internal representation.
"""

import numpy as np
from numba import types
from numba.core import cgutils
from numba.extending import unbox, box, NativeValue

from ._types import CSRType, CSCType


# =============================================================================
# Helper Functions
# =============================================================================

def _get_numpy_data_ptr(c, arr_obj, target_ptr_type):
    """Extract data pointer from a NumPy array object.
    
    Args:
        c: Unboxing context
        arr_obj: Python NumPy array object (LLVM value)
        target_ptr_type: Target LLVM pointer type
        
    Returns:
        LLVM pointer value
    """
    # Get arr.ctypes.data as integer
    ctypes_obj = c.pyapi.object_getattr_string(arr_obj, "ctypes")
    data_obj = c.pyapi.object_getattr_string(ctypes_obj, "data")
    data_int = c.pyapi.long_as_longlong(data_obj)
    c.pyapi.decref(data_obj)
    c.pyapi.decref(ctypes_obj)
    
    # Convert integer to typed pointer
    return c.builder.inttoptr(data_int, target_ptr_type)


def _get_module_path():
    """Dynamically determine the correct module path for sparse matrices.
    
    Returns the module path string that can be used to import sparse types.
    """
    # This will be evaluated at box time in the Python interpreter
    import sys
    if 'biosparse' in sys.modules:
        return "biosparse._binding._sparse"
    else:
        return "python._binding._sparse"


# =============================================================================
# CSR Unboxing: Python -> Numba
# =============================================================================

@unbox(CSRType)
def unbox_csr(typ, obj, c):
    """Convert a Python CSR object to Numba's internal CSRModel.
    
    This extracts all necessary data from the Python object for efficient
    access in JIT-compiled code.
    
    Args:
        typ: The Numba CSRType
        obj: The Python CSR object (LLVM value pointing to PyObject*)
        c: The unboxing context
        
    Returns:
        NativeValue containing the CSRModel struct
    """
    csr_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    
    # 1. Extract handle as integer
    handle_obj = c.pyapi.object_getattr_string(obj, "handle_as_int")
    handle_int = c.pyapi.long_as_longlong(handle_obj)
    c.pyapi.decref(handle_obj)
    csr_struct.handle = c.builder.inttoptr(handle_int, cgutils.voidptr_t)
    
    # 2. Extract dimensions
    nrows_obj = c.pyapi.object_getattr_string(obj, "nrows")
    ncols_obj = c.pyapi.object_getattr_string(obj, "ncols")
    nnz_obj = c.pyapi.object_getattr_string(obj, "nnz")
    
    csr_struct.nrows = c.pyapi.long_as_longlong(nrows_obj)
    csr_struct.ncols = c.pyapi.long_as_longlong(ncols_obj)
    csr_struct.nnz = c.pyapi.long_as_longlong(nnz_obj)
    
    c.pyapi.decref(nrows_obj)
    c.pyapi.decref(ncols_obj)
    c.pyapi.decref(nnz_obj)
    
    # 3. Call _prepare_numba_pointers() to get pointer arrays
    # This returns (values_ptrs, indices_ptrs, row_lens) as NumPy arrays
    prepare_fn = c.pyapi.object_getattr_string(obj, "_prepare_numba_pointers")
    empty_tuple = c.pyapi.tuple_new(0)
    ptr_tuple = c.pyapi.call(prepare_fn, empty_tuple)
    c.pyapi.decref(prepare_fn)
    c.pyapi.decref(empty_tuple)
    
    # 4. Extract pointer arrays from the tuple
    # Note: tuple_getitem returns a borrowed reference, so we must NOT decref these
    values_ptrs_arr = c.pyapi.tuple_getitem(ptr_tuple, 0)
    indices_ptrs_arr = c.pyapi.tuple_getitem(ptr_tuple, 1)
    row_lens_arr = c.pyapi.tuple_getitem(ptr_tuple, 2)

    # 5. Get data pointers from NumPy arrays
    csr_struct.values_ptrs = _get_numpy_data_ptr(c, values_ptrs_arr, csr_struct.values_ptrs.type)
    csr_struct.indices_ptrs = _get_numpy_data_ptr(c, indices_ptrs_arr, csr_struct.indices_ptrs.type)
    csr_struct.row_lens = _get_numpy_data_ptr(c, row_lens_arr, csr_struct.row_lens.type)

    # Only decref the tuple itself (which owns references to the arrays)
    # DO NOT decref values_ptrs_arr, indices_ptrs_arr, row_lens_arr - they are borrowed refs
    c.pyapi.decref(ptr_tuple)
    
    # 6. Set meminfo to NULL (Python owns the data)
    null_meminfo = c.context.get_constant_null(types.MemInfoPointer(types.voidptr))
    csr_struct.meminfo = null_meminfo
    
    # 7. Set owns_data to False (Python owns it)
    csr_struct.owns_data = cgutils.false_bit
    
    # 8. Check for errors
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    
    return NativeValue(csr_struct._getvalue(), is_error=is_error)


# =============================================================================
# CSR Boxing: Numba -> Python
# =============================================================================

@box(CSRType)
def box_csr(typ, val, c):
    """Convert a Numba CSRModel to a Python CSR object.
    
    This is called when a JIT function returns a CSR. We create a Python
    CSRF32/CSRF64 object that takes ownership of the handle.
    
    Args:
        typ: The Numba CSRType
        val: The LLVM value (CSRModel struct)
        c: The boxing context
        
    Returns:
        LLVM value pointing to the Python object
    """
    csr_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    
    # 1. Get the handle as an integer
    handle_int = c.builder.ptrtoint(csr_struct.handle, cgutils.intp_t)
    handle_obj = c.pyapi.long_from_longlong(handle_int)
    
    # 2. Determine the Python class name
    if typ.dtype == types.float32:
        cls_name = "CSRF32"
    else:
        cls_name = "CSRF64"
    
    # 3. Import module and class
    # Use insert_const_string to create LLVM string constant
    mod_name_ptr = c.context.insert_const_string(c.builder.module, "_binding._sparse")
    mod_obj = c.pyapi.import_module(mod_name_ptr)
    
    cls = c.pyapi.object_getattr_string(mod_obj, cls_name)
    
    # 4. Determine ownership - convert LLVM bool to Python bool
    owns_data = csr_struct.owns_data
    owns_obj = c.pyapi.bool_from_bool(owns_data)
    
    # 5. Call _from_handle(handle_int, owns_handle)
    from_handle = c.pyapi.object_getattr_string(cls, "_from_handle")
    args = c.pyapi.tuple_pack((handle_obj, owns_obj))
    result = c.pyapi.call(from_handle, args)
    
    # 6. If JIT owned the data, transfer ownership to Python
    # The meminfo will be released by NRT automatically
    
    # 7. Cleanup
    c.pyapi.decref(handle_obj)
    c.pyapi.decref(owns_obj)
    c.pyapi.decref(args)
    c.pyapi.decref(from_handle)
    c.pyapi.decref(cls)
    c.pyapi.decref(mod_obj)
    
    return result


# =============================================================================
# CSC Unboxing: Python -> Numba
# =============================================================================

@unbox(CSCType)
def unbox_csc(typ, obj, c):
    """Convert a Python CSC object to Numba's internal CSCModel.
    
    Args:
        typ: The Numba CSCType
        obj: The Python CSC object (LLVM value pointing to PyObject*)
        c: The unboxing context
        
    Returns:
        NativeValue containing the CSCModel struct
    """
    csc_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    
    # 1. Extract handle
    handle_obj = c.pyapi.object_getattr_string(obj, "handle_as_int")
    handle_int = c.pyapi.long_as_longlong(handle_obj)
    c.pyapi.decref(handle_obj)
    csc_struct.handle = c.builder.inttoptr(handle_int, cgutils.voidptr_t)
    
    # 2. Extract dimensions
    nrows_obj = c.pyapi.object_getattr_string(obj, "nrows")
    ncols_obj = c.pyapi.object_getattr_string(obj, "ncols")
    nnz_obj = c.pyapi.object_getattr_string(obj, "nnz")
    
    csc_struct.nrows = c.pyapi.long_as_longlong(nrows_obj)
    csc_struct.ncols = c.pyapi.long_as_longlong(ncols_obj)
    csc_struct.nnz = c.pyapi.long_as_longlong(nnz_obj)
    
    c.pyapi.decref(nrows_obj)
    c.pyapi.decref(ncols_obj)
    c.pyapi.decref(nnz_obj)
    
    # 3. Call _prepare_numba_pointers()
    prepare_fn = c.pyapi.object_getattr_string(obj, "_prepare_numba_pointers")
    empty_tuple = c.pyapi.tuple_new(0)
    ptr_tuple = c.pyapi.call(prepare_fn, empty_tuple)
    c.pyapi.decref(prepare_fn)
    c.pyapi.decref(empty_tuple)
    
    # 4. Extract pointer arrays
    # Note: tuple_getitem returns a borrowed reference, so we must NOT decref these
    values_ptrs_arr = c.pyapi.tuple_getitem(ptr_tuple, 0)
    indices_ptrs_arr = c.pyapi.tuple_getitem(ptr_tuple, 1)
    col_lens_arr = c.pyapi.tuple_getitem(ptr_tuple, 2)

    # 5. Get data pointers
    csc_struct.values_ptrs = _get_numpy_data_ptr(c, values_ptrs_arr, csc_struct.values_ptrs.type)
    csc_struct.indices_ptrs = _get_numpy_data_ptr(c, indices_ptrs_arr, csc_struct.indices_ptrs.type)
    csc_struct.col_lens = _get_numpy_data_ptr(c, col_lens_arr, csc_struct.col_lens.type)

    # Only decref the tuple itself (which owns references to the arrays)
    # DO NOT decref values_ptrs_arr, indices_ptrs_arr, col_lens_arr - they are borrowed refs
    c.pyapi.decref(ptr_tuple)
    
    # 6. Set meminfo to NULL (Python owns the data)
    null_meminfo = c.context.get_constant_null(types.MemInfoPointer(types.voidptr))
    csc_struct.meminfo = null_meminfo
    
    # 7. Set owns_data to False
    csc_struct.owns_data = cgutils.false_bit
    
    # 8. Check for errors
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    
    return NativeValue(csc_struct._getvalue(), is_error=is_error)


# =============================================================================
# CSC Boxing: Numba -> Python
# =============================================================================

@box(CSCType)
def box_csc(typ, val, c):
    """Convert a Numba CSCModel to a Python CSC object.
    
    Args:
        typ: The Numba CSCType
        val: The LLVM value (CSCModel struct)
        c: The boxing context
        
    Returns:
        LLVM value pointing to the Python object
    """
    csc_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    
    # 1. Get the handle as an integer
    handle_int = c.builder.ptrtoint(csc_struct.handle, cgutils.intp_t)
    handle_obj = c.pyapi.long_from_longlong(handle_int)
    
    # 2. Determine the Python class name
    if typ.dtype == types.float32:
        cls_name = "CSCF32"
    else:
        cls_name = "CSCF64"
    
    # 3. Import module and class
    mod_name_ptr = c.context.insert_const_string(c.builder.module, "_binding._sparse")
    mod_obj = c.pyapi.import_module(mod_name_ptr)
    
    cls = c.pyapi.object_getattr_string(mod_obj, cls_name)
    
    # 4. Determine ownership - convert LLVM bool to Python bool
    owns_data = csc_struct.owns_data
    owns_obj = c.pyapi.bool_from_bool(owns_data)
    
    # 5. Call _from_handle(handle_int, owns_handle)
    from_handle = c.pyapi.object_getattr_string(cls, "_from_handle")
    args = c.pyapi.tuple_pack((handle_obj, owns_obj))
    result = c.pyapi.call(from_handle, args)
    
    # 6. Cleanup
    c.pyapi.decref(handle_obj)
    c.pyapi.decref(owns_obj)
    c.pyapi.decref(args)
    c.pyapi.decref(from_handle)
    c.pyapi.decref(cls)
    c.pyapi.decref(mod_obj)
    
    return result
