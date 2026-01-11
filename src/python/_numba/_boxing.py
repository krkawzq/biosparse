import numpy as np
from numba import types
from numba.core import cgutils
from numba.extending import unbox, box, NativeValue

from ._types import CSRType, CSCType, CSRFloat32Type, CSRFloat64Type, CSCFloat32Type, CSCFloat64Type
from .._binding._sparse import CSRF32, CSRF64, CSCF32, CSCF64
from .._binding._cffi import ffi, lib


# =============================================================================
# CSR Unboxing: Python -> Numba
# =============================================================================

@unbox(CSRType)
def unbox_csr(typ, obj, c):
    """Convert a Python CSR object to Numba's internal representation.
    
    This is called when a Python CSR is passed to a JIT function.
    We extract all the data pointers once, avoiding repeated FFI calls in loops.
    
    Args:
        typ: The Numba type (CSRType)
        obj: The Python object (LLVM value pointing to PyObject*)
        c: The unboxing context
    
    Returns:
        NativeValue containing the CSR struct
    """
    csr_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    
    # Get handle_as_int from Python object
    handle_obj = c.pyapi.object_getattr_string(obj, "handle_as_int")
    handle_int = c.pyapi.long_as_longlong(handle_obj)
    c.pyapi.decref(handle_obj)
    csr_struct.handle = c.builder.inttoptr(handle_int, cgutils.voidptr_t)
    
    # Get dimensions
    nrows_obj = c.pyapi.object_getattr_string(obj, "nrows")
    ncols_obj = c.pyapi.object_getattr_string(obj, "ncols")
    nnz_obj = c.pyapi.object_getattr_string(obj, "nnz")
    
    csr_struct.nrows = c.pyapi.long_as_longlong(nrows_obj)
    csr_struct.ncols = c.pyapi.long_as_longlong(ncols_obj)
    csr_struct.nnz = c.pyapi.long_as_longlong(nnz_obj)
    
    c.pyapi.decref(nrows_obj)
    c.pyapi.decref(ncols_obj)
    c.pyapi.decref(nnz_obj)
    
    # For now, set pointer arrays to NULL
    # TODO: Call _prepare_numba_pointers() and extract the arrays
    null_voidptr = c.context.get_constant_null(types.voidptr)
    csr_struct.values_ptrs = c.builder.bitcast(null_voidptr, 
        csr_struct.values_ptrs.type)
    csr_struct.indices_ptrs = c.builder.bitcast(null_voidptr,
        csr_struct.indices_ptrs.type)
    csr_struct.row_lens = c.builder.bitcast(null_voidptr,
        csr_struct.row_lens.type)
    
    # Set meminfo to NULL (Python object owns the data)
    null_ptr = c.context.get_constant_null(types.MemInfoPointer(types.voidptr))
    csr_struct.meminfo = null_ptr
    
    # Check for errors
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    
    return NativeValue(csr_struct._getvalue(), is_error=is_error)


# =============================================================================
# CSR Boxing: Numba -> Python
# =============================================================================

@box(CSRType)
def box_csr(typ, val, c):
    """Convert a Numba CSR to a Python object.
    
    This is called when a JIT function returns a CSR.
    We create a Python CSRF32/CSRF64 object that owns the handle.
    
    Args:
        typ: The Numba type (CSRType)
        val: The LLVM value (CSR struct)
        c: The boxing context
    
    Returns:
        LLVM value pointing to the Python object
    """
    csr_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    
    # Get the handle as an integer
    handle_int = c.builder.ptrtoint(csr_struct.handle, cgutils.intp_t)
    handle_obj = c.pyapi.long_from_longlong(handle_int)
    
    # Import the appropriate class
    if typ.dtype == types.float32:
        mod = c.pyapi.import_module_noblock("scl._binding._sparse")
        cls = c.pyapi.object_getattr_string(mod, "CSRF32")
    else:
        mod = c.pyapi.import_module_noblock("scl._binding._sparse")
        cls = c.pyapi.object_getattr_string(mod, "CSRF64")
    
    # Call CSRF64._from_handle(handle_int, owns_handle=True)
    from_handle = c.pyapi.object_getattr_string(cls, "_from_handle")
    
    # Create True for owns_handle
    true_obj = c.pyapi.make_true()
    
    # Build args tuple
    args = c.pyapi.tuple_pack([handle_obj, true_obj])
    
    # Call _from_handle
    result = c.pyapi.call(from_handle, args)
    
    # If this handle was created in JIT (has meminfo), we need to clear
    # the meminfo to prevent double-free, since Python now owns it
    # TODO: Implement meminfo clearing
    
    # Cleanup
    c.pyapi.decref(handle_obj)
    c.pyapi.decref(true_obj)
    c.pyapi.decref(args)
    c.pyapi.decref(from_handle)
    c.pyapi.decref(cls)
    c.pyapi.decref(mod)
    
    return result


# =============================================================================
# CSC Unboxing: Python -> Numba
# =============================================================================

@unbox(CSCType)
def unbox_csc(typ, obj, c):
    """Convert a Python CSC object to Numba's internal representation."""
    csc_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    
    # Get handle_as_int
    handle_obj = c.pyapi.object_getattr_string(obj, "handle_as_int")
    handle_int = c.pyapi.long_as_longlong(handle_obj)
    c.pyapi.decref(handle_obj)
    csc_struct.handle = c.builder.inttoptr(handle_int, cgutils.voidptr_t)
    
    # Get dimensions
    nrows_obj = c.pyapi.object_getattr_string(obj, "nrows")
    ncols_obj = c.pyapi.object_getattr_string(obj, "ncols")
    nnz_obj = c.pyapi.object_getattr_string(obj, "nnz")
    
    csc_struct.nrows = c.pyapi.long_as_longlong(nrows_obj)
    csc_struct.ncols = c.pyapi.long_as_longlong(ncols_obj)
    csc_struct.nnz = c.pyapi.long_as_longlong(nnz_obj)
    
    c.pyapi.decref(nrows_obj)
    c.pyapi.decref(ncols_obj)
    c.pyapi.decref(nnz_obj)
    
    # For now, set pointer arrays to NULL
    null_voidptr = c.context.get_constant_null(types.voidptr)
    csc_struct.values_ptrs = c.builder.bitcast(null_voidptr,
        csc_struct.values_ptrs.type)
    csc_struct.indices_ptrs = c.builder.bitcast(null_voidptr,
        csc_struct.indices_ptrs.type)
    csc_struct.col_lens = c.builder.bitcast(null_voidptr,
        csc_struct.col_lens.type)
    
    # Set meminfo to NULL (Python owns the data)
    null_ptr = c.context.get_constant_null(types.MemInfoPointer(types.voidptr))
    csc_struct.meminfo = null_ptr
    
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    
    return NativeValue(csc_struct._getvalue(), is_error=is_error)


# =============================================================================
# CSC Boxing: Numba -> Python
# =============================================================================

@box(CSCType)
def box_csc(typ, val, c):
    """Convert a Numba CSC to a Python object."""
    csc_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    
    handle_int = c.builder.ptrtoint(csc_struct.handle, cgutils.intp_t)
    handle_obj = c.pyapi.long_from_longlong(handle_int)
    
    if typ.dtype == types.float32:
        mod = c.pyapi.import_module_noblock("scl._binding._sparse")
        cls = c.pyapi.object_getattr_string(mod, "CSCF32")
    else:
        mod = c.pyapi.import_module_noblock("scl._binding._sparse")
        cls = c.pyapi.object_getattr_string(mod, "CSCF64")
    
    from_handle = c.pyapi.object_getattr_string(cls, "_from_handle")
    true_obj = c.pyapi.make_true()
    args = c.pyapi.tuple_pack([handle_obj, true_obj])
    result = c.pyapi.call(from_handle, args)
    
    c.pyapi.decref(handle_obj)
    c.pyapi.decref(true_obj)
    c.pyapi.decref(args)
    c.pyapi.decref(from_handle)
    c.pyapi.decref(cls)
    c.pyapi.decref(mod)
    
    return result