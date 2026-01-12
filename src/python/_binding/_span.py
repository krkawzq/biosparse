"""Span wrapper for Python."""

from __future__ import annotations

import numpy as np
from typing import Optional, TYPE_CHECKING

from ._cffi import ffi, lib

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Span:
    """Base class for Span Python wrapper.
    
    Span is a core data structure in biosparse, representing a typed contiguous
    memory region.
    
    Attributes:
        handle: FFI handle (pointer to Rust Span structure).
        owns_handle: Whether this wrapper owns the handle (needs to be freed).
    """
    
    # Subclasses need to override these
    _dtype: np.dtype = None
    _data_fn: str = None
    _data_mut_fn: str = None
    _len_fn: str = None
    _flags_fn: str = None
    _is_view_fn: str = None
    _is_aligned_fn: str = None
    _is_mutable_fn: str = None
    _info_fn: str = None
    _byte_size_fn: str = None
    _clone_fn: str = None
    _free_fn: str = None
    
    def __init__(self, handle, owns_handle: bool = False):
        """Initialize Span wrapper.
        
        Args:
            handle: FFI handle.
            owns_handle: Whether this wrapper owns the handle. If True, the handle
                will be freed when the object is destroyed.
        """
        self._handle = handle
        self._owns_handle = owns_handle
    
    def __del__(self):
        """Free handle if owned."""
        if self._owns_handle and self._handle != ffi.NULL:
            getattr(lib, self._free_fn)(self._handle)
            self._handle = ffi.NULL

    @classmethod
    def _from_handle(cls, handle, owns_handle: bool = True):
        """Create a Span wrapper from an FFI handle.
        
        This is used internally by Numba boxing to create Python objects
        from handles created in JIT code.

        Args:
            handle: FFI handle (can be int or cffi pointer).
            owns_handle: Whether the wrapper should own the handle.

        Returns:
            A new Span wrapper instance.
        """
        if isinstance(handle, int):
            handle = ffi.cast("void*", handle)
        return cls(handle, owns_handle=owns_handle)
    
    @property
    def handle(self):
        """Get raw handle.
        
        Returns:
            FFI handle.
        """
        return self._handle

    @property
    def handle_as_int(self) -> int:
        """Returns the handle as an integer (for Numba interop).

        Returns:
            int: Handle value as integer.
        """
        return int(ffi.cast("uintptr_t", self._handle))
    
    @property
    def data_ptr(self) -> int:
        """Get data pointer address.
        
        Returns:
            Pointer address as integer.
        """
        ptr = getattr(lib, self._data_fn)(self._handle)
        return int(ffi.cast("uintptr_t", ptr))
    
    def __len__(self) -> int:
        """Get number of elements.
        
        Returns:
            Number of elements.
        """
        return getattr(lib, self._len_fn)(self._handle)
    
    @property
    def flags(self) -> int:
        """Get flags.
        
        Returns:
            Flags value.
        """
        return getattr(lib, self._flags_fn)(self._handle)
    
    @property
    def is_view(self) -> bool:
        """Check if in view mode.
        
        Returns:
            True if this is a view (does not own memory).
        """
        return getattr(lib, self._is_view_fn)(self._handle)
    
    @property
    def is_aligned(self) -> bool:
        """Check if data is aligned.
        
        Returns:
            True if data is properly aligned.
        """
        return getattr(lib, self._is_aligned_fn)(self._handle)
    
    @property
    def is_mutable(self) -> bool:
        """Check if mutable.
        
        Returns:
            True if data can be modified.
        """
        return getattr(lib, self._is_mutable_fn)(self._handle)
    
    @property
    def byte_size(self) -> int:
        """Get size in bytes.
        
        Returns:
            Size in bytes.
        """
        return getattr(lib, self._byte_size_fn)(self._handle)
    
    def to_numpy(self, copy: bool = False) -> NDArray:
        """Convert to NumPy array.
        
        Args:
            copy: Whether to copy data. If False, the returned array shares
                memory with the Span.
        
        Returns:
            NumPy array.
        """
        ptr = getattr(lib, self._data_fn)(self._handle)
        length = len(self)
        
        if ptr == ffi.NULL or length == 0:
            return np.array([], dtype=self._dtype)
        
        # Create NumPy array sharing memory
        arr = np.frombuffer(
            ffi.buffer(ptr, length * self._dtype.itemsize),
            dtype=self._dtype
        )
        
        if copy:
            return arr.copy()
        return arr
    
    def clone(self) -> "Span":
        """Clone Span.
        
        Returns:
            New Span with independent memory.
        """
        new_handle = getattr(lib, self._clone_fn)(self._handle)
        return self.__class__(new_handle, owns_handle=True)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"len={len(self)}, "
            f"is_view={self.is_view}, "
            f"is_aligned={self.is_aligned}, "
            f"is_mutable={self.is_mutable})"
        )


class SpanF32(Span):
    """Python wrapper for Span<f32>."""
    
    _dtype = np.dtype(np.float32)
    _data_fn = "span_f32_data"
    _data_mut_fn = "span_f32_data_mut"
    _len_fn = "span_f32_len"
    _flags_fn = "span_f32_flags"
    _is_view_fn = "span_f32_is_view"
    _is_aligned_fn = "span_f32_is_aligned"
    _is_mutable_fn = "span_f32_is_mutable"
    _info_fn = "span_f32_info"
    _byte_size_fn = "span_f32_byte_size"
    _clone_fn = "span_f32_clone"
    _free_fn = "span_f32_free"


class SpanF64(Span):
    """Python wrapper for Span<f64>."""
    
    _dtype = np.dtype(np.float64)
    _data_fn = "span_f64_data"
    _data_mut_fn = "span_f64_data_mut"
    _len_fn = "span_f64_len"
    _flags_fn = "span_f64_flags"
    _is_view_fn = "span_f64_is_view"
    _is_aligned_fn = "span_f64_is_aligned"
    _is_mutable_fn = "span_f64_is_mutable"
    _info_fn = "span_f64_info"
    _byte_size_fn = "span_f64_byte_size"
    _clone_fn = "span_f64_clone"
    _free_fn = "span_f64_free"


class SpanI32(Span):
    """Python wrapper for Span<i32>."""
    
    _dtype = np.dtype(np.int32)
    _data_fn = "span_i32_data"
    _data_mut_fn = "span_i32_data_mut"
    _len_fn = "span_i32_len"
    _flags_fn = "span_i32_flags"
    _is_view_fn = "span_i32_is_view"
    _is_aligned_fn = "span_i32_is_aligned"
    _is_mutable_fn = "span_i32_is_mutable"
    _info_fn = "span_i32_info"
    _byte_size_fn = "span_i32_byte_size"
    _clone_fn = "span_i32_clone"
    _free_fn = "span_i32_free"


class SpanI64(Span):
    """Python wrapper for Span<i64>."""
    
    _dtype = np.dtype(np.int64)
    _data_fn = "span_i64_data"
    _data_mut_fn = "span_i64_data_mut"
    _len_fn = "span_i64_len"
    _flags_fn = "span_i64_flags"
    _is_view_fn = "span_i64_is_view"
    _is_aligned_fn = "span_i64_is_aligned"
    _is_mutable_fn = "span_i64_is_mutable"
    _info_fn = "span_i64_info"
    _byte_size_fn = "span_i64_byte_size"
    _clone_fn = "span_i64_clone"
    _free_fn = "span_i64_free"
