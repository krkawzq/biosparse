"""Sparse matrix wrappers for Python."""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional, Union, TYPE_CHECKING

from ._cffi import ffi, lib, FfiResult
from ._span import Span, SpanF32, SpanF64, SpanI64

if TYPE_CHECKING:
    from numpy.typing import NDArray
    import scipy.sparse as sp


class CSR:
    """Base class for CSR (Compressed Sparse Row) sparse matrix Python wrapper.

    CSR (Compressed Sparse Row) format stores sparse matrices row by row.
    Non-zero elements of each row are stored in independent Spans.
    """

    _value_dtype: np.dtype = None  # Override in subclasses: value numpy dtype
    _index_dtype: np.dtype = np.dtype(np.int64)  # Index dtype (always int64)
    _span_value_class: type = None  # Span type for values
    _span_index_class: type = SpanI64  # Span type for indices (int64)
    _prefix: str = None  # FFI function prefix; override in subclasses

    def __init__(self, handle, owns_handle: bool = False):
        """Initialize a CSR wrapper.

        Args:
            handle: FFI handle to the matrix.
            owns_handle (bool): Whether this wrapper owns the underlying handle.
        """
        self._handle = handle
        self._owns_handle = owns_handle

    def __del__(self):
        """Frees the underlying FFI handle if owned by this object."""
        try:
            if self._owns_handle and self._handle is not None and self._handle != ffi.NULL:
                getattr(lib, f"{self._prefix}_free")(self._handle)
                self._handle = ffi.NULL
        except (TypeError, AttributeError):
            # Ignore errors during interpreter shutdown
            pass

    @classmethod
    def _from_handle(cls, handle, owns_handle: bool = True):
        """Create a CSR wrapper from an FFI handle.
        
        This is used internally by Numba boxing to create Python objects
        from handles created in JIT code.

        Args:
            handle: FFI handle (can be int or cffi pointer).
            owns_handle (bool): Whether the wrapper should own the handle.

        Returns:
            A new CSR wrapper instance.
        """
        # Convert integer handle to cffi pointer if needed
        if isinstance(handle, int):
            handle = ffi.cast("void*", handle)
        return cls(handle, owns_handle=owns_handle)

    @property
    def handle(self):
        """Returns the raw FFI handle to the matrix.

        Returns:
            The underlying FFI handle object.
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
    def shape(self) -> Tuple[int, int]:
        """Gets the shape of the matrix (rows, cols).

        Returns:
            Tuple[int, int]: Number of rows and columns.
        """
        s = getattr(lib, f"{self._prefix}_shape")(self._handle)
        return (s.rows, s.cols)

    @property
    def nrows(self) -> int:
        """Returns the number of rows.

        Returns:
            int: Number of rows.
        """
        return getattr(lib, f"{self._prefix}_rows")(self._handle)

    @property
    def ncols(self) -> int:
        """Returns the number of columns.

        Returns:
            int: Number of columns.
        """
        return getattr(lib, f"{self._prefix}_cols")(self._handle)

    @property
    def nnz(self) -> int:
        """Returns the number of non-zero elements.

        Returns:
            int: Number of non-zero elements.
        """
        return getattr(lib, f"{self._prefix}_nnz")(self._handle)

    # =========================
    # Validation methods
    # =========================

    @property
    def is_valid(self) -> bool:
        """Checks if the matrix structure is valid.

        Returns:
            bool: True if the structure is valid.
        """
        return getattr(lib, f"{self._prefix}_is_valid")(self._handle)

    @property
    def is_sorted(self) -> bool:
        """Checks if all row indices are sorted within each row.

        Returns:
            bool: True if all rows have sorted indices.
        """
        return getattr(lib, f"{self._prefix}_is_sorted")(self._handle)

    def validate(self) -> bool:
        """Performs a full check of structure, sorting, and index bounds.

        Returns:
            bool: True if all validations succeed.
        """
        return getattr(lib, f"{self._prefix}_validate")(self._handle)

    @property
    def indices_in_bounds(self) -> bool:
        """Checks that all indices are within allowed column bounds.

        Returns:
            bool: True if all indices are in-bounds.
        """
        return getattr(lib, f"{self._prefix}_indices_in_bounds")(self._handle)

    # =========================
    # Query methods
    # =========================

    @property
    def is_empty(self) -> bool:
        """Checks if the matrix has zero rows or columns.

        Returns:
            bool: True if the matrix is empty.
        """
        return getattr(lib, f"{self._prefix}_is_empty")(self._handle)

    @property
    def is_zero(self) -> bool:
        """Checks if the matrix has no non-zero elements.

        Returns:
            bool: True if nnz == 0.
        """
        return getattr(lib, f"{self._prefix}_is_zero")(self._handle)

    @property
    def sparsity(self) -> float:
        """Computes the sparsity (proportion of zero elements).

        Returns:
            float: Sparsity value in range [0.0, 1.0].
        """
        return getattr(lib, f"{self._prefix}_sparsity")(self._handle)

    @property
    def density(self) -> float:
        """Computes the density (proportion of non-zero elements).

        Returns:
            float: Density value in range [0.0, 1.0].
        """
        return getattr(lib, f"{self._prefix}_density")(self._handle)

    # =========================
    # Row access
    # =========================

    def row_values(self, row: int) -> Span:
        """Gets the value Span for a given row.

        Args:
            row (int): Row index.

        Returns:
            Span: Non-owning Span of values for that row.
        """
        handle = getattr(lib, f"{self._prefix}_row_values")(self._handle, row)
        return self._span_value_class(handle, owns_handle=False)

    def row_indices(self, row: int) -> SpanI64:
        """Gets the index Span for a given row.

        Args:
            row (int): Row index.

        Returns:
            SpanI64: Non-owning Span of indices for the row.
        """
        handle = getattr(lib, f"{self._prefix}_row_indices")(self._handle, row)
        return self._span_index_class(handle, owns_handle=False)

    def row_values_ptr(self, row: int) -> int:
        """Gets the pointer address to the values of a given row.

        Args:
            row (int): Row index.

        Returns:
            int: Address of value data as integer.
        """
        ptr = getattr(lib, f"{self._prefix}_row_values_ptr")(self._handle, row)
        return int(ffi.cast("uintptr_t", ptr))

    def row_indices_ptr(self, row: int) -> int:
        """Gets the pointer address to the indices of a given row.

        Args:
            row (int): Row index.

        Returns:
            int: Address of index data as integer.
        """
        ptr = getattr(lib, f"{self._prefix}_row_indices_ptr")(self._handle, row)
        return int(ffi.cast("uintptr_t", ptr))

    def row_len(self, row: int) -> int:
        """Returns the number of nonzeros in the specified row.

        Args:
            row (int): Row index.

        Returns:
            int: The number of nonzero elements in the row.
        """
        return getattr(lib, f"{self._prefix}_row_len")(self._handle, row)

    def row_to_numpy(self, row: int, copy: bool = False) -> Tuple["NDArray", "NDArray"]:
        """Returns values and indices of a row as NumPy arrays.

        Args:
            row (int): Row index.
            copy (bool): Whether to copy (True) or not (False).

        Returns:
            Tuple[NDArray, NDArray]: (values, indices) arrays for the row.
        """
        values_span = self.row_values(row)
        indices_span = self.row_indices(row)
        return values_span.to_numpy(copy), indices_span.to_numpy(copy)

    # =========================
    # Sorting
    # =========================

    def ensure_sorted(self) -> None:
        """Ensures all row indices are sorted in place."""
        getattr(lib, f"{self._prefix}_ensure_sorted")(self._handle)

    def ensure_sorted_checked(self) -> bool:
        """Ensures rows are sorted; returns whether sorting was actually performed.

        Returns:
            bool: True if sorting was done, False if already sorted.
        """
        return getattr(lib, f"{self._prefix}_ensure_sorted_checked")(self._handle)

    # =========================
    # Clone
    # =========================

    def clone(self) -> "CSR":
        """Returns a deep copy of this CSR matrix.

        Returns:
            CSR: New CSR matrix with its own independent memory.
        """
        new_handle = getattr(lib, f"{self._prefix}_clone")(self._handle)
        return self.__class__(new_handle, owns_handle=True)

    # =========================
    # Slicing
    # =========================

    def slice_rows(self, start: int, end: int) -> "CSR":
        """Returns a new CSR matrix with selected row range. Zero-copy if possible.

        Args:
            start (int): Start row (inclusive).
            end (int): End row (exclusive).

        Returns:
            CSR: New matrix with selected rows.
        """
        out_handle = ffi.new("void**")
        result = getattr(lib, f"{self._prefix}_slice_rows")(
            self._handle, start, end, out_handle
        )
        FfiResult.check(result, "slice_rows")
        return self.__class__(out_handle[0], owns_handle=True)

    def slice_cols(self, start: int, end: int) -> "CSR":
        """Returns a new CSR matrix with selected column range.

        Args:
            start (int): Start column (inclusive).
            end (int): End column (exclusive).

        Returns:
            CSR: New matrix with selected columns.
        """
        out_handle = ffi.new("void**")
        result = getattr(lib, f"{self._prefix}_slice_cols")(
            self._handle, start, end, out_handle
        )
        FfiResult.check(result, "slice_cols")
        return self.__class__(out_handle[0], owns_handle=True)

    def slice_rows_mask(self, mask: "NDArray") -> "CSR":
        """Returns a new CSR matrix by selecting rows using a boolean mask.

        Args:
            mask (NDArray): Boolean mask of length nrows.

        Returns:
            CSR: New matrix with only selected rows.
        """
        mask = np.asarray(mask, dtype=np.bool_)
        mask_ptr = ffi.cast("bool*", mask.ctypes.data)
        out_handle = ffi.new("void**")
        result = getattr(lib, f"{self._prefix}_slice_rows_mask")(
            self._handle, mask_ptr, len(mask), out_handle
        )
        FfiResult.check(result, "slice_rows_mask")
        return self.__class__(out_handle[0], owns_handle=True)

    def slice_cols_mask(self, mask: "NDArray") -> "CSR":
        """Returns a new CSR matrix by selecting columns using a boolean mask.

        Args:
            mask (NDArray): Boolean mask of length ncols.

        Returns:
            CSR: New matrix with only selected columns.
        """
        mask = np.asarray(mask, dtype=np.bool_)
        mask_ptr = ffi.cast("bool*", mask.ctypes.data)
        out_handle = ffi.new("void**")
        result = getattr(lib, f"{self._prefix}_slice_cols_mask")(
            self._handle, mask_ptr, len(mask), out_handle
        )
        FfiResult.check(result, "slice_cols_mask")
        return self.__class__(out_handle[0], owns_handle=True)

    def __getitem__(self, key):
        """Implements slicing: support 1D and 2D slices.

        Examples:
            csr[10:20]       # Row slicing
            csr[10:20, :]    # Row slicing
            csr[:, 5:15]     # Column slicing
            csr[10:20, 5:15] # Row and column slicing
        """
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or self.nrows
            return self.slice_rows(start, stop)
        elif isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            result = self

            # Handle row slicing
            if isinstance(row_key, slice):
                start = row_key.start or 0
                stop = row_key.stop or self.nrows
                if start != 0 or stop != self.nrows:
                    result = result.slice_rows(start, stop)

            # Handle column slicing
            if isinstance(col_key, slice):
                start = col_key.start or 0
                stop = col_key.stop or result.ncols
                if start != 0 or stop != result.ncols:
                    result = result.slice_cols(start, stop)

            return result
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    # =========================
    # Conversion methods
    # =========================

    def to_dense(self, order: str = "C") -> "NDArray":
        """Converts the CSR matrix to a dense NumPy array.

        Args:
            order (str): 'C' for row-major, 'F' for column-major.

        Returns:
            NDArray: Dense array with the same shape and dtype.
        """
        rows, cols = self.shape
        out = np.zeros((rows, cols), dtype=self._value_dtype, order=order)
        out_ptr = ffi.cast(f"{self._c_type}*", out.ctypes.data)
        col_major = (order == "F")
        result = getattr(lib, f"{self._prefix}_to_dense")(
            self._handle, out_ptr, out.size, col_major
        )
        FfiResult.check(result, "to_dense")
        return out

    def to_coo(self) -> Tuple["NDArray", "NDArray", "NDArray"]:
        """Converts the CSR matrix to COO (coordinate) format.

        Returns:
            Tuple[NDArray, NDArray, NDArray]: (row_indices, col_indices, data)
        """
        nnz = self.nnz
        row_indices = np.empty(nnz, dtype=np.int64)
        col_indices = np.empty(nnz, dtype=np.int64)
        data = np.empty(nnz, dtype=self._value_dtype)

        row_ptr = ffi.cast("int64_t*", row_indices.ctypes.data)
        col_ptr = ffi.cast("int64_t*", col_indices.ctypes.data)
        data_ptr = ffi.cast(f"{self._c_type}*", data.ctypes.data)

        result = getattr(lib, f"{self._prefix}_to_coo")(
            self._handle, row_ptr, col_ptr, data_ptr, nnz
        )
        FfiResult.check(result, "to_coo")
        return row_indices, col_indices, data

    def to_csc(self) -> "CSC":
        """Converts the CSR matrix to CSC format.

        Returns:
            CSC: The matrix in CSC form.
        """
        out_handle = ffi.new("void**")
        csc_prefix = self._prefix.replace("csr", "csc")
        result = getattr(lib, f"{csc_prefix}_from_csr")(self._handle, out_handle)
        FfiResult.check(result, "to_csc")
        csc_class = CSCF32 if self._value_dtype == np.float32 else CSCF64
        return csc_class(out_handle[0], owns_handle=True)

    def to_scipy(self) -> "sp.csr_matrix":
        """Converts the matrix to scipy.sparse.csr_matrix.

        Returns:
            sp.csr_matrix: The corresponding SciPy CSR matrix.
        """
        import scipy.sparse as sp
        row_indices, col_indices, data = self.to_coo()
        return sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=self.shape
        )

    # =========================
    # Class methods - constructors
    # =========================

    @classmethod
    def from_scipy(cls, mat: "sp.csr_matrix", copy: bool = True) -> "CSR":
        """Creates a CSR object from a SciPy CSR matrix.

        Args:
            mat (sp.csr_matrix): Input SciPy CSR matrix.
            copy (bool): If True, copy data; if False, create a view.

        Returns:
            CSR: New instance representing the matrix.
        """
        mat = mat.tocsr()

        if mat.indices.dtype != np.int64:
            mat.indices = mat.indices.astype(np.int64)
        if mat.indptr.dtype != np.int64:
            mat.indptr = mat.indptr.astype(np.int64)
        if mat.data.dtype != cls._value_dtype:
            mat.data = mat.data.astype(cls._value_dtype)

        rows, cols = mat.shape

        if copy:
            out_handle = ffi.new("void**")
            data_ptr = ffi.cast(f"{cls._c_type}*", mat.data.ctypes.data)
            indices_ptr = ffi.cast("int64_t*", mat.indices.ctypes.data)
            indptr_ptr = ffi.cast("int64_t*", mat.indptr.ctypes.data)

            result = getattr(lib, f"{cls._prefix}_from_scipy_copy")(
                rows, cols,
                data_ptr, len(mat.data),
                indices_ptr, len(mat.indices),
                indptr_ptr, len(mat.indptr),
                out_handle
            )
            FfiResult.check(result, "from_scipy")
            return cls(out_handle[0], owns_handle=True)
        else:
            # View mode: use external memory directly, user must keep array alive.
            data_ptr = ffi.cast(f"{cls._c_type}*", mat.data.ctypes.data)
            indices_ptr = ffi.cast("int64_t*", mat.indices.ctypes.data)
            indptr_ptr = ffi.cast("int64_t*", mat.indptr.ctypes.data)

            handle = getattr(lib, f"{cls._prefix}_from_scipy_view")(
                rows, cols, data_ptr, indices_ptr, indptr_ptr
            )
            if handle == ffi.NULL:
                raise RuntimeError("Failed to create CSR from scipy (view mode)")
            result = cls(handle, owns_handle=True)
            result._scipy_ref = mat  # Keep a reference to prevent deallocation
            return result

    @classmethod
    def from_coo(cls, rows: int, cols: int,
                 row_indices: "NDArray", col_indices: "NDArray", data: "NDArray") -> "CSR":
        """Creates a CSR object from COO-format data arrays.

        Args:
            rows (int): Number of rows.
            cols (int): Number of columns.
            row_indices (NDArray): Row index array.
            col_indices (NDArray): Column index array.
            data (NDArray): Values array.

        Returns:
            CSR: New matrix.
        """
        row_indices = np.asarray(row_indices, dtype=np.int64)
        col_indices = np.asarray(col_indices, dtype=np.int64)
        data = np.asarray(data, dtype=cls._value_dtype)

        out_handle = ffi.new("void**")
        row_ptr = ffi.cast("int64_t*", row_indices.ctypes.data)
        col_ptr = ffi.cast("int64_t*", col_indices.ctypes.data)
        data_ptr = ffi.cast(f"{cls._c_type}*", data.ctypes.data)

        result = getattr(lib, f"{cls._prefix}_from_coo")(
            rows, cols, row_ptr, col_ptr, data_ptr, len(data), out_handle
        )
        FfiResult.check(result, "from_coo")
        return cls(out_handle[0], owns_handle=True)

    # =========================
    # Stack operations (static)
    # =========================

    @classmethod
    def vstack(cls, matrices: List["CSR"]) -> "CSR":
        """Vertically stack (concatenate rows) multiple CSR matrices.

        Args:
            matrices (List[CSR]): List of CSR matrices.

        Returns:
            CSR: New matrix with all rows from input matrices.
        """
        if not matrices:
            raise ValueError("Cannot vstack empty list")

        handles = ffi.new(f"void*[{len(matrices)}]")
        for i, m in enumerate(matrices):
            handles[i] = m._handle

        out_handle = ffi.new("void**")
        result = getattr(lib, f"{cls._prefix}_vstack")(handles, len(matrices), out_handle)
        FfiResult.check(result, "vstack")
        return cls(out_handle[0], owns_handle=True)

    @classmethod
    def hstack(cls, matrices: List["CSR"]) -> "CSR":
        """Horizontally stack (concatenate columns) multiple CSR matrices.

        Args:
            matrices (List[CSR]): List of CSR matrices.

        Returns:
            CSR: New matrix with all columns from input matrices.
        """
        if not matrices:
            raise ValueError("Cannot hstack empty list")

        handles = ffi.new(f"void*[{len(matrices)}]")
        for i, m in enumerate(matrices):
            handles[i] = m._handle

        out_handle = ffi.new("void**")
        result = getattr(lib, f"{cls._prefix}_hstack")(handles, len(matrices), out_handle)
        FfiResult.check(result, "hstack")
        return cls(out_handle[0], owns_handle=True)

    def __repr__(self) -> str:
        """String representation, showing class, shape, nnz and density."""
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"nnz={self.nnz}, "
            f"density={self.density:.4f})"
        )


class CSRF32(CSR):
    """Python wrapper for CSR<f32, i64>."""

    _value_dtype = np.dtype(np.float32)
    _span_value_class = SpanF32
    _prefix = "csr_f32"
    _c_type = "float"


class CSRF64(CSR):
    """Python wrapper for CSR<f64, i64>."""

    _value_dtype = np.dtype(np.float64)
    _span_value_class = SpanF64
    _prefix = "csr_f64"
    _c_type = "double"


class CSC:
    """Base class for CSC (Compressed Sparse Column) sparse matrix Python wrapper.

    CSC (Compressed Sparse Column) format stores sparse matrices column by column.
    Non-zero elements of each column are stored in independent Spans.
    """

    _value_dtype: np.dtype = None
    _index_dtype: np.dtype = np.dtype(np.int64)
    _span_value_class: type = None
    _span_index_class: type = SpanI64
    _prefix: str = None

    def __init__(self, handle, owns_handle: bool = False):
        """Initialize a CSC wrapper.

        Args:
            handle: FFI handle to the matrix.
            owns_handle (bool): Whether this wrapper owns the underlying handle.
        """
        self._handle = handle
        self._owns_handle = owns_handle

    def __del__(self):
        """Frees the underlying FFI handle if owned by this object."""
        try:
            if self._owns_handle and self._handle is not None and self._handle != ffi.NULL:
                getattr(lib, f"{self._prefix}_free")(self._handle)
                self._handle = ffi.NULL
        except (TypeError, AttributeError):
            # Ignore errors during interpreter shutdown
            pass

    @classmethod
    def _from_handle(cls, handle, owns_handle: bool = True):
        """Create a CSC wrapper from an FFI handle.
        
        This is used internally by Numba boxing to create Python objects
        from handles created in JIT code.

        Args:
            handle: FFI handle (can be int or cffi pointer).
            owns_handle (bool): Whether the wrapper should own the handle.

        Returns:
            A new CSC wrapper instance.
        """
        # Convert integer handle to cffi pointer if needed
        if isinstance(handle, int):
            handle = ffi.cast("void*", handle)
        return cls(handle, owns_handle=owns_handle)

    @property
    def handle(self):
        """Returns the raw FFI handle to the matrix.

        Returns:
            The underlying FFI handle object.
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
    def shape(self) -> Tuple[int, int]:
        """Returns the shape as (rows, columns).

        Returns:
            Tuple[int, int]: Matrix dimensions.
        """
        s = getattr(lib, f"{self._prefix}_shape")(self._handle)
        return (s.rows, s.cols)

    @property
    def nrows(self) -> int:
        """Returns the number of rows.

        Returns:
            int: Number of rows.
        """
        return getattr(lib, f"{self._prefix}_rows")(self._handle)

    @property
    def ncols(self) -> int:
        """Returns the number of columns.

        Returns:
            int: Number of columns.
        """
        return getattr(lib, f"{self._prefix}_cols")(self._handle)

    @property
    def nnz(self) -> int:
        """Returns the number of nonzero values.

        Returns:
            int: Number of nonzeros.
        """
        return getattr(lib, f"{self._prefix}_nnz")(self._handle)

    # =========================
    # Validation methods
    # =========================

    @property
    def is_valid(self) -> bool:
        """Checks if the matrix structure is valid.

        Returns:
            bool: True if the structure is valid.
        """
        return getattr(lib, f"{self._prefix}_is_valid")(self._handle)

    @property
    def is_sorted(self) -> bool:
        """Checks if the indices for every column are sorted.

        Returns:
            bool: True if sorted.
        """
        return getattr(lib, f"{self._prefix}_is_sorted")(self._handle)

    def validate(self) -> bool:
        """Validates the structure, sorting, and index bounds.

        Returns:
            bool: True if all validation passes.
        """
        return getattr(lib, f"{self._prefix}_validate")(self._handle)

    @property
    def indices_in_bounds(self) -> bool:
        """Checks all indices are within allowed row bounds.

        Returns:
            bool: True if in bounds.
        """
        return getattr(lib, f"{self._prefix}_indices_in_bounds")(self._handle)

    # =========================
    # Query methods
    # =========================

    @property
    def is_empty(self) -> bool:
        """Checks if the matrix has zero rows or zero columns.

        Returns:
            bool: True if empty.
        """
        return getattr(lib, f"{self._prefix}_is_empty")(self._handle)

    @property
    def is_zero(self) -> bool:
        """Checks if the matrix has any nonzero values.

        Returns:
            bool: True if all-zero.
        """
        return getattr(lib, f"{self._prefix}_is_zero")(self._handle)

    @property
    def sparsity(self) -> float:
        """Computes the sparsity (proportion of zero elements).

        Returns:
            float: Value in [0.0, 1.0].
        """
        return getattr(lib, f"{self._prefix}_sparsity")(self._handle)

    @property
    def density(self) -> float:
        """Computes the density (proportion of nonzero elements).

        Returns:
            float: Value in [0.0, 1.0].
        """
        return getattr(lib, f"{self._prefix}_density")(self._handle)

    # =========================
    # Column access
    # =========================

    def col_values(self, col: int) -> Span:
        """Returns Span of values for the specified column.

        Args:
            col (int): Column index.

        Returns:
            Span: Non-owning view on values of the column.
        """
        handle = getattr(lib, f"{self._prefix}_col_values")(self._handle, col)
        return self._span_value_class(handle, owns_handle=False)

    def col_indices(self, col: int) -> SpanI64:
        """Returns Span of row indices for the specified column.

        Args:
            col (int): Column index.

        Returns:
            SpanI64: Non-owning view on row indices of the column.
        """
        handle = getattr(lib, f"{self._prefix}_col_indices")(self._handle, col)
        return self._span_index_class(handle, owns_handle=False)

    def col_values_ptr(self, col: int) -> int:
        """Returns pointer address to value data for the column.

        Args:
            col (int): Column index.

        Returns:
            int: Pointer as integer.
        """
        ptr = getattr(lib, f"{self._prefix}_col_values_ptr")(self._handle, col)
        return int(ffi.cast("uintptr_t", ptr))

    def col_indices_ptr(self, col: int) -> int:
        """Returns pointer address to row index data for the column.

        Args:
            col (int): Column index.

        Returns:
            int: Pointer as integer.
        """
        ptr = getattr(lib, f"{self._prefix}_col_indices_ptr")(self._handle, col)
        return int(ffi.cast("uintptr_t", ptr))

    def col_len(self, col: int) -> int:
        """Returns the number of nonzero elements in the column.

        Args:
            col (int): Column index.

        Returns:
            int: Nonzero count for the column.
        """
        return getattr(lib, f"{self._prefix}_col_len")(self._handle, col)

    def col_to_numpy(self, col: int, copy: bool = False) -> Tuple["NDArray", "NDArray"]:
        """Returns values and row indices of the column as numpy arrays.

        Args:
            col (int): Column index.
            copy (bool): Whether to copy data.

        Returns:
            Tuple[NDArray, NDArray]: (values, indices)
        """
        values_span = self.col_values(col)
        indices_span = self.col_indices(col)
        return values_span.to_numpy(copy), indices_span.to_numpy(copy)

    # =========================
    # Sorting
    # =========================

    def ensure_sorted(self) -> None:
        """Sorts the indices in all columns in-place."""
        getattr(lib, f"{self._prefix}_ensure_sorted")(self._handle)

    def ensure_sorted_checked(self) -> bool:
        """Sorts columns if necessary. Returns True if sorted, else False.

        Returns:
            bool: True if sorting performed, False if already sorted.
        """
        return getattr(lib, f"{self._prefix}_ensure_sorted_checked")(self._handle)

    # =========================
    # Clone
    # =========================

    def clone(self) -> "CSC":
        """Returns a deep copy of the CSC matrix.

        Returns:
            CSC: New object.
        """
        new_handle = getattr(lib, f"{self._prefix}_clone")(self._handle)
        return self.__class__(new_handle, owns_handle=True)

    # =========================
    # Slicing
    # =========================

    def slice_cols(self, start: int, end: int) -> "CSC":
        """Returns a new CSC with selected column range.

        Args:
            start (int): Start column, inclusive.
            end (int): End column, exclusive.

        Returns:
            CSC: Matrix with selected columns.
        """
        out_handle = ffi.new("void**")
        result = getattr(lib, f"{self._prefix}_slice_cols")(
            self._handle, start, end, out_handle
        )
        FfiResult.check(result, "slice_cols")
        return self.__class__(out_handle[0], owns_handle=True)

    def slice_rows(self, start: int, end: int) -> "CSC":
        """Returns a new CSC with selected row range.

        Args:
            start (int): Start row (inclusive).
            end (int): End row (exclusive).

        Returns:
            CSC: Matrix with selected rows.
        """
        out_handle = ffi.new("void**")
        result = getattr(lib, f"{self._prefix}_slice_rows")(
            self._handle, start, end, out_handle
        )
        FfiResult.check(result, "slice_rows")
        return self.__class__(out_handle[0], owns_handle=True)

    def slice_cols_mask(self, mask: "NDArray") -> "CSC":
        """Returns a new CSC by selecting columns using a boolean mask.

        Args:
            mask (NDArray): Boolean mask of length ncols.

        Returns:
            CSC: Matrix with selected columns.
        """
        mask = np.asarray(mask, dtype=np.bool_)
        mask_ptr = ffi.cast("bool*", mask.ctypes.data)
        out_handle = ffi.new("void**")
        result = getattr(lib, f"{self._prefix}_slice_cols_mask")(
            self._handle, mask_ptr, len(mask), out_handle
        )
        FfiResult.check(result, "slice_cols_mask")
        return self.__class__(out_handle[0], owns_handle=True)

    def slice_rows_mask(self, mask: "NDArray") -> "CSC":
        """Returns a new CSC by selecting rows using a boolean mask.

        Args:
            mask (NDArray): Boolean mask of length nrows.

        Returns:
            CSC: Matrix with selected rows.
        """
        mask = np.asarray(mask, dtype=np.bool_)
        mask_ptr = ffi.cast("bool*", mask.ctypes.data)
        out_handle = ffi.new("void**")
        result = getattr(lib, f"{self._prefix}_slice_rows_mask")(
            self._handle, mask_ptr, len(mask), out_handle
        )
        FfiResult.check(result, "slice_rows_mask")
        return self.__class__(out_handle[0], owns_handle=True)

    def __getitem__(self, key):
        """Supports matrix slicing syntax (columns-first for 1D).

        Returns:
            CSC: Sliced matrix.
        """
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or self.ncols
            return self.slice_cols(start, stop)
        elif isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            result = self

            if isinstance(row_key, slice):
                start = row_key.start or 0
                stop = row_key.stop or self.nrows
                if start != 0 or stop != self.nrows:
                    result = result.slice_rows(start, stop)

            if isinstance(col_key, slice):
                start = col_key.start or 0
                stop = col_key.stop or result.ncols
                if start != 0 or stop != result.ncols:
                    result = result.slice_cols(start, stop)

            return result
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    # =========================
    # Conversion methods
    # =========================

    def to_dense(self, order: str = "C") -> "NDArray":
        """Converts the CSC matrix to a dense NumPy array.

        Args:
            order (str): 'C' for row-major, 'F' for column-major.

        Returns:
            NDArray: Dense matrix.
        """
        rows, cols = self.shape
        out = np.zeros((rows, cols), dtype=self._value_dtype, order=order)
        out_ptr = ffi.cast(f"{self._c_type}*", out.ctypes.data)
        col_major = (order == "F")
        result = getattr(lib, f"{self._prefix}_to_dense")(
            self._handle, out_ptr, out.size, col_major
        )
        FfiResult.check(result, "to_dense")
        return out

    def to_coo(self) -> Tuple["NDArray", "NDArray", "NDArray"]:
        """Converts the CSC matrix to coordinate format.

        Returns:
            Tuple[NDArray, NDArray, NDArray]: (row_indices, col_indices, data)
        """
        nnz = self.nnz
        row_indices = np.empty(nnz, dtype=np.int64)
        col_indices = np.empty(nnz, dtype=np.int64)
        data = np.empty(nnz, dtype=self._value_dtype)

        row_ptr = ffi.cast("int64_t*", row_indices.ctypes.data)
        col_ptr = ffi.cast("int64_t*", col_indices.ctypes.data)
        data_ptr = ffi.cast(f"{self._c_type}*", data.ctypes.data)

        result = getattr(lib, f"{self._prefix}_to_coo")(
            self._handle, row_ptr, col_ptr, data_ptr, nnz
        )
        FfiResult.check(result, "to_coo")
        return row_indices, col_indices, data

    def to_csr(self) -> "CSR":
        """Converts the CSC matrix to CSR format.

        Returns:
            CSR: CSR-format matrix.
        """
        out_handle = ffi.new("void**")
        csr_prefix = self._prefix.replace("csc", "csr")
        result = getattr(lib, f"{csr_prefix}_from_csc")(self._handle, out_handle)
        FfiResult.check(result, "to_csr")
        csr_class = CSRF32 if self._value_dtype == np.float32 else CSRF64
        return csr_class(out_handle[0], owns_handle=True)

    def to_scipy(self) -> "sp.csc_matrix":
        """Converts the matrix to scipy.sparse.csc_matrix.

        Returns:
            sp.csc_matrix: The corresponding SciPy CSC matrix.
        """
        import scipy.sparse as sp
        row_indices, col_indices, data = self.to_coo()
        return sp.csc_matrix(
            (data, (row_indices, col_indices)),
            shape=self.shape
        )

    # =========================
    # Class methods - constructors
    # =========================

    @classmethod
    def from_scipy(cls, mat: "sp.csc_matrix", copy: bool = True) -> "CSC":
        """Creates a CSC object from SciPy CSC matrix.

        Args:
            mat (sp.csc_matrix): Input matrix.
            copy (bool): Whether to copy data.

        Returns:
            CSC: Matrix wrapping the same data.
        """
        mat = mat.tocsc()

        if mat.indices.dtype != np.int64:
            mat.indices = mat.indices.astype(np.int64)
        if mat.indptr.dtype != np.int64:
            mat.indptr = mat.indptr.astype(np.int64)
        if mat.data.dtype != cls._value_dtype:
            mat.data = mat.data.astype(cls._value_dtype)

        rows, cols = mat.shape

        if copy:
            out_handle = ffi.new("void**")
            data_ptr = ffi.cast(f"{cls._c_type}*", mat.data.ctypes.data)
            indices_ptr = ffi.cast("int64_t*", mat.indices.ctypes.data)
            indptr_ptr = ffi.cast("int64_t*", mat.indptr.ctypes.data)

            result = getattr(lib, f"{cls._prefix}_from_scipy_copy")(
                rows, cols,
                data_ptr, len(mat.data),
                indices_ptr, len(mat.indices),
                indptr_ptr, len(mat.indptr),
                out_handle
            )
            FfiResult.check(result, "from_scipy")
            return cls(out_handle[0], owns_handle=True)
        else:
            data_ptr = ffi.cast(f"{cls._c_type}*", mat.data.ctypes.data)
            indices_ptr = ffi.cast("int64_t*", mat.indices.ctypes.data)
            indptr_ptr = ffi.cast("int64_t*", mat.indptr.ctypes.data)

            handle = getattr(lib, f"{cls._prefix}_from_scipy_view")(
                rows, cols, data_ptr, indices_ptr, indptr_ptr
            )
            if handle == ffi.NULL:
                raise RuntimeError("Failed to create CSC from scipy (view mode)")

            result = cls(handle, owns_handle=True)
            result._scipy_ref = mat
            return result

    @classmethod
    def from_coo(cls, rows: int, cols: int,
                 row_indices: "NDArray", col_indices: "NDArray", data: "NDArray") -> "CSC":
        """Creates a CSC from COO-format arrays.

        Args:
            rows (int): Number of rows.
            cols (int): Number of columns.
            row_indices (NDArray): Row index array.
            col_indices (NDArray): Column index array.
            data (NDArray): Value array.

        Returns:
            CSC: The resulting matrix.
        """
        row_indices = np.asarray(row_indices, dtype=np.int64)
        col_indices = np.asarray(col_indices, dtype=np.int64)
        data = np.asarray(data, dtype=cls._value_dtype)

        out_handle = ffi.new("void**")
        row_ptr = ffi.cast("int64_t*", row_indices.ctypes.data)
        col_ptr = ffi.cast("int64_t*", col_indices.ctypes.data)
        data_ptr = ffi.cast(f"{cls._c_type}*", data.ctypes.data)

        result = getattr(lib, f"{cls._prefix}_from_coo")(
            rows, cols, row_ptr, col_ptr, data_ptr, len(data), out_handle
        )
        FfiResult.check(result, "from_coo")
        return cls(out_handle[0], owns_handle=True)

    # =========================
    # Stack operations (static)
    # =========================

    @classmethod
    def vstack(cls, matrices: List["CSC"]) -> "CSC":
        """Vertically stack (increase row count) several CSC matrices.

        Args:
            matrices (List[CSC]): List of CSC matrices.

        Returns:
            CSC: New matrix with extended rows.
        """
        if not matrices:
            raise ValueError("Cannot vstack empty list")

        handles = ffi.new(f"void*[{len(matrices)}]")
        for i, m in enumerate(matrices):
            handles[i] = m._handle

        out_handle = ffi.new("void**")
        result = getattr(lib, f"{cls._prefix}_vstack")(handles, len(matrices), out_handle)
        FfiResult.check(result, "vstack")
        return cls(out_handle[0], owns_handle=True)

    @classmethod
    def hstack(cls, matrices: List["CSC"]) -> "CSC":
        """Horizontally stack (increase column count) several CSC matrices.

        Args:
            matrices (List[CSC]): List of CSC matrices.

        Returns:
            CSC: Matrix with extended columns.
        """
        if not matrices:
            raise ValueError("Cannot hstack empty list")

        handles = ffi.new(f"void*[{len(matrices)}]")
        for i, m in enumerate(matrices):
            handles[i] = m._handle

        out_handle = ffi.new("void**")
        result = getattr(lib, f"{cls._prefix}_hstack")(handles, len(matrices), out_handle)
        FfiResult.check(result, "hstack")
        return cls(out_handle[0], owns_handle=True)

    def __repr__(self) -> str:
        """String representation, with shape, nnz, and density."""
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"nnz={self.nnz}, "
            f"density={self.density:.4f})"
        )


class CSCF32(CSC):
    """Python wrapper for CSC<f32, i64>."""

    _value_dtype = np.dtype(np.float32)
    _span_value_class = SpanF32
    _prefix = "csc_f32"
    _c_type = "float"


class CSCF64(CSC):
    """Python wrapper for CSC<f64, i64>."""

    _value_dtype = np.dtype(np.float64)
    _span_value_class = SpanF64
    _prefix = "csc_f64"
    _c_type = "double"


# =============================================================================
# Numba Support (Optional)
# =============================================================================
# Try to import numba extensions. If numba is not installed, this is silently
# skipped. The CSR/CSC classes will still work in pure Python mode.
#
# When numba is available, importing _numba registers:
#   - Type mappings (CSRF64 -> CSRType, etc.)
#   - Unbox/box functions for JIT argument passing
#   - Method overloads (get_row, shape, etc.)
#   - Iterator support (for row in csr:)

_NUMBA_AVAILABLE = False

try:
    # This import triggers all numba type registrations
    from .._numba import (
        CSRType,
        CSCType,
        CSRFloat32Type,
        CSRFloat64Type,
        CSCFloat32Type,
        CSCFloat64Type,
    )
    _NUMBA_AVAILABLE = True
except ImportError:
    # Numba not installed, continue without JIT support
    pass
except Exception as e:
    # Numba installed but extension failed to load
    import warnings
    warnings.warn(f"Failed to load numba extensions: {e}")


def is_numba_available() -> bool:
    """Check if Numba JIT support is available.
    
    Returns:
        True if numba is installed and extensions loaded successfully.
    """
    return _NUMBA_AVAILABLE
