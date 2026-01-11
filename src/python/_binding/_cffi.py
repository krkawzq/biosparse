"""CFFI bindings for scl-core."""

import os
import sys
from cffi import FFI

ffi = FFI()

# C function declarations
ffi.cdef("""
    // =========================================================================
    // Structures
    // =========================================================================
    
    typedef struct {
        void* data;
        size_t len;
        size_t element_size;
        size_t flags;
    } SpanInfo;
    
    typedef struct {
        int64_t rows;
        int64_t cols;
    } Shape;
    
    // =========================================================================
    // Constants
    // =========================================================================
    
    extern uint32_t SCL_CORE_ABI_VERSION;
    extern size_t SPAN_SIZE;
    extern size_t SPAN_FLAG_VIEW;
    extern size_t SPAN_FLAG_ALIGNED;
    extern size_t SPAN_FLAG_MUTABLE;
    
    // =========================================================================
    // Error codes
    // =========================================================================
    
    // FfiResult enum values (i32)
    // Ok = 0
    // NullPointer = -1
    // DimensionMismatch = -2
    // LengthMismatch = -3
    // InvalidIndptr = -4
    // IndexOutOfBounds = -5
    // BufferTooSmall = -6
    // AllocError = -7
    // EmptyInput = -8
    // Unknown = -99
    
    // =========================================================================
    // Span<f32> FFI
    // =========================================================================
    
    float* span_f32_data(void* handle);
    float* span_f32_data_mut(void* handle);
    size_t span_f32_len(void* handle);
    size_t span_f32_flags(void* handle);
    bool span_f32_is_view(void* handle);
    bool span_f32_is_aligned(void* handle);
    bool span_f32_is_mutable(void* handle);
    SpanInfo span_f32_info(void* handle);
    size_t span_f32_byte_size(void* handle);
    void* span_f32_clone(void* handle);
    void span_f32_free(void* handle);
    size_t span_f32_size(void);
    
    // =========================================================================
    // Span<f64> FFI
    // =========================================================================
    
    double* span_f64_data(void* handle);
    double* span_f64_data_mut(void* handle);
    size_t span_f64_len(void* handle);
    size_t span_f64_flags(void* handle);
    bool span_f64_is_view(void* handle);
    bool span_f64_is_aligned(void* handle);
    bool span_f64_is_mutable(void* handle);
    SpanInfo span_f64_info(void* handle);
    size_t span_f64_byte_size(void* handle);
    void* span_f64_clone(void* handle);
    void span_f64_free(void* handle);
    size_t span_f64_size(void);
    
    // =========================================================================
    // Span<i32> FFI
    // =========================================================================
    
    int32_t* span_i32_data(void* handle);
    int32_t* span_i32_data_mut(void* handle);
    size_t span_i32_len(void* handle);
    size_t span_i32_flags(void* handle);
    bool span_i32_is_view(void* handle);
    bool span_i32_is_aligned(void* handle);
    bool span_i32_is_mutable(void* handle);
    SpanInfo span_i32_info(void* handle);
    size_t span_i32_byte_size(void* handle);
    void* span_i32_clone(void* handle);
    void span_i32_free(void* handle);
    size_t span_i32_size(void);
    
    // =========================================================================
    // Span<i64> FFI
    // =========================================================================
    
    int64_t* span_i64_data(void* handle);
    int64_t* span_i64_data_mut(void* handle);
    size_t span_i64_len(void* handle);
    size_t span_i64_flags(void* handle);
    bool span_i64_is_view(void* handle);
    bool span_i64_is_aligned(void* handle);
    bool span_i64_is_mutable(void* handle);
    SpanInfo span_i64_info(void* handle);
    size_t span_i64_byte_size(void* handle);
    void* span_i64_clone(void* handle);
    void span_i64_free(void* handle);
    size_t span_i64_size(void);
    
    // =========================================================================
    // CSR<f32, i64> FFI
    // =========================================================================
    
    // Dimension queries
    int64_t csr_f32_rows(void* handle);
    int64_t csr_f32_cols(void* handle);
    Shape csr_f32_shape(void* handle);
    int64_t csr_f32_nnz(void* handle);
    
    // Internal array access
    void* csr_f32_values_vec_ptr(void* handle);
    void* csr_f32_indices_vec_ptr(void* handle);
    size_t csr_f32_values_vec_len(void* handle);
    
    // Single row access
    void* csr_f32_row_values(void* handle, size_t row);
    void* csr_f32_row_indices(void* handle, size_t row);
    float* csr_f32_row_values_ptr(void* handle, size_t row);
    int64_t* csr_f32_row_indices_ptr(void* handle, size_t row);
    size_t csr_f32_row_len(void* handle, size_t row);
    
    // Unchecked versions (no bounds checking)
    float* csr_f32_row_values_ptr_unchecked(void* handle, size_t row);
    int64_t* csr_f32_row_indices_ptr_unchecked(void* handle, size_t row);
    size_t csr_f32_row_len_unchecked(void* handle, size_t row);
    
    // Validation functions
    bool csr_f32_is_valid(void* handle);
    bool csr_f32_is_sorted(void* handle);
    bool csr_f32_validate(void* handle);
    bool csr_f32_indices_in_bounds(void* handle);
    
    // Query functions
    bool csr_f32_is_empty(void* handle);
    bool csr_f32_is_zero(void* handle);
    double csr_f32_sparsity(void* handle);
    double csr_f32_density(void* handle);
    
    // Cache management
    void csr_f32_invalidate_nnz(void* handle);
    void csr_f32_set_nnz(void* handle, int64_t nnz);
    bool csr_f32_has_nnz_cache(void* handle);
    
    // Clone and free
    void* csr_f32_clone(void* handle);
    void csr_f32_free(void* handle);
    size_t csr_f32_size(void);
    
    // Sorting
    void csr_f32_ensure_sorted(void* handle);
    bool csr_f32_ensure_sorted_checked(void* handle);
    
    // =========================================================================
    // CSR<f64, i64> FFI
    // =========================================================================
    
    // Dimension queries
    int64_t csr_f64_rows(void* handle);
    int64_t csr_f64_cols(void* handle);
    Shape csr_f64_shape(void* handle);
    int64_t csr_f64_nnz(void* handle);
    
    // Internal array access
    void* csr_f64_values_vec_ptr(void* handle);
    void* csr_f64_indices_vec_ptr(void* handle);
    size_t csr_f64_values_vec_len(void* handle);
    
    // Single row access
    void* csr_f64_row_values(void* handle, size_t row);
    void* csr_f64_row_indices(void* handle, size_t row);
    double* csr_f64_row_values_ptr(void* handle, size_t row);
    int64_t* csr_f64_row_indices_ptr(void* handle, size_t row);
    size_t csr_f64_row_len(void* handle, size_t row);
    
    // Unchecked versions (no bounds checking)
    double* csr_f64_row_values_ptr_unchecked(void* handle, size_t row);
    int64_t* csr_f64_row_indices_ptr_unchecked(void* handle, size_t row);
    size_t csr_f64_row_len_unchecked(void* handle, size_t row);
    
    // Validation functions
    bool csr_f64_is_valid(void* handle);
    bool csr_f64_is_sorted(void* handle);
    bool csr_f64_validate(void* handle);
    bool csr_f64_indices_in_bounds(void* handle);
    
    // Query functions
    bool csr_f64_is_empty(void* handle);
    bool csr_f64_is_zero(void* handle);
    double csr_f64_sparsity(void* handle);
    double csr_f64_density(void* handle);
    
    // Cache management
    void csr_f64_invalidate_nnz(void* handle);
    void csr_f64_set_nnz(void* handle, int64_t nnz);
    bool csr_f64_has_nnz_cache(void* handle);
    
    // Clone and free
    void* csr_f64_clone(void* handle);
    void csr_f64_free(void* handle);
    size_t csr_f64_size(void);
    
    // Sorting
    void csr_f64_ensure_sorted(void* handle);
    bool csr_f64_ensure_sorted_checked(void* handle);
    
    // =========================================================================
    // CSC<f32, i64> FFI
    // =========================================================================
    
    // Dimension queries
    int64_t csc_f32_rows(void* handle);
    int64_t csc_f32_cols(void* handle);
    Shape csc_f32_shape(void* handle);
    int64_t csc_f32_nnz(void* handle);
    
    // Internal array access
    void* csc_f32_values_vec_ptr(void* handle);
    void* csc_f32_indices_vec_ptr(void* handle);
    size_t csc_f32_values_vec_len(void* handle);
    
    // Single column access
    void* csc_f32_col_values(void* handle, size_t col);
    void* csc_f32_col_indices(void* handle, size_t col);
    float* csc_f32_col_values_ptr(void* handle, size_t col);
    int64_t* csc_f32_col_indices_ptr(void* handle, size_t col);
    size_t csc_f32_col_len(void* handle, size_t col);
    
    // Unchecked versions (no bounds checking)
    float* csc_f32_col_values_ptr_unchecked(void* handle, size_t col);
    int64_t* csc_f32_col_indices_ptr_unchecked(void* handle, size_t col);
    size_t csc_f32_col_len_unchecked(void* handle, size_t col);
    
    // Validation functions
    bool csc_f32_is_valid(void* handle);
    bool csc_f32_is_sorted(void* handle);
    bool csc_f32_validate(void* handle);
    bool csc_f32_indices_in_bounds(void* handle);
    
    // Query functions
    bool csc_f32_is_empty(void* handle);
    bool csc_f32_is_zero(void* handle);
    double csc_f32_sparsity(void* handle);
    double csc_f32_density(void* handle);
    
    // Cache management
    void csc_f32_invalidate_nnz(void* handle);
    void csc_f32_set_nnz(void* handle, int64_t nnz);
    bool csc_f32_has_nnz_cache(void* handle);
    
    // Clone and free
    void* csc_f32_clone(void* handle);
    void csc_f32_free(void* handle);
    size_t csc_f32_size(void);
    
    // Sorting
    void csc_f32_ensure_sorted(void* handle);
    bool csc_f32_ensure_sorted_checked(void* handle);
    
    // =========================================================================
    // CSC<f64, i64> FFI
    // =========================================================================
    
    // Dimension queries
    int64_t csc_f64_rows(void* handle);
    int64_t csc_f64_cols(void* handle);
    Shape csc_f64_shape(void* handle);
    int64_t csc_f64_nnz(void* handle);
    
    // Internal array access
    void* csc_f64_values_vec_ptr(void* handle);
    void* csc_f64_indices_vec_ptr(void* handle);
    size_t csc_f64_values_vec_len(void* handle);
    
    // Single column access
    void* csc_f64_col_values(void* handle, size_t col);
    void* csc_f64_col_indices(void* handle, size_t col);
    double* csc_f64_col_values_ptr(void* handle, size_t col);
    int64_t* csc_f64_col_indices_ptr(void* handle, size_t col);
    size_t csc_f64_col_len(void* handle, size_t col);
    
    // Unchecked versions (no bounds checking)
    double* csc_f64_col_values_ptr_unchecked(void* handle, size_t col);
    int64_t* csc_f64_col_indices_ptr_unchecked(void* handle, size_t col);
    size_t csc_f64_col_len_unchecked(void* handle, size_t col);
    
    // Validation functions
    bool csc_f64_is_valid(void* handle);
    bool csc_f64_is_sorted(void* handle);
    bool csc_f64_validate(void* handle);
    bool csc_f64_indices_in_bounds(void* handle);
    
    // Query functions
    bool csc_f64_is_empty(void* handle);
    bool csc_f64_is_zero(void* handle);
    double csc_f64_sparsity(void* handle);
    double csc_f64_density(void* handle);
    
    // Cache management
    void csc_f64_invalidate_nnz(void* handle);
    void csc_f64_set_nnz(void* handle, int64_t nnz);
    bool csc_f64_has_nnz_cache(void* handle);
    
    // Clone and free
    void* csc_f64_clone(void* handle);
    void csc_f64_free(void* handle);
    size_t csc_f64_size(void);
    
    // Sorting
    void csc_f64_ensure_sorted(void* handle);
    bool csc_f64_ensure_sorted_checked(void* handle);
    
    // =========================================================================
    // Conversion functions - scipy CSR/CSC format
    // =========================================================================
    
    // CSR from scipy (View mode)
    void* csr_f32_from_scipy_view(int64_t rows, int64_t cols, 
                                   float* data, int64_t* indices, int64_t* indptr);
    void* csr_f64_from_scipy_view(int64_t rows, int64_t cols,
                                   double* data, int64_t* indices, int64_t* indptr);
    
    // CSR from scipy (Copy mode)
    int32_t csr_f32_from_scipy_copy(int64_t rows, int64_t cols,
                                     float* data, size_t data_len,
                                     int64_t* indices, size_t indices_len,
                                     int64_t* indptr, size_t indptr_len,
                                     void** out_handle);
    int32_t csr_f64_from_scipy_copy(int64_t rows, int64_t cols,
                                     double* data, size_t data_len,
                                     int64_t* indices, size_t indices_len,
                                     int64_t* indptr, size_t indptr_len,
                                     void** out_handle);
    
    // CSC from scipy (View mode)
    void* csc_f32_from_scipy_view(int64_t rows, int64_t cols,
                                   float* data, int64_t* indices, int64_t* indptr);
    void* csc_f64_from_scipy_view(int64_t rows, int64_t cols,
                                   double* data, int64_t* indices, int64_t* indptr);
    
    // CSC from scipy (Copy mode)
    int32_t csc_f32_from_scipy_copy(int64_t rows, int64_t cols,
                                     float* data, size_t data_len,
                                     int64_t* indices, size_t indices_len,
                                     int64_t* indptr, size_t indptr_len,
                                     void** out_handle);
    int32_t csc_f64_from_scipy_copy(int64_t rows, int64_t cols,
                                     double* data, size_t data_len,
                                     int64_t* indices, size_t indices_len,
                                     int64_t* indptr, size_t indptr_len,
                                     void** out_handle);
    
    // =========================================================================
    // Conversion functions - COO format
    // =========================================================================
    
    int32_t csr_f32_from_coo(int64_t rows, int64_t cols,
                              int64_t* row_indices, int64_t* col_indices,
                              float* data, size_t nnz, void** out_handle);
    int32_t csr_f64_from_coo(int64_t rows, int64_t cols,
                              int64_t* row_indices, int64_t* col_indices,
                              double* data, size_t nnz, void** out_handle);
    int32_t csc_f32_from_coo(int64_t rows, int64_t cols,
                              int64_t* row_indices, int64_t* col_indices,
                              float* data, size_t nnz, void** out_handle);
    int32_t csc_f64_from_coo(int64_t rows, int64_t cols,
                              int64_t* row_indices, int64_t* col_indices,
                              double* data, size_t nnz, void** out_handle);
    
    // =========================================================================
    // Conversion functions - CSR <-> CSC interconversion
    // =========================================================================
    
    int32_t csc_f32_from_csr(void* csr_handle, void** out_handle);
    int32_t csc_f64_from_csr(void* csr_handle, void** out_handle);
    int32_t csr_f32_from_csc(void* csc_handle, void** out_handle);
    int32_t csr_f64_from_csc(void* csc_handle, void** out_handle);
    
    // =========================================================================
    // Conversion functions - Dense format
    // =========================================================================
    
    int32_t csr_f32_to_dense(void* handle, float* out, size_t out_len, bool col_major);
    int32_t csr_f64_to_dense(void* handle, double* out, size_t out_len, bool col_major);
    int32_t csc_f32_to_dense(void* handle, float* out, size_t out_len, bool col_major);
    int32_t csc_f64_to_dense(void* handle, double* out, size_t out_len, bool col_major);
    
    // =========================================================================
    // Conversion functions - COO output
    // =========================================================================
    
    int32_t csr_f32_to_coo(void* handle, int64_t* out_row_indices,
                            int64_t* out_col_indices, float* out_data, size_t out_len);
    int32_t csr_f64_to_coo(void* handle, int64_t* out_row_indices,
                            int64_t* out_col_indices, double* out_data, size_t out_len);
    int32_t csc_f32_to_coo(void* handle, int64_t* out_row_indices,
                            int64_t* out_col_indices, float* out_data, size_t out_len);
    int32_t csc_f64_to_coo(void* handle, int64_t* out_row_indices,
                            int64_t* out_col_indices, double* out_data, size_t out_len);
    
    // =========================================================================
    // Slicing operations
    // =========================================================================
    
    // CSR row slicing
    int32_t csr_f32_slice_rows(void* handle, int64_t row_start, int64_t row_end, void** out_handle);
    int32_t csr_f64_slice_rows(void* handle, int64_t row_start, int64_t row_end, void** out_handle);
    
    // CSR column slicing
    int32_t csr_f32_slice_cols(void* handle, int64_t col_start, int64_t col_end, void** out_handle);
    int32_t csr_f64_slice_cols(void* handle, int64_t col_start, int64_t col_end, void** out_handle);
    
    // CSC column slicing
    int32_t csc_f32_slice_cols(void* handle, int64_t col_start, int64_t col_end, void** out_handle);
    int32_t csc_f64_slice_cols(void* handle, int64_t col_start, int64_t col_end, void** out_handle);
    
    // CSC row slicing
    int32_t csc_f32_slice_rows(void* handle, int64_t row_start, int64_t row_end, void** out_handle);
    int32_t csc_f64_slice_rows(void* handle, int64_t row_start, int64_t row_end, void** out_handle);
    
    // CSR row mask slicing
    int32_t csr_f32_slice_rows_mask(void* handle, bool* mask, size_t mask_len, void** out_handle);
    int32_t csr_f64_slice_rows_mask(void* handle, bool* mask, size_t mask_len, void** out_handle);
    
    // CSR column mask slicing
    int32_t csr_f32_slice_cols_mask(void* handle, bool* mask, size_t mask_len, void** out_handle);
    int32_t csr_f64_slice_cols_mask(void* handle, bool* mask, size_t mask_len, void** out_handle);
    
    // CSC column mask slicing
    int32_t csc_f32_slice_cols_mask(void* handle, bool* mask, size_t mask_len, void** out_handle);
    int32_t csc_f64_slice_cols_mask(void* handle, bool* mask, size_t mask_len, void** out_handle);
    
    // CSC row mask slicing
    int32_t csc_f32_slice_rows_mask(void* handle, bool* mask, size_t mask_len, void** out_handle);
    int32_t csc_f64_slice_rows_mask(void* handle, bool* mask, size_t mask_len, void** out_handle);
    
    // =========================================================================
    // Stacking operations
    // =========================================================================
    
    // CSR vstack/hstack
    int32_t csr_f32_vstack(void** handles, size_t count, void** out_handle);
    int32_t csr_f64_vstack(void** handles, size_t count, void** out_handle);
    int32_t csr_f32_hstack(void** handles, size_t count, void** out_handle);
    int32_t csr_f64_hstack(void** handles, size_t count, void** out_handle);
    
    // CSC vstack/hstack
    int32_t csc_f32_vstack(void** handles, size_t count, void** out_handle);
    int32_t csc_f64_vstack(void** handles, size_t count, void** out_handle);
    int32_t csc_f32_hstack(void** handles, size_t count, void** out_handle);
    int32_t csc_f64_hstack(void** handles, size_t count, void** out_handle);
    
    // =========================================================================
    // Numba support - Batch pointer access
    // =========================================================================
    
    // Row/Column pointer info structures (for Numba unbox)
    typedef struct {
        void* values_ptr;
        void* indices_ptr;
        size_t len;
    } RowPointerInfo;
    
    typedef struct {
        void* values_ptr;
        void* indices_ptr;
        size_t len;
    } ColPointerInfo;
    
    // CSR batch pointer access (for Numba)
    int32_t csr_f32_get_row_pointers(void* handle, float** out_values_ptrs, int64_t** out_indices_ptrs, size_t* out_row_lens);
    int32_t csr_f64_get_row_pointers(void* handle, double** out_values_ptrs, int64_t** out_indices_ptrs, size_t* out_row_lens);
    
    // CSR single row info (for Numba)
    RowPointerInfo csr_f32_get_row_info(void* handle, size_t row);
    RowPointerInfo csr_f64_get_row_info(void* handle, size_t row);
    RowPointerInfo csr_f32_get_row_info_unchecked(void* handle, size_t row);
    RowPointerInfo csr_f64_get_row_info_unchecked(void* handle, size_t row);
    
    // CSR metadata (for Numba)
    int32_t csr_f32_get_metadata(void* handle, int64_t* out_nrows, int64_t* out_ncols, int64_t* out_nnz);
    int32_t csr_f64_get_metadata(void* handle, int64_t* out_nrows, int64_t* out_ncols, int64_t* out_nnz);
    
    // CSC batch pointer access (for Numba)
    int32_t csc_f32_get_col_pointers(void* handle, float** out_values_ptrs, int64_t** out_indices_ptrs, size_t* out_col_lens);
    int32_t csc_f64_get_col_pointers(void* handle, double** out_values_ptrs, int64_t** out_indices_ptrs, size_t* out_col_lens);
    
    // CSC single column info (for Numba)
    ColPointerInfo csc_f32_get_col_info(void* handle, size_t col);
    ColPointerInfo csc_f64_get_col_info(void* handle, size_t col);
    ColPointerInfo csc_f32_get_col_info_unchecked(void* handle, size_t col);
    ColPointerInfo csc_f64_get_col_info_unchecked(void* handle, size_t col);
    
    // CSC metadata (for Numba)
    int32_t csc_f32_get_metadata(void* handle, int64_t* out_nrows, int64_t* out_ncols, int64_t* out_nnz);
    int32_t csc_f64_get_metadata(void* handle, int64_t* out_nrows, int64_t* out_ncols, int64_t* out_nnz);
""")


# FFI result codes
class FfiResult:
    """FFI operation result codes."""
    Ok = 0
    NullPointer = -1
    DimensionMismatch = -2
    LengthMismatch = -3
    InvalidIndptr = -4
    IndexOutOfBounds = -5
    BufferTooSmall = -6
    AllocError = -7
    EmptyInput = -8
    Unknown = -99
    
    _messages = {
        0: "Success",
        -1: "Null pointer",
        -2: "Dimension mismatch",
        -3: "Length mismatch",
        -4: "Invalid indptr",
        -5: "Index out of bounds",
        -6: "Buffer too small",
        -7: "Allocation error",
        -8: "Empty input",
        -99: "Unknown error",
    }
    
    @classmethod
    def check(cls, code: int, context: str = "") -> None:
        """Check result code, raise exception on failure.
        
        Args:
            code: Result code to check.
            context: Optional context string for error message.
            
        Raises:
            RuntimeError: If code is not Ok.
        """
        if code != cls.Ok:
            msg = cls._messages.get(code, f"Unknown error code: {code}")
            if context:
                msg = f"{context}: {msg}"
            raise RuntimeError(msg)
    
    @classmethod
    def message(cls, code: int) -> str:
        """Get message corresponding to result code.
        
        Args:
            code: Result code.
            
        Returns:
            Error message string.
        """
        return cls._messages.get(code, f"Unknown error code: {code}")


def _find_library():
    """Find scl_core dynamic library.
    
    Returns:
        Path to the library file.
        
    Raises:
        RuntimeError: If library cannot be found.
    """
    # Library name (platform-dependent)
    if sys.platform == "win32":
        lib_name = "scl_core.dll"
    elif sys.platform == "darwin":
        lib_name = "libscl_core.dylib"
    else:
        lib_name = "libscl_core.so"
    
    # Search paths
    search_paths = [
        # Development mode: target relative to python directory
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "target", "release"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "target", "debug"),
        # Installation mode: same directory as Python package
        os.path.dirname(__file__),
        # System paths
        "/usr/local/lib",
        "/usr/lib",
    ]
    
    # Environment variable override
    if "SCL_CORE_LIB" in os.environ:
        return os.environ["SCL_CORE_LIB"]
    
    for path in search_paths:
        lib_path = os.path.join(path, lib_name)
        if os.path.exists(lib_path):
            return lib_path
    
    raise RuntimeError(
        f"Cannot find {lib_name}. "
        f"Please build the library with 'cargo build --release' "
        f"or set SCL_CORE_LIB environment variable."
    )


# Load dynamic library
lib = ffi.dlopen(_find_library())

# Export ABI version
ABI_VERSION = lib.SCL_CORE_ABI_VERSION

# Export Span flags
SPAN_FLAG_VIEW = lib.SPAN_FLAG_VIEW
SPAN_FLAG_ALIGNED = lib.SPAN_FLAG_ALIGNED
SPAN_FLAG_MUTABLE = lib.SPAN_FLAG_MUTABLE
