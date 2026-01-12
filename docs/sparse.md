# Sparse Matrix Module

The `biosparse._binding` module provides Python bindings for high-performance CSR (Compressed Sparse Row) and CSC (Compressed Sparse Column) sparse matrices implemented in Rust.

## Overview

This module exposes the following main classes:

- **CSRF32** / **CSRF64**: CSR sparse matrices with float32/float64 values
- **CSCF32** / **CSCF64**: CSC sparse matrices with float32/float64 values
- **SpanF32** / **SpanF64** / **SpanI32** / **SpanI64**: Typed memory views

## Quick Start

```python
import numpy as np
import scipy.sparse as sp
from biosparse import CSRF64, CSCF64

# Create from scipy
scipy_mat = sp.random(1000, 500, density=0.01, format='csr', dtype='float64')
csr = CSRF64.from_scipy(scipy_mat)

# Basic properties
print(csr.shape)      # (1000, 500)
print(csr.nnz)        # Number of non-zero elements
print(csr.density)    # ~0.01

# Slicing operations
sub = csr[100:200, 50:150]

# Convert back to scipy
scipy_back = csr.to_scipy()
```

---

## CSR Class API

### Constructors

#### `CSR.from_scipy(mat, copy=True)`
Create a CSR matrix from a scipy sparse matrix.

```python
import scipy.sparse as sp
from biosparse import CSRF64

scipy_mat = sp.random(1000, 1000, density=0.01, format='csr')
csr = CSRF64.from_scipy(scipy_mat)

# Zero-copy view (caller must keep scipy_mat alive)
csr_view = CSRF64.from_scipy(scipy_mat, copy=False)
```

**Parameters:**
- `mat`: scipy.sparse matrix (will be converted to CSR if needed)
- `copy`: If True, copy data; if False, create a view

**Returns:** New CSR instance

---

#### `CSR.from_coo(rows, cols, row_indices, col_indices, data)`
Create a CSR matrix from COO (coordinate) format arrays.

```python
import numpy as np
from biosparse import CSRF64

rows, cols = 100, 50
row_idx = np.array([0, 0, 1, 2, 2], dtype=np.int64)
col_idx = np.array([0, 2, 1, 0, 3], dtype=np.int64)
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

csr = CSRF64.from_coo(rows, cols, row_idx, col_idx, data)
```

**Parameters:**
- `rows`, `cols`: Matrix dimensions
- `row_indices`, `col_indices`: Coordinate arrays
- `data`: Value array

---

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `shape` | `Tuple[int, int]` | Matrix dimensions `(nrows, ncols)` |
| `nrows` | `int` | Number of rows |
| `ncols` | `int` | Number of columns |
| `nnz` | `int` | Number of non-zero elements |
| `density` | `float` | Proportion of non-zero elements (0.0-1.0) |
| `sparsity` | `float` | Proportion of zero elements (0.0-1.0) |
| `is_empty` | `bool` | True if matrix has zero rows or columns |
| `is_zero` | `bool` | True if matrix has no non-zero elements |
| `is_valid` | `bool` | True if matrix structure is valid |
| `is_sorted` | `bool` | True if all row indices are sorted |
| `indices_in_bounds` | `bool` | True if all indices are within bounds |
| `handle` | `cffi pointer` | Raw FFI handle (for advanced use) |
| `handle_as_int` | `int` | Handle as integer (for Numba interop) |

---

### Row Access Methods

#### `row_values(row)`
Get the value Span for a row.

```python
values_span = csr.row_values(0)
values_arr = values_span.to_numpy()  # Convert to NumPy
```

#### `row_indices(row)`
Get the index Span for a row.

```python
indices_span = csr.row_indices(0)
indices_arr = indices_span.to_numpy()
```

#### `row_to_numpy(row, copy=False)`
Get row data as NumPy arrays.

```python
values, indices = csr.row_to_numpy(0)
print(f"Row 0: values={values}, indices={indices}")
```

#### `row(row, copy=False)`
Simplified alias for `row_to_numpy()`.

```python
values, indices = csr.row(5)
```

#### `row_len(row)`
Get the number of non-zeros in a row.

```python
nnz_in_row = csr.row_len(0)
```

#### `col(col)`
Extract column data (non-contiguous, requires traversing all rows).

```python
values, row_indices = csr.col(10)
```

---

### Element Access

#### `csr[i, j]`
Access a single element.

```python
value = csr[10, 20]  # Returns float (0.0 if position is sparse)
```

#### `get(row, col, default=0.0)`
Safe element access with default value.

```python
value = csr.get(10, 20)  # Returns 0.0 if not found or out of bounds
value = csr.get(10, 20, default=-1.0)  # Custom default
```

---

### Slicing Operations

#### `slice_rows(start, end)`
Extract a row range.

```python
sub = csr.slice_rows(10, 20)  # Rows 10-19
```

#### `slice_cols(start, end)`
Extract a column range.

```python
sub = csr.slice_cols(5, 15)  # Columns 5-14
```

#### `slice_rows_mask(mask)`
Select rows using a boolean mask.

```python
mask = np.array([True, False, True, ...])
sub = csr.slice_rows_mask(mask)
```

#### `slice_cols_mask(mask)`
Select columns using a boolean mask.

```python
mask = np.array([True, False, True, ...])
sub = csr.slice_cols_mask(mask)
```

#### Python Slicing Syntax

```python
# Row slicing
sub = csr[10:20]
sub = csr[10:20, :]

# Column slicing
sub = csr[:, 5:15]

# Combined slicing
sub = csr[10:20, 5:15]

# Single element
value = csr[10, 5]
```

---

### Conversion Methods

#### `to_dense(order='C')`
Convert to dense NumPy array.

```python
dense = csr.to_dense()        # Row-major (C order)
dense = csr.to_dense('F')     # Column-major (Fortran order)
```

#### `to_coo()`
Convert to COO format.

```python
row_indices, col_indices, data = csr.to_coo()
```

#### `to_csc()`
Convert to CSC format.

```python
csc = csr.to_csc()
```

#### `T()`
Transpose (returns CSC).

```python
csc_transposed = csr.T()
```

#### `to_scipy()`
Convert to scipy.sparse.csr_matrix.

```python
scipy_mat = csr.to_scipy()
```

---

### Stacking Operations

#### `CSR.vstack(matrices)`
Vertically stack matrices (concatenate rows).

```python
from biosparse import CSRF64

stacked = CSRF64.vstack([csr1, csr2, csr3])
```

#### `CSR.hstack(matrices)`
Horizontally stack matrices (concatenate columns).

```python
stacked = CSRF64.hstack([csr1, csr2, csr3])
```

---

### Utility Methods

#### `clone()`
Create a deep copy.

```python
csr_copy = csr.clone()
```

#### `ensure_sorted()`
Sort row indices in place.

```python
csr.ensure_sorted()
```

#### `ensure_sorted_checked()`
Sort if needed, return whether sorting was performed.

```python
did_sort = csr.ensure_sorted_checked()
```

#### `validate()`
Full validation (structure + sorting + bounds).

```python
is_valid = csr.validate()
```

---

## CSC Class API

CSC (Compressed Sparse Column) has a similar API to CSR, but oriented around columns:

| CSR Method | CSC Equivalent |
|------------|----------------|
| `row_values(i)` | `col_values(j)` |
| `row_indices(i)` | `col_indices(j)` |
| `row_to_numpy(i)` | `col_to_numpy(j)` |
| `row(i)` | `col(j)` |
| `row_len(i)` | `col_len(j)` |

### CSC-Specific Conversions

```python
csc = CSCF64.from_scipy(scipy_csc_mat)

# Convert to CSR
csr = csc.to_csr()

# Transpose (returns CSR)
csr_transposed = csc.T()
```

---

## Span Class API

Span represents a typed contiguous memory view.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_view` | `bool` | True if this is a view (does not own memory) |
| `is_aligned` | `bool` | True if data is SIMD-aligned |
| `is_mutable` | `bool` | True if data can be modified |
| `byte_size` | `int` | Size in bytes |
| `data_ptr` | `int` | Data pointer as integer |

### Methods

#### `to_numpy(copy=False)`
Convert to NumPy array.

```python
span = csr.row_values(0)
arr = span.to_numpy()        # View (shares memory)
arr = span.to_numpy(True)    # Copy
```

#### `clone()`
Create a deep copy.

```python
span_copy = span.clone()
```

---

## Numba Integration

When Numba is available, CSR/CSC objects can be passed directly to JIT functions:

```python
from numba import njit
from biosparse import CSRF64
import scipy.sparse as sp

@njit
def sum_all_values(csr):
    """Sum all non-zero values in a CSR matrix."""
    total = 0.0
    for values, indices in csr:  # Iterate over rows
        for val in values:
            total += val
    return total

# Create CSR
scipy_mat = sp.random(1000, 500, density=0.01, format='csr')
csr = CSRF64.from_scipy(scipy_mat)

# Call JIT function
result = sum_all_values(csr)
```

See `numba_binding.md` for detailed Numba integration documentation.

---

## Performance Tips

1. **Use zero-copy views when possible**: `from_scipy(mat, copy=False)` avoids copying data.

2. **Prefer row access for CSR, column access for CSC**: These are O(1) operations.

3. **Avoid repeated slicing in loops**: Extract the slice once before the loop.

4. **Use Numba for hot paths**: Direct iteration in Numba JIT is much faster than Python.

5. **Ensure indices are sorted**: Some operations require sorted indices. Call `ensure_sorted()` once if needed.

---

## Type Reference

| Class | Value Type | Index Type |
|-------|------------|------------|
| `CSRF32` | `float32` | `int64` |
| `CSRF64` | `float64` | `int64` |
| `CSCF32` | `float32` | `int64` |
| `CSCF64` | `float64` | `int64` |
| `SpanF32` | `float32` | - |
| `SpanF64` | `float64` | - |
| `SpanI32` | `int32` | - |
| `SpanI64` | `int64` | - |
