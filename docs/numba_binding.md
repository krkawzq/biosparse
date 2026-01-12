# Numba Integration Module

The `biosparse._numba` module provides Numba type extensions that enable CSR/CSC sparse matrices to be used directly in JIT-compiled functions with zero-copy efficiency.

## Overview

This module implements:

- **Type System**: Numba type representations for CSR/CSC matrices
- **Boxing/Unboxing**: Conversion between Python objects and Numba's internal representation
- **Method Overloads**: JIT-compatible implementations of matrix operations
- **Iterator Support**: Efficient row/column iteration in JIT code

## Quick Start

```python
from numba import njit
from biosparse import CSRF64
import scipy.sparse as sp
import numpy as np

# Create a CSR matrix
scipy_mat = sp.random(1000, 500, density=0.01, format='csr', dtype='float64')
csr = CSRF64.from_scipy(scipy_mat)

@njit
def row_sums(csr):
    """Compute sum of each row."""
    n_rows = csr.nrows
    result = np.empty(n_rows, dtype=np.float64)
    
    for i in range(n_rows):
        values, indices = csr.row_to_numpy(i)
        result[i] = np.sum(values)
    
    return result

# Call the JIT function
sums = row_sums(csr)
```

---

## Type System

### Available Types

| Python Class | Numba Type | Description |
|--------------|------------|-------------|
| `CSRF32` | `CSRFloat32Type` | CSR with float32 values |
| `CSRF64` | `CSRFloat64Type` | CSR with float64 values |
| `CSCF32` | `CSCFloat32Type` | CSC with float32 values |
| `CSCF64` | `CSCFloat64Type` | CSC with float64 values |

### Type Inference

Types are automatically inferred when passing CSR/CSC objects to JIT functions:

```python
from numba import njit
from biosparse import CSRF64, CSCF64

@njit
def process_csr(csr):
    return csr.nrows  # Numba knows csr is CSRFloat64Type

@njit
def process_csc(csc):
    return csc.ncols  # Numba knows csc is CSCFloat64Type

csr = CSRF64.from_scipy(scipy_mat)
process_csr(csr)  # Type inference happens automatically
```

---

## Available Properties in JIT Code

### CSR Properties

```python
@njit
def inspect_csr(csr):
    # Basic dimensions
    nrows = csr.nrows       # Number of rows (int64)
    ncols = csr.ncols       # Number of columns (int64)
    nnz = csr.nnz           # Number of non-zeros (int64)
    
    # Derived properties
    shape = csr.shape       # (nrows, ncols) tuple
    density = csr.density   # nnz / (nrows * ncols)
    sparsity = csr.sparsity # 1 - density
    
    # Boolean flags
    is_empty = csr.is_empty # nrows == 0 or ncols == 0
    is_zero = csr.is_zero   # nnz == 0
    
    return nrows, ncols, nnz
```

### CSC Properties

```python
@njit
def inspect_csc(csc):
    # Same properties as CSR
    nrows = csc.nrows
    ncols = csc.ncols
    nnz = csc.nnz
    shape = csc.shape
    density = csc.density
    sparsity = csc.sparsity
    is_empty = csc.is_empty
    is_zero = csc.is_zero
    
    return nrows, ncols, nnz
```

---

## Available Methods in JIT Code

### CSR Row Access

#### `row_to_numpy(row_idx, copy=False)`
Get row data as NumPy arrays.

```python
@njit
def process_row(csr, row_idx):
    values, indices = csr.row_to_numpy(row_idx)
    # values: float64[::1] or float32[::1] array
    # indices: int64[::1] array
    return np.sum(values)
```

#### `row(row_idx, copy=False)`
Alias for `row_to_numpy()`.

```python
@njit
def process_row(csr, row_idx):
    values, indices = csr.row(row_idx)
    return np.sum(values)
```

#### `row_len(row_idx)`
Get the number of non-zeros in a row.

```python
@njit
def get_row_lengths(csr):
    lengths = np.empty(csr.nrows, dtype=np.int64)
    for i in range(csr.nrows):
        lengths[i] = csr.row_len(i)
    return lengths
```

#### `col(col_idx)`
Extract column data (traverses all rows).

```python
@njit
def get_column(csr, col_idx):
    values, row_indices = csr.col(col_idx)
    return values, row_indices
```

#### `get(row_idx, col_idx, default=0.0)`
Safe element access with default value.

```python
@njit
def safe_access(csr, i, j):
    # Returns 0.0 if position is sparse or out of bounds
    return csr.get(i, j)

@njit
def safe_access_custom(csr, i, j):
    # Custom default value
    return csr.get(i, j, -1.0)
```

---

### CSC Column Access

#### `col_to_numpy(col_idx, copy=False)`
Get column data as NumPy arrays.

```python
@njit
def process_col(csc, col_idx):
    values, indices = csc.col_to_numpy(col_idx)
    return np.sum(values)
```

#### `col(col_idx, copy=False)`
Alias for `col_to_numpy()`.

#### `col_len(col_idx)`
Get the number of non-zeros in a column.

#### `row(row_idx)`
Extract row data (traverses all columns).

#### `get(row_idx, col_idx, default=0.0)`
Safe element access.

---

## Iterator Support

### CSR Row Iteration

CSR matrices support direct iteration over rows:

```python
@njit
def sum_all_csr(csr):
    """Sum all non-zero values using row iteration."""
    total = 0.0
    
    for values, indices in csr:
        # values: float64[::1] array of row values
        # indices: int64[::1] array of column indices
        for val in values:
            total += val
    
    return total
```

### CSC Column Iteration

CSC matrices support iteration over columns:

```python
@njit
def sum_all_csc(csc):
    """Sum all non-zero values using column iteration."""
    total = 0.0
    
    for values, indices in csc:
        # values: float64[::1] array of column values
        # indices: int64[::1] array of row indices
        for val in values:
            total += val
    
    return total
```

### Enumerated Iteration

For row/column indices during iteration:

```python
@njit
def row_statistics(csr):
    """Compute mean of each row."""
    result = np.empty(csr.nrows, dtype=np.float64)
    
    row_idx = 0
    for values, indices in csr:
        n = len(values)
        if n > 0:
            result[row_idx] = np.sum(values) / n
        else:
            result[row_idx] = 0.0
        row_idx += 1
    
    return result
```

---

## len() Builtin Support

```python
@njit
def get_lengths(csr, csc):
    # len(csr) returns number of rows
    n_rows = len(csr)
    
    # len(csc) returns number of columns
    n_cols = len(csc)
    
    return n_rows, n_cols
```

---

## Complete Examples

### Example 1: Row-wise Normalization

```python
from numba import njit
from biosparse import CSRF64
import numpy as np
import scipy.sparse as sp

@njit
def compute_row_sums(csr):
    """Compute sum of each row (accounting for zeros)."""
    n_rows = csr.nrows
    n_cols = csr.ncols
    sums = np.empty(n_rows, dtype=np.float64)
    
    row_idx = 0
    for values, indices in csr:
        # Sum of non-zero values (zeros sum to 0 anyway)
        sums[row_idx] = np.sum(values)
        row_idx += 1
    
    return sums

@njit
def compute_row_means(csr):
    """Compute mean of each row (including zeros)."""
    n_rows = csr.nrows
    n_cols = float(csr.ncols)
    means = np.empty(n_rows, dtype=np.float64)
    
    row_idx = 0
    for values, indices in csr:
        means[row_idx] = np.sum(values) / n_cols
        row_idx += 1
    
    return means

# Usage
scipy_mat = sp.random(1000, 500, density=0.01, format='csr')
csr = CSRF64.from_scipy(scipy_mat)

sums = compute_row_sums(csr)
means = compute_row_means(csr)
```

### Example 2: Feature-wise Statistics

```python
from numba import njit
from biosparse import CSRF64
from biosparse.optim import assume, vectorize
import numpy as np

@njit
def compute_feature_stats(csr):
    """Compute mean and variance for each row (feature)."""
    n_rows = csr.nrows
    N = float(csr.ncols)
    
    assume(n_rows > 0)
    assume(N > 0)
    
    means = np.empty(n_rows, dtype=np.float64)
    variances = np.empty(n_rows, dtype=np.float64)
    
    row_idx = 0
    for values, indices in csr:
        # Compute sum and sum of squares
        row_sum = 0.0
        row_sq_sum = 0.0
        
        vectorize(8)
        for j in range(len(values)):
            val = values[j]
            row_sum += val
            row_sq_sum += val * val
        
        # Mean
        mu = row_sum / N
        
        # Variance using computational formula
        var = (row_sq_sum - row_sum * mu) / (N - 1.0) if N > 1.0 else 0.0
        if var < 0.0:
            var = 0.0
        
        means[row_idx] = mu
        variances[row_idx] = var
        row_idx += 1
    
    return means, variances
```

### Example 3: Sparse Matrix-Vector Multiplication

```python
from numba import njit
from biosparse import CSRF64
import numpy as np

@njit
def spmv(csr, x):
    """Sparse matrix-vector multiplication: y = A @ x"""
    n_rows = csr.nrows
    y = np.zeros(n_rows, dtype=np.float64)
    
    row_idx = 0
    for values, col_indices in csr:
        dot = 0.0
        for j in range(len(values)):
            dot += values[j] * x[col_indices[j]]
        y[row_idx] = dot
        row_idx += 1
    
    return y

# Usage
scipy_mat = sp.random(1000, 500, density=0.01, format='csr')
csr = CSRF64.from_scipy(scipy_mat)
x = np.random.randn(500)

y = spmv(csr, x)
```

### Example 4: Binary Search in Sparse Row

```python
from numba import njit
from biosparse import CSRF64
import numpy as np

@njit
def get_element(csr, row, col):
    """Get element at (row, col) using binary search."""
    values, indices = csr.row_to_numpy(row)
    
    # Binary search
    pos = np.searchsorted(indices, col)
    
    if pos < len(indices) and indices[pos] == col:
        return values[pos]
    return 0.0

@njit
def get_submatrix_dense(csr, rows, cols):
    """Extract a dense submatrix from sparse matrix."""
    n_rows = len(rows)
    n_cols = len(cols)
    result = np.zeros((n_rows, n_cols), dtype=np.float64)
    
    for i in range(n_rows):
        row_idx = rows[i]
        values, indices = csr.row_to_numpy(row_idx)
        
        for j in range(n_cols):
            col_idx = cols[j]
            pos = np.searchsorted(indices, col_idx)
            if pos < len(indices) and indices[pos] == col_idx:
                result[i, j] = values[pos]
    
    return result
```

### Example 5: Group-wise Aggregation

```python
from numba import njit
from biosparse import CSRF64
import numpy as np

@njit
def group_means(csr, group_ids, n_groups):
    """Compute mean per group for each row.
    
    Args:
        csr: CSR sparse matrix (features x samples)
        group_ids: Group assignment for each column
        n_groups: Number of groups
    
    Returns:
        (n_rows, n_groups) array of means
    """
    n_rows = csr.nrows
    n_cols = csr.ncols
    
    # Count elements per group
    group_counts = np.zeros(n_groups, dtype=np.int64)
    for j in range(n_cols):
        g = group_ids[j]
        if g >= 0 and g < n_groups:
            group_counts[g] += 1
    
    result = np.zeros((n_rows, n_groups), dtype=np.float64)
    
    row_idx = 0
    for values, col_indices in csr:
        # Accumulate sums per group
        group_sums = np.zeros(n_groups, dtype=np.float64)
        
        for j in range(len(values)):
            col = col_indices[j]
            g = group_ids[col]
            if g >= 0 and g < n_groups:
                group_sums[g] += values[j]
        
        # Compute means
        for g in range(n_groups):
            if group_counts[g] > 0:
                result[row_idx, g] = group_sums[g] / group_counts[g]
        
        row_idx += 1
    
    return result
```

---

## Performance Tips

### 1. Avoid Repeated Row/Column Access

```python
# BAD: Accesses row data twice
@njit
def bad_example(csr, row):
    n = csr.row_len(row)
    values, indices = csr.row_to_numpy(row)
    return n, np.sum(values)

# GOOD: Access once
@njit
def good_example(csr, row):
    values, indices = csr.row_to_numpy(row)
    return len(values), np.sum(values)
```

### 2. Use Iterator for Full Traversal

```python
# BAD: Indexing in loop
@njit
def bad_sum(csr):
    total = 0.0
    for i in range(csr.nrows):
        values, _ = csr.row_to_numpy(i)
        total += np.sum(values)
    return total

# GOOD: Use iterator
@njit
def good_sum(csr):
    total = 0.0
    for values, _ in csr:
        total += np.sum(values)
    return total
```

### 3. Use Optimization Intrinsics

```python
from biosparse.optim import assume, vectorize, likely, unlikely

@njit
def optimized_sum(csr):
    assume(csr.nrows > 0)
    
    total = 0.0
    for values, _ in csr:
        n = len(values)
        assume(n >= 0)
        
        vectorize(8)
        for j in range(n):
            total += values[j]
    
    return total
```

### 4. Preallocate Output Arrays

```python
# BAD: Growing lists
@njit
def bad_collect(csr, threshold):
    result = []
    for values, _ in csr:
        for v in values:
            if v > threshold:
                result.append(v)  # Slow!
    return result

# GOOD: Preallocate
@njit
def good_collect(csr, threshold):
    # First pass: count
    count = 0
    for values, _ in csr:
        for v in values:
            if v > threshold:
                count += 1
    
    # Allocate
    result = np.empty(count, dtype=np.float64)
    
    # Second pass: fill
    idx = 0
    for values, _ in csr:
        for v in values:
            if v > threshold:
                result[idx] = v
                idx += 1
    
    return result
```

---

## Limitations

1. **Read-only**: CSR/CSC objects cannot be modified in JIT code
2. **No slicing**: `csr[i:j, k:l]` slicing is not supported in JIT (use Python)
3. **No construction**: Cannot create new CSR/CSC in JIT (use Python)
4. **Return values**: Returning CSR/CSC from JIT is experimental

---

## Checking Numba Availability

```python
from biosparse._binding._sparse import is_numba_available

if is_numba_available():
    print("Numba JIT support is available")
else:
    print("Numba not installed or extension failed to load")
```
