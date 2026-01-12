# Chapter 1: Data Structure Integration - Using CSR/CSC in Numba

## Overview

biosparse provides high-performance sparse matrix types (`CSR` and `CSC`) that work seamlessly with Numba JIT compilation. This chapter teaches you how to use these data structures effectively in optimized kernels.

## The biosparse Type System

```
┌─────────────────────────────────────────────────────────────────────┐
│                      biosparse Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│  Python Layer                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  CSRF32, CSRF64, CSCF32, CSCF64                                 ││
│  │  - Full Python API (slicing, iteration, conversion)             ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              ↕ Boxing/Unboxing                      │
│  Numba Layer                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  CSRType, CSCType (Numba type system)                           ││
│  │  - JIT-compiled access methods                                  ││
│  │  - Iterator support                                             ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              ↕ FFI                                  │
│  Rust Core                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  High-performance sparse matrix implementation                  ││
│  │  - Zero-copy operations where possible                          ││
│  │  - SIMD-optimized routines                                      ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## Creating Sparse Matrices

### From SciPy

```python
import scipy.sparse as sp
from biosparse import CSRF64, CSCF64

# Create a SciPy sparse matrix
scipy_csr = sp.random(1000, 500, density=0.1, format='csr', dtype='float64')

# Convert to biosparse (with copy - safe, independent)
csr = CSRF64.from_scipy(scipy_csr, copy=True)

# Convert to biosparse (zero-copy view - fast, shares memory)
csr_view = CSRF64.from_scipy(scipy_csr, copy=False)
# Warning: scipy_csr must stay alive while using csr_view!
```

### From COO Format

```python
import numpy as np
from biosparse import CSRF64

# COO data
row_indices = np.array([0, 0, 1, 2, 2], dtype=np.int64)
col_indices = np.array([0, 2, 1, 0, 2], dtype=np.int64)
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

# Create CSR matrix (3 rows, 3 cols)
csr = CSRF64.from_coo(3, 3, row_indices, col_indices, data)
```

## CSR vs CSC: Choosing the Right Format

| Operation | CSR (Row-major) | CSC (Column-major) |
|-----------|-----------------|-------------------|
| Row iteration | ✅ Fast (contiguous) | ❌ Slow (scattered) |
| Column iteration | ❌ Slow (scattered) | ✅ Fast (contiguous) |
| Row slicing | ✅ O(1) | ❌ O(nnz) |
| Column slicing | ❌ O(nnz) | ✅ O(1) |
| Gene × Cell matrix (by gene) | ✅ Use CSR | |
| Gene × Cell matrix (by cell) | | ✅ Use CSC |

### Converting Between Formats

```python
from biosparse import CSRF64

# CSR to CSC (transpose operation)
csr = CSRF64.from_scipy(scipy_mat)
csc = csr.T()  # Returns CSC

# CSC back to CSR
csr_again = csc.T()  # Returns CSR

# Explicit conversion (same orientation)
csc_same = csr.to_csc()  # CSR → CSC without transpose
csr_same = csc.to_csr()  # CSC → CSR without transpose
```

## Using Sparse Matrices in Numba

### Basic JIT Function

```python
from numba import njit
from biosparse import CSRF64
import biosparse._numba  # Required: registers Numba types

@njit
def get_matrix_info(csr):
    """Access basic properties in JIT code."""
    nrows = csr.nrows
    ncols = csr.ncols
    nnz = csr.nnz
    return nrows, ncols, nnz

# Usage
csr = CSRF64.from_scipy(scipy_mat)
info = get_matrix_info(csr)  # Works!
```

### Two Ways to Access Row/Column Data

biosparse provides two patterns for accessing row (CSR) or column (CSC) data:

#### Pattern 1: Iterator (Simple, but not parallelizable)

```python
from numba import njit

@njit
def row_sums_iterator(csr):
    """Sum each row using iterator."""
    n_rows = csr.nrows
    result = np.empty(n_rows, dtype=np.float64)
    
    row_idx = 0
    for values, indices in csr:  # Iterate over rows
        total = 0.0
        for j in range(len(values)):
            total += values[j]
        result[row_idx] = total
        row_idx += 1
    
    return result
```

**Pros**: Clean, Pythonic syntax  
**Cons**: Cannot use with `prange` for parallelization

#### Pattern 2: Indexed Access (Parallelizable)

```python
from numba import njit, prange
from biosparse.optim import parallel_jit, assume

@parallel_jit
def row_sums_parallel(csr):
    """Sum each row using indexed access + prange."""
    n_rows = csr.nrows
    assume(n_rows > 0)
    
    result = np.empty(n_rows, dtype=np.float64)
    
    for row in prange(n_rows):  # Parallel!
        values, indices = csr.row_to_numpy(row)
        total = 0.0
        for j in range(len(values)):
            total += values[j]
        result[row] = total
    
    return result
```

**Pros**: Can use `prange` for multi-threaded execution  
**Cons**: Slightly more verbose

### ⚠️ Critical: Iterator vs prange

**You CANNOT use iterators inside prange loops!**

```python
# ❌ WRONG: This will not parallelize correctly
@parallel_jit
def wrong_approach(csr):
    for values, indices in csr:  # Iterator cannot be parallelized
        # ... this runs sequentially
        pass

# ✅ CORRECT: Use indexed access with prange
@parallel_jit  
def correct_approach(csr):
    for row in prange(csr.nrows):  # This parallelizes
        values, indices = csr.row_to_numpy(row)
        # ... this runs in parallel
```

## Common Operations in JIT

### Element Access

```python
@njit
def get_element(csr, row, col):
    """Get single element (binary search)."""
    values, indices = csr.row_to_numpy(row)
    
    # Binary search for column
    pos = np.searchsorted(indices, col)
    if pos < len(indices) and indices[pos] == col:
        return values[pos]
    else:
        return 0.0  # Sparse position
```

### Row/Column Statistics

```python
@parallel_jit
def row_statistics(csr):
    """Compute mean and variance per row."""
    n_rows = csr.nrows
    n_cols = csr.ncols
    assume(n_rows > 0)
    assume(n_cols > 0)
    
    N = float(n_cols)
    means = np.empty(n_rows, dtype=np.float64)
    variances = np.empty(n_rows, dtype=np.float64)
    
    for row in prange(n_rows):
        values, _ = csr.row_to_numpy(row)
        
        # Sum (including implicit zeros)
        row_sum = 0.0
        row_sq_sum = 0.0
        for j in range(len(values)):
            v = values[j]
            row_sum += v
            row_sq_sum += v * v
        
        mean = row_sum / N
        var = (row_sq_sum / N) - mean * mean
        
        means[row] = mean
        variances[row] = max(0.0, var)  # Numerical stability
    
    return means, variances
```

### Grouped Aggregation

```python
@parallel_jit
def grouped_row_sums(csr, group_ids, n_groups):
    """Sum values by group for each row."""
    n_rows = csr.nrows
    assume(n_rows > 0)
    assume(n_groups > 0)
    
    result = np.zeros((n_rows, n_groups), dtype=np.float64)
    
    for row in prange(n_rows):
        values, col_indices = csr.row_to_numpy(row)
        
        for j in range(len(values)):
            col = col_indices[j]
            g = group_ids[col]
            if g >= 0 and g < n_groups:
                result[row, g] += values[j]
    
    return result
```

## ⚠️ Critical Pitfall: Race Conditions with prange

When using `prange`, you must avoid **race conditions** - multiple threads writing to the same memory location.

### ❌ WRONG: Shared Array Accumulation

```python
@parallel_jit
def count_groups_WRONG(group_ids, n_groups):
    """DON'T DO THIS - Race condition!"""
    n = len(group_ids)
    counts = np.zeros(n_groups, dtype=np.int64)
    
    for i in prange(n):  # Multiple threads...
        g = group_ids[i]
        counts[g] += 1    # ...write to same location! RACE!
    
    return counts
```

**What happens**: Thread A and Thread B both read `counts[0] = 5`, both compute `5 + 1 = 6`, both write `6`. Result: `counts[0] = 6` instead of `7`.

### ✅ CORRECT: Sequential for Shared Writes

```python
from biosparse.optim import fast_jit

@fast_jit  # Not parallel_jit!
def count_groups_CORRECT(group_ids, n_groups):
    """Sequential loop - no race condition."""
    n = len(group_ids)
    counts = np.zeros(n_groups, dtype=np.int64)
    
    for i in range(n):  # Sequential, not prange
        g = group_ids[i]
        if g >= 0 and g < n_groups:
            counts[g] += 1
    
    return counts
```

### ✅ CORRECT: Parallel with Per-Row Output

```python
@parallel_jit
def row_group_sums(csr, group_ids, n_groups):
    """Each row writes to its own output slice - safe!"""
    n_rows = csr.nrows
    result = np.zeros((n_rows, n_groups), dtype=np.float64)
    
    for row in prange(n_rows):  # Each row is independent
        values, indices = csr.row_to_numpy(row)
        for j in range(len(values)):
            g = group_ids[indices[j]]
            result[row, g] += values[j]  # Safe: each row has own slice
    
    return result
```

## Slicing Operations

biosparse supports efficient slicing both in Python and JIT:

### Python Slicing

```python
csr = CSRF64.from_scipy(scipy_mat)

# Row slicing
subset = csr[100:200]           # Rows 100-199
subset = csr.slice_rows(100, 200)  # Same thing

# Column slicing  
subset = csr[:, 50:100]         # Columns 50-99
subset = csr.slice_cols(50, 100)   # Same thing

# Both
subset = csr[100:200, 50:100]   # Rows 100-199, Cols 50-99

# Single element
val = csr[10, 20]  # Returns float (0.0 if sparse)
```

### JIT Slicing

```python
@njit
def process_subset(csr, row_start, row_end):
    """Slice and process in JIT."""
    # Slice rows (returns new CSR)
    subset = csr.slice_rows(row_start, row_end)
    
    # Now process subset
    total = 0.0
    for values, _ in subset:
        for v in values:
            total += v
    
    return total
```

## Conversion Operations

### In Python

```python
# To dense NumPy array
dense = csr.to_dense()  # Row-major by default
dense_f = csr.to_dense(order='F')  # Column-major

# To SciPy
scipy_mat = csr.to_scipy()

# To COO format
rows, cols, data = csr.to_coo()

# Clone (deep copy)
csr_copy = csr.clone()
```

### Format Conversion in JIT

```python
@njit
def transpose_and_process(csr):
    """Convert CSR to CSC and iterate columns."""
    csc = csr.T()  # CSR → CSC (transpose)
    
    n_cols = csc.ncols
    col_sums = np.empty(n_cols, dtype=np.float64)
    
    col_idx = 0
    for values, indices in csc:  # Iterate columns
        col_sums[col_idx] = np.sum(values)
        col_idx += 1
    
    return col_sums
```

## Stacking Operations

```python
# Vertical stack (concatenate rows)
combined = CSRF64.vstack([csr1, csr2, csr3])

# Horizontal stack (concatenate columns)
combined = CSRF64.hstack([csr1, csr2, csr3])
```

## Summary: Best Practices

| Scenario | Recommended Approach |
|----------|---------------------|
| Simple row iteration | `for values, indices in csr:` |
| Parallel row processing | `for row in prange(n): csr.row_to_numpy(row)` |
| Shared accumulation | Use `range()` not `prange()` |
| Per-row output | Safe with `prange()` |
| Column-wise operations | Convert to CSC first: `csc = csr.T()` |
| Large matrix, few operations | Use `copy=False` for zero-copy |
| Long-lived matrix | Use `copy=True` for independence |

---

**Next Chapter**: [Chapter 2: LLVM Intrinsics](./2_intrinsics.md)
