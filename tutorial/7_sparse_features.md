# Chapter 7: biosparse Sparse Matrix Features

## Overview

biosparse provides high-performance sparse matrix types designed specifically for scientific computing workloads like single-cell RNA sequencing analysis.

## Matrix Types

### CSR (Compressed Sparse Row)

Stores data row-by-row. Optimal for:
- Row-wise operations (gene statistics)
- Row slicing
- Row iteration

```python
from biosparse import CSRF32, CSRF64

# Float64 CSR (most common)
csr = CSRF64.from_scipy(scipy_mat)

# Float32 CSR (memory efficient)
csr_f32 = CSRF32.from_scipy(scipy_mat.astype(np.float32))
```

### CSC (Compressed Sparse Column)

Stores data column-by-column. Optimal for:
- Column-wise operations (cell statistics)
- Column slicing
- Column iteration

```python
from biosparse import CSCF32, CSCF64

# Create CSC from scipy
csc = CSCF64.from_scipy(scipy_csc)

# Or convert from CSR
csc = csr.T()  # Transpose: CSR → CSC
```

## Creating Matrices

### From SciPy

```python
import scipy.sparse as sp
from biosparse import CSRF64

# With copy (default) - independent memory
csr = CSRF64.from_scipy(scipy_mat, copy=True)

# Without copy - shares memory, faster but scipy_mat must stay alive
csr_view = CSRF64.from_scipy(scipy_mat, copy=False)
```

### From COO Format

```python
from biosparse import CSRF64
import numpy as np

# COO arrays
rows = np.array([0, 0, 1, 2, 2], dtype=np.int64)
cols = np.array([0, 2, 1, 0, 2], dtype=np.int64)
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

# Create 3x3 matrix
csr = CSRF64.from_coo(3, 3, rows, cols, data)
```

## Basic Properties

```python
csr = CSRF64.from_scipy(scipy_mat)

# Dimensions
print(f"Shape: {csr.shape}")      # (nrows, ncols)
print(f"Rows: {csr.nrows}")       # Number of rows
print(f"Columns: {csr.ncols}")    # Number of columns
print(f"Non-zeros: {csr.nnz}")    # Number of stored elements

# Sparsity metrics
print(f"Density: {csr.density}")   # nnz / (nrows * ncols)
print(f"Sparsity: {csr.sparsity}") # 1 - density

# Validation
print(f"Valid: {csr.is_valid}")    # Structure is valid
print(f"Sorted: {csr.is_sorted}")  # Indices sorted within rows
print(f"In bounds: {csr.indices_in_bounds}")  # All indices valid
```

## Accessing Data

### Row Access (CSR)

```python
# Get row as (values, indices) tuple
values, indices = csr.row_to_numpy(row_idx)

# Or using the .row() method
values, indices = csr.row(row_idx)

# Number of non-zeros in a row
nnz_in_row = csr.row_len(row_idx)

# Raw pointers (for advanced use)
values_ptr = csr.row_values_ptr(row_idx)
indices_ptr = csr.row_indices_ptr(row_idx)
```

### Column Access (CSC)

```python
# Get column as (values, indices) tuple
values, indices = csc.col_to_numpy(col_idx)

# Or using the .col() method
values, indices = csc.col(col_idx)

# Number of non-zeros in a column
nnz_in_col = csc.col_len(col_idx)
```

### Element Access

```python
# Single element (uses binary search)
val = csr[row, col]  # Returns 0.0 for sparse positions

# Safe access with default
val = csr.get(row, col, default=0.0)

# Out-of-bounds returns default (no exception)
val = csr.get(999999, 0, default=-1.0)  # Returns -1.0
```

## Slicing

### Row Slicing (CSR - O(1))

```python
# Slice rows [100, 200)
subset = csr[100:200]
subset = csr.slice_rows(100, 200)

# First 100 rows
first_100 = csr[:100]

# Last 100 rows
last_100 = csr[-100:]  # Negative indexing works
```

### Column Slicing (CSR - O(nnz))

```python
# Slice columns [50, 150)
subset = csr[:, 50:150]
subset = csr.slice_cols(50, 150)
```

### Combined Slicing

```python
# Rows [100, 200), Columns [50, 150)
subset = csr[100:200, 50:150]
```

### Mask-Based Slicing

```python
# Select rows by boolean mask
row_mask = np.array([True, False, True, ...])  # Length == nrows
subset = csr.slice_rows_mask(row_mask)

# Select columns by boolean mask
col_mask = np.array([True, True, False, ...])  # Length == ncols
subset = csr.slice_cols_mask(col_mask)
```

## Stacking Operations

### Vertical Stack (Concatenate Rows)

```python
# Stack multiple CSR matrices vertically
combined = CSRF64.vstack([csr1, csr2, csr3])

# Result: nrows = csr1.nrows + csr2.nrows + csr3.nrows
#         ncols = same for all (must match)
```

### Horizontal Stack (Concatenate Columns)

```python
# Stack multiple CSR matrices horizontally
combined = CSRF64.hstack([csr1, csr2, csr3])

# Result: nrows = same for all (must match)
#         ncols = csr1.ncols + csr2.ncols + csr3.ncols
```

## Format Conversion

### CSR ↔ CSC

```python
# CSR to CSC (transpose)
csc = csr.T()

# CSC to CSR (transpose)
csr = csc.T()

# Without transpose (just format change)
csc = csr.to_csc()  # Same data, different format
csr = csc.to_csr()  # Same data, different format
```

### To Dense

```python
# Row-major (C order)
dense = csr.to_dense()

# Column-major (Fortran order)
dense_f = csr.to_dense(order='F')
```

### To SciPy

```python
# Convert back to scipy.sparse
scipy_csr = csr.to_scipy()
```

### To COO

```python
# Get COO arrays
rows, cols, data = csr.to_coo()
```

## Cloning

```python
# Deep copy - completely independent
csr_copy = csr.clone()

# Modify copy without affecting original
# csr_copy.modify(...)  # Original unchanged
```

## Iteration

### In Python

```python
# Iterate over rows (CSR)
for row_idx in range(csr.nrows):
    values, indices = csr.row_to_numpy(row_idx)
    print(f"Row {row_idx}: {len(values)} non-zeros")

# Iterate over columns (CSC)
for col_idx in range(csc.ncols):
    values, indices = csc.col_to_numpy(col_idx)
    print(f"Column {col_idx}: {len(values)} non-zeros")
```

### In Numba (Iterator Pattern)

```python
from numba import njit
import biosparse._numba  # Register types

@njit
def iterate_rows(csr):
    """Iterate using iterator pattern."""
    row_sums = []
    for values, indices in csr:  # Yields (values, indices) per row
        row_sums.append(np.sum(values))
    return row_sums
```

### In Numba (Indexed Pattern - Parallelizable)

```python
from numba import njit, prange
from biosparse.optim import parallel_jit

@parallel_jit
def iterate_rows_parallel(csr):
    """Iterate using indexed access - can use prange."""
    n_rows = csr.nrows
    row_sums = np.empty(n_rows, dtype=np.float64)
    
    for row in prange(n_rows):  # Parallel!
        values, indices = csr.row_to_numpy(row)
        row_sums[row] = np.sum(values)
    
    return row_sums
```

## Using in JIT Functions

### Complete Example

```python
from biosparse import CSRF64
from biosparse.optim import parallel_jit, assume, likely, vectorize
import numpy as np
from numba import prange

@parallel_jit
def compute_gene_statistics(csr, min_cells=3):
    """
    Compute statistics for each gene (row).
    
    Args:
        csr: Gene × Cell expression matrix
        min_cells: Minimum cells with expression
    
    Returns:
        means: Mean expression per gene
        variances: Variance per gene
        n_expressing: Number of expressing cells per gene
    """
    n_genes = csr.nrows
    n_cells = csr.ncols
    N = float(n_cells)
    
    # Assumptions
    assume(n_genes > 0)
    assume(n_cells > 0)
    assume(min_cells >= 0)
    
    # Output arrays
    means = np.empty(n_genes, dtype=np.float64)
    variances = np.empty(n_genes, dtype=np.float64)
    n_expressing = np.empty(n_genes, dtype=np.int64)
    
    for gene in prange(n_genes):
        values, _ = csr.row_to_numpy(gene)
        n_nnz = len(values)
        
        # Count expressing cells
        n_expressing[gene] = n_nnz
        
        # Skip genes with too few expressing cells
        if n_nnz < min_cells:
            means[gene] = 0.0
            variances[gene] = 0.0
            continue
        
        # Compute mean (including zeros)
        total = 0.0
        sq_total = 0.0
        
        vectorize(8)
        for j in range(n_nnz):
            v = values[j]
            total += v
            sq_total += v * v
        
        mean = total / N
        var = (sq_total / N) - mean * mean
        
        # Numerical stability
        if var < 0.0:
            var = 0.0
        
        means[gene] = mean
        variances[gene] = var
    
    return means, variances, n_expressing


# Usage
import scipy.sparse as sp

# Create test data
np.random.seed(42)
scipy_mat = sp.random(20000, 50000, density=0.05, format='csr')
csr = CSRF64.from_scipy(scipy_mat)

# Run optimized computation
means, variances, n_expr = compute_gene_statistics(csr, min_cells=10)
print(f"Computed statistics for {len(means)} genes")
```

## Performance Characteristics

| Operation | CSR | CSC | Notes |
|-----------|-----|-----|-------|
| `row_to_numpy(i)` | O(1) | O(nnz) | CSR is row-contiguous |
| `col_to_numpy(j)` | O(nnz) | O(1) | CSC is column-contiguous |
| `slice_rows(a, b)` | O(1) | O(nnz) | CSR row slicing is cheap |
| `slice_cols(a, b)` | O(nnz) | O(1) | CSC column slicing is cheap |
| `[i, j]` element | O(log nnz_row) | O(log nnz_col) | Binary search |
| `T()` transpose | O(nnz) | O(nnz) | Creates new matrix |
| `clone()` | O(nnz) | O(nnz) | Deep copy |

## Memory Layout

```
CSR Memory Layout:
┌─────────────────────────────────────────────────────────────┐
│ Row 0: values=[1.0, 2.0]  indices=[0, 3]                    │
│ Row 1: values=[3.0]       indices=[2]                       │
│ Row 2: values=[4.0, 5.0, 6.0]  indices=[0, 1, 4]            │
└─────────────────────────────────────────────────────────────┘

CSC Memory Layout:
┌─────────────────────────────────────────────────────────────┐
│ Col 0: values=[1.0, 4.0]  indices=[0, 2]                    │
│ Col 1: values=[5.0]       indices=[2]                       │
│ Col 2: values=[3.0]       indices=[1]                       │
│ Col 3: values=[2.0]       indices=[0]                       │
│ Col 4: values=[6.0]       indices=[2]                       │
└─────────────────────────────────────────────────────────────┘
```

## Best Practices Summary

1. **Choose the right format**:
   - CSR for row operations (gene-level stats)
   - CSC for column operations (cell-level stats)

2. **Use zero-copy when safe**:
   ```python
   # Fast but requires scipy_mat to stay alive
   csr = CSRF64.from_scipy(scipy_mat, copy=False)
   ```

3. **Use indexed access for parallel code**:
   ```python
   for row in prange(n_rows):
       values, indices = csr.row_to_numpy(row)  # Not iterator!
   ```

4. **Transpose for cross-cutting operations**:
   ```python
   csc = csr.T()  # Now column access is O(1)
   for col in prange(csc.ncols):
       values, indices = csc.col_to_numpy(col)
   ```

5. **Validate inputs in production**:
   ```python
   assert csr.is_valid, "Invalid matrix structure"
   assert csr.is_sorted, "Indices not sorted"
   ```

---

## Conclusion

This tutorial has covered the complete biosparse optimization framework:

| Chapter | Key Takeaway |
|---------|-------------|
| 0 | Compiler hints can outperform hand-optimized C++ |
| 1 | CSR/CSC integrate seamlessly with Numba |
| 2 | `assume`, `likely`, `unlikely` remove defensive code |
| 3 | `vectorize`, `interleave` control loop optimization |
| 4 | Layer optimizations: algorithm → parallel → hints |
| 5 | Avoid race conditions, document assumptions |
| 6 | Verify correctness, benchmark performance |
| 7 | Choose right format, use indexed access for prange |

**Happy optimizing!** 🚀
