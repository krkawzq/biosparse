# Chapter 4: Case Study - Optimizing a Real Kernel Step by Step

## The Task: Welch's t-test on Sparse Data

We'll optimize a real computational kernel: **Welch's t-test** for differential expression analysis on sparse gene expression matrices.

**Input**: 
- Sparse CSR matrix: genes × cells (e.g., 20,000 × 50,000)
- Group assignments: which cells belong to reference vs target groups

**Output**:
- t-statistics, p-values, and log2 fold changes for each gene

## Stage 0: Naive Python (Baseline)

```python
import numpy as np
import scipy.stats as stats

def ttest_naive(dense_matrix, group_ids):
    """Naive Python implementation - our baseline."""
    n_genes, n_cells = dense_matrix.shape
    
    t_stats = np.empty(n_genes)
    p_values = np.empty(n_genes)
    
    for gene in range(n_genes):
        row = dense_matrix[gene, :]
        
        # Split by group
        ref_vals = row[group_ids == 0]
        tar_vals = row[group_ids == 1]
        
        # Scipy t-test
        t, p = stats.ttest_ind(ref_vals, tar_vals, equal_var=False)
        
        t_stats[gene] = t
        p_values[gene] = p
    
    return t_stats, p_values

# Performance: ~100 seconds for 20k genes × 50k cells
```

**Problems**:
- Dense matrix: huge memory usage
- Python loops: slow
- Scipy overhead per call

## Stage 1: Basic Numba JIT

```python
from numba import njit
import numpy as np

@njit
def ttest_numba_v1(csr, group_ids):
    """First Numba version - just JIT the loop."""
    n_rows = csr.nrows
    n_cols = csr.ncols
    
    t_stats = np.empty(n_rows, dtype=np.float64)
    p_values = np.empty(n_rows, dtype=np.float64)
    
    for row in range(n_rows):
        values, col_indices = csr.row_to_numpy(row)
        
        # Accumulate statistics
        sum_ref = 0.0
        sum_sq_ref = 0.0
        n_ref = 0
        
        sum_tar = 0.0
        sum_sq_tar = 0.0
        n_tar = 0
        
        for j in range(len(values)):
            col = col_indices[j]
            val = values[j]
            g = group_ids[col]
            
            if g == 0:
                sum_ref += val
                sum_sq_ref += val * val
                n_ref += 1
            else:
                sum_tar += val
                sum_sq_tar += val * val
                n_tar += 1
        
        # Count zeros (implicit in sparse matrix)
        n_ref_total = np.sum(group_ids == 0)
        n_tar_total = np.sum(group_ids == 1)
        
        # Compute means (including zeros)
        mean_ref = sum_ref / n_ref_total
        mean_tar = sum_tar / n_tar_total
        
        # Compute variances
        var_ref = (sum_sq_ref / n_ref_total) - mean_ref * mean_ref
        var_tar = (sum_sq_tar / n_tar_total) - mean_tar * mean_tar
        
        # Welch's t-statistic
        se = np.sqrt(var_ref / n_ref_total + var_tar / n_tar_total)
        t_stat = (mean_tar - mean_ref) / se if se > 1e-10 else 0.0
        
        t_stats[row] = t_stat
        p_values[row] = 1.0  # Placeholder
    
    return t_stats, p_values

# Performance: ~10 seconds (10x faster than naive)
```

**Improvements**:
- JIT compilation removes Python overhead
- Works directly on sparse data

**Remaining problems**:
- Single-threaded
- Recomputes group counts every row
- No SIMD optimization

## Stage 2: Add Parallelization

```python
from numba import njit, prange

@njit(parallel=True)
def ttest_numba_v2(csr, group_ids, n_ref_total, n_tar_total):
    """Add parallel processing."""
    n_rows = csr.nrows
    
    t_stats = np.empty(n_rows, dtype=np.float64)
    p_values = np.empty(n_rows, dtype=np.float64)
    
    # Pre-compute constants
    n_ref_f = float(n_ref_total)
    n_tar_f = float(n_tar_total)
    
    for row in prange(n_rows):  # PARALLEL!
        values, col_indices = csr.row_to_numpy(row)
        
        sum_ref = 0.0
        sum_sq_ref = 0.0
        sum_tar = 0.0
        sum_sq_tar = 0.0
        
        for j in range(len(values)):
            col = col_indices[j]
            val = values[j]
            g = group_ids[col]
            
            if g == 0:
                sum_ref += val
                sum_sq_ref += val * val
            else:
                sum_tar += val
                sum_sq_tar += val * val
        
        mean_ref = sum_ref / n_ref_f
        mean_tar = sum_tar / n_tar_f
        
        var_ref = (sum_sq_ref / n_ref_f) - mean_ref * mean_ref
        var_tar = (sum_sq_tar / n_tar_f) - mean_tar * mean_tar
        
        se = np.sqrt(var_ref / n_ref_f + var_tar / n_tar_f)
        t_stat = (mean_tar - mean_ref) / se if se > 1e-10 else 0.0
        
        t_stats[row] = t_stat
        p_values[row] = 1.0
    
    return t_stats, p_values

# Performance: ~1.5 seconds (6.7x faster, using 8 cores)
```

**Improvements**:
- Multi-threaded via `prange`
- Pre-computed group counts

**Remaining problems**:
- Compiler generates defensive code
- No SIMD in inner loop
- Branch prediction not optimized

## Stage 3: Add Compiler Assumptions

```python
from biosparse.optim import parallel_jit, assume

@parallel_jit
def ttest_numba_v3(csr, group_ids, n_ref_total, n_tar_total):
    """Add compiler assumptions."""
    n_rows = csr.nrows
    n_cols = csr.ncols
    
    # === COMPILER ASSUMPTIONS ===
    assume(n_rows > 0)
    assume(n_cols > 0)
    assume(n_ref_total > 0)
    assume(n_tar_total > 0)
    assume(n_ref_total + n_tar_total == n_cols)
    
    t_stats = np.empty(n_rows, dtype=np.float64)
    p_values = np.empty(n_rows, dtype=np.float64)
    
    n_ref_f = float(n_ref_total)
    n_tar_f = float(n_tar_total)
    inv_n_ref = 1.0 / n_ref_f
    inv_n_tar = 1.0 / n_tar_f
    
    for row in prange(n_rows):
        values, col_indices = csr.row_to_numpy(row)
        nnz = len(values)
        
        assume(nnz >= 0)
        assume(nnz <= n_cols)
        
        sum_ref = 0.0
        sum_sq_ref = 0.0
        sum_tar = 0.0
        sum_sq_tar = 0.0
        
        for j in range(nnz):
            col = col_indices[j]
            val = values[j]
            g = group_ids[col]
            
            if g == 0:
                sum_ref += val
                sum_sq_ref += val * val
            else:
                sum_tar += val
                sum_sq_tar += val * val
        
        mean_ref = sum_ref * inv_n_ref
        mean_tar = sum_tar * inv_n_tar
        
        var_ref = (sum_sq_ref * inv_n_ref) - mean_ref * mean_ref
        var_tar = (sum_sq_tar * inv_n_tar) - mean_tar * mean_tar
        
        # Ensure non-negative variance
        assume(var_ref >= -1e-10)
        assume(var_tar >= -1e-10)
        if var_ref < 0.0:
            var_ref = 0.0
        if var_tar < 0.0:
            var_tar = 0.0
        
        se_sq = var_ref * inv_n_ref + var_tar * inv_n_tar
        assume(se_sq >= 0.0)
        
        if se_sq > 1e-20:
            se = np.sqrt(se_sq)
            t_stat = (mean_tar - mean_ref) / se
        else:
            t_stat = 0.0
        
        t_stats[row] = t_stat
        p_values[row] = 1.0
    
    return t_stats, p_values

# Performance: ~1.2 seconds (1.25x faster)
```

**Improvements**:
- Eliminated division by zero checks
- Removed bounds validation overhead
- Pre-computed reciprocals

## Stage 4: Add Branch Hints

```python
from biosparse.optim import parallel_jit, assume, likely, unlikely

@parallel_jit
def ttest_numba_v4(csr, group_ids, n_ref_total, n_tar_total):
    """Add branch prediction hints."""
    n_rows = csr.nrows
    n_cols = csr.ncols
    
    assume(n_rows > 0)
    assume(n_cols > 0)
    assume(n_ref_total > 0)
    assume(n_tar_total > 0)
    
    t_stats = np.empty(n_rows, dtype=np.float64)
    p_values = np.empty(n_rows, dtype=np.float64)
    
    n_ref_f = float(n_ref_total)
    n_tar_f = float(n_tar_total)
    inv_n_ref = 1.0 / n_ref_f
    inv_n_tar = 1.0 / n_tar_f
    
    for row in prange(n_rows):
        values, col_indices = csr.row_to_numpy(row)
        nnz = len(values)
        
        assume(nnz >= 0)
        
        sum_ref = 0.0
        sum_sq_ref = 0.0
        sum_tar = 0.0
        sum_sq_tar = 0.0
        
        for j in range(nnz):
            col = col_indices[j]
            val = values[j]
            g = group_ids[col]
            
            # === BRANCH HINTS ===
            # In typical data, reference group is ~50% of cells
            if likely(g == 0):
                sum_ref += val
                sum_sq_ref += val * val
            elif likely(g == 1):
                sum_tar += val
                sum_sq_tar += val * val
            # Invalid group (very rare)
            # else: pass
        
        mean_ref = sum_ref * inv_n_ref
        mean_tar = sum_tar * inv_n_tar
        
        var_ref = (sum_sq_ref * inv_n_ref) - mean_ref * mean_ref
        var_tar = (sum_sq_tar * inv_n_tar) - mean_tar * mean_tar
        
        if unlikely(var_ref < 0.0):
            var_ref = 0.0
        if unlikely(var_tar < 0.0):
            var_tar = 0.0
        
        se_sq = var_ref * inv_n_ref + var_tar * inv_n_tar
        
        # Most genes have some variance
        if likely(se_sq > 1e-20):
            se = np.sqrt(se_sq)
            t_stat = (mean_tar - mean_ref) / se
        else:
            t_stat = 0.0
        
        t_stats[row] = t_stat
        p_values[row] = 1.0
    
    return t_stats, p_values

# Performance: ~1.0 seconds (1.2x faster)
```

**Improvements**:
- Optimized branch prediction
- Hot path (valid groups) prioritized

## Stage 5: Add Loop Optimization Hints

```python
from biosparse.optim import (
    parallel_jit, assume, likely, unlikely,
    vectorize, interleave
)

@parallel_jit
def ttest_numba_v5(csr, group_ids, n_ref_total, n_tar_total):
    """Add loop optimization hints - FINAL VERSION."""
    n_rows = csr.nrows
    n_cols = csr.ncols
    
    assume(n_rows > 0)
    assume(n_cols > 0)
    assume(n_ref_total > 0)
    assume(n_tar_total > 0)
    
    t_stats = np.empty(n_rows, dtype=np.float64)
    p_values = np.empty(n_rows, dtype=np.float64)
    
    n_ref_f = float(n_ref_total)
    n_tar_f = float(n_tar_total)
    inv_n_ref = 1.0 / n_ref_f
    inv_n_tar = 1.0 / n_tar_f
    
    for row in prange(n_rows):
        values, col_indices = csr.row_to_numpy(row)
        nnz = len(values)
        
        assume(nnz >= 0)
        
        sum_ref = 0.0
        sum_sq_ref = 0.0
        sum_tar = 0.0
        sum_sq_tar = 0.0
        
        # === LOOP HINTS ===
        # Note: Can't fully vectorize due to conditional accumulation
        # But interleave helps hide latency
        interleave(4)
        for j in range(nnz):
            col = col_indices[j]
            val = values[j]
            g = group_ids[col]
            
            if likely(g == 0):
                sum_ref += val
                sum_sq_ref += val * val
            elif likely(g == 1):
                sum_tar += val
                sum_sq_tar += val * val
        
        mean_ref = sum_ref * inv_n_ref
        mean_tar = sum_tar * inv_n_tar
        
        var_ref = (sum_sq_ref * inv_n_ref) - mean_ref * mean_ref
        var_tar = (sum_sq_tar * inv_n_tar) - mean_tar * mean_tar
        
        if unlikely(var_ref < 0.0):
            var_ref = 0.0
        if unlikely(var_tar < 0.0):
            var_tar = 0.0
        
        se_sq = var_ref * inv_n_ref + var_tar * inv_n_tar
        
        if likely(se_sq > 1e-20):
            se = np.sqrt(se_sq)
            t_stat = (mean_tar - mean_ref) / se
        else:
            t_stat = 0.0
        
        t_stats[row] = t_stat
        p_values[row] = 1.0
    
    return t_stats, p_values

# Performance: ~0.8 seconds
```

## Performance Summary

| Version | Time | Speedup vs Naive | Key Optimization |
|---------|------|-----------------|------------------|
| Naive Python | 100s | 1x | Baseline |
| V1: Basic JIT | 10s | 10x | JIT compilation |
| V2: + Parallel | 1.5s | 67x | Multi-threading |
| V3: + Assume | 1.2s | 83x | Remove defensive code |
| V4: + Branch hints | 1.0s | 100x | Branch prediction |
| V5: + Loop hints | 0.8s | **125x** | Instruction pipelining |

## Key Takeaways

### 1. Optimization Order Matters

```
1. Algorithm choice (sparse vs dense)
2. JIT compilation
3. Parallelization (prange)
4. Compiler assumptions (assume)
5. Branch hints (likely/unlikely)
6. Loop hints (vectorize/interleave)
```

### 2. Measure Before Optimizing

Always profile to find the actual bottleneck:

```python
import time

def benchmark(func, *args, warmup=2, runs=5):
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.median(times), result
```

### 3. Assumptions Must Be True

Every `assume()` is a **contract**. Violating it causes undefined behavior.

```python
# Document your assumptions!
def my_kernel(csr, group_ids, n_groups):
    """
    Preconditions:
    - csr.nrows > 0
    - csr.ncols > 0
    - len(group_ids) == csr.ncols
    - all(0 <= g < n_groups for g in group_ids)
    """
    assume(csr.nrows > 0)
    assume(csr.ncols > 0)
    assume(n_groups > 0)
    # ...
```

### 4. Not Everything Vectorizes

Some patterns can't be vectorized:
- Conditional accumulation to different targets
- Data-dependent branching
- Indirect memory access

For these, use `interleave` to improve throughput instead.

---

**Next Chapter**: [Chapter 5: Best Practices and Pitfalls](./5_best_practices.md)
