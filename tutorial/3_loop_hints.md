# Chapter 3: Loop Hints - Controlling Loop Optimizations

## Why Loop Hints?

Loops are where programs spend most of their time. Modern compilers can optimize loops through:
- **Vectorization**: Using SIMD instructions (process 4-16 elements at once)
- **Unrolling**: Reducing loop overhead by processing multiple iterations
- **Interleaving**: Using multiple accumulators to hide latency
- **Distribution**: Splitting loops into simpler parts

However, compilers are **conservative** - they won't apply aggressive optimizations unless they're certain it's safe. Loop hints let you tell the compiler: **"Yes, this optimization is safe and beneficial."**

## How Loop Hints Work in biosparse

Loop hints work through a two-stage process:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1: Compilation                                                │
│                                                                     │
│   @optimized_jit                                                    │
│   def my_func():                                                    │
│       vectorize(8)  ← Inserts marker into LLVM IR                   │
│       for i in range(n):                                            │
│           ...                                                       │
│                                                                     │
│   Marker: __BIOSPARSE_LOOP_VECTORIZE_8__                            │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 2: IR Processing                                              │
│                                                                     │
│   IRProcessor scans for markers                                     │
│       ↓                                                             │
│   Finds next loop after marker                                      │
│       ↓                                                             │
│   Adds LLVM loop metadata:                                          │
│       !llvm.loop.vectorize.width = 8                                │
│       ↓                                                             │
│   LLVM optimizer respects the metadata                              │
└─────────────────────────────────────────────────────────────────────┘
```

**Important**: Loop hints require `@optimized_jit`, `@fast_jit`, or `@parallel_jit`. They don't work with plain `@njit`.

## `vectorize(width)` - SIMD Vectorization

### What It Does

Tells the compiler to vectorize the next loop with a specific SIMD width.

### Choosing the Right Width

| CPU Feature | Register Size | float32 Width | float64 Width |
|-------------|--------------|---------------|---------------|
| SSE | 128-bit | 4 | 2 |
| AVX/AVX2 | 256-bit | 8 | 4 |
| AVX-512 | 512-bit | 16 | 8 |

```python
from biosparse.optim import parallel_jit, vectorize, assume

@parallel_jit
def dot_product_f64(a, b):
    """Dot product for float64 arrays."""
    n = len(a)
    assume(n > 0)
    assume(len(b) >= n)
    
    result = 0.0
    
    vectorize(4)  # AVX: 4 x float64 = 256 bits
    for i in range(n):
        result += a[i] * b[i]
    
    return result

@parallel_jit
def dot_product_f32(a, b):
    """Dot product for float32 arrays."""
    n = len(a)
    assume(n > 0)
    assume(len(b) >= n)
    
    result = 0.0
    
    vectorize(8)  # AVX: 8 x float32 = 256 bits
    for i in range(n):
        result += a[i] * b[i]
    
    return result
```

### Best Practice: Ensure Alignment with `assume`

For best vectorization results, tell the compiler the loop count is a multiple of the vector width:

```python
@parallel_jit
def sum_array_optimal(arr):
    n = len(arr)
    assume(n > 0)
    assume(n % 8 == 0)  # No remainder loop needed!
    
    total = 0.0
    
    vectorize(8)
    for i in range(n):
        total += arr[i]
    
    return total
```

Without `assume(n % 8 == 0)`, the compiler must generate:
1. A vectorized main loop
2. A scalar "tail" loop for the remaining 0-7 elements

### Vectorization in Nested Loops

```python
@parallel_jit
def matrix_vector_multiply(A, x, y):
    """y = A @ x, where A is (m, n)."""
    m, n = A.shape
    assume(m > 0)
    assume(n > 0)
    assume(n % 8 == 0)
    
    for i in prange(m):  # Outer: parallel
        total = 0.0
        
        vectorize(8)  # Inner: vectorized
        for j in range(n):
            total += A[i, j] * x[j]
        
        y[i] = total
```

## `unroll(count)` - Loop Unrolling

### What It Does

Loop unrolling reduces the overhead of loop control (increment, compare, branch) by processing multiple iterations in one pass.

### Usage

```python
from biosparse.optim import parallel_jit, unroll

@parallel_jit
def unrolled_sum(arr):
    n = len(arr)
    total = 0.0
    
    unroll(4)  # Process 4 iterations per loop pass
    for i in range(n):
        total += arr[i]
    
    return total

# Compiler generates something like:
# for i in range(0, n, 4):
#     total += arr[i]
#     total += arr[i+1]
#     total += arr[i+2]
#     total += arr[i+3]
```

### Full Unrolling

Use `unroll(0)` to fully unroll a loop (all iterations become straight-line code):

```python
@parallel_jit
def process_fixed_size():
    """Process exactly 8 elements - fully unroll."""
    result = 0.0
    
    unroll(0)  # Full unroll
    for i in range(8):
        result += compute(i)
    
    return result

# Compiler generates:
# result += compute(0)
# result += compute(1)
# result += compute(2)
# ... (no loop at all)
```

### When to Use Unrolling

| Scenario | Recommendation |
|----------|---------------|
| Small, fixed-size loop | `unroll(0)` - full unroll |
| Medium loop (10-100 iterations) | `unroll(4)` or `unroll(8)` |
| Large loop | Usually not needed (vectorize instead) |
| Loop with heavy body | Lower unroll count (avoid code bloat) |

## `interleave(count)` - Multiple Accumulators

### The Problem: Data Dependencies

Consider this sum:

```python
total = 0.0
for i in range(n):
    total += arr[i]  # Each iteration depends on previous!
```

This has a **loop-carried dependency**: each `total += arr[i]` must wait for the previous addition to complete. On modern CPUs, addition takes ~4 cycles, so you can only do 1 addition every 4 cycles.

### The Solution: Multiple Accumulators

```python
from biosparse.optim import parallel_jit, interleave

@parallel_jit
def sum_interleaved(arr):
    n = len(arr)
    total = 0.0
    
    interleave(4)  # Use 4 independent accumulators
    for i in range(n):
        total += arr[i]
    
    return total
```

The compiler transforms this to:

```python
# Conceptually:
total0 = total1 = total2 = total3 = 0.0
for i in range(0, n, 4):
    total0 += arr[i]
    total1 += arr[i+1]  # Independent of total0!
    total2 += arr[i+2]  # Independent of total0, total1!
    total3 += arr[i+3]  # Independent!
total = total0 + total1 + total2 + total3
```

Now 4 additions happen in parallel, achieving ~4x throughput.

### Combining vectorize and interleave

For maximum performance, combine both:

```python
@parallel_jit
def sum_maximum_throughput(arr):
    n = len(arr)
    assume(n > 0)
    assume(n % 32 == 0)  # 8 (vector) x 4 (interleave) = 32
    
    total = 0.0
    
    vectorize(8)    # 8-wide SIMD
    interleave(4)   # 4 accumulators
    for i in range(n):
        total += arr[i]
    
    return total

# Effective: 8 * 4 = 32 elements processed per "logical iteration"
```

### When to Use Interleaving

| Scenario | Recommendation |
|----------|---------------|
| Reduction (sum, product, max) | `interleave(4)` |
| Dot product | `interleave(4)` |
| No loop-carried dependency | Not needed |
| Limited registers | Lower count |

## `distribute()` - Loop Fission

### What It Does

Splits a loop with multiple independent operations into separate loops.

### Why It Helps

```python
# Before distribute:
for i in range(n):
    b[i] = a[i] * 2      # Operation 1
    c[i] = a[i] + 1      # Operation 2 (independent!)

# After distribute:
for i in range(n):
    b[i] = a[i] * 2      # Better cache locality for b

for i in range(n):
    c[i] = a[i] + 1      # Better cache locality for c
```

### Usage

```python
from biosparse.optim import parallel_jit, distribute

@parallel_jit
def independent_operations(a, b, c):
    n = len(a)
    
    distribute()  # Split into separate loops
    for i in range(n):
        b[i] = a[i] * 2
        c[i] = a[i] + 1
```

### When to Use Distribution

| Scenario | Recommendation |
|----------|---------------|
| Multiple independent array writes | Use `distribute()` |
| Operations share intermediate values | Don't distribute |
| Large arrays, memory-bound | Distribution helps cache |
| Small arrays | Overhead may hurt |

## `pipeline(stages)` - Software Pipelining

### What It Does

Hints the compiler to overlap iterations of a loop, similar to CPU instruction pipelining.

### Usage

```python
from biosparse.optim import parallel_jit, pipeline

@parallel_jit
def pipelined_transform(arr):
    n = len(arr)
    
    pipeline(3)  # 3-stage pipeline
    for i in range(n):
        # Stage 1: Load
        val = arr[i]
        # Stage 2: Compute
        result = val * 2 + 1
        # Stage 3: Store
        arr[i] = result
```

### When to Use Pipelining

- Long loops with regular memory access
- Operations with multiple stages (load, compute, store)
- Less common than vectorize/unroll/interleave

## Combining Loop Hints: A Complete Example

Here's a real-world optimized kernel using multiple hints:

```python
from biosparse.optim import (
    parallel_jit, assume, likely, unlikely,
    vectorize, interleave
)
import numpy as np
from numba import prange

@parallel_jit
def compute_row_statistics(csr):
    """
    Compute mean and variance for each row of a sparse matrix.
    
    Demonstrates:
    - prange for outer parallelism
    - vectorize for inner SIMD
    - interleave for reduction throughput
    - assume for enabling optimizations
    - likely/unlikely for branch prediction
    """
    n_rows = csr.nrows
    n_cols = csr.ncols
    N = float(n_cols)
    
    # === Assumptions ===
    assume(n_rows > 0)
    assume(n_cols > 0)
    assume(N > 0.0)
    
    # Output arrays
    means = np.empty(n_rows, dtype=np.float64)
    variances = np.empty(n_rows, dtype=np.float64)
    
    # === Parallel outer loop ===
    for row in prange(n_rows):
        values, _ = csr.row_to_numpy(row)
        n_nnz = len(values)
        
        assume(n_nnz >= 0)
        
        # === Vectorized, interleaved reduction ===
        row_sum = 0.0
        row_sq_sum = 0.0
        
        vectorize(8)
        interleave(4)
        for j in range(n_nnz):
            v = values[j]
            row_sum += v
            row_sq_sum += v * v
        
        # Compute statistics
        mean = row_sum / N
        
        # Variance: E[X^2] - E[X]^2
        # Note: This accounts for implicit zeros
        var = (row_sq_sum / N) - mean * mean
        
        # Numerical stability: variance can't be negative
        if unlikely(var < 0.0):
            var = 0.0
        
        means[row] = mean
        variances[row] = var
    
    return means, variances
```

## Placement Rules for Loop Hints

### Rule 1: Place Hint Immediately Before Loop

```python
# ✅ CORRECT
vectorize(8)
for i in range(n):
    ...

# ❌ WRONG - code between hint and loop
vectorize(8)
x = 1  # Don't put code here!
y = 2
for i in range(n):
    ...
```

### Rule 2: One Hint Per Loop

```python
# ✅ CORRECT - multiple hints for same loop
vectorize(8)
interleave(4)
for i in range(n):
    ...

# ❌ WRONG - hint for loop that doesn't follow
vectorize(8)
for i in range(10):
    ...
vectorize(4)  # This hint has no effect!
for j in range(20):
    ...
```

### Rule 3: Only the Next Loop is Affected

```python
vectorize(8)  # Only affects the immediately following loop
for i in range(n):
    total += arr[i]

# This loop is NOT vectorized by the above hint
for j in range(m):
    process(j)
```

## Summary: Loop Hint Selection Guide

| Goal | Hint | Typical Value |
|------|------|---------------|
| SIMD processing | `vectorize(width)` | 4 (f64), 8 (f32) |
| Reduce loop overhead | `unroll(count)` | 4-8 |
| Hide latency in reductions | `interleave(count)` | 4 |
| Separate independent ops | `distribute()` | - |
| Pipeline stages | `pipeline(stages)` | 2-4 |

### Recommended Combinations

| Loop Type | Recommended Hints |
|-----------|------------------|
| Simple sum/reduction | `vectorize(8)` + `interleave(4)` |
| Dot product | `vectorize(8)` + `interleave(4)` |
| Element-wise transform | `vectorize(8)` |
| Small fixed-size loop | `unroll(0)` |
| Multi-output loop | `distribute()` + `vectorize(8)` |

---

**Next Chapter**: [Chapter 4: Case Study - Optimizing a Real Kernel](./4_case_study.md)
