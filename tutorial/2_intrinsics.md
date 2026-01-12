# Chapter 2: LLVM Intrinsics - Telling the Compiler What You Know

## The Core Idea

LLVM intrinsics are **direct instructions to the compiler**. Unlike regular code that the compiler must analyze, intrinsics provide **guaranteed** information that enables aggressive optimizations.

```python
# Without hints: Compiler must be defensive
def sum_array(arr):
    n = len(arr)
    # Compiler thinks: "n could be 0, negative, or huge"
    # Generates: bounds check, empty array check, overflow protection
    ...

# With hints: Compiler knows the truth
def sum_array_optimized(arr):
    n = len(arr)
    assume(n > 0)      # "I guarantee n is positive"
    assume(n <= 10000) # "I guarantee n is bounded"
    # Compiler removes defensive code, enables aggressive optimization
    ...
```

## `assume(condition)` - The Most Powerful Hint

### What It Does

`assume(condition)` tells LLVM: **"This condition is ALWAYS true. Optimize accordingly."**

The compiler can then:
- Eliminate dead branches
- Remove bounds checks
- Enable loop optimizations
- Perform constant propagation

### Basic Usage

```python
from biosparse.optim import parallel_jit, assume
import numpy as np

@parallel_jit
def process_data(arr):
    n = len(arr)
    
    # Tell compiler about array properties
    assume(n > 0)           # Not empty
    assume(n % 8 == 0)      # Multiple of 8 → full vectorization
    assume(n <= 1000000)    # Bounded size → no overflow worry
    
    total = 0.0
    for i in range(n):
        total += arr[i]
    return total
```

### What Each Assume Enables

| Assumption | Compiler Optimization |
|------------|----------------------|
| `assume(n > 0)` | Removes "if n == 0 return" check |
| `assume(n % 8 == 0)` | No scalar tail loop needed |
| `assume(n <= MAX)` | Can use 32-bit loop counter |
| `assume(ptr != null)` | Removes null checks |
| `assume(a <= b)` | Enables range-based optimizations |

### Real Example: Matrix Operations

```python
@parallel_jit
def matrix_row_sums(csr, group_ids, n_groups):
    n_rows = csr.nrows
    n_cols = csr.ncols
    
    # Structural assumptions
    assume(n_rows > 0)
    assume(n_cols > 0)
    assume(n_groups > 0)
    
    # Relationship assumptions
    assume(n_groups <= n_cols)  # Can't have more groups than columns
    assume(len(group_ids) >= n_cols)  # group_ids covers all columns
    
    result = np.empty((n_rows, n_groups), dtype=np.float64)
    
    for row in prange(n_rows):
        values, indices = csr.row_to_numpy(row)
        n_nnz = len(values)
        
        assume(n_nnz >= 0)  # Non-negative count
        assume(n_nnz <= n_cols)  # Can't exceed column count
        
        # ... rest of computation
```

### ⚠️ Warning: Undefined Behavior

If an assumption is **violated at runtime**, behavior is **undefined**. The program may:
- Produce wrong results
- Crash
- Appear to work (but be silently wrong)

```python
@njit
def dangerous(arr):
    assume(len(arr) > 0)  # PROMISE: array is not empty
    return arr[0]

# If called with empty array:
dangerous(np.array([]))  # UNDEFINED BEHAVIOR!
```

**Rule**: Only use `assume` for conditions you can **guarantee**.

## `likely(condition)` / `unlikely(condition)` - Branch Prediction

### What They Do

These hints tell the compiler which branch is the **common case**:
- `likely(cond)`: The `if` branch is usually taken
- `unlikely(cond)`: The `if` branch is rarely taken

### Why It Matters

Modern CPUs use **branch prediction**. Mispredictions cost 10-20 cycles. The compiler can:
- Reorder code to put the common path first
- Use conditional moves instead of branches
- Optimize instruction cache layout

### Usage Patterns

```python
from biosparse.optim import parallel_jit, likely, unlikely

@parallel_jit
def process_with_validation(arr):
    n = len(arr)
    total = 0.0
    errors = 0
    
    for i in range(n):
        val = arr[i]
        
        # Common case: valid data
        if likely(val >= 0):
            total += val
        else:
            # Rare case: error handling
            if unlikely(val == -999):
                errors += 1
            else:
                total += abs(val)
    
    return total, errors
```

### Real-World Patterns

```python
@parallel_jit
def sparse_computation(csr, group_ids, n_groups):
    for row in prange(csr.nrows):
        values, indices = csr.row_to_numpy(row)
        
        for j in range(len(values)):
            g = group_ids[indices[j]]
            
            # Most elements belong to valid groups
            if likely(g >= 0 and g < n_groups):
                process(values[j], g)
            # Edge case: invalid group (shouldn't happen often)
            else:
                handle_invalid(j)
```

### Combining with `assume`

```python
@parallel_jit
def optimized_kernel(csr, threshold):
    for row in prange(csr.nrows):
        values, _ = csr.row_to_numpy(row)
        n = len(values)
        
        assume(n >= 0)  # Guarantee: non-negative length
        
        positive_sum = 0.0
        for j in range(n):
            v = values[j]
            # In biological data, most expression values are positive
            if likely(v > threshold):
                positive_sum += v
        
        # Error case: no positive values (rare)
        if unlikely(positive_sum == 0.0):
            handle_empty_row(row)
```

## `prefetch_read(ptr, locality)` / `prefetch_write(ptr, locality)`

### What They Do

Prefetch hints tell the CPU to **load data into cache before it's needed**, hiding memory latency.

### Locality Levels

| Level | Meaning | Use Case |
|-------|---------|----------|
| 0 | No temporal locality | Data used once, then discarded |
| 1 | Low temporal locality | Data might be reused once |
| 2 | Medium temporal locality | Data will be reused several times |
| 3 | High temporal locality | Data will be reused many times soon |

### Basic Usage

```python
from biosparse.optim import parallel_jit, prefetch_read

@parallel_jit
def sum_with_prefetch(arr):
    n = len(arr)
    total = 0.0
    
    PREFETCH_DISTANCE = 16  # Prefetch 16 elements ahead
    
    for i in range(n):
        # Prefetch future data
        if i + PREFETCH_DISTANCE < n:
            prefetch_read(arr[i + PREFETCH_DISTANCE:].ctypes.data, 3)
        
        total += arr[i]
    
    return total
```

### When Prefetching Helps

Prefetching is most effective when:
- Memory access pattern is **predictable** (sequential or strided)
- Data is **not already in cache**
- There's enough work between prefetch and use to hide latency

```python
@parallel_jit
def process_sparse_with_prefetch(csr, lookup_table):
    """Prefetch lookup table entries."""
    n_rows = csr.nrows
    result = np.empty(n_rows, dtype=np.float64)
    
    for row in prange(n_rows):
        values, indices = csr.row_to_numpy(row)
        
        total = 0.0
        for j in range(len(values)):
            idx = indices[j]
            
            # Prefetch next lookup
            if j + 4 < len(indices):
                next_idx = indices[j + 4]
                prefetch_read(lookup_table[next_idx:].ctypes.data, 2)
            
            total += values[j] * lookup_table[idx]
        
        result[row] = total
    
    return result
```

### ⚠️ When NOT to Prefetch

- Small arrays (already in cache)
- Random access patterns (unpredictable)
- Memory bandwidth-bound code (prefetch adds overhead)

## `assume_aligned(ptr, alignment)`

### What It Does

Tells the compiler that a pointer is aligned to a specific byte boundary, enabling **aligned SIMD loads**.

### Why Alignment Matters

| Access Type | Instruction | Cycles |
|-------------|-------------|--------|
| Unaligned load | `vmovdqu` | 1-3 |
| Aligned load | `vmovdqa` | 1 |
| Cross-cache-line | Multiple | 3-10 |

### Usage

```python
from biosparse.optim import parallel_jit, assume_aligned

@parallel_jit
def aligned_sum(arr):
    n = len(arr)
    ptr = arr.ctypes.data
    
    # Promise: array is 32-byte aligned (for AVX)
    assume_aligned(ptr, 32)
    
    total = 0.0
    for i in range(n):
        total += arr[i]
    
    return total
```

### Creating Aligned Arrays

```python
import numpy as np

# NumPy arrays are typically 16-byte aligned, but you can ensure 32/64:
def aligned_array(shape, dtype, alignment=32):
    """Create an array aligned to specific byte boundary."""
    size = np.prod(shape) * np.dtype(dtype).itemsize
    buf = np.empty(size + alignment, dtype=np.uint8)
    offset = alignment - (buf.ctypes.data % alignment)
    return np.frombuffer(buf[offset:offset+size].data, dtype=dtype).reshape(shape)

arr = aligned_array((1024,), np.float64, alignment=32)
```

## `unreachable()` - Dead Code Elimination

### What It Does

Marks code as **impossible to reach**. If reached, behavior is undefined.

### Usage

```python
from biosparse.optim import fast_jit, unreachable

@fast_jit
def process_mode(mode, x):
    """Process based on mode. Caller guarantees mode is 0, 1, or 2."""
    if mode == 0:
        return x + 1
    elif mode == 1:
        return x * 2
    elif mode == 2:
        return x ** 2
    else:
        unreachable()  # Should never get here
        # Compiler removes this branch entirely

@fast_jit
def safe_divide(a, b):
    """Divide a by b. Caller guarantees b != 0."""
    if b == 0:
        unreachable()
    return a / b  # No division-by-zero check generated
```

## `invariant_start` / `invariant_end` - Read-Only Memory

### What They Do

Mark a memory region as **immutable** for a scope, enabling aggressive load hoisting and CSE.

### Usage

```python
from biosparse.optim import fast_jit, invariant_start, invariant_end

@fast_jit
def process_readonly(arr):
    """Process array that won't change during this function."""
    n = len(arr)
    size = n * 8  # 8 bytes per float64
    ptr = arr.ctypes.data
    
    # Mark as read-only
    token = invariant_start(size, ptr)
    
    # LLVM knows arr won't change - can hoist loads out of loops
    total = 0.0
    for i in range(n):
        for j in range(100):
            total += arr[i]  # arr[i] can be loaded once, not 100 times
    
    invariant_end(token, size, ptr)
    return total
```

## Practical Example: Combining Intrinsics

Here's a real-world kernel showing how intrinsics work together:

```python
from biosparse.optim import parallel_jit, assume, likely, unlikely, prefetch_read
import numpy as np
from numba import prange

@parallel_jit
def optimized_grouped_stats(csr, group_ids, n_groups):
    """
    Compute mean per group for each row.
    
    Demonstrates:
    - assume: structural guarantees
    - likely/unlikely: branch optimization
    - prefetch: memory latency hiding
    """
    n_rows = csr.nrows
    n_cols = csr.ncols
    
    # === Structural assumptions ===
    assume(n_rows > 0)
    assume(n_cols > 0)
    assume(n_groups > 0)
    assume(n_groups <= 100)  # Reasonable bound
    assume(len(group_ids) >= n_cols)
    
    # Output: (n_rows, n_groups)
    sums = np.zeros((n_rows, n_groups), dtype=np.float64)
    counts = np.zeros((n_rows, n_groups), dtype=np.int64)
    
    for row in prange(n_rows):
        values, col_indices = csr.row_to_numpy(row)
        nnz = len(values)
        
        assume(nnz >= 0)
        assume(nnz <= n_cols)
        
        for j in range(nnz):
            col = col_indices[j]
            val = values[j]
            
            # Prefetch next group lookup
            if j + 8 < nnz:
                next_col = col_indices[j + 8]
                prefetch_read(group_ids[next_col:].ctypes.data, 2)
            
            g = group_ids[col]
            
            # Most columns should have valid groups
            if likely(g >= 0 and g < n_groups):
                sums[row, g] += val
                counts[row, g] += 1
            # Invalid group (data error, rare)
            elif unlikely(g < 0):
                pass  # Skip negative groups
    
    # Compute means
    result = np.empty((n_rows, n_groups), dtype=np.float64)
    for row in prange(n_rows):
        for g in range(n_groups):
            c = counts[row, g]
            if likely(c > 0):
                result[row, g] = sums[row, g] / float(c)
            else:
                result[row, g] = 0.0
    
    return result
```

## Summary: When to Use Each Intrinsic

| Intrinsic | Use When | Effect |
|-----------|----------|--------|
| `assume(cond)` | You can guarantee a condition | Eliminates branches, enables optimizations |
| `likely(cond)` | Branch is taken >70% of time | Optimizes code layout |
| `unlikely(cond)` | Branch is taken <30% of time | Moves cold code out of hot path |
| `prefetch_read` | Predictable access, large data | Hides memory latency |
| `assume_aligned` | Array is known-aligned | Enables aligned SIMD |
| `unreachable` | Code path is impossible | Eliminates dead code |
| `invariant_*` | Data is read-only in scope | Enables load hoisting |

---

**Next Chapter**: [Chapter 3: Loop Hints](./3_loop_hints.md)
