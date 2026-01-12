# Optimization Toolkit Module

The `biosparse.optim` module provides low-level optimization tools for Numba JIT-compiled functions, including LLVM intrinsics and loop optimization hints.

## Overview

This module includes:

- **LLVM Intrinsics**: Low-level compiler hints (`assume`, `likely`, `unlikely`, etc.)
- **Loop Hints**: Vectorization, unrolling, and other loop optimizations
- **Enhanced JIT Decorators**: `@optimized_jit`, `@fast_jit`, `@parallel_jit`
- **IR Processing**: LLVM IR post-processing for loop metadata

## Quick Start

```python
from biosparse.optim import optimized_jit, assume, likely, vectorize
import numpy as np

@optimized_jit(fastmath=True)
def fast_sum(arr):
    n = len(arr)
    assume(n > 0)
    assume(n % 8 == 0)  # Help vectorizer
    
    vectorize(8)
    total = 0.0
    for i in range(n):
        if likely(arr[i] > 0):
            total += arr[i]
    return total

# Call the function
arr = np.random.randn(1024)
result = fast_sum(arr)
```

---

## LLVM Intrinsics

### `assume(condition)`

Tell the compiler that a condition is always true, enabling aggressive optimizations.

```python
from numba import njit
from biosparse.optim import assume

@njit
def safe_divide(a, b):
    assume(b != 0)  # Promise: b is never zero
    return a / b

@njit
def process_array(arr):
    n = len(arr)
    assume(n > 0)           # Array is not empty
    assume(n % 4 == 0)      # Length is multiple of 4 (helps vectorizer)
    assume(n <= 1000000)    # Bounded size
    
    total = 0.0
    for i in range(n):
        total += arr[i]
    return total
```

**Parameters:**
- `condition`: Boolean expression assumed to be true

**Warning:** If the assumption is violated at runtime, behavior is undefined!

---

### `likely(condition)` / `unlikely(condition)`

Hint that a branch is likely or unlikely to be taken, improving branch prediction.

```python
from numba import njit
from biosparse.optim import likely, unlikely

@njit
def process_with_branch(arr):
    total = 0.0
    for x in arr:
        if likely(x > 0):
            # This path is optimized for (common case)
            total += x
        else:
            # Rare case
            total -= x
    return total

@njit
def validate_input(x):
    if unlikely(x < 0):
        # Error path - rarely executed
        return -1
    # Normal path
    return x * 2
```

**Use Cases:**
- `likely`: Common case, success path, valid input
- `unlikely`: Error handling, edge cases, rare conditions

---

### `unreachable()`

Mark code as unreachable, enabling dead code elimination.

```python
from numba import njit
from biosparse.optim import unreachable

@njit
def safe_access(arr, idx):
    # Caller guarantees valid index
    if idx < 0 or idx >= len(arr):
        unreachable()
    return arr[idx]

@njit
def switch_on_mode(mode, x):
    if mode == 0:
        return x + 1
    elif mode == 1:
        return x * 2
    elif mode == 2:
        return x ** 2
    else:
        unreachable()  # Caller guarantees mode is 0, 1, or 2
```

**Warning:** If the unreachable point is reached, the program will crash!

---

### `prefetch_read(ptr, locality)` / `prefetch_write(ptr, locality)`

Hint to prefetch memory into cache.

```python
from numba import njit
from biosparse.optim import prefetch_read
import numpy as np

@njit
def sum_with_prefetch(arr):
    n = len(arr)
    total = 0.0
    
    for i in range(n):
        # Prefetch 16 elements ahead
        if i + 16 < n:
            prefetch_read(arr[i + 16:].ctypes.data, 3)
        total += arr[i]
    
    return total
```

**Parameters:**
- `ptr`: Pointer to memory (use `arr.ctypes.data` or `arr[i:].ctypes.data`)
- `locality`: Temporal locality hint
  - `0`: No temporal locality (won't be reused)
  - `1`: Low temporal locality
  - `2`: Medium temporal locality
  - `3`: High temporal locality (will be reused soon)

---

### `assume_aligned(ptr, align)`

Assume a pointer is aligned to a specific boundary.

```python
from numba import njit
from biosparse.optim import assume_aligned

@njit
def process_aligned(arr):
    ptr = arr.ctypes.data
    aligned_ptr = assume_aligned(ptr, 32)  # 32-byte alignment for AVX
    # Now LLVM can use aligned vector instructions
    ...
```

**Parameters:**
- `ptr`: Pointer value
- `align`: Alignment in bytes (must be power of 2)

---

### `invariant_start(size, ptr)` / `invariant_end(token, size, ptr)`

Mark a memory region as immutable for a scope.

```python
from numba import njit
from biosparse.optim import invariant_start, invariant_end

@njit
def read_only_access(arr):
    size = len(arr) * 8  # 8 bytes per float64
    ptr = arr.ctypes.data
    
    # Mark as read-only
    token = invariant_start(size, ptr)
    
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]  # LLVM knows arr won't change
    
    invariant_end(token, size, ptr)
    return total
```

**Warning:** Writing to invariant memory is undefined behavior!

---

## Loop Hints

Loop hints require using `@optimized_jit` instead of `@njit`.

### `vectorize(width)`

Hint that the next loop should be vectorized with a specific width.

```python
from biosparse.optim import optimized_jit, vectorize

@optimized_jit
def sum_vectorized(arr):
    n = len(arr)
    total = 0.0
    
    vectorize(8)  # Vectorize with width 8 (AVX-256 for float32)
    for i in range(n):
        total += arr[i]
    
    return total
```

**Common Widths:**

| Width | Register | Data Type |
|-------|----------|-----------|
| 4 | SSE/AVX | float64 |
| 8 | AVX/AVX2 | float32 |
| 8 | AVX-512 | float64 |
| 16 | AVX-512 | float32 |

---

### `unroll(count)`

Hint that the next loop should be unrolled.

```python
from biosparse.optim import optimized_jit, unroll

@optimized_jit
def small_loop():
    result = 0.0
    
    unroll(4)  # Unroll 4 iterations
    for i in range(4):
        result += process(i)
    
    return result

@optimized_jit
def full_unroll():
    unroll(0)  # 0 means full unroll
    for i in range(8):
        process(i)
```

**Parameters:**
- `count`: Number of iterations to unroll (0 = full unroll)

---

### `interleave(count)`

Hint loop interleaving count for better instruction-level parallelism.

```python
from biosparse.optim import optimized_jit, interleave

@optimized_jit
def sum_interleaved(arr):
    n = len(arr)
    total = 0.0
    
    interleave(4)  # Use 4 accumulators
    for i in range(n):
        total += arr[i]
    
    return total
```

**Use Case:** Reduction loops (sum, product) to hide latency.

---

### `distribute()`

Hint that the loop should be split into separate loops.

```python
from biosparse.optim import optimized_jit, distribute

@optimized_jit
def process_arrays(a, b, c):
    distribute()  # Split into two loops
    for i in range(len(a)):
        b[i] = a[i] * 2    # First operation
        c[i] = a[i] + 1    # Independent operation
```

**Use Case:** When loop body contains independent operations.

---

### `pipeline(stages)`

Hint software pipelining for the loop.

```python
from biosparse.optim import optimized_jit, pipeline

@optimized_jit
def pipelined_process(arr):
    pipeline(3)  # 3-stage pipeline
    for i in range(len(arr)):
        arr[i] = arr[i] * 2 + 1
```

---

## JIT Decorators

### `@optimized_jit`

Enhanced JIT decorator with loop hint processing.

```python
from biosparse.optim import optimized_jit

@optimized_jit
def my_func(arr):
    ...

# With options
@optimized_jit(fastmath=True, parallel=True)
def fast_func(arr):
    ...

# All options
@optimized_jit(
    process_hints=True,   # Process loop hints (default: True)
    verbose=False,        # Debug logging (default: False)
    recompile=False,      # Experimental recompilation (default: False)
    nogil=True,           # Release GIL (default: True)
    cache=False,          # Cache to disk (default: False)
    parallel=False,       # Auto-parallelization (default: False)
    fastmath=False,       # Fast math (default: False)
    boundscheck=False,    # Bounds checking (default: False)
)
def full_options(arr):
    ...
```

---

### `@fast_jit`

Shorthand for `@optimized_jit(fastmath=True)`.

```python
from biosparse.optim import fast_jit

@fast_jit
def fast_sum(arr):
    total = 0.0
    for x in arr:
        total += x
    return total
```

---

### `@parallel_jit`

Shorthand for `@optimized_jit(parallel=True, fastmath=True)`.

```python
from biosparse.optim import parallel_jit
from numba import prange

@parallel_jit
def parallel_sum(arr):
    n = len(arr)
    total = 0.0
    
    for i in prange(n):  # Parallel loop
        total += arr[i]
    
    return total
```

---

## Debugging Utilities

### `inspect_hints(func)`

Print loop hints found in a compiled function.

```python
from biosparse.optim import optimized_jit, vectorize, inspect_hints

@optimized_jit
def my_func(arr):
    vectorize(8)
    for i in range(len(arr)):
        arr[i] *= 2

# Compile first
my_func(np.array([1.0, 2.0, 3.0]))

# Inspect hints
inspect_hints(my_func)
# Output:
# Signature: (float64[::1],)
# ----------------------------------------
#   vectorize(8) at line 5
```

---

### `get_modified_ir(func, signature=None)`

Get the modified LLVM IR with loop metadata.

```python
from biosparse.optim import optimized_jit, get_modified_ir

@optimized_jit
def my_func(arr):
    vectorize(8)
    for i in range(len(arr)):
        arr[i] *= 2

my_func(np.array([1.0, 2.0]))
ir = get_modified_ir(my_func)
print(ir)  # LLVM IR with loop metadata
```

---

### Logging Control

```python
from biosparse.optim import set_log_level, enable_debug, disable_logging
import logging

# Set log level
set_log_level(logging.DEBUG)

# Enable debug mode
enable_debug()

# Disable all logging
disable_logging()
```

---

## Complete Examples

### Example 1: Optimized Vector Operations

```python
import numpy as np
from biosparse.optim import optimized_jit, assume, vectorize

@optimized_jit(fastmath=True)
def dot_product(a, b):
    """Optimized dot product."""
    n = len(a)
    assume(n > 0)
    assume(len(b) >= n)
    
    result = 0.0
    
    vectorize(8)
    for i in range(n):
        result += a[i] * b[i]
    
    return result

@optimized_jit(fastmath=True)
def saxpy(a, x, y, result):
    """SAXPY: result = a * x + y"""
    n = len(x)
    assume(n > 0)
    assume(len(y) >= n)
    assume(len(result) >= n)
    
    vectorize(8)
    for i in range(n):
        result[i] = a * x[i] + y[i]

# Usage
a = np.random.randn(1024)
b = np.random.randn(1024)
dot = dot_product(a, b)

result = np.empty(1024)
saxpy(2.0, a, b, result)
```

### Example 2: Sparse Matrix Row Sum

```python
import numpy as np
from biosparse.optim import parallel_jit, assume, vectorize
from biosparse import CSRF64

@parallel_jit
def sparse_row_sums(csr):
    """Compute sum of each row."""
    n_rows = csr.nrows
    assume(n_rows > 0)
    
    result = np.empty(n_rows, dtype=np.float64)
    
    row_idx = 0
    for values, indices in csr:
        row_sum = 0.0
        n = len(values)
        
        assume(n >= 0)
        
        vectorize(8)
        for j in range(n):
            row_sum += values[j]
        
        result[row_idx] = row_sum
        row_idx += 1
    
    return result
```

### Example 3: Conditional Processing with Branch Hints

```python
import numpy as np
from biosparse.optim import optimized_jit, likely, unlikely, assume

@optimized_jit(fastmath=True)
def process_with_threshold(arr, threshold):
    """Process array with threshold, optimized for mostly-positive case."""
    n = len(arr)
    assume(n > 0)
    
    positive_sum = 0.0
    negative_count = 0
    
    for i in range(n):
        x = arr[i]
        
        if likely(x >= threshold):
            # Common case: above threshold
            positive_sum += x
        else:
            # Rare case: below threshold
            if unlikely(x < 0):
                negative_count += 1
    
    return positive_sum, negative_count
```

### Example 4: Memory Prefetching

```python
import numpy as np
from biosparse.optim import optimized_jit, prefetch_read, assume

@optimized_jit
def sum_with_prefetch(arr):
    """Sum with software prefetching."""
    n = len(arr)
    assume(n > 0)
    
    total = 0.0
    PREFETCH_DISTANCE = 16
    
    for i in range(n):
        # Prefetch ahead
        if i + PREFETCH_DISTANCE < n:
            prefetch_read(arr[i + PREFETCH_DISTANCE:].ctypes.data, 3)
        
        total += arr[i]
    
    return total
```

### Example 5: Combined Optimizations

```python
import numpy as np
from numba import prange
from biosparse.optim import parallel_jit, assume, vectorize, likely

@parallel_jit
def optimized_filter_sum(arr, mask):
    """Sum masked elements with all optimizations."""
    n = len(arr)
    assume(n > 0)
    assume(len(mask) >= n)
    
    # Parallel outer loop
    partial_sums = np.zeros(n, dtype=np.float64)
    
    for i in prange(n):
        if likely(mask[i]):
            partial_sums[i] = arr[i]
    
    # Final reduction
    total = 0.0
    vectorize(8)
    for i in range(n):
        total += partial_sums[i]
    
    return total
```

---

## Best Practices

### 1. Use `assume` for Bounds

```python
@optimized_jit
def good_example(arr):
    n = len(arr)
    assume(n > 0)
    assume(n <= 1000000)  # Reasonable upper bound
    
    for i in range(n):
        # Compiler knows: 0 <= i < n <= 1000000
        ...
```

### 2. Use `likely`/`unlikely` for Branches

```python
# BAD: No hints
if condition:
    common_case()
else:
    rare_case()

# GOOD: With hints
if likely(condition):
    common_case()
else:
    rare_case()
```

### 3. Align Vectorization Width with Data Type

| Data Type | SSE (128-bit) | AVX (256-bit) | AVX-512 |
|-----------|---------------|---------------|---------|
| float32 | 4 | 8 | 16 |
| float64 | 2 | 4 | 8 |

```python
# float64 array
vectorize(4)  # AVX-256

# float32 array
vectorize(8)  # AVX-256
```

### 4. Place Hints Before Loops

```python
# CORRECT: Hint immediately before loop
vectorize(8)
for i in range(n):
    ...

# WRONG: Code between hint and loop
vectorize(8)
x = 1  # Don't put code here!
for i in range(n):
    ...
```

### 5. Combine with Numba's Parallel

```python
from numba import prange
from biosparse.optim import parallel_jit, vectorize

@parallel_jit
def combined(arr):
    n = len(arr)
    
    # Outer loop: parallel
    for i in prange(n):
        inner_sum = 0.0
        
        # Inner loop: vectorized
        vectorize(8)
        for j in range(1000):
            inner_sum += arr[i] * j
```

---

## API Reference

### Intrinsics

| Function | Signature | Description |
|----------|-----------|-------------|
| `assume(cond)` | `void(bool)` | Assert condition is true |
| `likely(cond)` | `bool(bool)` | Hint branch is likely |
| `unlikely(cond)` | `bool(bool)` | Hint branch is unlikely |
| `unreachable()` | `void()` | Mark code unreachable |
| `prefetch_read(ptr, loc)` | `void(ptr, i32)` | Prefetch for read |
| `prefetch_write(ptr, loc)` | `void(ptr, i32)` | Prefetch for write |
| `assume_aligned(ptr, align)` | `ptr(ptr, i64)` | Assert alignment |
| `invariant_start(size, ptr)` | `ptr(i64, ptr)` | Start invariant region |
| `invariant_end(tok, size, ptr)` | `void(ptr, i64, ptr)` | End invariant region |

### Loop Hints

| Function | Signature | Description |
|----------|-----------|-------------|
| `vectorize(width)` | `void(intp)` | Hint vectorization width |
| `unroll(count)` | `void(intp)` | Hint unroll count (0=full) |
| `interleave(count)` | `void(intp)` | Hint interleave count |
| `distribute()` | `void()` | Enable loop distribution |
| `pipeline(stages)` | `void(intp)` | Hint software pipelining |

### Decorators

| Decorator | Equivalent |
|-----------|------------|
| `@optimized_jit` | `@njit` + hint processing |
| `@fast_jit` | `@optimized_jit(fastmath=True)` |
| `@parallel_jit` | `@optimized_jit(parallel=True, fastmath=True)` |

### Utilities

| Function | Description |
|----------|-------------|
| `inspect_hints(func)` | Print loop hints |
| `get_modified_ir(func)` | Get modified LLVM IR |
| `set_log_level(level)` | Set logging level |
| `enable_debug()` | Enable debug logging |
| `disable_logging()` | Disable all logging |
