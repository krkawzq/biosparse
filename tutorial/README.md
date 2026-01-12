# biosparse Optimization Tutorial

A comprehensive guide to writing high-performance numerical code with biosparse and Numba.

## Why This Tutorial?

Modern compilers are **conservative** - they generate defensive code because they don't know what you know about your data. This tutorial teaches you how to **communicate your knowledge** to the compiler using LLVM intrinsics and loop hints, achieving performance that can **exceed hand-optimized C++**.

## Tutorial Structure

| Chapter | Title | Description |
|---------|-------|-------------|
| [0](./0_philosophy.md) | **The Philosophy** | Why Numba + hints can outperform C++ |
| [1](./1_data_structures.md) | **Data Structures** | Using CSR/CSC sparse matrices in Numba |
| [2](./2_intrinsics.md) | **LLVM Intrinsics** | `assume`, `likely`, `unlikely`, `prefetch` |
| [3](./3_loop_hints.md) | **Loop Hints** | `vectorize`, `unroll`, `interleave`, `distribute` |
| [4](./4_case_study.md) | **Case Study** | Optimizing a real kernel step-by-step |
| [5](./5_best_practices.md) | **Best Practices** | Common pitfalls and how to avoid them |
| [6](./6_debugging.md) | **Debugging** | Verification and performance analysis |
| [7](./7_sparse_features.md) | **Sparse Features** | Complete biosparse sparse matrix API |

## Quick Start

```python
from biosparse.optim import parallel_jit, assume, likely, vectorize
from biosparse import CSRF64
import numpy as np
from numba import prange

@parallel_jit
def optimized_row_sums(csr):
    """Compute row sums with full optimization."""
    n_rows = csr.nrows
    
    # Tell compiler what we know
    assume(n_rows > 0)
    
    result = np.empty(n_rows, dtype=np.float64)
    
    for row in prange(n_rows):
        values, _ = csr.row_to_numpy(row)
        
        total = 0.0
        vectorize(8)  # SIMD hint
        for v in values:
            total += v
        
        result[row] = total
    
    return result

# Usage
import scipy.sparse as sp
scipy_mat = sp.random(10000, 5000, density=0.01, format='csr')
csr = CSRF64.from_scipy(scipy_mat)

sums = optimized_row_sums(csr)
```

## Key Concepts

### The Optimization Stack

```
┌─────────────────────────────────────────┐
│ 1. Algorithm Choice                     │  ← Most impact
│    (sparse vs dense, O(n) vs O(n²))     │
├─────────────────────────────────────────┤
│ 2. Parallelization                      │
│    (@parallel_jit + prange)             │
├─────────────────────────────────────────┤
│ 3. Compiler Assumptions                 │
│    (assume, likely, unlikely)           │
├─────────────────────────────────────────┤
│ 4. Loop Hints                           │  ← Least impact
│    (vectorize, unroll, interleave)      │
└─────────────────────────────────────────┘
```

### Available Tools

**JIT Decorators:**
- `@optimized_jit` - Base optimized JIT
- `@fast_jit` - With `fastmath=True`
- `@parallel_jit` - With `parallel=True, fastmath=True`

**LLVM Intrinsics:**
- `assume(condition)` - Assert condition is true
- `likely(condition)` - Branch usually taken
- `unlikely(condition)` - Branch rarely taken
- `prefetch_read(ptr, locality)` - Prefetch for reading
- `assume_aligned(ptr, alignment)` - Assert pointer alignment

**Loop Hints:**
- `vectorize(width)` - SIMD vectorization
- `unroll(count)` - Loop unrolling
- `interleave(count)` - Multiple accumulators
- `distribute()` - Loop fission

## Prerequisites

- Python 3.8+
- NumPy
- Numba
- biosparse

```bash
pip install biosparse
```

## License

MIT License - See repository root for details.
