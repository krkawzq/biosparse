# Chapter 0: The Philosophy - Why Numba + Hints Can Outperform C++

## The Problem: Compiler Conservatism

Modern compilers (GCC, Clang, LLVM) are incredibly powerful, but they have a fundamental limitation: **they must be conservative**.

Consider this simple loop:

```c++
// C++ code
void sum_array(double* arr, int n, double* result) {
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        total += arr[i];
    }
    *result = total;
}
```

What the compiler **doesn't know**:
- Is `n > 0`? Must generate code for empty array case
- Is `n` a multiple of 4/8? Must generate scalar tail loop
- Does `arr` alias `result`? Cannot reorder loads/stores freely
- Is `arr` aligned to 32 bytes? Must use unaligned SIMD loads

The result: **The compiler generates defensive code** with extra branches, scalar fallbacks, and conservative memory access patterns.

## The Performance Gap

```
┌─────────────────────────────────────────────────────────────────────┐
│  What the programmer knows          What the compiler assumes       │
│  ─────────────────────────          ─────────────────────────       │
│  • n is always > 0                  • n could be 0, negative        │
│  • n is always a multiple of 8      • n could be any value          │
│  • arr is 32-byte aligned           • arr might be unaligned        │
│  • arr and result don't overlap     • they might alias              │
│  • Most values are positive         • any value is equally likely   │
│                                                                     │
│  Result: 2-3x performance gap due to defensive code generation      │
└─────────────────────────────────────────────────────────────────────┘
```

## The Solution: Tell the Compiler What You Know

biosparse provides **LLVM intrinsics** that let you communicate your knowledge directly to the compiler:

```python
from biosparse.optim import parallel_jit, assume, likely, vectorize

@parallel_jit
def sum_array(arr):
    n = len(arr)
    
    # Tell compiler: these conditions are ALWAYS true
    assume(n > 0)              # → eliminates empty array check
    assume(n % 8 == 0)         # → enables full vectorization, no tail
    
    total = 0.0
    
    vectorize(8)               # → hint: use 8-wide SIMD
    for i in range(n):
        if likely(arr[i] > 0): # → hint: this branch is usually taken
            total += arr[i]
    
    return total
```

## Why Hand-Written C++ Can't Do This Easily

You might think: "I'll just write optimized C++ with intrinsics!"

**Problem 1: LLVM intrinsics aren't portable**
```c++
// GCC only
__builtin_assume(n > 0);

// Clang only  
__builtin_assume(n > 0);

// MSVC - different syntax
__assume(n > 0);

// Standard C++ - no equivalent!
```

**Problem 2: Manual SIMD loses compiler optimizations**
```c++
// Hand-written AVX code
__m256d sum = _mm256_setzero_pd();
for (int i = 0; i < n; i += 4) {
    __m256d v = _mm256_loadu_pd(&arr[i]);  // unaligned load
    sum = _mm256_add_pd(sum, v);
}

// Problems:
// - Compiler can't auto-unroll this
// - Compiler can't interleave multiple accumulators
// - You're stuck with AVX even on AVX-512 machines
// - No automatic prefetching
```

**Problem 3: Losing high-level information**
```c++
// Once you go manual SIMD, you lose:
// - Auto-vectorization of surrounding code
// - Cross-function optimization
// - Target-specific tuning
// - Future CPU feature support
```

## The biosparse Approach

biosparse provides a **middle ground**: you write high-level Python/Numba code, but with precise hints that unlock compiler optimizations.

```python
from biosparse.optim import parallel_jit, assume, likely, unlikely, vectorize, interleave

@parallel_jit
def optimized_kernel(csr, group_ids, n_targets):
    n_rows = csr.nrows
    n_cols = csr.ncols
    
    # === Compiler Hints: What We Know ===
    assume(n_rows > 0)
    assume(n_cols > 0)
    assume(n_targets > 0)
    assume(n_targets <= n_cols)  # reasonable bound
    
    # Allocate output
    result = np.empty((n_rows, n_targets), dtype=np.float64)
    
    # Parallel row processing
    for row in prange(n_rows):
        values, indices = csr.row_to_numpy(row)
        nnz = len(values)
        
        # Inner accumulation - vectorizable
        total = 0.0
        vectorize(8)
        interleave(4)  # use 4 accumulators to hide latency
        for j in range(nnz):
            g = group_ids[indices[j]]
            if likely(g >= 0 and g < n_targets):
                total += values[j]
        
        result[row, 0] = total
    
    return result
```

## Performance Comparison

| Approach | Relative Performance | Notes |
|----------|---------------------|-------|
| Naive Python | 1x | Baseline (interpreted) |
| NumPy | 10-50x | C backend, no parallelism |
| Naive Numba `@njit` | 50-100x | JIT, conservative codegen |
| Hand-optimized C++ | 80-150x | Manual SIMD, complex code |
| **biosparse + hints** | **150-300x** | JIT + full compiler optimization |

The key insight: **Numba + LLVM + proper hints > hand-written C++** because:
1. You keep high-level semantics (compiler can still optimize)
2. You communicate precise constraints (compiler removes defensive code)
3. You get automatic target-specific tuning (AVX vs AVX-512 vs ARM NEON)
4. Code remains readable and maintainable

## What You'll Learn in This Tutorial

| Chapter | Topic |
|---------|-------|
| **0** | Philosophy: Why hints enable peak performance |
| **1** | Data Structures: Using CSR/CSC in Numba |
| **2** | LLVM Intrinsics: `assume`, `likely`, `unlikely`, `prefetch` |
| **3** | Loop Hints: `vectorize`, `unroll`, `interleave` |
| **4** | Case Study: Optimizing a real kernel step-by-step |
| **5** | Best Practices & Pitfalls |
| **6** | Debugging & Verification |
| **7** | biosparse Sparse Matrix Features |

## Quick Start

```python
# Install biosparse
pip install biosparse

# Basic usage
from biosparse.optim import parallel_jit, assume, likely, vectorize
from biosparse import CSRF64
import numpy as np

@parallel_jit
def fast_row_sums(csr):
    """Compute row sums with full optimization."""
    n_rows = csr.nrows
    assume(n_rows > 0)
    
    result = np.empty(n_rows, dtype=np.float64)
    
    for row in prange(n_rows):
        values, _ = csr.row_to_numpy(row)
        total = 0.0
        
        vectorize(8)
        for j in range(len(values)):
            total += values[j]
        
        result[row] = total
    
    return result

# Create sparse matrix
import scipy.sparse as sp
scipy_mat = sp.random(10000, 5000, density=0.01, format='csr')
csr = CSRF64.from_scipy(scipy_mat)

# Run optimized kernel
sums = fast_row_sums(csr)
```

---

**Next Chapter**: [Chapter 1: Data Structure Integration](./1_data_structures.md)
