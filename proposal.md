**Title:**
**[RFC] Refactoring Scanpy's Numba Backend: C-Style Kernel Layer & Biology-Optimized Sparse Matrices**

---

## 1. Motivation: Current State of Scanpy's Numba Backend

This RFC proposes a **gradual refactoring** of Scanpy's Numba backend to replace the current ad-hoc, poorly optimized JIT implementations with a structured, high-performance kernel layer.

### The Problem: Chaotic Numba Usage

The current Numba usage in Scanpy is **disorganized and inefficient**. A prime example is [`scanpy/preprocessing/_highly_variable_genes.py`](https://github.com/scverse/scanpy/blob/main/src/scanpy/preprocessing/_highly_variable_genes.py):

```python
# Current Scanpy pattern - problematic
@njit(cache=True, parallel=False)  # ← Parallelism DISABLED
def _sum_and_sum_squares_clipped(indices, data, n_cols, clip_val, nnz):
    squared_batch_counts_sum = np.zeros(n_cols, dtype=np.float64)
    batch_counts_sum = np.zeros(n_cols, dtype=np.float64)
    for i in range(nnz):
        val = data[i]
        if val > clip_val[indices[i]]:
            val = clip_val[indices[i]]
        squared_batch_counts_sum[indices[i]] += val ** 2  # ← Race condition!
        batch_counts_sum[indices[i]] += val               # ← Race condition!
    return squared_batch_counts_sum, batch_counts_sum
```

**Critical Issues:**

| Problem | Description |
|---------|-------------|
| **Race Conditions** | Parallelism disabled (`parallel=False`) because race conditions on shared arrays were not properly handled via reduction or atomics |
| **No Optimization Hints** | Missing `assume()`, `vectorize()`, `likely()` → LLVM cannot optimize aggressively |
| **Pythonic Style** | High-level constructs prevent SIMD vectorization and loop unrolling |
| **Tight Coupling** | Business logic (string methods, flavor dispatch) mixed into JIT functions → Object mode fallbacks |
| **Unverified Compilation** | No CI checks for optimal parallel patterns or Python C-API callbacks |

This pattern repeats across Scanpy's codebase, leaving **1-2 orders of magnitude of performance** on the table.

---

## 2. Proposal: A Decoupled Kernel Architecture

### 2.1 The Kernel Layer

We propose establishing a **dedicated kernel layer** containing pure computational kernels written in **C-style Python**. All kernels must satisfy the following requirements (enforced via CI):

#### Requirement 1: Optimal Parallelization (CI-Verified)

Use `numba.parallel_diagnostics()` to verify kernels achieve **optimal parallel patterns**:

```python
# CI check example
from numba import njit, prange

@njit(parallel=True)
def kernel(...):
    for i in prange(n):  # Must show as "PARALLEL" in diagnostics
        ...

# Verify in CI:
# kernel.parallel_diagnostics(level=4) should show no "NOT PARALLEL" warnings
```

#### Requirement 2: No Python Callbacks (CI-Verified)

Use `inspect_llvm()` to verify **zero Python C-API calls** in compiled code:

```python
# CI check: No calls to scipy.loess, statsmodels, or any interpreted code
llvm_ir = kernel.inspect_llvm()
assert 'NRT_' not in llvm_ir  # No Numba runtime calls for Python objects
assert 'PyObject' not in llvm_ir  # No Python object manipulation
```

#### Requirement 3: LLVM Optimization Hints

Use compiler hints to enable optimizations that **exceed hand-written C/C++**:

```python
from biosparse.optim import parallel_jit, assume, likely, vectorize, unroll

@parallel_jit(boundscheck=False)
def optimized_kernel(csr, n_targets):
    n_rows = csr.nrows
    
    # Tell LLVM what we know → eliminates defensive code
    assume(n_rows > 0)
    assume(n_targets > 0)
    
    for row in prange(n_rows):
        values, indices = csr.row_to_numpy(row)
        
        total = 0.0
        vectorize(8)  # SIMD hint
        unroll(4)     # Loop unrolling hint
        for j in range(len(values)):
            if likely(values[j] > 0):  # Branch prediction
                total += values[j]
```

Reference implementation: [`biosparse/optim`](https://github.com/krkawzq/biosparse/tree/main/src/biosparse/optim)

#### Requirement 4: Strict Decoupling

Kernels must be **pure computation** with no business logic:

```python
# ✅ GOOD: Pure kernel - accepts only primitive data structures
@parallel_jit
def mwu_core(csr, group_ids, group_counts, n_targets, out_U, out_tie):
    ...

# ❌ BAD: Mixed with business logic
@njit
def compute_hvg(adata, flavor='seurat', n_top_genes=2000):  # String params!
    if flavor == 'seurat':  # Business logic in kernel!
        ...
```

**Kernel Inputs:** Only NumPy arrays, SciPy CSR/CSC components, or biosparse structures.  
**No:** Strings, AnnData objects, flavor switches, parameter validation.

### 2.2 The Frontend Layer

Refactor `scanpy.tools` and `scanpy.preprocessing` as **pure dispatchers**:

```
┌─────────────────────────────────────────────────────────────────┐
│  FRONTEND: scanpy.preprocessing / scanpy.tools                  │
│  ─────────────────────────────────────────────────────────────  │
│  • Parse arguments & validate parameters                        │
│  • Handle flavor selection (seurat, cell_ranger, etc.)          │
│  • AnnData I/O (read adata.X, write adata.var)                  │
│  • Convert data structures for kernel consumption               │
│  • NO mathematical computation                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ dispatch
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  BACKEND: kernel layer (new module)                              │
│  ─────────────────────────────────────────────────────────────  │
│  • Pure C-style Numba JIT kernels                               │
│  • Verified parallel patterns (CI)                               │
│  • Verified no Python callbacks (CI)                             │
│  • LLVM optimization hints (assume, vectorize, etc.)            │
│  • Only primitive data structures                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Sparse Matrix: Biology-First Data Structure

### The Problem with SciPy Sparse

SciPy's CSR/CSC matrices are designed for **general linear algebra**, not biological data analysis. Key limitations:

| Operation | SciPy Behavior | Biology Need |
|-----------|---------------|--------------|
| Slicing `X[1000:2000, :]` | **Copies data** | Zero-cost view |
| Stacking `vstack([A, B])` | **Allocates new arrays** | Efficient concatenation |
| Numba integration | Requires unpacking to arrays | Native JIT support |
| Memory layout | General-purpose | Optimized for cell×gene access |

### Recommendation: biosparse

[**biosparse**](https://github.com/krkawzq/biosparse) provides sparse matrices designed for biological data:

```python
from biosparse import CSRF64

# Zero-copy from scipy
csr = CSRF64.from_scipy(scipy_mat, copy=False)

# Zero-cost slicing (views, not copies)
subset = csr[1000:2000, :]       # No data copy!
genes_subset = csr[:, gene_mask] # No data copy!

# Efficient stacking for dataset merging
merged = CSRF64.vstack([dataset1, dataset2, dataset3])

# Native Numba integration
@parallel_jit
def my_kernel(csr: CSR):
    for row in prange(csr.nrows):
        values, indices = csr.row_to_numpy(row)  # Direct access
        ...
```

**Designed for million-scale cell datasets** where SciPy's copy-on-slice becomes a critical bottleneck.

---

## 4. Case Study: biosparse

I have implemented [**biosparse**](https://github.com/krkawzq/biosparse) as a proof-of-concept demonstrating this architecture. It is a case study project implementing selected kernels to validate the proposed approach.

### 4.1 Implemented Kernels

Currently implemented (for case study purposes):

| Kernel | Description | Location |
|--------|-------------|----------|
| **HVG** | Seurat, Seurat V3, Cell Ranger, Pearson residuals | [`kernel/hvg.py`](https://github.com/krkawzq/biosparse/tree/main/src/biosparse/kernel/hvg.py) |
| **MWU** | Mann-Whitney U test | [`kernel/mwu.py`](https://github.com/krkawzq/biosparse/tree/main/src/biosparse/kernel/mwu.py) |
| **t-test** | Welch's and Student's t-test | [`kernel/ttest.py`](https://github.com/krkawzq/biosparse/tree/main/src/biosparse/kernel/ttest.py) |

All kernels: [`biosparse/kernel`](https://github.com/krkawzq/biosparse/tree/main/src/biosparse/kernel)

### 4.2 Performance Results

**biosparse achieves 20-100x speedup** over Scanpy's current implementations:

| Kernel | Speedup vs Scanpy |
|--------|-------------------|
| HVG (Seurat) | **30x** |
| HVG (Seurat V3) | **30x** |
| HVG (Pearson) | **80x** |
| Mann-Whitney U | **25x** |

Benchmarks: [`biosparse/benchmarks`](https://github.com/krkawzq/biosparse/tree/main/benchmarks)

### 4.3 Implementation Style

All kernels are written in **pure Python** but follow **C-style conventions**:

```python
@parallel_jit(cache=True, boundscheck=False)
def _mwu_core(
    csr: CSR,
    group_ids: np.ndarray,
    group_counts: np.ndarray,
    n_targets: int,
    out_U1: np.ndarray,
    out_tie: np.ndarray,
    out_sum_ref: np.ndarray,
    out_sum_tar: np.ndarray,
    # Thread-local pre-allocated buffers (indexed by get_thread_id())
    tl_buf_ref: np.ndarray,      # (n_threads, max_nnz + 1)
    tl_buf_tar: np.ndarray,      # (n_threads, n_targets, max_nnz + 1)
    tl_n_tar_nz: np.ndarray,     # (n_threads, n_targets)
    tl_sum_tar: np.ndarray,      # (n_threads, n_targets)
) -> None:
    """Core MWU - C-style optimized kernel with thread-local buffers.
    
    Thread-local buffers eliminate heap allocation overhead in prange loop.
    Buffers are indexed by get_thread_id() for zero-contention access.
    """
    n_rows = csr.nrows
    n_ref = group_counts[0]
    
    # Precompute constants (avoid per-iteration computation)
    n_ref_f = float(n_ref)
    half_n1_n1p1 = 0.5 * n_ref_f * (n_ref_f + 1.0)
    
    # Compiler hints
    assume(n_rows > 0)
    assume(n_targets > 0)
    assume(n_ref > 0)
    
    for row in prange(n_rows):
        # Get thread-local buffer via thread ID (ZERO heap allocation!)
        tid = get_thread_id()
        buf_ref = tl_buf_ref[tid]
        buf_tar = tl_buf_tar[tid]
        n_tar_nz = tl_n_tar_nz[tid]
        sum_tar = tl_sum_tar[tid]
        
        values, col_indices = csr.row_to_numpy(row)
        nnz = len(values)
        
        n_ref_nz = 0
        sum_ref = 0.0
        
        # Reset thread-local counters (no allocation, just zero-init)
        vectorize(4)
        unroll(4)
        for t in range(n_targets):
            n_tar_nz[t] = 0
            sum_tar[t] = 0.0
        
        # Single-pass scan with branch prediction
        for j in range(nnz):
            col = col_indices[j]
            val = float(values[j])  # Explicit cast
            g = group_ids[col]
            
            if g == 0:
                buf_ref[n_ref_nz] = val
                sum_ref += val
                n_ref_nz += 1
            elif likely(g > 0):
                if likely(g <= n_targets):
                    t = g - 1
                    buf_tar[t, n_tar_nz[t]] = val
                    sum_tar[t] += val
                    n_tar_nz[t] += 1
        
        # ... (sorting and rank computation)
```

**C-Style Optimization Techniques Used:**
- **Thread-local buffers** - pre-allocate once, index by `get_thread_id()` to eliminate heap allocation in prange
- All constants inlined / precomputed
- `assume()` for bounds elimination
- `likely()`/`unlikely()` for branch prediction
- `vectorize()`/`unroll()` for SIMD and ILP
- `np.empty()` + manual init (faster than `np.zeros` in prange)
- Explicit type casts to avoid implicit conversions
- Insertion sort for small arrays, numpy sort for large
- Binary search instead of linear scan
- Single-pass algorithms where possible

---

## 5. Technical Challenge: Dynamic Loop Hints

### The Problem

biosparse's `optim` module is currently a **lightweight toolkit**. Numba's support for loop hints is limited, particularly around dynamic vectorization width dispatch.

Current approach uses hardcoded hints:

```python
vectorize(8)  # Hardcoded 8-wide SIMD
for i in range(n):
    ...
```

This can confuse LLVM's middle-end during type-based dispatch. For example, an `f64` kernel entering a loop marked `vectorize(8)` assumes 512-bit vectors (AVX-512), but:
- `f32` data could use 16 lanes at the same vector width
- Not all CPUs support AVX-512
- LLVM's cost model might prefer different widths

### Seeking Input

Since biosparse intentionally avoids C/C++ extensions, we need a **pure Numba/Python solution** for:

1. **Compile-time type inspection** to determine optimal vector width
2. **Runtime CPU feature detection** for SIMD capability dispatch
3. **Better integration** with LLVM's auto-vectorization cost model

**We welcome suggestions from maintainers on solving this within the Numba ecosystem.**

Optimization toolkit: [`biosparse/optim`](https://github.com/krkawzq/biosparse/tree/main/src/biosparse/optim)

---

## 6. Conclusion

biosparse demonstrates that **Scanpy's current Numba usage has 1-2 orders of magnitude optimization headroom**. The existing codebase suffers from:

- Race conditions handled by disabling parallelism
- No compiler optimization hints
- Tight coupling between business logic and computation
- Suboptimal data structures for biological workloads

### Proposed Actions

1. **Establish a kernel layer** with CI-enforced quality standards
2. **Refactor frontend modules** to pure dispatchers
3. **Adopt biology-optimized sparse matrices** (e.g., biosparse)
4. **Incrementally migrate kernels** to C-style implementations

### Resources

| Resource | Link |
|----------|------|
| biosparse repository | https://github.com/krkawzq/biosparse |
| Kernel implementations | [`biosparse/kernel`](https://github.com/krkawzq/biosparse/tree/main/src/biosparse/kernel) |
| Optimization toolkit | [`biosparse/optim`](https://github.com/krkawzq/biosparse/tree/main/src/biosparse/optim) |
| Benchmarks | [`biosparse/benchmarks`](https://github.com/krkawzq/biosparse/tree/main/benchmarks) |

---

*biosparse is a case study project I developed to demonstrate this architecture. I'm happy to discuss implementation details, contribute to refactoring efforts, or provide additional benchmarks.*
