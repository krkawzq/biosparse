<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<h1 align="center">🧬 biosparse</h1>

<p align="center">
  <strong>Sparse matrices. Reimagined for biology.</strong>
</p>

<p align="center">
  1000x faster than scipy. 10-100x faster than scanpy.<br>
  Zero-cost slicing. Numba-native. Production-ready.
</p>

---

## Why biosparse?

biosparse is built on three pillars:

### 1️⃣ Biology-First Sparse Matrices

A custom sparse matrix format designed for how biologists actually work:

- **Zero-cost slicing & stacking** - Subset genes/cells without copying data
- **scipy/numpy compatible** - `from_scipy()`, `to_scipy()`, works with your existing code
- **Memory efficient** - Views instead of copies, reduced memory footprint

```python
from biosparse import CSRF64
import scipy.sparse as sp

# From scipy (zero-copy available)
csr = CSRF64.from_scipy(scipy_mat, copy=False)

# Zero-cost operations
subset = csr[1000:2000, :]           # No data copy
stacked = CSRF64.vstack([csr1, csr2])  # Efficient concatenation

# Back to scipy when needed
scipy_mat = csr.to_scipy()
```

### 2️⃣ High-Performance Kernels

Battle-tested algorithms built on our sparse matrix, compiled with Numba JIT:

| Algorithm | vs scipy | vs scanpy |
|-----------|----------|-----------|
| Sparse nonlinear ops | **1,000 - 10,000x** | - |
| HVG selection | - | **10 - 100x** |
| Mann-Whitney U | - | **10 - 100x** |
| t-test | - | **10 - 100x** |

*Speedup scales with core count*

**Supported:**
- HVG: Seurat, Seurat V3, Cell Ranger, Pearson residuals
- Stats: Mann-Whitney U, Welch's t-test, Student's t-test, MMD

### 3️⃣ Numba Optimization Toolkit

The secret sauce: tools that make Numba JIT **outperform hand-written C++**.

```python
from biosparse.optim import parallel_jit, assume, vectorize, likely

@parallel_jit
def my_kernel(csr):
    assume(csr.nrows > 0)  # Enable compiler optimizations
    
    for row in prange(csr.nrows):
        values, indices = csr.row_to_numpy(row)
        
        vectorize(8)  # SIMD hint
        for v in values:
            if likely(v > 0):  # Branch prediction
                # ...
```

**Includes:**
- LLVM intrinsics: `assume`, `likely`, `unlikely`, `prefetch`
- Loop hints: `vectorize`, `unroll`, `interleave`, `distribute`
- [Complete tutorial](./tutorial/) - 7 chapters from basics to expert

---

## Quick Start

```bash
pip install biosparse
```

```python
from biosparse import CSRF64
from biosparse.kernel import hvg

# Load your data
import scanpy as sc
adata = sc.read_h5ad("data.h5ad")

# Convert (zero-copy)
csr = CSRF64.from_scipy(adata.X.T)

# 100x faster HVG selection
indices, mask, *_ = hvg.hvg_seurat_v3(csr, n_top_genes=2000)

# Use with scanpy
adata.var['highly_variable'] = mask.astype(bool)
```

---

## Documentation

| Resource | Description |
|----------|-------------|
| [Tutorial](./tutorial/) | 7-chapter guide: from basics to outperforming C++ |
| [Sparse API](./docs/sparse.md) | CSR/CSC matrix reference |
| [Kernels](./docs/kernels.md) | HVG, MWU, t-test documentation |
| [Optimization](./docs/optimizations.md) | LLVM intrinsics & loop hints |

---

## License

MIT

---

<p align="center">
  <strong>Sparse. Fast. Biological.</strong>
</p>
