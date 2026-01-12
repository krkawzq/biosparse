<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<h1 align="center">🧬 biosparse</h1>

<p align="center">
  <strong>Make your single-cell analysis 10x faster</strong>
</p>

<p align="center">
  biosparse is a high-performance drop-in replacement for scanpy.<br>
  Same code. Same results. 10x the speed.
</p>

---

## 😩 The Problem

Single-cell RNA-seq datasets are getting massive. When processing 1M+ cells:

- **HVG selection** takes 5+ minutes
- **Differential expression** takes even longer
- **Interactive exploration** becomes a waiting game

## ✨ The Solution

biosparse rewrites core computations in Rust + Numba:

## 🚀 Quick Start

```bash
pip install biosparse
```

```python
# Just change one import
from biosparse import CSRF64
from biosparse.kernel import hvg

# Your scanpy data
import scanpy as sc
adata = sc.read_h5ad("your_data.h5ad")

# Convert (zero-copy, nearly instant)
csr = CSRF64.from_scipy(adata.X.T)  # genes x cells

# 🚀 10x faster HVG selection
indices, mask, *_ = hvg.hvg_seurat_v3(csr, n_top_genes=2000)

# Use results directly in scanpy
adata.var['highly_variable'] = mask.astype(bool)
```

## 🎯 Supported Algorithms

### Highly Variable Genes
- ✅ Seurat (binning + z-score)
- ✅ Seurat V3 (VST + LOESS) 
- ✅ Cell Ranger (percentile + median/MAD)
- ✅ Pearson residuals

### Differential Expression
- ✅ Mann-Whitney U test
- ✅ Welch's t-test
- ✅ Student's t-test

### Other
- ✅ Maximum Mean Discrepancy (MMD)
- ✅ Basic statistics (mean, variance, dispersion)

## 💡 Why So Fast?

It's not just "rewriting in another language":

1. **Rust core** - Sparse matrix ops implemented in Rust
2. **Zero-copy** - Shares memory with scipy, no data copying
3. **Parallelized** - All algorithms auto-parallelize across cores
4. **Compiler hints** - LLVM intrinsics generate optimal machine code

## 🔄 scanpy Compatible

biosparse results match scanpy **exactly**:

```python
# biosparse
idx_bio, mask_bio, *_ = hvg.hvg_seurat_v3(csr, n_top_genes=2000)

# scanpy  
sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000)

# Same results
assert np.allclose(adata.var['highly_variable'], mask_bio)
```

## 📖 Documentation

- [Tutorial](./tutorial/) - Learn biosparse from scratch
- [API Docs](./docs/) - Complete API reference

## 📄 License

MIT

---

<p align="center">
  <strong>Stop waiting. Start discovering.</strong><br>
  <sub>biosparse - Built for single-cell bioinformatics</sub>
</p>
