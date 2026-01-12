"""Test optimized HVG module."""
from biosparse.kernel.hvg import (
    compute_moments, compute_dispersion, select_top_k, select_top_k_sorted,
    hvg_seurat, hvg_cell_ranger, hvg_seurat_v3, hvg_pearson_residuals
)
from biosparse import CSRF64
import scipy.sparse as sp
import numpy as np
import time

# Create test data
np.random.seed(42)
n_genes, n_cells = 2000, 5000
density = 0.2
data = np.random.exponential(2.0, size=(n_genes, n_cells))
data[np.random.random(size=data.shape) > density] = 0
scipy_csr = sp.csr_matrix(data)
csr = CSRF64.from_scipy(scipy_csr)

print(f'Test matrix: {n_genes} genes x {n_cells} cells, density={density}')
print()

# Warm up JIT compilation
print('JIT compilation...')
_ = compute_moments(csr, ddof=1)
_ = hvg_seurat(csr, 100)
_ = hvg_cell_ranger(csr, 100)
_ = hvg_seurat_v3(csr, 100, span=0.3)
_ = hvg_pearson_residuals(csr, 100)
print()

# Benchmark
n_runs = 5
print(f'Benchmarking ({n_runs} runs each):')

t0 = time.perf_counter()
for _ in range(n_runs):
    means, vars_ = compute_moments(csr, ddof=1)
print(f'  compute_moments:     {(time.perf_counter()-t0)/n_runs*1000:.3f} ms')

t0 = time.perf_counter()
for _ in range(n_runs):
    disps = compute_dispersion(means, vars_)
print(f'  compute_dispersion:  {(time.perf_counter()-t0)/n_runs*1000:.3f} ms')

t0 = time.perf_counter()
for _ in range(n_runs):
    idx, mask, m, d, dn = hvg_seurat(csr, 100)
print(f'  hvg_seurat:          {(time.perf_counter()-t0)/n_runs*1000:.3f} ms')

t0 = time.perf_counter()
for _ in range(n_runs):
    idx, mask, m, d, dn = hvg_cell_ranger(csr, 100)
print(f'  hvg_cell_ranger:     {(time.perf_counter()-t0)/n_runs*1000:.3f} ms')

t0 = time.perf_counter()
for _ in range(n_runs):
    idx, mask, m, v, vn = hvg_seurat_v3(csr, 100, span=0.3)
print(f'  hvg_seurat_v3:       {(time.perf_counter()-t0)/n_runs*1000:.3f} ms')

t0 = time.perf_counter()
for _ in range(n_runs):
    idx, mask, m, v, rv = hvg_pearson_residuals(csr, 100)
print(f'  hvg_pearson:         {(time.perf_counter()-t0)/n_runs*1000:.3f} ms')

print()
print('Correctness validation:')
idx, mask, m, d, dn = hvg_seurat(csr, 100)
print(f'  hvg_seurat: {mask.sum()} genes selected')
print(f'  means: min={m.min():.4f}, max={m.max():.4f}')
print(f'  dispersions: min={d.min():.4f}, max={d.max():.4f}')

# Verify sorted output
sorted_idx, _ = select_top_k_sorted(dn, 10)
scores = [dn[i] for i in sorted_idx]
print(f'  Top 10 norm dispersions: {[f"{s:.2f}" for s in scores]}')

# Verify descending order
for i in range(1, len(sorted_idx)):
    assert dn[sorted_idx[i-1]] >= dn[sorted_idx[i]], 'Sort order wrong!'
print('  Sort order: CORRECT')

print()
print('All tests passed!')
