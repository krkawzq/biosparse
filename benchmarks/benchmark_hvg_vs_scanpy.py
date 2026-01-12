#!/usr/bin/env python
"""Benchmark: biosparse HVG vs Scanpy HVG Core Operators.

直接对比核心算子性能:
1. biosparse.kernel.hvg (hvg_seurat, hvg_cell_ranger, hvg_seurat_v3, hvg_pearson_residuals)
2. scanpy preprocessing._highly_variable_genes 核心算子

Usage:
    python benchmarks/benchmark_hvg_vs_scanpy.py
    python benchmarks/benchmark_hvg_vs_scanpy.py --quick
    python benchmarks/benchmark_hvg_vs_scanpy.py --large
    python benchmarks/benchmark_hvg_vs_scanpy.py --flavor seurat_v3
"""

import argparse
import time
import warnings
import math
import numpy as np
import numba
from numba import njit, prange
from scipy import stats
import scipy.sparse as sp

warnings.filterwarnings('ignore')

# Check dependencies
try:
    from biosparse._binding import CSRF64
    import biosparse._numba
    from biosparse.kernel.hvg import (
        hvg_seurat, hvg_cell_ranger, hvg_seurat_v3,
        hvg_pearson_residuals, compute_moments, compute_dispersion
    )
    BIOSPARSE_AVAILABLE = True
except ImportError as e:
    BIOSPARSE_AVAILABLE = False
    print(f"WARNING: biosparse not available: {e}")

try:
    from skmisc.loess import loess as skmisc_loess
    SKMISC_AVAILABLE = True
except ImportError:
    SKMISC_AVAILABLE = False
    print("WARNING: skmisc not available, seurat_v3 benchmarks will be limited")

try:
    from fast_array_utils import stats as fau_stats
    FAU_AVAILABLE = True
except ImportError:
    FAU_AVAILABLE = False


# =============================================================================
# Scanpy-Style Core Operators (extracted from scanpy)
# =============================================================================

def scanpy_mean_var(data: np.ndarray, ddof: int = 1):
    """Compute mean and variance per gene (column)."""
    means = data.mean(axis=0)
    vars_ = data.var(axis=0, ddof=ddof)
    return means, vars_


def scanpy_mean_var_sparse(csr_mat, ddof: int = 1):
    """Compute mean and variance for sparse matrix (CSR, genes x cells)."""
    n_genes, n_cells = csr_mat.shape
    means = np.asarray(csr_mat.mean(axis=1)).ravel()
    
    # For variance, we need: E[X^2] - E[X]^2, with ddof correction
    # E[X^2] = sum(x^2) / n
    sq_data = csr_mat.copy()
    sq_data.data = sq_data.data ** 2
    sq_means = np.asarray(sq_data.mean(axis=1)).ravel()
    
    # Var = n/(n-ddof) * (E[X^2] - E[X]^2)
    n = float(n_cells)
    vars_ = (n / (n - ddof)) * (sq_means - means ** 2)
    vars_ = np.maximum(vars_, 0)  # numerical stability
    
    return means, vars_


def scanpy_dispersion(means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
    """Compute dispersion = var / mean."""
    eps = 1e-12
    return np.where(means > eps, vars_ / means, 0.0)


# Scanpy numba clipping kernel (from _highly_variable_genes.py)
@njit(cache=True, parallel=False)
def scanpy_sum_and_sum_squares_clipped(
    indices: np.ndarray,
    data: np.ndarray,
    n_cols: int,
    clip_val: np.ndarray,
    nnz: int,
):
    """Scanpy's clipped sum and sum-of-squares computation for CSR."""
    squared_batch_counts_sum = np.zeros(n_cols, dtype=np.float64)
    batch_counts_sum = np.zeros(n_cols, dtype=np.float64)
    for i in numba.prange(nnz):
        idx = indices[i]
        element = min(np.float64(data[i]), clip_val[idx])
        squared_batch_counts_sum[idx] += element ** 2
        batch_counts_sum[idx] += element
    return squared_batch_counts_sum, batch_counts_sum


def scanpy_hvg_seurat(X: np.ndarray, n_top_genes: int, n_bins: int = 20,
                      min_mean: float = 0.0125, max_mean: float = 3.0,
                      min_disp: float = 0.5, max_disp: float = np.inf):
    """Scanpy's Seurat HVG algorithm (simplified core).
    
    Args:
        X: Dense matrix (cells x genes)
        n_top_genes: Number of top genes to select
        n_bins: Number of bins for normalization
        min_mean, max_mean: Mean cutoffs
        min_disp, max_disp: Dispersion cutoffs
    
    Returns:
        indices, mask, means, dispersions, dispersions_norm
    """
    n_cells, n_genes = X.shape
    
    # Compute mean and variance (per gene = per column)
    means, vars_ = scanpy_mean_var(X, ddof=1)
    
    # Compute dispersion
    eps = 1e-12
    means_for_disp = means.copy()
    means_for_disp[means_for_disp == 0] = eps
    dispersions = vars_ / means_for_disp
    
    # Log transform for binning
    log_dispersions = np.log(np.maximum(dispersions, eps))
    log_means = np.log1p(means)
    
    # Bin by mean
    bins = np.linspace(log_means.min(), log_means.max(), n_bins + 1)
    bin_indices = np.clip(np.digitize(log_means, bins) - 1, 0, n_bins - 1)
    
    # Per-bin mean and std of log(dispersion)
    dispersions_norm = np.zeros(n_genes, dtype=np.float64)
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_mean = log_dispersions[mask].mean()
            bin_std = log_dispersions[mask].std() if mask.sum() > 1 else 1.0
            if bin_std < eps:
                bin_std = 1.0
            dispersions_norm[mask] = (log_dispersions[mask] - bin_mean) / bin_std
    
    # Apply cutoffs (log scale for means)
    log_min = np.log1p(min_mean)
    log_max = np.log1p(max_mean)
    valid = (log_means >= log_min) & (log_means <= log_max) & \
            (dispersions_norm >= min_disp) & (dispersions_norm <= max_disp)
    
    # Select top k
    scores = np.where(valid, dispersions_norm, -np.inf)
    indices = np.argsort(scores)[::-1][:n_top_genes]
    mask = np.zeros(n_genes, dtype=np.uint8)
    mask[indices] = 1
    
    return indices, mask, means, dispersions, dispersions_norm


def scanpy_hvg_cell_ranger(X: np.ndarray, n_top_genes: int):
    """Scanpy's Cell Ranger HVG algorithm (simplified core).
    
    Uses percentile bins and median/MAD normalization.
    """
    n_cells, n_genes = X.shape
    
    # Compute mean and variance
    means, vars_ = scanpy_mean_var(X, ddof=1)
    
    # Compute dispersion
    eps = 1e-12
    means_for_disp = means.copy()
    means_for_disp[means_for_disp == 0] = eps
    dispersions = vars_ / means_for_disp
    
    # Cell Ranger uses percentile bins (5% steps from 10% to 100%)
    percentiles = np.arange(10, 105, 5)
    edges = np.percentile(means, percentiles)
    edges = np.r_[-np.inf, edges, np.inf]
    
    bin_indices = np.digitize(means, edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(percentiles))
    n_bins = len(percentiles) + 1
    
    # Per-bin median and MAD normalization
    dispersions_norm = np.zeros(n_genes, dtype=np.float64)
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_disp = dispersions[mask]
            median = np.median(bin_disp)
            mad = np.median(np.abs(bin_disp - median)) if len(bin_disp) > 1 else 1.0
            if mad < eps:
                mad = 1.0
            dispersions_norm[mask] = (dispersions[mask] - median) / mad
    
    # Select top k
    indices = np.argsort(dispersions_norm)[::-1][:n_top_genes]
    mask = np.zeros(n_genes, dtype=np.uint8)
    mask[indices] = 1
    
    return indices, mask, means, dispersions, dispersions_norm


def scanpy_hvg_seurat_v3(csr_mat, n_top_genes: int, span: float = 0.3):
    """Scanpy's Seurat V3 HVG algorithm (exact match to scanpy implementation).
    
    Uses VST (variance-stabilizing transformation) with LOESS regression.
    
    Args:
        csr_mat: scipy.sparse.csr_matrix (genes x cells)
        n_top_genes: Number of top genes to select
        span: LOESS span parameter
    
    Returns:
        indices, mask, means, variances, variances_norm
    """
    if not SKMISC_AVAILABLE:
        raise ImportError("skmisc is required for seurat_v3")
    
    n_genes, n_cells = csr_mat.shape
    
    # Compute mean and variance
    means, variances = scanpy_mean_var_sparse(csr_mat, ddof=1)
    
    # Filter constant genes: var > 0 (matching scanpy exactly!)
    not_const = variances > 0
    
    # LOESS regression on log10 scale
    # CRITICAL: NO EPS added to log10 - must match scanpy exactly!
    estimat_var = np.zeros(n_genes, dtype=np.float64)
    if not_const.sum() > 0:
        y = np.log10(variances[not_const])
        x = np.log10(means[not_const])  # NO EPS! Match scanpy line 190
        
        model = skmisc_loess(x, y, span=span, degree=2)
        model.fit()
        estimat_var[not_const] = model.outputs.fitted_values
    
    reg_std = np.sqrt(10 ** estimat_var)
    
    # Clip values (per gene)
    clip_val = reg_std * np.sqrt(n_cells) + means
    
    # CRITICAL: Convert to cells x genes format for scanpy's clipping kernel!
    # scanpy's _sum_and_sum_squares_clipped expects CSR with:
    #   - rows = cells
    #   - columns = genes (indices point to genes)
    csr_cells_genes = csr_mat.T.tocsr()  # Transpose: (genes x cells) -> (cells x genes)
    
    # Now indices are gene indices, which is what scanpy expects
    sq_sum, val_sum = scanpy_sum_and_sum_squares_clipped(
        csr_cells_genes.indices.astype(np.int64),
        csr_cells_genes.data.astype(np.float64),
        n_genes,  # n_cols = n_genes
        clip_val,
        csr_cells_genes.nnz
    )
    
    # Normalized variance
    variances_norm = np.zeros(n_genes, dtype=np.float64)
    eps = 1e-12
    for i in range(n_genes):
        if reg_std[i] > eps:
            factor = 1.0 / ((n_cells - 1) * reg_std[i] ** 2)
            variances_norm[i] = factor * (
                n_cells * means[i] ** 2 + sq_sum[i] - 2 * val_sum[i] * means[i]
            )
    
    # Select top k
    indices = np.argsort(variances_norm)[::-1][:n_top_genes]
    mask = np.zeros(n_genes, dtype=np.uint8)
    mask[indices] = 1
    
    return indices, mask, means, variances, variances_norm


def scanpy_hvg_pearson_residuals(X: np.ndarray, n_top_genes: int,
                                  theta: float = 100.0, clip: float = -1.0):
    """Scanpy's Pearson residuals HVG algorithm.
    
    Computes variance of clipped Pearson residuals.
    """
    n_cells, n_genes = X.shape
    
    if clip < 0:
        clip = np.sqrt(n_cells)
    
    # Sums
    row_sums = X.sum(axis=1)  # per cell
    col_sums = X.sum(axis=0)  # per gene
    total = X.sum()
    
    # Compute residual variance per gene
    residual_vars = np.zeros(n_genes, dtype=np.float64)
    eps = 1e-12
    
    for g in range(n_genes):
        residuals = np.zeros(n_cells, dtype=np.float64)
        gene_sum = col_sums[g]
        
        for c in range(n_cells):
            mu = gene_sum * row_sums[c] / total
            if mu > eps:
                var = mu + mu ** 2 / theta
                res = (X[c, g] - mu) / np.sqrt(var)
            else:
                res = 0.0
            
            # Clip
            res = np.clip(res, -clip, clip)
            residuals[c] = res
        
        residual_vars[g] = residuals.var()
    
    # Compute basic moments
    means = X.mean(axis=0)
    variances = X.var(axis=0, ddof=1)
    
    # Select top k
    indices = np.argsort(residual_vars)[::-1][:n_top_genes]
    mask = np.zeros(n_genes, dtype=np.uint8)
    mask[indices] = 1
    
    return indices, mask, means, variances, residual_vars


# Optimized Pearson residuals (vectorized)
def scanpy_hvg_pearson_residuals_vectorized(X: np.ndarray, n_top_genes: int,
                                             theta: float = 100.0, clip: float = -1.0):
    """Vectorized Pearson residuals (faster than loop version)."""
    n_cells, n_genes = X.shape
    
    if clip < 0:
        clip = np.sqrt(n_cells)
    
    # Sums
    row_sums = X.sum(axis=1, keepdims=True)  # (n_cells, 1)
    col_sums = X.sum(axis=0, keepdims=True)  # (1, n_genes)
    total = X.sum()
    
    # Expected values: mu[c,g] = row_sums[c] * col_sums[g] / total
    mu = row_sums * col_sums / total
    
    # Variance: var = mu + mu^2 / theta
    var = mu + mu ** 2 / theta
    var = np.maximum(var, 1e-12)
    
    # Residuals
    residuals = (X - mu) / np.sqrt(var)
    residuals = np.clip(residuals, -clip, clip)
    
    # Variance per gene
    residual_vars = residuals.var(axis=0)
    
    # Basic moments
    means = X.mean(axis=0)
    variances = X.var(axis=0, ddof=1)
    
    # Select top k
    indices = np.argsort(residual_vars)[::-1][:n_top_genes]
    mask = np.zeros(n_genes, dtype=np.uint8)
    mask[indices] = 1
    
    return indices, mask, means, variances, residual_vars


# =============================================================================
# Utilities
# =============================================================================

def format_time(seconds: float) -> str:
    """Format time with appropriate unit."""
    if seconds < 0.001:
        return f"{seconds*1e6:.1f}μs"
    elif seconds < 1:
        return f"{seconds*1e3:.2f}ms"
    else:
        return f"{seconds:.2f}s"


def create_test_data(n_genes: int, n_cells: int, density: float, seed: int = 42):
    """Create test data for both biosparse and scanpy.
    
    Returns:
        scipy_csr: scipy sparse matrix (genes x cells) for biosparse
        bio_csr: biosparse CSR matrix (genes x cells)
        dense_cells_genes: dense matrix (cells x genes) for scanpy
    """
    np.random.seed(seed)
    
    # Create sparse matrix (genes x cells) - biosparse format
    scipy_csr = sp.random(n_genes, n_cells, density=density, format='csr', dtype=np.float64)
    
    # Set realistic values (raw counts for seurat_v3, log-normalized for others)
    scipy_csr.data[:] = np.abs(np.random.randn(len(scipy_csr.data))) * 2
    
    # Create biosparse matrix (genes x cells)
    bio_csr = CSRF64.from_scipy(scipy_csr) if BIOSPARSE_AVAILABLE else None
    
    # Create dense matrix (cells x genes) for scanpy dispersion-based methods
    dense_cells_genes = scipy_csr.T.toarray()
    
    return scipy_csr, bio_csr, dense_cells_genes


def warmup_jit():
    """Warmup JIT compilation with realistic data sizes.
    
    Note: Numba JIT compilation is cached per function signature.
    We need to warmup with data sizes similar to actual benchmarks.
    """
    print("Warming up JIT compilation...")
    
    # Warmup biosparse with realistic sizes
    if BIOSPARSE_AVAILABLE:
        # Use larger data for better warmup (2000 x 1000 is typical small benchmark)
        scipy_warm = sp.random(2000, 1000, density=0.05, format='csr', dtype=np.float64)
        scipy_warm.data[:] = np.abs(np.random.randn(len(scipy_warm.data))) * 2
        bio_warm = CSRF64.from_scipy(scipy_warm)
        
        try:
            # Call each function 2x to ensure stable JIT
            for _ in range(2):
                _ = hvg_seurat(bio_warm, n_top_genes=100)
            for _ in range(2):
                _ = hvg_cell_ranger(bio_warm, n_top_genes=100)
            for _ in range(2):
                _ = hvg_seurat_v3(bio_warm, n_top_genes=100)
            for _ in range(2):
                _ = hvg_pearson_residuals(bio_warm, n_top_genes=100)
            print("  biosparse HVG kernels: OK")
        except Exception as e:
            print(f"  Warmup warning: {e}")
    
    # Warmup scanpy operators
    dense_warm = np.abs(np.random.randn(1000, 2000).astype(np.float64)) * 2
    _ = scanpy_hvg_seurat(dense_warm, n_top_genes=100)
    _ = scanpy_hvg_cell_ranger(dense_warm, n_top_genes=100)
    print("  scanpy-style operators: OK")
    
    print("JIT warmup complete.\n")


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_moments(bio_csr, dense_mat: np.ndarray, n_runs: int = 5):
    """Benchmark basic moments computation (mean, var)."""
    results = {}
    
    # Scanpy numpy (dense_mat is cells x genes, compute per-gene stats)
    def run_scanpy():
        return scanpy_mean_var(dense_mat, ddof=1)
    
    # Warmup: 2 calls
    scanpy_result = run_scanpy()
    _ = run_scanpy()
    
    start = time.perf_counter()
    for _ in range(n_runs):
        scanpy_result = run_scanpy()
    results['scanpy_numpy'] = (time.perf_counter() - start) / n_runs
    
    # Biosparse
    if BIOSPARSE_AVAILABLE:
        def run_bio():
            return compute_moments(bio_csr, ddof=1)
        
        # Warmup: 2 calls for stable JIT
        bio_result = run_bio()
        _ = run_bio()
        
        start = time.perf_counter()
        for _ in range(n_runs):
            bio_result = run_bio()
        results['biosparse'] = (time.perf_counter() - start) / n_runs
        
        # Verify correctness (means and variances should match)
        means_match = np.allclose(bio_result[0], scanpy_result[0], rtol=1e-10)
        vars_match = np.allclose(bio_result[1], scanpy_result[1], rtol=1e-10)
        results['correct'] = means_match and vars_match
    
    return results


def benchmark_seurat(bio_csr, dense_mat: np.ndarray, n_top_genes: int, n_runs: int = 5):
    """Benchmark Seurat flavor HVG."""
    results = {}
    
    # Scanpy
    def run_scanpy():
        return scanpy_hvg_seurat(dense_mat, n_top_genes)
    
    # Warmup: 2 calls
    _ = run_scanpy()
    _ = run_scanpy()
    
    start = time.perf_counter()
    for _ in range(n_runs):
        scanpy_result = run_scanpy()
    results['scanpy'] = (time.perf_counter() - start) / n_runs
    
    # Biosparse
    if BIOSPARSE_AVAILABLE:
        def run_bio():
            return hvg_seurat(bio_csr, n_top_genes)
        
        # Warmup: 2 calls for stable JIT
        _ = run_bio()
        _ = run_bio()
        
        start = time.perf_counter()
        for _ in range(n_runs):
            bio_result = run_bio()
        results['biosparse'] = (time.perf_counter() - start) / n_runs
        
        # Verify results
        scanpy_top = set(scanpy_result[0][:n_top_genes])
        bio_top = set(bio_result[0][:n_top_genes])
        results['overlap'] = len(scanpy_top & bio_top) / n_top_genes
    
    return results


def benchmark_cell_ranger(bio_csr, dense_mat: np.ndarray, n_top_genes: int, n_runs: int = 5):
    """Benchmark Cell Ranger flavor HVG."""
    results = {}
    
    # Scanpy
    def run_scanpy():
        return scanpy_hvg_cell_ranger(dense_mat, n_top_genes)
    
    # Warmup: 2 calls
    _ = run_scanpy()
    _ = run_scanpy()
    
    start = time.perf_counter()
    for _ in range(n_runs):
        scanpy_result = run_scanpy()
    results['scanpy'] = (time.perf_counter() - start) / n_runs
    
    # Biosparse
    if BIOSPARSE_AVAILABLE:
        def run_bio():
            return hvg_cell_ranger(bio_csr, n_top_genes)
        
        # Warmup: 2 calls for stable JIT
        _ = run_bio()
        _ = run_bio()
        
        start = time.perf_counter()
        for _ in range(n_runs):
            bio_result = run_bio()
        results['biosparse'] = (time.perf_counter() - start) / n_runs
        
        # Verify results
        scanpy_top = set(scanpy_result[0][:n_top_genes])
        bio_top = set(bio_result[0][:n_top_genes])
        results['overlap'] = len(scanpy_top & bio_top) / n_top_genes
    
    return results


def benchmark_seurat_v3(scipy_csr, bio_csr, n_top_genes: int, n_runs: int = 5):
    """Benchmark Seurat V3 flavor HVG (VST + LOESS)."""
    results = {}
    scanpy_result = None
    
    # Scanpy (needs CSR genes x cells format)
    if SKMISC_AVAILABLE:
        def run_scanpy():
            return scanpy_hvg_seurat_v3(scipy_csr, n_top_genes)
        
        try:
            # Warmup: 2 calls to ensure stable timing
            _ = run_scanpy()
            _ = run_scanpy()
            
            start = time.perf_counter()
            for _ in range(n_runs):
                scanpy_result = run_scanpy()
            results['scanpy'] = (time.perf_counter() - start) / n_runs
        except Exception as e:
            print(f"  Scanpy seurat_v3 error: {e}")
            results['scanpy'] = float('inf')
    else:
        results['scanpy'] = float('inf')
    
    # Biosparse - uses biosparse's parallel LOESS
    if BIOSPARSE_AVAILABLE:
        def run_bio():
            return hvg_seurat_v3(bio_csr, n_top_genes)
        
        try:
            # Warmup
            _ = run_bio()
            _ = run_bio()
            
            start = time.perf_counter()
            for _ in range(n_runs):
                bio_result = run_bio()
            results['biosparse'] = (time.perf_counter() - start) / n_runs
            
            # Verify overlap with scanpy
            if scanpy_result is not None:
                scanpy_top = set(scanpy_result[0][:n_top_genes])
                bio_top = set(bio_result[0][:n_top_genes])
                results['overlap'] = len(scanpy_top & bio_top) / n_top_genes
                
        except Exception as e:
            print(f"  Biosparse seurat_v3 error: {e}")
            import traceback
            traceback.print_exc()
            results['biosparse'] = float('inf')
    
    return results


def benchmark_pearson_residuals(bio_csr, dense_mat: np.ndarray, n_top_genes: int, n_runs: int = 3):
    """Benchmark Pearson residuals HVG."""
    results = {}
    
    # Scanpy (vectorized version)
    def run_scanpy():
        return scanpy_hvg_pearson_residuals_vectorized(dense_mat, n_top_genes)
    
    # Warmup: 2 calls
    _ = run_scanpy()
    _ = run_scanpy()
    
    start = time.perf_counter()
    for _ in range(n_runs):
        scanpy_result = run_scanpy()
    results['scanpy'] = (time.perf_counter() - start) / n_runs
    
    # Biosparse
    if BIOSPARSE_AVAILABLE:
        def run_bio():
            return hvg_pearson_residuals(bio_csr, n_top_genes)
        
        try:
            # Warmup: 2 calls for stable JIT
            _ = run_bio()
            _ = run_bio()
            
            start = time.perf_counter()
            for _ in range(n_runs):
                bio_result = run_bio()
            results['biosparse'] = (time.perf_counter() - start) / n_runs
            
            # Verify results
            scanpy_top = set(scanpy_result[0][:n_top_genes])
            bio_top = set(bio_result[0][:n_top_genes])
            results['overlap'] = len(scanpy_top & bio_top) / n_top_genes
        except Exception as e:
            print(f"  Biosparse pearson error: {e}")
            results['biosparse'] = float('inf')
    
    return results


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(n_genes: int, n_cells: int, density: float, n_top_genes: int = 2000, n_runs: int = 5):
    """Run full HVG benchmark comparison."""
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {n_genes} genes x {n_cells} cells, {density:.0%} density, top {n_top_genes} genes")
    print(f"{'='*80}")
    
    # Create test data
    print("\nCreating test data...")
    scipy_csr, bio_csr, dense_cells_genes = create_test_data(n_genes, n_cells, density)
    
    print(f"  biosparse CSR: {n_genes} genes x {n_cells} cells (genes x cells)")
    print(f"  scanpy dense:  {n_cells} cells x {n_genes} genes (cells x genes)")
    print(f"  NNZ: {scipy_csr.nnz:,} ({scipy_csr.nnz / (n_genes * n_cells) * 100:.1f}%)")
    
    all_results = {}
    
    # 1. Moments benchmark
    print(f"\n[1/5] Benchmarking Moments (mean, variance)...")
    moments_results = benchmark_moments(bio_csr, dense_cells_genes, n_runs)
    all_results['moments'] = moments_results
    
    scanpy_t = moments_results.get('scanpy_numpy', 0)
    bio_t = moments_results.get('biosparse', float('inf'))
    speedup = scanpy_t / bio_t if bio_t > 0 and bio_t < float('inf') else 0
    correct = moments_results.get('correct', False)
    correct_str = "✓" if correct else "✗"
    print(f"  scanpy (numpy): {format_time(scanpy_t)}")
    print(f"  biosparse:      {format_time(bio_t)} ({speedup:.2f}x, {correct_str})")
    
    # 2. Seurat benchmark
    print(f"\n[2/5] Benchmarking Seurat flavor...")
    seurat_results = benchmark_seurat(bio_csr, dense_cells_genes, n_top_genes, n_runs)
    all_results['seurat'] = seurat_results
    
    scanpy_t = seurat_results.get('scanpy', 0)
    bio_t = seurat_results.get('biosparse', float('inf'))
    speedup = scanpy_t / bio_t if bio_t > 0 and bio_t < float('inf') else 0
    overlap = seurat_results.get('overlap', 0)
    print(f"  scanpy:    {format_time(scanpy_t)}")
    print(f"  biosparse: {format_time(bio_t)} ({speedup:.2f}x, {overlap*100:.0f}% overlap)")
    
    # 3. Cell Ranger benchmark
    print(f"\n[3/5] Benchmarking Cell Ranger flavor...")
    cellranger_results = benchmark_cell_ranger(bio_csr, dense_cells_genes, n_top_genes, n_runs)
    all_results['cell_ranger'] = cellranger_results
    
    scanpy_t = cellranger_results.get('scanpy', 0)
    bio_t = cellranger_results.get('biosparse', float('inf'))
    speedup = scanpy_t / bio_t if bio_t > 0 and bio_t < float('inf') else 0
    overlap = cellranger_results.get('overlap', 0)
    print(f"  scanpy:    {format_time(scanpy_t)}")
    print(f"  biosparse: {format_time(bio_t)} ({speedup:.2f}x, {overlap*100:.0f}% overlap)")
    
    # 4. Seurat V3 benchmark
    print(f"\n[4/5] Benchmarking Seurat V3 flavor (VST + LOESS)...")
    seuratv3_results = benchmark_seurat_v3(scipy_csr, bio_csr, n_top_genes, n_runs)
    all_results['seurat_v3'] = seuratv3_results
    
    scanpy_t = seuratv3_results.get('scanpy', float('inf'))
    bio_t = seuratv3_results.get('biosparse', float('inf'))
    speedup = scanpy_t / bio_t if bio_t > 0 and bio_t < float('inf') and scanpy_t < float('inf') else 0
    overlap = seuratv3_results.get('overlap', 0)
    if scanpy_t < float('inf'):
        print(f"  scanpy:    {format_time(scanpy_t)}")
    else:
        print(f"  scanpy:    N/A (skmisc not available)")
    if bio_t < float('inf'):
        print(f"  biosparse: {format_time(bio_t)} ({speedup:.2f}x, {overlap*100:.0f}% overlap)")
    else:
        print(f"  biosparse: N/A")
    
    # 5. Pearson residuals benchmark
    print(f"\n[5/5] Benchmarking Pearson Residuals flavor...")
    pearson_results = benchmark_pearson_residuals(bio_csr, dense_cells_genes, n_top_genes, n_runs)
    all_results['pearson'] = pearson_results
    
    scanpy_t = pearson_results.get('scanpy', 0)
    bio_t = pearson_results.get('biosparse', float('inf'))
    speedup = scanpy_t / bio_t if bio_t > 0 and bio_t < float('inf') else 0
    overlap = pearson_results.get('overlap', 0)
    print(f"  scanpy:    {format_time(scanpy_t)}")
    if bio_t < float('inf'):
        print(f"  biosparse: {format_time(bio_t)} ({speedup:.2f}x, {overlap*100:.0f}% overlap)")
    else:
        print(f"  biosparse: N/A")
    
    return all_results


def run_scaling_benchmark(quick: bool = False, large: bool = False, flavor: str = None):
    """Run scaling benchmarks across different sizes."""

    warmup_jit()

    if quick:
        configs = [
            (5000, 10000, 0.2, 2000),   # (n_genes, n_cells, density, n_top)
            (5000, 20000, 0.2, 3000),
        ]
        n_runs = 3
    elif large:
        configs = [
            (2000, 15000, 0.2, 1000),
            (5000, 25000, 0.2, 2000),
            (10000, 25000, 0.2, 2000),
            (15000, 25000, 0.2, 2000),
        ]
        n_runs = 3
    else:
        configs = [
            (10000, 50000, 0.2, 5000),
            (20000, 50000, 0.2, 5000),
            (50000, 100000, 0.2, 5000),
            (80000, 200000, 0.2, 5000),
        ]
        n_runs = 3
    
    print("\n" + "="*100)
    print("SCALING BENCHMARK: biosparse HVG vs Scanpy HVG")
    print("="*100)
    
    all_results = []
    
    for n_genes, n_cells, density, n_top in configs:
        try:
            results = run_benchmark(n_genes, n_cells, density, n_top, n_runs)
            
            # Collect speedups
            row = {
                'n_genes': n_genes,
                'n_cells': n_cells,
                'density': density,
                'n_top': n_top,
            }
            
            for method in ['moments', 'seurat', 'cell_ranger', 'seurat_v3', 'pearson']:
                if method in results:
                    r = results[method]
                    scanpy_t = r.get('scanpy', r.get('scanpy_numpy', float('inf')))
                    bio_t = r.get('biosparse', float('inf'))
                    if scanpy_t < float('inf') and bio_t < float('inf') and bio_t > 0:
                        row[f'{method}_speedup'] = scanpy_t / bio_t
                        row[f'{method}_scanpy'] = scanpy_t
                        row[f'{method}_biosparse'] = bio_t
                    else:
                        row[f'{method}_speedup'] = 0
            
            all_results.append(row)
            
        except Exception as e:
            print(f"\nError with {n_genes}x{n_cells}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    print(f"\n{'='*120}")
    print("SUMMARY TABLE")
    print(f"{'='*120}")
    
    headers = ['Genes', 'Cells', 'Density', 'Moments', 'Seurat', 'CellRanger', 'SeuratV3', 'Pearson']
    print(f"\n{'Genes':<10} {'Cells':<10} {'Density':<8} {'Moments':<12} {'Seurat':<12} {'CellRanger':<12} {'SeuratV3':<12} {'Pearson':<12}")
    print("-" * 100)
    
    for r in all_results:
        moments_s = f"{r.get('moments_speedup', 0):.2f}x" if r.get('moments_speedup', 0) > 0 else "N/A"
        seurat_s = f"{r.get('seurat_speedup', 0):.2f}x" if r.get('seurat_speedup', 0) > 0 else "N/A"
        cr_s = f"{r.get('cell_ranger_speedup', 0):.2f}x" if r.get('cell_ranger_speedup', 0) > 0 else "N/A"
        v3_s = f"{r.get('seurat_v3_speedup', 0):.2f}x" if r.get('seurat_v3_speedup', 0) > 0 else "N/A"
        pr_s = f"{r.get('pearson_speedup', 0):.2f}x" if r.get('pearson_speedup', 0) > 0 else "N/A"
        
        print(f"{r['n_genes']:<10} {r['n_cells']:<10} {r['density']:<8.0%} "
              f"{moments_s:<12} {seurat_s:<12} {cr_s:<12} {v3_s:<12} {pr_s:<12}")
    
    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL SPEEDUP STATISTICS")
    print(f"{'='*80}")
    
    for method in ['moments', 'seurat', 'cell_ranger', 'seurat_v3', 'pearson']:
        speedups = [r.get(f'{method}_speedup', 0) for r in all_results if r.get(f'{method}_speedup', 0) > 0]
        if speedups:
            print(f"\n{method.upper()}:")
            print(f"  Average speedup: {np.mean(speedups):.2f}x")
            print(f"  Median speedup:  {np.median(speedups):.2f}x")
            print(f"  Range:           {min(speedups):.2f}x - {max(speedups):.2f}x")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="HVG vs Scanpy Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode with smaller matrices")
    parser.add_argument("--large", action="store_true", help="Large-scale benchmark")
    parser.add_argument("--genes", type=int, default=None, help="Number of genes")
    parser.add_argument("--cells", type=int, default=None, help="Number of cells")
    parser.add_argument("--density", type=float, default=0.1, help="Sparsity density")
    parser.add_argument("--top", type=int, default=2000, help="Number of top genes")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--flavor", type=str, default=None, 
                        choices=['seurat', 'cell_ranger', 'seurat_v3', 'pearson'],
                        help="Only benchmark specific flavor")
    args = parser.parse_args()
    
    print("="*80)
    print("HVG vs SCANPY BENCHMARK")
    print("="*80)
    print(f"biosparse available: {BIOSPARSE_AVAILABLE}")
    print(f"skmisc available:    {SKMISC_AVAILABLE}")
    print(f"fast_array_utils:    {FAU_AVAILABLE}")
    
    if not BIOSPARSE_AVAILABLE:
        print("\nERROR: biosparse is required")
        return
    
    if args.genes and args.cells:
        # Single benchmark
        warmup_jit()
        run_benchmark(args.genes, args.cells, args.density, args.top, args.runs)
    else:
        # Scaling benchmark
        run_scaling_benchmark(quick=args.quick, large=args.large, flavor=args.flavor)


if __name__ == "__main__":
    main()
