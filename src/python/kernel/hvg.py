"""Highly Variable Gene (HVG) Selection.

Vectorized implementation of HVG selection algorithms:
    - compute_dispersion: Compute variance/mean ratio
    - normalize_dispersion: Z-score normalize dispersion
    - select_top_k: Select top k genes by score
    - compute_moments: Compute mean and variance from sparse matrix

All functions return arrays and are optimized with Numba JIT.
Uses project's CSR sparse matrix type.
"""

import numpy as np
from numba import prange

from optim import parallel_jit, assume, vectorize
from _binding import CSR

# Import for type hints only
import _numba  # noqa: F401 - registers CSR/CSC types

__all__ = [
    'compute_dispersion',
    'normalize_dispersion',
    'select_top_k',
    'compute_moments',
    'compute_clipped_moments',
    'select_hvg_by_dispersion',
]


# =============================================================================
# Dispersion Computation
# =============================================================================

@parallel_jit
def compute_dispersion(means: np.ndarray, vars: np.ndarray) -> np.ndarray:
    """Compute dispersion = variance / mean.
    
    Args:
        means: Mean values per gene
        vars: Variance values per gene
    
    Returns:
        Dispersion values array
    """
    EPSILON = 1e-12
    
    n = len(means)
    assume(n > 0)
    assume(len(vars) >= n)
    
    out = np.empty(n, dtype=np.float64)
    
    vectorize(8)
    for i in prange(n):
        m = means[i]
        if m > EPSILON:
            out[i] = vars[i] / m
        else:
            out[i] = 0.0
    
    return out


@parallel_jit
def normalize_dispersion(
    dispersions: np.ndarray,
    means: np.ndarray,
    min_mean: float,
    max_mean: float
) -> np.ndarray:
    """Normalize dispersion values by z-score.
    
    Genes outside [min_mean, max_mean] range are set to -inf.
    
    Args:
        dispersions: Raw dispersion values
        means: Mean values per gene
        min_mean: Minimum mean threshold
        max_mean: Maximum mean threshold
    
    Returns:
        Normalized dispersion values
    """
    NEG_INF = -np.inf
    
    n = len(dispersions)
    assume(n > 0)
    assume(len(means) >= n)
    
    out = np.empty(n, dtype=np.float64)
    
    # First pass: compute mean and std of valid dispersions
    valid_sum = 0.0
    valid_sq_sum = 0.0
    valid_count = 0
    
    for i in range(n):
        m = means[i]
        d = dispersions[i]
        if m >= min_mean and m <= max_mean and d > 0.0:
            valid_sum += d
            valid_sq_sum += d * d
            valid_count += 1
    
    if valid_count == 0:
        for i in prange(n):
            out[i] = NEG_INF
        return out
    
    disp_mean = valid_sum / valid_count
    disp_var = (valid_sq_sum / valid_count) - disp_mean * disp_mean
    disp_std = np.sqrt(disp_var) if disp_var > 0.0 else 1.0
    inv_std = 1.0 / disp_std
    
    # Second pass: normalize
    vectorize(8)
    for i in prange(n):
        m = means[i]
        d = dispersions[i]
        if m >= min_mean and m <= max_mean and d > 0.0:
            out[i] = (d - disp_mean) * inv_std
        else:
            out[i] = NEG_INF
    
    return out


@parallel_jit
def select_top_k(scores: np.ndarray, k: int) -> tuple:
    """Select top k genes by score using partial sort.
    
    Args:
        scores: Score values per gene
        k: Number of top genes to select
    
    Returns:
        (indices, mask): Top k indices and binary mask
    """
    n = len(scores)
    assume(n > 0)
    assume(k > 0)
    assume(k <= n)
    
    out_indices = np.empty(k, dtype=np.int64)
    out_mask = np.zeros(n, dtype=np.uint8)
    
    # Create index array
    indices = np.arange(n, dtype=np.int64)
    
    # Partial sort: find top k
    for i in range(k):
        max_idx = i
        max_val = scores[indices[i]]
        
        for j in range(i + 1, n):
            if scores[indices[j]] > max_val:
                max_idx = j
                max_val = scores[indices[j]]
        
        # Swap
        if max_idx != i:
            tmp = indices[i]
            indices[i] = indices[max_idx]
            indices[max_idx] = tmp
        
        out_indices[i] = indices[i]
        out_mask[indices[i]] = 1
    
    return out_indices, out_mask


# =============================================================================
# Moment Computation for Sparse Matrices (CSR type)
# =============================================================================

@parallel_jit
def compute_moments(csr: CSR, ddof: int) -> tuple:
    """Compute per-row mean and variance for CSR sparse matrix.
    
    Args:
        csr: CSR sparse matrix (CSRF32 or CSRF64)
        ddof: Delta degrees of freedom for variance
    
    Returns:
        (means, vars): Per-row means and variances
    """
    n_rows = csr.nrows
    N = float(csr.ncols)
    denom = N - float(ddof)
    
    assume(n_rows > 0)
    
    out_means = np.empty(n_rows, dtype=np.float64)
    out_vars = np.empty(n_rows, dtype=np.float64)
    
    row_idx = 0
    for values, indices in csr:
        row_sum = 0.0
        row_sq_sum = 0.0
        n_nnz = len(values)
        
        vectorize(8)
        for j in range(n_nnz):
            val = values[j]
            row_sum += val
            row_sq_sum += val * val
        
        mu = row_sum / N
        var = 0.0
        if denom > 0.0:
            var = (row_sq_sum - row_sum * mu) / denom
        if var < 0.0:
            var = 0.0
        
        out_means[row_idx] = mu
        out_vars[row_idx] = var
        row_idx += 1
    
    return out_means, out_vars


@parallel_jit
def compute_clipped_moments(csr: CSR, clip_vals: np.ndarray) -> tuple:
    """Compute per-row mean and variance with clipping for VST.
    
    Args:
        csr: CSR sparse matrix (CSRF32 or CSRF64)
        clip_vals: Clip values per row
    
    Returns:
        (means, vars): Per-row means and variances
    """
    n_rows = csr.nrows
    N = float(csr.ncols)
    N_minus_1 = N - 1.0
    
    assume(n_rows > 0)
    assume(len(clip_vals) >= n_rows)
    
    out_means = np.empty(n_rows, dtype=np.float64)
    out_vars = np.empty(n_rows, dtype=np.float64)
    
    row_idx = 0
    for values, indices in csr:
        clip = clip_vals[row_idx]
        row_sum = 0.0
        row_sq_sum = 0.0
        n_nnz = len(values)
        
        vectorize(8)
        for j in range(n_nnz):
            val = values[j]
            if val > clip:
                val = clip
            row_sum += val
            row_sq_sum += val * val
        
        mu = row_sum / N
        var = 0.0
        if N > 1.0:
            var = (row_sq_sum - N * mu * mu) / N_minus_1
        if var < 0.0:
            var = 0.0
        
        out_means[row_idx] = mu
        out_vars[row_idx] = var
        row_idx += 1
    
    return out_means, out_vars


# =============================================================================
# Complete HVG Selection
# =============================================================================

@parallel_jit
def select_hvg_by_dispersion(csr: CSR, n_top: int) -> tuple:
    """Select highly variable genes by dispersion.
    
    Args:
        csr: CSR sparse matrix (genes x cells)
        n_top: Number of top genes to select
    
    Returns:
        (indices, mask, dispersions): Selected gene info
    """
    EPSILON = 1e-12
    n_rows = csr.nrows
    N = float(csr.ncols)
    
    assume(n_rows > 0)
    assume(n_top > 0)
    assume(n_top <= n_rows)
    
    out_dispersions = np.empty(n_rows, dtype=np.float64)
    
    # Compute dispersions
    row_idx = 0
    for values, indices in csr:
        row_sum = 0.0
        row_sq_sum = 0.0
        n_nnz = len(values)
        
        vectorize(8)
        for j in range(n_nnz):
            val = values[j]
            row_sum += val
            row_sq_sum += val * val
        
        mu = row_sum / N
        var = (row_sq_sum - row_sum * mu) / (N - 1.0) if N > 1.0 else 0.0
        if var < 0.0:
            var = 0.0
        
        if mu > EPSILON:
            out_dispersions[row_idx] = var / mu
        else:
            out_dispersions[row_idx] = 0.0
        
        row_idx += 1
    
    # Select top k (sequential)
    out_indices = np.empty(n_top, dtype=np.int64)
    out_mask = np.zeros(n_rows, dtype=np.uint8)
    
    idx_arr = np.arange(n_rows, dtype=np.int64)
    
    for i in range(n_top):
        max_idx = i
        max_val = out_dispersions[idx_arr[i]]
        
        for j in range(i + 1, n_rows):
            if out_dispersions[idx_arr[j]] > max_val:
                max_idx = j
                max_val = out_dispersions[idx_arr[j]]
        
        if max_idx != i:
            tmp = idx_arr[i]
            idx_arr[i] = idx_arr[max_idx]
            idx_arr[max_idx] = tmp
        
        out_indices[i] = idx_arr[i]
        out_mask[idx_arr[i]] = 1
    
    return out_indices, out_mask, out_dispersions
