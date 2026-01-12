"""Regression Functions for HVG Selection - FULLY OPTIMIZED.

Provides Numba-optimized regression algorithms:
    - loess_fit: LOESS (Locally Estimated Scatterplot Smoothing)
    - weighted_polyfit: Weighted polynomial fitting (degree 1-2)

Optimization Techniques Applied:
    1. ALL constants INLINED
    2. boundscheck=False, nogil=True on ALL functions
    3. math.* functions (compile to single instructions)
    4. Pre-compute reciprocals
    5. vectorize(8), interleave(4) hints
    6. [OPT-7] Fused tricube weight computation (one pass)
    7. [OPT-9] vectorize in weighted_polyfit_2
    8. FULLY PARALLELIZED with prange
"""

import math
import numpy as np
from numba import njit, prange

from biosparse.optim import (
    parallel_jit, fast_jit, assume, vectorize, 
    interleave, unroll, likely, unlikely
)

__all__ = [
    'loess_fit',
    'loess_fit_sorted',
    'loess_fit_parallel',
    'weighted_polyfit_1',
    'weighted_polyfit_2',
    'tricube_weight',
]


# =============================================================================
# Weight Functions - [OPT-7] Fused tricube computation
# =============================================================================

@fast_jit(cache=True, inline='always', boundscheck=False)
def tricube_weight(d: float) -> float:
    """Tricube weight: w(d) = (1 - |d|^3)^3 for |d| < 1, else 0."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    ONE = 1.0
    
    if d >= ONE:
        return ZERO
    if d <= ZERO:
        return ONE
    
    d3 = d * d * d
    t = ONE - d3
    return t * t * t


@fast_jit(cache=True, inline='always', boundscheck=False)
def _compute_tricube_weights_fused(
    x: np.ndarray,
    x_i: float,
    neighbor_indices: np.ndarray,
    k: int,
    weights: np.ndarray,
    distances: np.ndarray
) -> float:
    """Compute tricube weights in single pass. [OPT-7]
    
    Fuses distance computation and weight calculation:
    1. First, compute all distances and find max (single pass)
    2. Then, compute weights using cached distances (no recomputation)
    """
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    ONE = 1.0
    
    # Single pass: compute distances and find max
    max_dist = ZERO
    
    vectorize(8)
    for j in range(k):
        idx = neighbor_indices[j]
        dist = math.fabs(x[idx] - x_i)
        distances[j] = dist
        if dist > max_dist:
            max_dist = dist
    
    # Compute weights using cached distances
    if max_dist > ZERO:
        inv_max = ONE / max_dist
        
        vectorize(8)
        interleave(4)
        for j in range(k):
            d = distances[j] * inv_max
            # Inline tricube to avoid function call
            if d >= ONE:
                weights[j] = ZERO
            else:
                d3 = d * d * d
                t = ONE - d3
                weights[j] = t * t * t
    else:
        vectorize(8)
        for j in range(k):
            weights[j] = ONE
    
    return max_dist


# =============================================================================
# Weighted Polynomial Fitting - [OPT-9] vectorize added to polyfit_2
# =============================================================================

@fast_jit(cache=True, inline='always', boundscheck=False)
def weighted_polyfit_1(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, n: int
) -> tuple:
    """Weighted linear regression: y = a + b*x."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    ONE = 1.0
    EPS = 1e-15
    
    sw = ZERO
    swx = ZERO
    swy = ZERO
    swxx = ZERO
    swxy = ZERO
    
    vectorize(8)
    interleave(4)
    for i in range(n):
        w = weights[i]
        xi = x[i]
        yi = y[i]
        wxi = w * xi
        sw += w
        swx += wxi
        swy += w * yi
        swxx += wxi * xi
        swxy += wxi * yi
    
    det = sw * swxx - swx * swx
    
    if math.fabs(det) < EPS:
        return swy / sw if sw > ZERO else ZERO, ZERO
    
    inv_det = ONE / det
    a = (swxx * swy - swx * swxy) * inv_det
    b = (sw * swxy - swx * swy) * inv_det
    
    return a, b


@fast_jit(cache=True, inline='always', boundscheck=False)
def weighted_polyfit_2(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, n: int
) -> tuple:
    """Weighted quadratic regression: y = a + b*x + c*x^2. [OPT-9] vectorized."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    ONE = 1.0
    EPS = 1e-15
    
    sw = ZERO
    swx = ZERO
    swx2 = ZERO
    swx3 = ZERO
    swx4 = ZERO
    swy = ZERO
    swxy = ZERO
    swx2y = ZERO
    
    # [OPT-9] Add vectorize hint
    vectorize(8)
    interleave(4)
    for i in range(n):
        w = weights[i]
        xi = x[i]
        yi = y[i]
        x2 = xi * xi
        wx2 = w * x2
        wxi = w * xi
        
        sw += w
        swx += wxi
        swx2 += wx2
        swx3 += wx2 * xi
        swx4 += wx2 * x2
        swy += w * yi
        swxy += wxi * yi
        swx2y += wx2 * yi
    
    # Cramer's rule for 3x3 system
    det = (sw * (swx2 * swx4 - swx3 * swx3)
           - swx * (swx * swx4 - swx3 * swx2)
           + swx2 * (swx * swx3 - swx2 * swx2))
    
    if math.fabs(det) < EPS:
        a, b = weighted_polyfit_1(x, y, weights, n)
        return a, b, ZERO
    
    inv_det = ONE / det
    
    a = ((swy * (swx2 * swx4 - swx3 * swx3)
          - swx * (swxy * swx4 - swx3 * swx2y)
          + swx2 * (swxy * swx3 - swx2 * swx2y)) * inv_det)
    
    b = ((sw * (swxy * swx4 - swx3 * swx2y)
          - swy * (swx * swx4 - swx3 * swx2)
          + swx2 * (swx * swx2y - swxy * swx2)) * inv_det)
    
    c = ((sw * (swx2 * swx2y - swxy * swx3)
          - swx * (swx * swx2y - swxy * swx2)
          + swy * (swx * swx3 - swx2 * swx2)) * inv_det)
    
    return a, b, c


# =============================================================================
# K-Nearest Neighbors (Binary Search + Expansion)
# =============================================================================

@fast_jit(cache=True, inline='always', boundscheck=False)
def _find_k_nearest_sorted(
    x_sorted: np.ndarray, x_i: float, n: int, k: int,
    neighbor_indices: np.ndarray
) -> None:
    """Find k nearest neighbors in sorted array. O(log n + k)."""
    assume(n > 0)
    assume(k > 0)
    assume(k <= n)
    
    # Binary search for insertion point
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) >> 1
        if x_sorted[mid] < x_i:
            lo = mid + 1
        else:
            hi = mid
    
    # Expand from insertion point
    left = lo - 1
    right = lo
    count = 0
    
    while count < k:
        if left < 0:
            neighbor_indices[count] = right
            right += 1
        elif right >= n:
            neighbor_indices[count] = left
            left -= 1
        else:
            d_left = x_i - x_sorted[left]
            d_right = x_sorted[right] - x_i
            if d_left <= d_right:
                neighbor_indices[count] = left
                left -= 1
            else:
                neighbor_indices[count] = right
                right += 1
        count += 1


# =============================================================================
# LOESS - FULLY PARALLELIZED + OPTIMIZED
# =============================================================================

@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def loess_fit_sorted(
    x: np.ndarray, y: np.ndarray, span: float = 0.3, degree: int = 2
) -> np.ndarray:
    """LOESS fit for sorted x. PARALLEL over all points."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    ONE = 1.0
    
    n = len(x)
    assume(n > 0)
    assume(span > ZERO)
    assume(span <= ONE)
    
    # Number of neighbors
    k = int(span * float(n))
    k_min = degree + 1
    if k < k_min:
        k = k_min
    if k > n:
        k = n
    
    assume(k > 0)
    
    fitted = np.empty(n, dtype=np.float64)
    
    # PARALLEL: Each point's local fit is independent
    for i in prange(n):
        x_i = x[i]
        
        # Thread-local arrays
        neighbor_idx = np.empty(k, dtype=np.int64)
        weights = np.empty(k, dtype=np.float64)
        distances = np.empty(k, dtype=np.float64)
        x_local = np.empty(k, dtype=np.float64)
        y_local = np.empty(k, dtype=np.float64)
        
        # Find k nearest
        _find_k_nearest_sorted(x, x_i, n, k, neighbor_idx)
        
        # Extract local data
        vectorize(8)
        for j in range(k):
            idx = neighbor_idx[j]
            x_local[j] = x[idx]
            y_local[j] = y[idx]
        
        # Compute weights [OPT-7] fused
        _compute_tricube_weights_fused(x, x_i, neighbor_idx, k, weights, distances)
        
        # Polynomial fit
        if degree == 1:
            a, b = weighted_polyfit_1(x_local, y_local, weights, k)
            fitted[i] = a + b * x_i
        else:
            a, b, c = weighted_polyfit_2(x_local, y_local, weights, k)
            fitted[i] = a + b * x_i + c * x_i * x_i
    
    return fitted


@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def loess_fit_parallel(
    x: np.ndarray, y: np.ndarray, span: float = 0.3, degree: int = 2
) -> np.ndarray:
    """Parallel LOESS for unsorted x. Sort once, fit parallel."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    ONE = 1.0
    
    n = len(x)
    assume(n > 0)
    assume(span > ZERO)
    assume(span <= ONE)
    
    k = int(span * float(n))
    k_min = degree + 1
    if k < k_min:
        k = k_min
    if k > n:
        k = n
    
    assume(k > 0)
    
    # Sort once (sequential, but only O(n log n))
    sort_idx = np.argsort(x)
    x_sorted = np.empty(n, dtype=np.float64)
    y_sorted = np.empty(n, dtype=np.float64)
    
    vectorize(8)
    for i in range(n):
        idx = sort_idx[i]
        x_sorted[i] = x[idx]
        y_sorted[i] = y[idx]
    
    # PARALLEL fit
    fitted_sorted = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        x_i = x_sorted[i]
        
        neighbor_idx = np.empty(k, dtype=np.int64)
        weights = np.empty(k, dtype=np.float64)
        distances = np.empty(k, dtype=np.float64)
        x_local = np.empty(k, dtype=np.float64)
        y_local = np.empty(k, dtype=np.float64)
        
        _find_k_nearest_sorted(x_sorted, x_i, n, k, neighbor_idx)
        
        vectorize(8)
        for j in range(k):
            idx = neighbor_idx[j]
            x_local[j] = x_sorted[idx]
            y_local[j] = y_sorted[idx]
        
        # [OPT-7] Fused weight computation
        _compute_tricube_weights_fused(x_sorted, x_i, neighbor_idx, k, weights, distances)
        
        if degree == 1:
            a, b = weighted_polyfit_1(x_local, y_local, weights, k)
            fitted_sorted[i] = a + b * x_i
        else:
            a, b, c = weighted_polyfit_2(x_local, y_local, weights, k)
            fitted_sorted[i] = a + b * x_i + c * x_i * x_i
    
    # Unsort results
    fitted = np.empty(n, dtype=np.float64)
    for i in range(n):
        fitted[sort_idx[i]] = fitted_sorted[i]
    
    return fitted


# Alias for convenience
loess_fit = loess_fit_parallel


# =============================================================================
# HVG Utility Functions - PARALLELIZED + boundscheck=False
# =============================================================================

@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def compute_vst_clip_values(
    means: np.ndarray, fitted_log_var: np.ndarray, n_cells: int
) -> np.ndarray:
    """VST clip values: clip = reg_std * sqrt(n) + mean."""
    # === INLINE CONSTANTS ===
    TEN = 10.0
    
    n = len(means)
    assume(n > 0)
    assume(n_cells > 0)
    
    clip_vals = np.empty(n, dtype=np.float64)
    sqrt_n = math.sqrt(float(n_cells))
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        reg_std = math.sqrt(math.pow(TEN, fitted_log_var[i]))
        clip_vals[i] = reg_std * sqrt_n + means[i]
    
    return clip_vals


@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def compute_normalized_variance(
    means: np.ndarray,
    sum_clipped: np.ndarray,
    sum_sq_clipped: np.ndarray,
    reg_std: np.ndarray,
    n_cells: int
) -> np.ndarray:
    """Normalized variance for Seurat V3."""
    # === INLINE CONSTANTS ===
    EPS = 1e-12
    ZERO = 0.0
    ONE = 1.0
    TWO = 2.0
    
    n_genes = len(means)
    assume(n_genes > 0)
    assume(n_cells > 1)
    
    norm_var = np.empty(n_genes, dtype=np.float64)
    n_f = float(n_cells)
    inv_nm1 = ONE / (n_f - ONE)
    
    vectorize(8)
    interleave(4)
    for i in prange(n_genes):
        mean = means[i]
        std = reg_std[i]
        
        if likely(std > EPS):
            std_sq = std * std
            inv_factor = inv_nm1 / std_sq
            val = n_f * mean * mean + sum_sq_clipped[i] - TWO * sum_clipped[i] * mean
            norm_var[i] = inv_factor * val
        else:
            norm_var[i] = ZERO
    
    return norm_var


@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def compute_reg_std_and_clip(
    fitted_log_var: np.ndarray, means: np.ndarray, n_cells: int
) -> tuple:
    """Compute reg_std and clip_vals in single parallel pass."""
    # === INLINE CONSTANTS ===
    TEN = 10.0
    
    n = len(means)
    assume(n > 0)
    assume(n_cells > 0)
    
    reg_std = np.empty(n, dtype=np.float64)
    clip_vals = np.empty(n, dtype=np.float64)
    sqrt_n = math.sqrt(float(n_cells))
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        rs = math.sqrt(math.pow(TEN, fitted_log_var[i]))
        reg_std[i] = rs
        clip_vals[i] = rs * sqrt_n + means[i]
    
    return reg_std, clip_vals
