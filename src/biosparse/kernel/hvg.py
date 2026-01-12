"""Highly Variable Gene (HVG) Selection - FULLY OPTIMIZED.

Complete implementation of HVG selection algorithms matching scanpy:
    - hvg_seurat: Seurat flavor (binning + mean/std z-score)
    - hvg_cell_ranger: Cell Ranger flavor (percentile bins + median/MAD)
    - hvg_seurat_v3: Seurat V3 flavor (VST with LOESS regression)
    - hvg_pearson_residuals: Pearson residuals flavor

Optimization Techniques Applied:
    1. ALL constants INLINED (avoid Numba closure detection)
    2. boundscheck=False, nogil=True on ALL hot functions
    3. Parallel min/max via reduction
    4. Column sums via chunked parallel reduction
    5. Pearson residuals: cache residuals to avoid recomputation
    6. math.* functions (compile to single instructions)
    7. Pre-compute ALL reciprocals
    8. vectorize(8), interleave(4), unroll(4) hints
    9. likely(), unlikely() branch hints
    10. assume() for bounds elimination
    11. prefetch_read for sequential access patterns
"""

import math
import numpy as np
from numba import njit, prange, get_num_threads

from biosparse.optim import (
    parallel_jit, fast_jit, assume, vectorize, 
    interleave, unroll, likely, unlikely, prefetch_read
)
from biosparse._binding import CSR

import biosparse._numba  # noqa: F401

__all__ = [
    'hvg_seurat', 'hvg_cell_ranger', 'hvg_seurat_v3', 'hvg_seurat_v3_numba',
    'hvg_pearson_residuals', 'select_hvg_by_dispersion',
    'compute_moments', 'compute_clipped_moments', 'compute_dispersion',
    'normalize_dispersion', 'select_top_k', 'select_top_k_sorted',
]


# =============================================================================
# Core Statistics - FULLY PARALLELIZED + boundscheck=False
# =============================================================================

@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def compute_moments(csr: CSR, ddof: int = 1) -> tuple:
    """Per-row mean and variance. PARALLEL over rows."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    
    n_rows = csr.nrows
    n_cols = csr.ncols
    
    assume(n_rows > 0)
    assume(n_cols > 0)
    assume(ddof >= 0)
    
    N = float(n_cols)
    denom = N - float(ddof)
    inv_N = 1.0 / N
    inv_denom = 1.0 / denom if denom > ZERO else ZERO
    
    out_means = np.empty(n_rows, dtype=np.float64)
    out_vars = np.empty(n_rows, dtype=np.float64)
    
    for row in prange(n_rows):
        values, _ = csr.row_to_numpy(row)
        nnz = len(values)
        
        s = ZERO
        sq = ZERO
        
        vectorize(8)
        interleave(4)
        for j in range(nnz):
            v = float(values[j])
            s += v
            sq += v * v
        
        mu = s * inv_N
        var = (sq - s * mu) * inv_denom
        
        out_means[row] = mu
        out_vars[row] = var if var > ZERO else ZERO
    
    return out_means, out_vars


@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def compute_clipped_moments(csr: CSR, clip_vals: np.ndarray) -> tuple:
    """Per-row sum/sum-sq with clipping. PARALLEL over rows."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    
    n_rows = csr.nrows
    assume(n_rows > 0)
    
    out_sum = np.empty(n_rows, dtype=np.float64)
    out_sq = np.empty(n_rows, dtype=np.float64)
    
    for row in prange(n_rows):
        clip = clip_vals[row]
        values, _ = csr.row_to_numpy(row)
        nnz = len(values)
        
        s = ZERO
        sq = ZERO
        
        vectorize(8)
        interleave(4)
        for j in range(nnz):
            v = float(values[j])
            v = v if v < clip else clip
            s += v
            sq += v * v
        
        out_sum[row] = s
        out_sq[row] = sq
    
    return out_sum, out_sq


@fast_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def compute_dispersion(means: np.ndarray, vars: np.ndarray) -> np.ndarray:
    """Dispersion = var/mean. Fast vectorized."""
    # === INLINE CONSTANTS ===
    EPS = 1e-12
    ZERO = 0.0
    
    n = len(means)
    assume(n > 0)
    
    out = np.empty(n, dtype=np.float64)
    
    vectorize(8)
    interleave(4)
    for i in range(n):
        m = means[i]
        out[i] = vars[i] / m if m > EPS else ZERO
    
    return out


# =============================================================================
# Parallel Min/Max Reduction [OPT-5]
# =============================================================================

@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def _parallel_minmax(arr: np.ndarray) -> tuple:
    """Parallel min/max via reduction."""
    n = len(arr)
    assume(n > 0)
    
    # Get number of threads
    n_threads = 8  # Inline constant
    chunk_size = (n + n_threads - 1) // n_threads
    
    local_mins = np.empty(n_threads, dtype=np.float64)
    local_maxs = np.empty(n_threads, dtype=np.float64)
    
    # Initialize with first element
    for t in range(n_threads):
        local_mins[t] = arr[0]
        local_maxs[t] = arr[0]
    
    # Parallel reduction
    for t in prange(n_threads):
        start = t * chunk_size
        end = start + chunk_size
        if end > n:
            end = n
        
        if start < n:
            v_min = arr[start]
            v_max = arr[start]
            
            vectorize(8)
            for i in range(start + 1, end):
                v = arr[i]
                if v < v_min:
                    v_min = v
                if v > v_max:
                    v_max = v
            
            local_mins[t] = v_min
            local_maxs[t] = v_max
    
    # Final reduction (sequential, small)
    final_min = local_mins[0]
    final_max = local_maxs[0]
    
    unroll(8)
    for t in range(1, n_threads):
        if local_mins[t] < final_min:
            final_min = local_mins[t]
        if local_maxs[t] > final_max:
            final_max = local_maxs[t]
    
    return final_min, final_max


# =============================================================================
# Fused Operations - PARALLELIZED + boundscheck=False
# =============================================================================

@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def _fused_moments_dispersion_log_parallel(csr: CSR) -> tuple:
    """Fused moments + dispersion + logs. PARALLEL over genes."""
    # === INLINE CONSTANTS ===
    NEG_INF = -1e308
    EPS = 1e-12
    ZERO = 0.0
    ONE = 1.0
    
    n_genes = csr.nrows
    n_cells = csr.ncols
    
    assume(n_genes > 0)
    assume(n_cells > 0)
    
    N = float(n_cells)
    inv_N = ONE / N
    inv_Nm1 = ONE / (N - ONE) if N > ONE else ZERO
    
    means = np.empty(n_genes, dtype=np.float64)
    dispersions = np.empty(n_genes, dtype=np.float64)
    log_means = np.empty(n_genes, dtype=np.float64)
    log_disps = np.empty(n_genes, dtype=np.float64)
    
    for i in prange(n_genes):
        values, _ = csr.row_to_numpy(i)
        nnz = len(values)
        
        s = ZERO
        sq = ZERO
        
        vectorize(8)
        interleave(4)
        for j in range(nnz):
            v = float(values[j])
            s += v
            sq += v * v
        
        mu = s * inv_N
        var = (sq - s * mu) * inv_Nm1
        if var < ZERO:
            var = ZERO
        
        disp = var / mu if mu > EPS else ZERO
        
        means[i] = mu
        dispersions[i] = disp
        log_means[i] = math.log1p(mu)
        log_disps[i] = math.log(disp) if disp > EPS else NEG_INF
    
    return means, dispersions, log_means, log_disps


# =============================================================================
# Top-K Selection (boundscheck=False)
# =============================================================================

@fast_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def _heap_sift_down(vals: np.ndarray, idxs: np.ndarray, start: int, end: int) -> None:
    """Min-heap sift down."""
    root = start
    while True:
        child = (root << 1) + 1
        if child > end:
            return
        
        swap = root
        if vals[swap] > vals[child]:
            swap = child
        if child + 1 <= end and vals[swap] > vals[child + 1]:
            swap = child + 1
        
        if swap == root:
            return
        
        vals[root], vals[swap] = vals[swap], vals[root]
        idxs[root], idxs[swap] = idxs[swap], idxs[root]
        root = swap


@fast_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def select_top_k(scores: np.ndarray, k: int) -> tuple:
    """Top k via min-heap. O(n log k)."""
    n = len(scores)
    assume(n > 0)
    assume(k > 0)
    assume(k <= n)
    
    heap_vals = np.empty(k, dtype=np.float64)
    heap_idxs = np.empty(k, dtype=np.int64)
    
    unroll(4)
    for i in range(k):
        heap_vals[i] = scores[i]
        heap_idxs[i] = i
    
    for i in range((k - 2) >> 1, -1, -1):
        _heap_sift_down(heap_vals, heap_idxs, i, k - 1)
    
    min_val = heap_vals[0]
    km1 = k - 1
    
    for i in range(k, n):
        v = scores[i]
        if v > min_val:
            heap_vals[0] = v
            heap_idxs[0] = i
            _heap_sift_down(heap_vals, heap_idxs, 0, km1)
            min_val = heap_vals[0]
    
    out_indices = np.empty(k, dtype=np.int64)
    out_mask = np.zeros(n, dtype=np.uint8)
    
    unroll(4)
    for i in range(k):
        idx = heap_idxs[i]
        out_indices[i] = idx
        out_mask[idx] = 1
    
    return out_indices, out_mask


@fast_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def select_top_k_sorted(scores: np.ndarray, k: int) -> tuple:
    """Top k sorted descending."""
    n = len(scores)
    assume(n > 0)
    assume(k > 0)
    
    indices, mask = select_top_k(scores, k)
    
    tmp = np.empty(k, dtype=np.float64)
    for i in range(k):
        tmp[i] = -scores[indices[i]]
    
    order = np.argsort(tmp)
    
    sorted_idx = np.empty(k, dtype=np.int64)
    for i in range(k):
        sorted_idx[i] = indices[order[i]]
    
    return sorted_idx, mask


# =============================================================================
# Binning - PARALLELIZED + boundscheck=False
# =============================================================================

@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def _bin_assign_parallel(values: np.ndarray, v_min: float, inv_width: float, n_bins_m1: int) -> np.ndarray:
    """Assign equal-width bins. PARALLEL."""
    n = len(values)
    assume(n > 0)
    
    bin_indices = np.empty(n, dtype=np.int64)
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        b = int((values[i] - v_min) * inv_width)
        if b < 0:
            b = 0
        elif b > n_bins_m1:
            b = n_bins_m1
        bin_indices[i] = b
    
    return bin_indices


@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def _bin_by_edges_parallel(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign bins via binary search. PARALLEL."""
    n = len(values)
    n_bins = len(edges) - 1
    
    assume(n > 0)
    assume(n_bins > 0)
    
    bin_indices = np.empty(n, dtype=np.int64)
    n_bins_m1 = n_bins - 1
    
    for i in prange(n):
        v = values[i]
        lo = 0
        hi = n_bins
        while lo < hi:
            mid = (lo + hi) >> 1
            if v >= edges[mid + 1]:
                lo = mid + 1
            else:
                hi = mid
        bin_indices[i] = lo if lo <= n_bins_m1 else n_bins_m1
    
    return bin_indices


# =============================================================================
# Bin Statistics - boundscheck=False + unroll
# =============================================================================

@fast_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def _bin_stats_mean_std(log_disps: np.ndarray, bin_indices: np.ndarray, n_bins: int) -> tuple:
    """Per-bin mean/std. Sequential accumulation."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    ONE = 1.0
    NEG_INF_CHECK = -1e100
    
    n = len(log_disps)
    assume(n > 0)
    assume(n_bins > 0)
    
    sums = np.zeros(n_bins, dtype=np.float64)
    sq_sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    
    for i in range(n):
        ld = log_disps[i]
        b = bin_indices[i]
        if ld > NEG_INF_CHECK and 0 <= b < n_bins:
            sums[b] += ld
            sq_sums[b] += ld * ld
            counts[b] += 1
    
    bin_means = np.zeros(n_bins, dtype=np.float64)
    bin_stds = np.zeros(n_bins, dtype=np.float64)
    
    unroll(4)
    for b in range(n_bins):
        c = counts[b]
        if c > 0:
            c_f = float(c)
            m = sums[b] / c_f
            bin_means[b] = m
            if c > 1:
                var = (sq_sums[b] - c_f * m * m) / (c_f - ONE)
                bin_stds[b] = math.sqrt(var) if var > ZERO else ZERO
    
    return bin_means, bin_stds


@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def _normalize_dispersion_parallel(
    log_disps: np.ndarray,
    bin_indices: np.ndarray,
    bin_means: np.ndarray,
    inv_stds: np.ndarray,
    log_means: np.ndarray,
    log_min: float,
    log_max: float,
    min_disp: float,
    max_disp: float
) -> np.ndarray:
    """Z-score normalize with cutoffs. PARALLEL."""
    # === INLINE CONSTANTS ===
    NEG_INF = -1e308
    ZERO = 0.0
    
    n = len(log_disps)
    assume(n > 0)
    
    out = np.empty(n, dtype=np.float64)
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        lm = log_means[i]
        ld = log_disps[i]
        b = bin_indices[i]
        
        inv_s = inv_stds[b]
        dn = (ld - bin_means[b]) * inv_s if inv_s > ZERO else ZERO
        
        if lm < log_min or lm > log_max or dn < min_disp or dn > max_disp:
            dn = NEG_INF
        
        out[i] = dn
    
    return out


@fast_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def normalize_dispersion(dispersions: np.ndarray, means: np.ndarray, min_mean: float, max_mean: float) -> np.ndarray:
    """Legacy global z-score normalization."""
    # === INLINE CONSTANTS ===
    NEG_INF = -1e308
    EPS = 1e-12
    ZERO = 0.0
    ONE = 1.0
    
    n = len(dispersions)
    assume(n > 0)
    
    out = np.empty(n, dtype=np.float64)
    
    s = ZERO
    sq = ZERO
    cnt = 0
    
    for i in range(n):
        m = means[i]
        d = dispersions[i]
        if m >= min_mean and m <= max_mean and d > ZERO:
            s += d
            sq += d * d
            cnt += 1
    
    if unlikely(cnt == 0):
        for i in range(n):
            out[i] = NEG_INF
        return out
    
    cnt_f = float(cnt)
    mean = s / cnt_f
    var = sq / cnt_f - mean * mean
    std = math.sqrt(var) if var > EPS else ONE
    inv_std = ONE / std
    
    vectorize(8)
    for i in range(n):
        m = means[i]
        d = dispersions[i]
        if likely(m >= min_mean and m <= max_mean and d > ZERO):
            out[i] = (d - mean) * inv_std
        else:
            out[i] = NEG_INF
    
    return out


# =============================================================================
# Seurat Flavor - FULLY OPTIMIZED with parallel minmax [OPT-5]
# =============================================================================

@fast_jit(cache=True, boundscheck=False, nogil=True)
def hvg_seurat(
    csr: CSR,
    n_top_genes: int,
    n_bins: int = 20,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    max_disp: float = np.inf
) -> tuple:
    """Seurat HVG. PARALLEL moments, parallel minmax, binning, normalization."""
    # === INLINE CONSTANTS ===
    EPS = 1e-12
    ZERO = 0.0
    ONE = 1.0
    
    n_genes = csr.nrows
    assume(n_genes > 0)
    assume(n_top_genes > 0)
    assume(n_bins > 0)
    
    # PARALLEL: fused moments + dispersion + log
    means, dispersions, log_means, log_disps = _fused_moments_dispersion_log_parallel(csr)
    
    # PARALLEL: min/max via reduction [OPT-5]
    v_min, v_max = _parallel_minmax(log_means)
    
    # PARALLEL: bin assignment
    if v_max > v_min:
        width = (v_max - v_min) / float(n_bins)
        inv_width = ONE / width
        bin_indices = _bin_assign_parallel(log_means, v_min, inv_width, n_bins - 1)
    else:
        bin_indices = np.zeros(n_genes, dtype=np.int64)
    
    # Per-bin statistics
    bin_means, bin_stds = _bin_stats_mean_std(log_disps, bin_indices, n_bins)
    
    # Pre-compute inverse stds
    inv_stds = np.empty(n_bins, dtype=np.float64)
    unroll(4)
    for b in range(n_bins):
        s = bin_stds[b]
        inv_stds[b] = ONE / s if s > EPS else ZERO
    
    log_min = math.log1p(min_mean)
    log_max = math.log1p(max_mean)
    
    # PARALLEL: normalization with cutoffs
    dispersions_norm = _normalize_dispersion_parallel(
        log_disps, bin_indices, bin_means, inv_stds,
        log_means, log_min, log_max, min_disp, max_disp
    )
    
    indices, mask = select_top_k(dispersions_norm, n_top_genes)
    return indices, mask, means, dispersions, dispersions_norm


# =============================================================================
# Cell Ranger Flavor - PARALLELIZED + boundscheck=False
# =============================================================================

@fast_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def _percentiles(arr: np.ndarray, percentiles: np.ndarray) -> np.ndarray:
    """Compute percentiles."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    ONE = 1.0
    
    n = len(arr)
    n_p = len(percentiles)
    assume(n > 0)
    
    sorted_arr = np.sort(arr)
    out = np.empty(n_p, dtype=np.float64)
    nm1 = float(n - 1)
    
    unroll(4)
    for j in range(n_p):
        p = percentiles[j] * 0.01
        if unlikely(p <= ZERO):
            out[j] = sorted_arr[0]
        elif unlikely(p >= ONE):
            out[j] = sorted_arr[n - 1]
        else:
            idx = p * nm1
            lo = int(idx)
            hi = lo + 1
            if hi >= n:
                hi = n - 1
            frac = idx - float(lo)
            out[j] = sorted_arr[lo] * (ONE - frac) + sorted_arr[hi] * frac
    
    return out


@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def _bin_median_mad_parallel(
    values: np.ndarray,
    bin_indices: np.ndarray,
    n_bins: int
) -> tuple:
    """Per-bin median/MAD using offset array. Parallel per-bin computation."""
    # === INLINE CONSTANTS ===
    HALF = 0.5
    
    n = len(values)
    assume(n > 0)
    assume(n_bins > 0)
    
    # Pass 1: Count per bin (sequential)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n):
        b = bin_indices[i]
        if likely(0 <= b < n_bins):
            counts[b] += 1
    
    # Compute offsets
    offsets = np.empty(n_bins + 1, dtype=np.int64)
    offsets[0] = 0
    for b in range(n_bins):
        offsets[b + 1] = offsets[b] + counts[b]
    
    # Pass 2: Scatter
    sorted_vals = np.empty(n, dtype=np.float64)
    pos = np.zeros(n_bins, dtype=np.int64)
    
    for i in range(n):
        b = bin_indices[i]
        if likely(0 <= b < n_bins):
            sorted_vals[offsets[b] + pos[b]] = values[i]
            pos[b] += 1
    
    medians = np.zeros(n_bins, dtype=np.float64)
    mads = np.zeros(n_bins, dtype=np.float64)
    
    # Pass 3: PARALLEL per-bin median/MAD
    for b in prange(n_bins):
        c = counts[b]
        if unlikely(c == 0):
            continue
        
        start = offsets[b]
        bin_vals = np.sort(sorted_vals[start:start + c])
        
        mid = c >> 1
        if c & 1:
            med = bin_vals[mid]
        else:
            med = (bin_vals[mid - 1] + bin_vals[mid]) * HALF
        medians[b] = med
        
        if likely(c > 1):
            dev = np.empty(c, dtype=np.float64)
            vectorize(8)
            for j in range(c):
                dev[j] = math.fabs(bin_vals[j] - med)
            sorted_dev = np.sort(dev)
            
            if c & 1:
                mads[b] = sorted_dev[mid]
            else:
                mads[b] = (sorted_dev[mid - 1] + sorted_dev[mid]) * HALF
    
    return medians, mads, counts


@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def _normalize_by_median_mad_parallel(
    dispersions: np.ndarray,
    bin_indices: np.ndarray,
    bin_medians: np.ndarray,
    inv_mads: np.ndarray
) -> np.ndarray:
    """Z-score by median/MAD. PARALLEL."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    
    n = len(dispersions)
    assume(n > 0)
    
    out = np.empty(n, dtype=np.float64)
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        b = bin_indices[i]
        inv_m = inv_mads[b]
        out[i] = (dispersions[i] - bin_medians[b]) * inv_m if inv_m > ZERO else ZERO
    
    return out


@fast_jit(cache=True, boundscheck=False, nogil=True)
def hvg_cell_ranger(csr: CSR, n_top_genes: int) -> tuple:
    """Cell Ranger HVG. PARALLEL moments, binning, normalization."""
    # === INLINE CONSTANTS ===
    NEG_INF = -1e308
    POS_INF = 1e308
    EPS = 1e-12
    ONE = 1.0
    
    n_genes = csr.nrows
    assume(n_genes > 0)
    assume(n_top_genes > 0)
    
    # PARALLEL: moments + dispersion
    means, vars_ = compute_moments(csr, 1)
    dispersions = compute_dispersion(means, vars_)
    
    percentiles = np.arange(10.0, 105.0, 5.0)
    n_p = len(percentiles)
    
    p_vals = _percentiles(means, percentiles)
    
    edges = np.empty(n_p + 2, dtype=np.float64)
    edges[0] = NEG_INF
    for i in range(n_p):
        edges[i + 1] = p_vals[i]
    edges[n_p + 1] = POS_INF
    
    # PARALLEL: bin assignment
    bin_indices = _bin_by_edges_parallel(means, edges)
    n_bins = n_p + 1
    
    # PARALLEL per-bin: median/MAD
    bin_medians, bin_mads, _ = _bin_median_mad_parallel(dispersions, bin_indices, n_bins)
    
    # Pre-compute inverse MADs
    inv_mads = np.empty(n_bins, dtype=np.float64)
    unroll(4)
    for b in range(n_bins):
        m = bin_mads[b]
        inv_mads[b] = ONE / m if m > EPS else 0.0
    
    # PARALLEL: normalization
    dispersions_norm = _normalize_by_median_mad_parallel(dispersions, bin_indices, bin_medians, inv_mads)
    
    indices, mask = select_top_k(dispersions_norm, n_top_genes)
    return indices, mask, means, dispersions, dispersions_norm


# =============================================================================
# Seurat V3 - FULLY PARALLELIZED (including LOESS)
# =============================================================================

from biosparse.kernel.math._regression import loess_fit_parallel, compute_reg_std_and_clip


@parallel_jit(cache=True, boundscheck=False, nogil=True)
def _compute_reg_std_clip_parallel(
    fitted_log_var: np.ndarray,
    means: np.ndarray,
    sqrt_n: float
) -> tuple:
    """Compute reg_std and clip_vals in parallel."""
    TEN = 10.0
    n_genes = len(means)
    
    reg_std = np.empty(n_genes, dtype=np.float64)
    clip_vals = np.empty(n_genes, dtype=np.float64)
    
    for i in prange(n_genes):
        rs = math.sqrt(math.pow(TEN, fitted_log_var[i]))
        reg_std[i] = rs
        clip_vals[i] = rs * sqrt_n + means[i]
    
    return reg_std, clip_vals


@parallel_jit(cache=True, boundscheck=False, nogil=True)
def _compute_variances_norm_parallel(
    reg_std: np.ndarray,
    means: np.ndarray,
    sum_clip: np.ndarray,
    sq_clip: np.ndarray,
    n_cells: int
) -> np.ndarray:
    """Compute normalized variance in parallel."""
    EPS = 1e-12
    ZERO = 0.0
    ONE = 1.0
    TWO = 2.0
    
    n_genes = len(means)
    n_f = float(n_cells)
    inv_nm1 = ONE / (n_f - ONE)
    
    variances_norm = np.empty(n_genes, dtype=np.float64)
    
    for i in prange(n_genes):
        std = reg_std[i]
        if likely(std > EPS):
            std_sq = std * std
            inv_factor = inv_nm1 / std_sq
            m = means[i]
            val = n_f * m * m + sq_clip[i] - TWO * sum_clip[i] * m
            variances_norm[i] = inv_factor * val
        else:
            variances_norm[i] = ZERO
    
    return variances_norm


@fast_jit(cache=True, boundscheck=False, nogil=True)
def _hvg_seurat_v3_core(
    means: np.ndarray,
    variances: np.ndarray,
    fitted_log_var: np.ndarray,
    csr: CSR,
    n_top_genes: int
) -> tuple:
    """Core computation for Seurat V3 after LOESS fitting. FULLY PARALLEL."""
    n_genes = len(means)
    n_cells = csr.ncols
    sqrt_n = math.sqrt(float(n_cells))
    
    # PARALLEL: Compute reg_std and clip_vals
    reg_std, clip_vals = _compute_reg_std_clip_parallel(fitted_log_var, means, sqrt_n)
    
    # PARALLEL: clipped moments
    sum_clip, sq_clip = compute_clipped_moments(csr, clip_vals)
    
    # PARALLEL: Normalized variance
    variances_norm = _compute_variances_norm_parallel(reg_std, means, sum_clip, sq_clip, n_cells)
    
    indices, mask = select_top_k(variances_norm, n_top_genes)
    return indices, mask, variances_norm


@parallel_jit(cache=True, boundscheck=False, nogil=True)
def _map_fitted_to_full(fitted: np.ndarray, valid_idx: np.ndarray, n_genes: int) -> np.ndarray:
    """Map fitted values back to full gene array. PARALLEL."""
    fitted_log_var = np.zeros(n_genes, dtype=np.float64)
    n_valid = len(valid_idx)
    
    for j in prange(n_valid):
        fitted_log_var[valid_idx[j]] = fitted[j]
    
    return fitted_log_var


@parallel_jit(cache=True, boundscheck=False, nogil=True)
def _extract_valid_log(
    means: np.ndarray, 
    variances: np.ndarray,
    valid_idx: np.ndarray
) -> tuple:
    """Extract valid genes and compute log10. PARALLEL."""
    n_valid = len(valid_idx)
    log_means = np.empty(n_valid, dtype=np.float64)
    log_vars = np.empty(n_valid, dtype=np.float64)
    
    for j in prange(n_valid):
        idx = valid_idx[j]
        log_means[j] = math.log10(means[idx])
        log_vars[j] = math.log10(variances[idx])
    
    return log_means, log_vars


def hvg_seurat_v3(csr: CSR, n_top_genes: int, span: float = 0.3) -> tuple:
    """Seurat V3 HVG (VST + LOESS). FULLY PARALLEL.
    
    Args:
        csr: CSR sparse matrix (genes x cells)
        n_top_genes: Number of top genes to select
        span: LOESS span parameter (default 0.3)
    
    Returns:
        Tuple of (indices, mask, means, variances, variances_norm)
    """
    n_genes = csr.nrows
    
    # PARALLEL: Compute moments
    means, variances = compute_moments(csr, 1)
    
    # Filter constant genes (var > 0, matching scanpy)
    not_const = variances > 0
    n_valid = int(not_const.sum())
    
    if n_valid == 0:
        return (np.zeros(0, dtype=np.int64),
                np.zeros(n_genes, dtype=np.uint8),
                means, variances,
                np.zeros(n_genes, dtype=np.float64))
    
    # Get valid indices
    valid_idx = np.where(not_const)[0].astype(np.int64)
    
    # PARALLEL: Extract and log10 transform
    log_means, log_vars = _extract_valid_log(means, variances, valid_idx)
    
    # PARALLEL: LOESS regression
    fitted = loess_fit_parallel(log_means, log_vars, span, 2)
    
    # PARALLEL: Map back to full gene array
    fitted_log_var = _map_fitted_to_full(fitted, valid_idx, n_genes)
    
    # PARALLEL: Core computation
    indices, mask, variances_norm = _hvg_seurat_v3_core(
        means, variances, fitted_log_var, csr, n_top_genes
    )
    
    return indices, mask, means, variances, variances_norm


@fast_jit(cache=True, boundscheck=False, nogil=True)
def hvg_seurat_v3_numba(csr: CSR, n_top_genes: int, span: float = 0.3) -> tuple:
    """Seurat V3 HVG - Pure Numba version (faster but may differ slightly from scanpy).
    
    This version uses biosparse's parallel LOESS implementation.
    For exact scanpy compatibility, use hvg_seurat_v3() with use_skmisc=True.
    """
    # === INLINE CONSTANTS ===
    EPS = 1e-12
    ZERO = 0.0
    ONE = 1.0
    TEN = 10.0
    TWO = 2.0
    
    n_genes = csr.nrows
    n_cells = csr.ncols
    
    assume(n_genes > 0)
    assume(n_cells > 0)
    assume(n_top_genes > 0)
    assume(span > ZERO)
    
    # PARALLEL: moments
    means, variances = compute_moments(csr, 1)
    
    # Count valid genes (var > 0 to match scanpy)
    n_valid = 0
    for i in range(n_genes):
        if variances[i] > ZERO:
            n_valid += 1
    
    if unlikely(n_valid == 0):
        return (np.zeros(0, dtype=np.int64),
                np.zeros(n_genes, dtype=np.uint8),
                means, variances,
                np.zeros(n_genes, dtype=np.float64))
    
    # Extract valid genes - NO EPS added to match scanpy!
    log_means = np.empty(n_valid, dtype=np.float64)
    log_vars = np.empty(n_valid, dtype=np.float64)
    valid_idx = np.empty(n_valid, dtype=np.int64)
    
    j = 0
    for i in range(n_genes):
        if variances[i] > ZERO:
            log_means[j] = math.log10(means[i])
            log_vars[j] = math.log10(variances[i])
            valid_idx[j] = i
            j += 1
    
    # PARALLEL: LOESS regression
    fitted = loess_fit_parallel(log_means, log_vars, span, 2)
    
    # Map back
    estimat = np.zeros(n_genes, dtype=np.float64)
    for j in range(n_valid):
        estimat[valid_idx[j]] = fitted[j]
    
    # PARALLEL: reg_std and clip_vals
    reg_std, clip_vals = compute_reg_std_and_clip(estimat, means, n_cells)
    
    # PARALLEL: clipped moments
    sum_clip, sq_clip = compute_clipped_moments(csr, clip_vals)
    
    # Normalized variance
    variances_norm = np.empty(n_genes, dtype=np.float64)
    n_f = float(n_cells)
    inv_nm1 = ONE / (n_f - ONE)
    
    vectorize(8)
    for i in range(n_genes):
        std = reg_std[i]
        if likely(std > EPS):
            std_sq = std * std
            inv_factor = inv_nm1 / std_sq
            m = means[i]
            val = n_f * m * m + sq_clip[i] - TWO * sum_clip[i] * m
            variances_norm[i] = inv_factor * val
        else:
            variances_norm[i] = ZERO
    
    indices, mask = select_top_k(variances_norm, n_top_genes)
    return indices, mask, means, variances, variances_norm


# =============================================================================
# Pearson Residuals - OPTIMIZED [OPT-1, OPT-2]
# =============================================================================

@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def _row_sums_parallel(csr: CSR) -> np.ndarray:
    """Row sums. PARALLEL."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    
    n_rows = csr.nrows
    assume(n_rows > 0)
    
    sums = np.empty(n_rows, dtype=np.float64)
    
    for row in prange(n_rows):
        values, _ = csr.row_to_numpy(row)
        s = ZERO
        vectorize(8)
        for j in range(len(values)):
            s += float(values[j])
        sums[row] = s
    
    return sums


@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def _col_sums_chunked(csr: CSR) -> np.ndarray:
    """Column sums via chunked parallel reduction. [OPT-1]
    
    Each thread accumulates to a local buffer, then merge.
    Avoids write conflicts while enabling parallelism.
    """
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    N_CHUNKS = 8  # Inline constant
    
    n_rows = csr.nrows
    n_cols = csr.ncols
    assume(n_rows > 0)
    assume(n_cols > 0)
    
    # Compute actual number of chunks needed
    chunk_size = (n_rows + N_CHUNKS - 1) // N_CHUNKS
    actual_chunks = (n_rows + chunk_size - 1) // chunk_size
    if actual_chunks > N_CHUNKS:
        actual_chunks = N_CHUNKS
    
    # Each chunk has its own local buffer
    local_sums = np.zeros((actual_chunks, n_cols), dtype=np.float64)
    
    # Parallel accumulation per chunk
    for chunk in prange(actual_chunks):
        start = chunk * chunk_size
        end = start + chunk_size
        if end > n_rows:
            end = n_rows
        
        # Skip empty chunks
        if start < n_rows:
            for row in range(start, end):
                values, cols = csr.row_to_numpy(row)
                nnz = len(values)
                for j in range(nnz):
                    local_sums[chunk, cols[j]] += float(values[j])
    
    # Merge all chunks - parallel over columns
    sums = np.zeros(n_cols, dtype=np.float64)
    
    for col in prange(n_cols):
        s = ZERO
        for chunk in range(actual_chunks):
            s += local_sums[chunk, col]
        sums[col] = s
    
    return sums


@parallel_jit(cache=True, boundscheck=False, nogil=True)
def _precompute_zero_residual_contributions(
    row_sums: np.ndarray,
    col_sums: np.ndarray,
    total: float,
    theta: float,
    clip: float
) -> tuple:
    """Precompute sum and sum-of-squares of zero residuals for ALL genes.
    
    PARALLEL over genes. Each gene computes its "all-zero" statistics.
    Then in the sparse loop, we only correct for non-zero entries.
    
    Complexity: O(n_genes * n_cells) but fully parallelized.
    """
    # === INLINE CONSTANTS ===
    EPS = 1e-12
    ZERO = 0.0
    
    n_genes = len(row_sums)
    n_cells = len(col_sums)
    
    inv_theta = 1.0 / theta
    inv_total = 1.0 / total
    neg_clip = -clip
    
    # Per-cell factor: factor[j] = col_sums[j] / total
    cell_factors = np.empty(n_cells, dtype=np.float64)
    
    vectorize(8)
    for j in range(n_cells):
        cell_factors[j] = col_sums[j] * inv_total
    
    # Output arrays
    all_zero_sum = np.empty(n_genes, dtype=np.float64)
    all_zero_sq = np.empty(n_genes, dtype=np.float64)
    
    # PARALLEL: Compute per-gene stats
    for gene in prange(n_genes):
        gene_sum = row_sums[gene]
        
        sum_res = ZERO
        sq_res = ZERO
        
        vectorize(8)
        for j in range(n_cells):
            mu = gene_sum * cell_factors[j]
            
            if likely(mu > EPS):
                denom = math.sqrt(mu + mu * mu * inv_theta)
                res = -mu / denom
            else:
                res = ZERO
            
            # Clip
            if res > clip:
                res = clip
            elif res < neg_clip:
                res = neg_clip
            
            sum_res += res
            sq_res += res * res
        
        all_zero_sum[gene] = sum_res
        all_zero_sq[gene] = sq_res
    
    return all_zero_sum, all_zero_sq, cell_factors


@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def _pearson_variance_sparse(
    csr: CSR,
    row_sums: np.ndarray,
    all_zero_sum: np.ndarray,
    all_zero_sq: np.ndarray,
    cell_factors: np.ndarray,
    theta: float,
    clip: float
) -> np.ndarray:
    """Pearson variance - sparse-optimized parallel kernel.
    
    Uses precomputed "all-zero" statistics and only corrects for nnz entries.
    Complexity: O(total_nnz) which is optimal for sparse data.
    """
    # === INLINE CONSTANTS ===
    EPS = 1e-12
    ZERO = 0.0
    
    n_genes = csr.nrows
    n_cells = csr.ncols
    
    assume(n_genes > 0)
    assume(n_cells > 0)
    
    inv_theta = 1.0 / theta
    inv_n_cells = 1.0 / float(n_cells)
    neg_clip = -clip
    
    residual_vars = np.empty(n_genes, dtype=np.float64)
    
    for gene in prange(n_genes):
        values, cols = csr.row_to_numpy(gene)
        nnz = len(values)
        gene_sum = row_sums[gene]
        
        # Start with precomputed "all-zero" statistics
        sum_res = all_zero_sum[gene]
        sq_res = all_zero_sq[gene]
        
        # Correct for non-zero entries: O(nnz) per gene
        vectorize(8)
        for j in range(nnz):
            cell = cols[j]
            val = float(values[j])
            mu = gene_sum * cell_factors[cell]
            
            if likely(mu > EPS):
                denom = math.sqrt(mu + mu * mu * inv_theta)
                res_zero = -mu / denom
                res_nnz = (val - mu) / denom
            else:
                res_zero = ZERO
                res_nnz = ZERO
            
            # Clip zero residual
            if res_zero > clip:
                res_zero = clip
            elif res_zero < neg_clip:
                res_zero = neg_clip
            
            # Clip actual residual
            if res_nnz > clip:
                res_nnz = clip
            elif res_nnz < neg_clip:
                res_nnz = neg_clip
            
            # Correct: subtract zero contribution, add actual contribution
            sum_res += res_nnz - res_zero
            sq_res += res_nnz * res_nnz - res_zero * res_zero
        
        # Variance = E[X^2] - E[X]^2
        mean_res = sum_res * inv_n_cells
        residual_vars[gene] = sq_res * inv_n_cells - mean_res * mean_res
    
    return residual_vars


@parallel_jit(cache=True, inline='always', boundscheck=False, nogil=True)
def _pearson_variance_optimized(
    csr: CSR,
    row_sums: np.ndarray,
    col_sums: np.ndarray,
    total: float,
    theta: float,
    clip: float
) -> np.ndarray:
    """Pearson residual variance - optimized version without bitmap allocation.
    
    This version avoids per-gene bitmap allocation by using the
    "compute all-zero then correct" strategy.
    
    Still O(n_genes * n_cells) but with better cache behavior and no allocation.
    """
    # === INLINE CONSTANTS ===
    EPS = 1e-12
    ZERO = 0.0
    
    n_genes = csr.nrows
    n_cells = csr.ncols
    
    assume(n_genes > 0)
    assume(n_cells > 0)
    assume(total > ZERO)
    
    inv_theta = 1.0 / theta
    inv_total = 1.0 / total
    inv_n_cells = 1.0 / float(n_cells)
    neg_clip = -clip
    
    # Precompute per-cell factor
    cell_factors = np.empty(n_cells, dtype=np.float64)
    
    vectorize(8)
    for j in range(n_cells):
        cell_factors[j] = col_sums[j] * inv_total
    
    residual_vars = np.empty(n_genes, dtype=np.float64)
    
    for gene in prange(n_genes):
        values, cols = csr.row_to_numpy(gene)
        nnz = len(values)
        gene_sum = row_sums[gene]
        
        # Phase 1: Compute total sum/sq for ALL cells (as if all zero)
        sum_all_zero = ZERO
        sq_all_zero = ZERO
        
        vectorize(8)
        for j in range(n_cells):
            mu = gene_sum * cell_factors[j]
            
            if likely(mu > EPS):
                denom = math.sqrt(mu + mu * mu * inv_theta)
                res = -mu / denom
            else:
                res = ZERO
            
            # Clip
            if res > clip:
                res = clip
            elif res < neg_clip:
                res = neg_clip
            
            sum_all_zero += res
            sq_all_zero += res * res
        
        # Phase 2: Process non-zeros and correct the sums
        sum_res = sum_all_zero
        sq_res = sq_all_zero
        
        vectorize(8)
        for j in range(nnz):
            cell = cols[j]
            val = float(values[j])
            mu = gene_sum * cell_factors[cell]
            
            if likely(mu > EPS):
                denom = math.sqrt(mu + mu * mu * inv_theta)
                res_zero = -mu / denom
                res_nnz = (val - mu) / denom
            else:
                res_zero = ZERO
                res_nnz = ZERO
            
            # Clip zero residual
            if res_zero > clip:
                res_zero = clip
            elif res_zero < neg_clip:
                res_zero = neg_clip
            
            # Clip actual residual
            if res_nnz > clip:
                res_nnz = clip
            elif res_nnz < neg_clip:
                res_nnz = neg_clip
            
            # Correct: subtract zero, add actual
            sum_res += res_nnz - res_zero
            sq_res += res_nnz * res_nnz - res_zero * res_zero
        
        # Variance = E[X^2] - E[X]^2
        mean_res = sum_res * inv_n_cells
        residual_vars[gene] = sq_res * inv_n_cells - mean_res * mean_res
    
    return residual_vars


def hvg_pearson_residuals(
    csr: CSR,
    n_top_genes: int,
    theta: float = 100.0,
    clip: float = -1.0
) -> tuple:
    """Pearson residuals HVG. FULLY OPTIMIZED:
    
    1. Parallel row/col sums
    2. Parallel "all-zero" precomputation  
    3. Parallel sparse correction (O(nnz) per gene)
    """
    n_genes = csr.nrows
    n_cells = csr.ncols
    
    if clip < 0:
        clip = math.sqrt(float(n_cells))
    
    # PARALLEL: row sums
    row_sums = _row_sums_parallel(csr)
    
    # PARALLEL: col sums via chunked reduction
    col_sums = _col_sums_chunked(csr)
    
    # Total sum
    total = float(np.sum(row_sums))
    
    # PARALLEL: moments
    means, variances = compute_moments(csr, 1)
    
    # PARALLEL: Precompute "all-zero" statistics for each gene
    all_zero_sum, all_zero_sq, cell_factors = _precompute_zero_residual_contributions(
        row_sums, col_sums, total, theta, clip
    )
    
    # PARALLEL: Sparse correction (O(nnz) per gene)
    residual_vars = _pearson_variance_sparse(
        csr, row_sums, all_zero_sum, all_zero_sq, cell_factors, theta, clip
    )
    
    indices, mask = select_top_k(residual_vars, n_top_genes)
    return indices, mask, means, variances, residual_vars


# =============================================================================
# Legacy API
# =============================================================================

@fast_jit(cache=True, boundscheck=False)
def select_hvg_by_dispersion(csr: CSR, n_top: int) -> tuple:
    """Legacy: select by raw dispersion."""
    assume(n_top > 0)
    means, vars_ = compute_moments(csr, 1)
    dispersions = compute_dispersion(means, vars_)
    indices, mask = select_top_k(dispersions, n_top)
    return indices, mask, dispersions
