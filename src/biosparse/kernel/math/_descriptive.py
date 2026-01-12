"""Descriptive Statistics Functions.

C-style Numba-optimized implementations:
    - median, mad: Median and Median Absolute Deviation
    - quantile, percentile: Quantile computation
    - argsort_full, argpartition: Sorting utilities
    - assign_bins_*: Binning utilities
    - group_mean_std, group_median_mad: Per-group statistics
    - zscore_by_bin, zscore_by_bin_mad: Z-score normalization

Optimization Philosophy:
    - ALL constants INLINED (avoid Numba closure detection overhead)
    - Use math.* functions (compile to single CPU instructions)
    - Pre-compute ALL reciprocals (division -> multiplication)
    - Bit operations: >> for /2, & for %2
    - SIMD hints: vectorize(8), interleave(4)
    - Branch prediction: likely(), unlikely()
    - assume() for bounds elimination
"""

import math
import sys
import numpy as np
from numba import njit, prange

from biosparse.optim import (
    parallel_jit, fast_jit, assume, vectorize, 
    interleave, unroll, likely, unlikely
)

# =============================================================================
# Platform-specific cache settings
# =============================================================================
# WORKAROUND: Numba cache has a bug on Linux where loading cached functions
# that involve recursive call chains causes segmentation faults.
# See: https://github.com/numba/numba/issues/XXXX (to be reported)
# This affects functions like argsort_full -> _qsort_idx (recursive).
# On Windows, the linker resolves these dependencies correctly.
_CACHE_RECURSIVE = sys.platform == 'win32'

__all__ = [
    'median', 'mad', 'quantile', 'percentile', 'quantiles_batch',
    'argsort_full', 'argpartition',
    'assign_bins_equal_width', 'assign_bins_by_quantiles', 'compute_bin_edges_quantile',
    'group_mean_std', 'group_median_mad',
    'zscore_by_bin', 'zscore_by_bin_mad',
]


# =============================================================================
# Sorting Utilities - Inline helpers
# Note: These functions use _CACHE_RECURSIVE because they are part of
# recursive call chains (argsort_full -> _qsort_idx -> recursive calls).
# =============================================================================

@fast_jit(cache=_CACHE_RECURSIVE, inline='always')
def _swap(indices: np.ndarray, i: int, j: int) -> None:
    """Swap two elements. Inlined to zero overhead."""
    t = indices[i]
    indices[i] = indices[j]
    indices[j] = t


@fast_jit(cache=_CACHE_RECURSIVE, inline='always')
def _insertion_sort_idx(arr: np.ndarray, idx: np.ndarray, lo: int, hi: int) -> None:
    """Insertion sort for small subarrays (< 16 elements)."""
    for i in range(lo + 1, hi + 1):
        j = i
        while j > lo and arr[idx[j - 1]] > arr[idx[j]]:
            _swap(idx, j - 1, j)
            j -= 1


@fast_jit(cache=_CACHE_RECURSIVE, inline='always')
def _median3(arr: np.ndarray, idx: np.ndarray, lo: int, mid: int, hi: int) -> None:
    """Median-of-three pivot selection. Sorts lo, mid, hi."""
    if arr[idx[lo]] > arr[idx[mid]]:
        _swap(idx, lo, mid)
    if arr[idx[lo]] > arr[idx[hi]]:
        _swap(idx, lo, hi)
    if arr[idx[mid]] > arr[idx[hi]]:
        _swap(idx, mid, hi)


@fast_jit(cache=_CACHE_RECURSIVE)
def _qsort_idx(arr: np.ndarray, idx: np.ndarray, lo: int, hi: int) -> None:
    """Quicksort indices. Median-of-3 pivot, insertion sort for small."""
    # Inline constant for threshold
    THRESHOLD = 16
    
    if hi - lo < THRESHOLD:
        _insertion_sort_idx(arr, idx, lo, hi)
        return
    
    mid = (lo + hi) >> 1  # Bit shift for /2
    _median3(arr, idx, lo, mid, hi)
    
    _swap(idx, mid, hi - 1)
    pivot = arr[idx[hi - 1]]
    
    i = lo
    j = hi - 1
    
    while True:
        i += 1
        while i < hi and arr[idx[i]] < pivot:
            i += 1
        j -= 1
        while j > lo and arr[idx[j]] > pivot:
            j -= 1
        if i >= j:
            break
        _swap(idx, i, j)
    
    _swap(idx, i, hi - 1)
    
    # Recurse on smaller partition first
    if i - lo < hi - i:
        _qsort_idx(arr, idx, lo, i - 1)
        _qsort_idx(arr, idx, i + 1, hi)
    else:
        _qsort_idx(arr, idx, i + 1, hi)
        _qsort_idx(arr, idx, lo, i - 1)


@fast_jit(cache=_CACHE_RECURSIVE, inline='always')
def argsort_full(arr: np.ndarray) -> np.ndarray:
    """Argsort using optimized quicksort."""
    # === INLINE CONSTANTS ===
    THRESHOLD = 16
    
    n = len(arr)
    assume(n > 0)
    
    idx = np.arange(n, dtype=np.int64)
    
    if unlikely(n <= 1):
        return idx
    
    if n <= THRESHOLD:
        _insertion_sort_idx(arr, idx, 0, n - 1)
    else:
        _qsort_idx(arr, idx, 0, n - 1)
    
    return idx


@fast_jit(cache=_CACHE_RECURSIVE, inline='always')
def _partition(arr: np.ndarray, idx: np.ndarray, lo: int, hi: int) -> int:
    """Partition for quickselect."""
    mid = (lo + hi) >> 1
    _median3(arr, idx, lo, mid, hi)
    
    pivot = arr[idx[mid]]
    _swap(idx, mid, hi)
    
    store = lo
    for i in range(lo, hi):
        if arr[idx[i]] < pivot:
            _swap(idx, i, store)
            store += 1
    
    _swap(idx, store, hi)
    return store


@fast_jit(cache=_CACHE_RECURSIVE, inline='always')
def _qselect(arr: np.ndarray, idx: np.ndarray, lo: int, hi: int, k: int) -> None:
    """Quickselect: partition around k-th smallest."""
    while lo < hi:
        p = _partition(arr, idx, lo, hi)
        if p == k:
            return
        elif p < k:
            lo = p + 1
        else:
            hi = p - 1


@fast_jit(cache=_CACHE_RECURSIVE, inline='always')
def argpartition(arr: np.ndarray, k: int) -> np.ndarray:
    """Partial argsort: first k are k smallest (unordered)."""
    n = len(arr)
    assume(n > 0)
    assume(k >= 0)
    assume(k < n)
    
    idx = np.arange(n, dtype=np.int64)
    
    if unlikely(n <= 1):
        return idx
    
    _qselect(arr, idx, 0, n - 1, k)
    return idx


# =============================================================================
# Core Statistics
# =============================================================================

@fast_jit(cache=True, inline='always')
def median(arr: np.ndarray) -> float:
    """Median using Numba's optimized sort."""
    # === INLINE CONSTANTS ===
    HALF = 0.5
    
    n = len(arr)
    assume(n > 0)
    
    if unlikely(n == 1):
        return float(arr[0])
    
    sorted_arr = np.sort(arr.astype(np.float64))
    
    mid = n >> 1  # n / 2
    
    if n & 1:  # Odd: n % 2 == 1
        return sorted_arr[mid]
    else:
        return (sorted_arr[mid - 1] + sorted_arr[mid]) * HALF


@fast_jit(cache=True, inline='always')
def mad(arr: np.ndarray, center: float = np.nan) -> float:
    """Median Absolute Deviation = median(|X - median(X)|)."""
    # === INLINE CONSTANTS ===
    HALF = 0.5
    
    n = len(arr)
    assume(n > 0)
    
    if math.isnan(center):
        center = median(arr)
    
    # Compute deviations - vectorizable
    dev = np.empty(n, dtype=np.float64)
    
    vectorize(8)
    interleave(4)
    for i in range(n):
        dev[i] = math.fabs(arr[i] - center)
    
    sorted_dev = np.sort(dev)
    
    mid = n >> 1
    if n & 1:
        return sorted_dev[mid]
    else:
        return (sorted_dev[mid - 1] + sorted_dev[mid]) * HALF


@fast_jit(cache=True, inline='always')
def quantile(arr: np.ndarray, q: float) -> float:
    """Q-th quantile (0 <= q <= 1) with linear interpolation."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    ONE = 1.0
    
    n = len(arr)
    assume(n > 0)
    assume(q >= ZERO)
    assume(q <= ONE)
    
    if unlikely(n == 1):
        return float(arr[0])
    
    sorted_arr = np.sort(arr.astype(np.float64))
    nm1 = float(n - 1)
    
    if unlikely(q <= ZERO):
        return sorted_arr[0]
    if unlikely(q >= ONE):
        return sorted_arr[n - 1]
    
    idx = q * nm1
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        hi = n - 1
    
    frac = idx - float(lo)
    return sorted_arr[lo] * (ONE - frac) + sorted_arr[hi] * frac


@fast_jit(cache=True, inline='always')
def percentile(arr: np.ndarray, p: float) -> float:
    """P-th percentile (0 <= p <= 100)."""
    # === INLINE CONSTANT ===
    INV_100 = 0.01
    
    assume(p >= 0.0)
    assume(p <= 100.0)
    return quantile(arr, p * INV_100)


@fast_jit(cache=True, inline='always')
def quantiles_batch(arr: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Multiple quantiles at once (more efficient than loop)."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    ONE = 1.0
    
    n = len(arr)
    nq = len(qs)
    assume(n > 0)
    assume(nq > 0)
    
    sorted_arr = np.sort(arr.astype(np.float64))
    nm1 = float(n - 1)
    
    out = np.empty(nq, dtype=np.float64)
    
    vectorize(4)
    for j in range(nq):
        q = qs[j]
        if unlikely(q <= ZERO):
            out[j] = sorted_arr[0]
        elif unlikely(q >= ONE):
            out[j] = sorted_arr[n - 1]
        else:
            idx = q * nm1
            lo = int(idx)
            hi = lo + 1
            if hi >= n:
                hi = n - 1
            frac = idx - float(lo)
            out[j] = sorted_arr[lo] * (ONE - frac) + sorted_arr[hi] * frac
    
    return out


# =============================================================================
# Binning Utilities
# =============================================================================

@parallel_jit(cache=True, inline='always')
def assign_bins_equal_width(values: np.ndarray, n_bins: int) -> tuple:
    """Assign to equal-width bins. Returns (bin_indices, bin_edges)."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    ONE = 1.0
    
    n = len(values)
    assume(n > 0)
    assume(n_bins > 0)
    
    # Find min/max in single pass
    v_min = values[0]
    v_max = values[0]
    
    for i in range(1, n):
        v = values[i]
        if v < v_min:
            v_min = v
        if v > v_max:
            v_max = v
    
    if unlikely(v_max == v_min):
        idx = np.zeros(n, dtype=np.int64)
        edges = np.array([v_min, v_max + ONE], dtype=np.float64)
        return idx, edges
    
    width = (v_max - v_min) / float(n_bins)
    inv_w = ONE / width
    n_bins_m1 = n_bins - 1
    
    edges = np.empty(n_bins + 1, dtype=np.float64)
    for i in range(n_bins + 1):
        edges[i] = v_min + float(i) * width
    
    idx = np.empty(n, dtype=np.int64)
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        b = int((values[i] - v_min) * inv_w)
        # Clamp
        if b < 0:
            b = 0
        elif b > n_bins_m1:
            b = n_bins_m1
        idx[i] = b
    
    return idx, edges


@fast_jit(cache=True, inline='always')
def compute_bin_edges_quantile(values: np.ndarray, percentiles: np.ndarray) -> np.ndarray:
    """Compute bin edges from percentiles (0-100)."""
    # === INLINE CONSTANT ===
    INV_100 = 0.01
    
    n_e = len(percentiles)
    assume(n_e > 0)
    
    qs = np.empty(n_e, dtype=np.float64)
    
    vectorize(8)
    for i in range(n_e):
        qs[i] = percentiles[i] * INV_100
    
    return quantiles_batch(values, qs)


@parallel_jit(cache=True, inline='always')
def assign_bins_by_quantiles(values: np.ndarray, percentiles: np.ndarray) -> tuple:
    """Assign to bins by percentile edges. Returns (bin_indices, bin_edges)."""
    n = len(values)
    n_p = len(percentiles)
    n_bins = n_p - 1
    
    assume(n > 0)
    assume(n_bins > 0)
    
    edges = compute_bin_edges_quantile(values, percentiles)
    
    idx = np.empty(n, dtype=np.int64)
    n_bins_m1 = n_bins - 1
    
    for i in prange(n):
        v = values[i]
        # Binary search
        lo = 0
        hi = n_bins
        while lo < hi:
            mid = (lo + hi) >> 1
            if v >= edges[mid + 1]:
                lo = mid + 1
            else:
                hi = mid
        if lo > n_bins_m1:
            lo = n_bins_m1
        idx[i] = lo
    
    return idx, edges


# =============================================================================
# Group Statistics
# =============================================================================

@fast_jit(cache=True, inline='always')
def group_mean_std(values: np.ndarray, bin_indices: np.ndarray, n_bins: int) -> tuple:
    """Per-bin mean and std."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    ONE = 1.0
    
    n = len(values)
    assume(n > 0)
    assume(n_bins > 0)
    
    sums = np.zeros(n_bins, dtype=np.float64)
    sq_sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    
    for i in range(n):
        b = bin_indices[i]
        if likely(0 <= b < n_bins):
            v = float(values[i])
            sums[b] += v
            sq_sums[b] += v * v
            counts[b] += 1
    
    means = np.zeros(n_bins, dtype=np.float64)
    stds = np.zeros(n_bins, dtype=np.float64)
    
    for b in range(n_bins):
        c = counts[b]
        if likely(c > 0):
            c_f = float(c)
            inv_c = ONE / c_f
            m = sums[b] * inv_c
            means[b] = m
            if c > 1:
                var = (sq_sums[b] - c_f * m * m) / (c_f - ONE)
                stds[b] = math.sqrt(var) if var > ZERO else ZERO
    
    return means, stds, counts


@fast_jit(cache=True, inline='always')
def _median_from_sorted(sorted_arr: np.ndarray, n: int) -> float:
    """Median from pre-sorted array."""
    # === INLINE CONSTANT ===
    HALF = 0.5
    
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_arr[0]
    
    mid = n >> 1
    if n & 1:
        return sorted_arr[mid]
    else:
        return (sorted_arr[mid - 1] + sorted_arr[mid]) * HALF


@fast_jit(cache=True, inline='always')
def _mad_from_sorted(sorted_arr: np.ndarray, n: int, center: float) -> float:
    """MAD from pre-sorted array."""
    # === INLINE CONSTANT ===
    HALF = 0.5
    
    if n == 0:
        return 0.0
    
    dev = np.empty(n, dtype=np.float64)
    
    vectorize(8)
    for i in range(n):
        dev[i] = math.fabs(sorted_arr[i] - center)
    
    sorted_dev = np.sort(dev)
    
    mid = n >> 1
    if n & 1:
        return sorted_dev[mid]
    else:
        return (sorted_dev[mid - 1] + sorted_dev[mid]) * HALF


@fast_jit(cache=True, inline='always')
def group_median_mad(values: np.ndarray, bin_indices: np.ndarray, n_bins: int) -> tuple:
    """Per-bin median and MAD using offset array (O(n) total)."""
    # === INLINE CONSTANTS ===
    ZERO = 0.0
    HALF = 0.5
    
    n = len(values)
    assume(n > 0)
    assume(n_bins > 0)
    
    # Pass 1: Count per bin
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n):
        b = bin_indices[i]
        if likely(0 <= b < n_bins):
            counts[b] += 1
    
    # Compute offsets (prefix sum)
    offsets = np.empty(n_bins + 1, dtype=np.int64)
    offsets[0] = 0
    for b in range(n_bins):
        offsets[b + 1] = offsets[b] + counts[b]
    
    # Pass 2: Scatter values
    sorted_by_bin = np.empty(n, dtype=np.float64)
    pos = np.zeros(n_bins, dtype=np.int64)
    
    for i in range(n):
        b = bin_indices[i]
        if likely(0 <= b < n_bins):
            sorted_by_bin[offsets[b] + pos[b]] = float(values[i])
            pos[b] += 1
    
    # Pass 3: Compute per-bin median/MAD
    medians = np.zeros(n_bins, dtype=np.float64)
    mads = np.zeros(n_bins, dtype=np.float64)
    
    for b in range(n_bins):
        c = counts[b]
        if unlikely(c == 0):
            continue
        
        start = offsets[b]
        bin_vals = np.sort(sorted_by_bin[start:start + c])
        
        # Median
        mid = c >> 1
        if c & 1:
            med = bin_vals[mid]
        else:
            med = (bin_vals[mid - 1] + bin_vals[mid]) * HALF
        medians[b] = med
        
        # MAD
        if likely(c > 1):
            dev = np.empty(c, dtype=np.float64)
            for j in range(c):
                dev[j] = math.fabs(bin_vals[j] - med)
            sorted_dev = np.sort(dev)
            
            if c & 1:
                mads[b] = sorted_dev[mid]
            else:
                mads[b] = (sorted_dev[mid - 1] + sorted_dev[mid]) * HALF
    
    return medians, mads, counts


# =============================================================================
# Z-score Normalization
# =============================================================================

@parallel_jit(cache=True, inline='always')
def zscore_by_bin(
    values: np.ndarray,
    bin_indices: np.ndarray,
    bin_means: np.ndarray,
    bin_stds: np.ndarray,
    out: np.ndarray
) -> None:
    """Z-score using per-bin mean/std. out[i] = (v[i] - mean[b]) / std[b]."""
    # === INLINE CONSTANTS ===
    EPS = 1e-12
    ZERO = 0.0
    ONE = 1.0
    
    n = len(values)
    n_bins = len(bin_stds)
    assume(n > 0)
    
    # Pre-compute inverse stds
    inv_stds = np.empty(n_bins, dtype=np.float64)
    for b in range(n_bins):
        s = bin_stds[b]
        inv_stds[b] = ONE / s if s > EPS else ZERO
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        b = bin_indices[i]
        inv_s = inv_stds[b]
        out[i] = (values[i] - bin_means[b]) * inv_s if inv_s > ZERO else ZERO


@parallel_jit(cache=True, inline='always')
def zscore_by_bin_mad(
    values: np.ndarray,
    bin_indices: np.ndarray,
    bin_medians: np.ndarray,
    bin_mads: np.ndarray,
    out: np.ndarray
) -> None:
    """Z-score using per-bin median/MAD. out[i] = (v[i] - median[b]) / mad[b]."""
    # === INLINE CONSTANTS ===
    EPS = 1e-12
    ZERO = 0.0
    ONE = 1.0
    
    n = len(values)
    n_bins = len(bin_mads)
    assume(n > 0)
    
    # Pre-compute inverse MADs
    inv_mads = np.empty(n_bins, dtype=np.float64)
    for b in range(n_bins):
        m = bin_mads[b]
        inv_mads[b] = ONE / m if m > EPS else ZERO
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        b = bin_indices[i]
        inv_m = inv_mads[b]
        out[i] = (values[i] - bin_medians[b]) * inv_m if inv_m > ZERO else ZERO
