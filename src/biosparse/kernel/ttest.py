"""T-Test for Sparse Matrices.

Optimized implementation of Student's and Welch's t-test
for CSR sparse matrices.

Design:
    - group_ids: 0 = reference group, 1/2/3... = target groups
    - One-vs-all: computes ref vs target_i for all targets at once
    - Output shape: (n_rows, n_targets) for multi-target results

Algorithm (optimized):
    - Batch p-value computation (separated from t-statistic computation)
    - Uses exact Student's t-distribution CDF for small df
    - Uses normal approximation for df > 30
    - Vectorized with prange for parallel row processing

Computes per-row t-statistics, p-values, and log2 fold change.
Uses project's CSR sparse matrix type.
"""

import math
import numpy as np
from numba import prange, njit

from biosparse.optim import (
    parallel_jit, fast_jit, assume, vectorize, unroll, 
    interleave, likely, unlikely
)
from biosparse._binding import CSR
from biosparse.kernel.math._tdist import t_test_pvalue

# Import for type hints only
import biosparse._numba  # noqa: F401 - registers CSR/CSC types

__all__ = [
    'ttest',
    'welch_ttest',
    'student_ttest',
]


# =============================================================================
# Batch p-value computation (separated from t-statistic computation)
# =============================================================================

@parallel_jit(cache=True, inline='always')
def _compute_ttest_pvalues_batch(
    t_stats: np.ndarray,
    dfs: np.ndarray,
    out_p: np.ndarray
) -> None:
    """Batch compute two-sided p-values from t-statistics.
    
    Separating p-value computation allows better vectorization.
    Uses exact t-distribution for df <= 30, normal approximation otherwise.
    
    Args:
        t_stats: T-statistics array (flattened)
        dfs: Degrees of freedom array
        out_p: Output p-values array
    """
    INV_SQRT2 = 0.7071067811865475
    
    n = len(t_stats)
    assume(n > 0)
    assume(len(dfs) >= n)
    assume(len(out_p) >= n)
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        df = dfs[i]
        
        if unlikely(df <= 0.0):
            out_p[i] = 1.0
            continue
        
        abs_t = abs(t_stats[i])
        
        if likely(df > 30.0):
            # Normal approximation (most common case for biological data)
            sf = 0.5 * math.erfc(abs_t * INV_SQRT2)
            out_p[i] = 2.0 * sf
        else:
            # Exact t-distribution for small df
            out_p[i] = t_test_pvalue(t_stats[i], df, 0)


# =============================================================================
# Online variance computation (Welford's algorithm)
# =============================================================================

@fast_jit(cache=True, inline='always')
def _welford_update(count: int, mean: float, m2: float, x: float) -> tuple:
    """Single step of Welford's online variance algorithm.
    
    Numerically stable compared to naive sum of squares.
    
    Args:
        count: Current count
        mean: Current mean
        m2: Current M2 (sum of squared differences from mean)
        x: New value
    
    Returns:
        (new_count, new_mean, new_m2)
    """
    count += 1
    delta = x - mean
    mean += delta / float(count)
    delta2 = x - mean
    m2 += delta * delta2
    return count, mean, m2


@fast_jit(cache=True, inline='always')
def _welford_finalize(count: int, mean: float, m2: float, ddof: int) -> tuple:
    """Finalize Welford's algorithm to get variance.
    
    Args:
        count: Total count
        mean: Final mean
        m2: Final M2
        ddof: Delta degrees of freedom (typically 1 for sample variance)
    
    Returns:
        (mean, variance)
    """
    if count < ddof + 1:
        return mean, 0.0
    
    var = m2 / float(count - ddof)
    if var < 0.0:
        var = 0.0
    return mean, var


# =============================================================================
# T-Test for Sparse Matrix (One-vs-All Design)
# =============================================================================

@parallel_jit(cache=True)
def ttest(
    csr: CSR,
    group_ids: np.ndarray,
    n_targets: int,
    use_welch: bool = True
) -> tuple:
    """Perform t-test: reference (group 0) vs all targets (groups 1..n_targets).
    
    Optimized implementation with batch p-value computation.
    
    Args:
        csr: CSR sparse matrix (CSRF32 or CSRF64), genes x cells
        group_ids: Group assignment for each column (cell)
                   0 = reference, 1..n_targets = target groups
        n_targets: Number of target groups (excludes reference)
        use_welch: If True, use Welch's t-test; else Student's t-test
    
    Returns:
        (t_stats, p_values, log2_fc):
            Each has shape (n_rows, n_targets)
    """
    # Inline constants
    EPS = 1e-9
    SIGMA_MIN = 1e-15
    
    n_rows = csr.nrows
    n_cols = csr.ncols
    
    # Compiler optimization hints
    assume(n_rows > 0)
    assume(n_cols > 0)
    assume(n_targets > 0)
    assume(n_targets <= 256)  # Reasonable upper bound
    
    # Count elements in each group (sequential, small)
    n_groups = n_targets + 1
    group_counts = np.zeros(n_groups, dtype=np.int64)
    
    for i in range(n_cols):
        g = group_ids[i]
        if likely(g >= 0 and g < n_groups):
            group_counts[g] += 1
    
    n_ref = group_counts[0]
    assume(n_ref > 0)
    assume(n_ref <= n_cols)
    
    n_ref_f = float(n_ref)
    inv_n_ref = 1.0 / n_ref_f
    
    # Allocate output arrays: (n_rows, n_targets)
    out_t_stats = np.empty((n_rows, n_targets), dtype=np.float64)
    out_p_values = np.empty((n_rows, n_targets), dtype=np.float64)
    out_log2_fc = np.empty((n_rows, n_targets), dtype=np.float64)
    
    # Intermediate arrays for batch p-value computation
    total_pairs = n_rows * n_targets
    t_stats_flat = np.empty(total_pairs, dtype=np.float64)
    dfs_flat = np.empty(total_pairs, dtype=np.float64)
    
    for row in prange(n_rows):
        values, col_indices = csr.row_to_numpy(row)
        nnz = len(values)
        
        # Accumulate sums for each group (thread-local)
        sum_ref = 0.0
        sum_sq_ref = 0.0
        n_ref_nz = 0
        
        sum_tar = np.zeros(n_targets, dtype=np.float64)
        sum_sq_tar = np.zeros(n_targets, dtype=np.float64)
        n_tar_nz = np.zeros(n_targets, dtype=np.int64)
        
        # Main accumulation loop - optimized for sparse data
        for j in range(nnz):
            col_idx = col_indices[j]
            val = float(values[j])
            g = group_ids[col_idx]
            
            if g == 0:
                sum_ref += val
                sum_sq_ref += val * val
                n_ref_nz += 1
            elif likely(g > 0 and g <= n_targets):
                t = g - 1  # target index
                sum_tar[t] += val
                sum_sq_tar[t] += val * val
                n_tar_nz[t] += 1
        
        # Reference mean (including zeros)
        mean_ref = sum_ref * inv_n_ref
        
        # Reference variance: var = (sq_sum - n * mean^2) / (n - 1)
        var_ref = 0.0
        if likely(n_ref > 1):
            var_numer_ref = sum_sq_ref - n_ref_f * mean_ref * mean_ref
            var_ref = var_numer_ref / (n_ref_f - 1.0)
            if unlikely(var_ref < 0.0):
                var_ref = 0.0
        
        # Process each target
        for t in range(n_targets):
            flat_idx = row * n_targets + t
            n_tar = group_counts[t + 1]
            
            if unlikely(n_tar == 0):
                out_t_stats[row, t] = 0.0
                out_log2_fc[row, t] = 0.0
                t_stats_flat[flat_idx] = 0.0
                dfs_flat[flat_idx] = 1.0
                continue
            
            n_tar_f = float(n_tar)
            inv_n_tar = 1.0 / n_tar_f
            
            # Target mean (including zeros)
            mean_tar = sum_tar[t] * inv_n_tar
            
            # Log2 fold change
            out_log2_fc[row, t] = np.log2((mean_tar + EPS) / (mean_ref + EPS))
            
            # Target variance
            var_tar = 0.0
            if likely(n_tar > 1):
                var_numer_tar = sum_sq_tar[t] - n_tar_f * mean_tar * mean_tar
                var_tar = var_numer_tar / (n_tar_f - 1.0)
                if unlikely(var_tar < 0.0):
                    var_tar = 0.0
            
            # Compute t-statistic
            mean_diff = mean_tar - mean_ref
            t_stat = 0.0
            df = 1.0
            
            if use_welch:
                # Welch's t-test (unequal variances)
                se_sq = var_ref * inv_n_ref + var_tar * inv_n_tar
                if likely(se_sq > SIGMA_MIN):
                    se = math.sqrt(se_sq)
                    t_stat = mean_diff / se
                    
                    # Welch-Satterthwaite degrees of freedom
                    v1_n1 = var_ref * inv_n_ref
                    v2_n2 = var_tar * inv_n_tar
                    sum_v = v1_n1 + v2_n2
                    
                    if likely(sum_v > 1e-12):
                        denom = (v1_n1 * v1_n1) / (n_ref_f - 1.0) + (v2_n2 * v2_n2) / (n_tar_f - 1.0)
                        if likely(denom > 0.0):
                            df = (sum_v * sum_v) / denom
            else:
                # Student's t-test (pooled variance)
                pooled_df = n_ref_f + n_tar_f - 2.0
                if likely(pooled_df > 0.0):
                    df1 = n_ref_f - 1.0
                    df2 = n_tar_f - 1.0
                    pooled_var = (df1 * var_ref + df2 * var_tar) / pooled_df
                    se_sq = pooled_var * (inv_n_ref + inv_n_tar)
                    
                    if likely(se_sq > SIGMA_MIN):
                        se = math.sqrt(se_sq)
                        t_stat = mean_diff / se
                        df = pooled_df
            
            out_t_stats[row, t] = t_stat
            t_stats_flat[flat_idx] = t_stat
            dfs_flat[flat_idx] = df
    
    # Batch compute p-values (vectorized)
    p_flat = np.empty(total_pairs, dtype=np.float64)
    _compute_ttest_pvalues_batch(t_stats_flat, dfs_flat, p_flat)
    
    # Reshape p-values back to (n_rows, n_targets)
    for row in range(n_rows):
        for t in range(n_targets):
            out_p_values[row, t] = p_flat[row * n_targets + t]
    
    return out_t_stats, out_p_values, out_log2_fc


@fast_jit(cache=True)
def welch_ttest(
    csr: CSR,
    group_ids: np.ndarray,
    n_targets: int
) -> tuple:
    """Welch's t-test (convenience wrapper).
    
    Args:
        csr: CSR sparse matrix
        group_ids: Group assignment (0 = ref, 1..n_targets = targets)
        n_targets: Number of target groups
    
    Returns:
        (t_stats, p_values, log2_fc): Each with shape (n_rows, n_targets)
    """
    return ttest(csr, group_ids, n_targets, True)


@fast_jit(cache=True)
def student_ttest(
    csr: CSR,
    group_ids: np.ndarray,
    n_targets: int
) -> tuple:
    """Student's t-test (convenience wrapper).
    
    Args:
        csr: CSR sparse matrix
        group_ids: Group assignment (0 = ref, 1..n_targets = targets)
        n_targets: Number of target groups
    
    Returns:
        (t_stats, p_values, log2_fc): Each with shape (n_rows, n_targets)
    """
    return ttest(csr, group_ids, n_targets, False)
