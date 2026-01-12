"""T-Test for Sparse Matrices.

Vectorized implementation of Student's and Welch's t-test
for CSR sparse matrices.

Design:
    - group_ids: 0 = reference group, 1/2/3... = target groups
    - One-vs-all: computes ref vs target_i for all targets at once
    - Output shape: (n_rows, n_targets) for multi-target results

Computes per-row t-statistics, p-values, and log2 fold change.
Uses project's CSR sparse matrix type.

P-values are computed using the exact Student's t-distribution CDF,
matching scipy.special.stdtr implementation.
"""

import numpy as np
from numba import prange

from biosparse.optim import parallel_jit, assume, vectorize
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
# T-Test for Sparse Matrix (One-vs-All Design)
# =============================================================================

@parallel_jit
def ttest(
    csr: CSR,
    group_ids: np.ndarray,
    n_targets: int,
    use_welch: bool = True
) -> tuple:
    """Perform t-test: reference (group 0) vs all targets (groups 1..n_targets).
    
    Optimized with prange for parallel row processing.
    
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
    EPS = 1e-9
    SIGMA_MIN = 1e-15
    
    n_rows = csr.nrows
    n_cols = csr.ncols
    
    assume(n_rows > 0)
    assume(n_targets > 0)
    
    # Count elements in each group (sequential, small)
    n_groups = n_targets + 1
    group_counts = np.zeros(n_groups, dtype=np.int64)
    
    for i in range(n_cols):
        g = group_ids[i]
        if g >= 0 and g < n_groups:
            group_counts[g] += 1
    
    n_ref = group_counts[0]
    assume(n_ref > 0)
    
    n_ref_f = float(n_ref)
    inv_n_ref = 1.0 / n_ref_f
    
    # Allocate output arrays: (n_rows, n_targets)
    out_t_stats = np.empty((n_rows, n_targets), dtype=np.float64)
    out_p_values = np.empty((n_rows, n_targets), dtype=np.float64)
    out_log2_fc = np.empty((n_rows, n_targets), dtype=np.float64)
    
    row = 0
    for values, col_indices in csr:
        nnz = len(values)
        
        # Accumulate sums for each group (thread-local)
        sum_ref = 0.0
        sum_sq_ref = 0.0
        n_ref_nz = 0
        
        sum_tar = np.zeros(n_targets, dtype=np.float64)
        sum_sq_tar = np.zeros(n_targets, dtype=np.float64)
        n_tar_nz = np.zeros(n_targets, dtype=np.int64)
        
        for j in range(nnz):
            col_idx = col_indices[j]
            val = float(values[j])
            g = group_ids[col_idx]
            
            if g == 0:
                sum_ref += val
                sum_sq_ref += val * val
                n_ref_nz += 1
            elif g > 0 and g <= n_targets:
                t = g - 1  # target index
                sum_tar[t] += val
                sum_sq_tar[t] += val * val
                n_tar_nz[t] += 1
        
        # Reference mean (including zeros)
        mean_ref = sum_ref * inv_n_ref
        
        # Reference variance: var = (sq_sum - n * mean^2) / (n - 1)
        var_ref = 0.0
        if n_ref > 1:
            var_numer_ref = sum_sq_ref - n_ref_f * mean_ref * mean_ref
            var_ref = var_numer_ref / (n_ref_f - 1.0)
            if var_ref < 0.0:
                var_ref = 0.0
        
        # Process each target
        for t in range(n_targets):
            n_tar = group_counts[t + 1]
            
            if n_tar == 0:
                out_t_stats[row, t] = 0.0
                out_p_values[row, t] = 1.0
                out_log2_fc[row, t] = 0.0
                continue
            
            n_tar_f = float(n_tar)
            inv_n_tar = 1.0 / n_tar_f
            n_tar_nz_t = n_tar_nz[t]
            
            # Target mean (including zeros)
            mean_tar = sum_tar[t] * inv_n_tar
            
            # Log2 fold change
            out_log2_fc[row, t] = np.log2((mean_tar + EPS) / (mean_ref + EPS))
            
            # Target variance: var = (sq_sum - n * mean^2) / (n - 1)
            var_tar = 0.0
            if n_tar > 1:
                var_numer_tar = sum_sq_tar[t] - n_tar_f * mean_tar * mean_tar
                var_tar = var_numer_tar / (n_tar_f - 1.0)
                if var_tar < 0.0:
                    var_tar = 0.0
            
            # Compute t-statistic
            mean_diff = mean_tar - mean_ref
            t_stat = 0.0
            p_val = 1.0
            
            if use_welch:
                # Welch's t-test (unequal variances)
                se_sq = var_ref * inv_n_ref + var_tar * inv_n_tar
                if se_sq > SIGMA_MIN:
                    se = np.sqrt(se_sq)
                    t_stat = mean_diff / se
                    
                    # Welch-Satterthwaite degrees of freedom
                    v1_n1 = var_ref * inv_n_ref
                    v2_n2 = var_tar * inv_n_tar
                    sum_v = v1_n1 + v2_n2
                    
                    if sum_v > 1e-12:
                        denom = (v1_n1 * v1_n1) / (n_ref_f - 1.0) + (v2_n2 * v2_n2) / (n_tar_f - 1.0)
                        df = (sum_v * sum_v) / denom
                    else:
                        df = 1.0
                    
                    # P-value using exact t-distribution (two-sided)
                    p_val = t_test_pvalue(t_stat, df, 0)
            else:
                # Student's t-test (pooled variance)
                pooled_df = n_ref_f + n_tar_f - 2.0
                if pooled_df > 0.0:
                    df1 = n_ref_f - 1.0
                    df2 = n_tar_f - 1.0
                    pooled_var = (df1 * var_ref + df2 * var_tar) / pooled_df
                    se = np.sqrt(pooled_var * (inv_n_ref + inv_n_tar))
                    
                    if se > SIGMA_MIN:
                        t_stat = mean_diff / se
                        
                        # P-value using exact t-distribution (two-sided)
                        p_val = t_test_pvalue(t_stat, pooled_df, 0)
            
            out_t_stats[row, t] = t_stat
            out_p_values[row, t] = p_val
        
        row += 1
    
    return out_t_stats, out_p_values, out_log2_fc


@fast_jit
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


@fast_jit
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
