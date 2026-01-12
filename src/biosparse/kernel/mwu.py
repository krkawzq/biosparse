"""Mann-Whitney U Test for Sparse Matrices.

Vectorized implementation of MWU test for CSR sparse matrices.
Computes per-row U-statistics, p-values, log2FC, and AUROC.

Design:
    - group_ids: 0 = reference group, 1/2/3... = target groups
    - One-vs-all: computes ref vs target_i for all targets at once
    - Output shape: (n_rows, n_targets) for multi-target results

Uses normal approximation with tie correction (matches scipy.stats.mannwhitneyu):
    mu = n1 * n2 / 2
    var = (n1 * n2 / 12) * (N + 1 - tie_correction)
    z = (U - mu - continuity_correction) / sigma
    p = norm.sf(z) * factor

Alternative hypothesis options:
    0 = two-sided (default)
    1 = greater (x > y)
    -1 = less (x < y)

Uses project's CSR sparse matrix type.
"""

import math
import numpy as np
from numba import prange

from biosparse.optim import parallel_jit, assume, likely, unlikely
from biosparse._binding import CSR

# Import for type hints only
import biosparse._numba  # noqa: F401 - registers CSR/CSC types

__all__ = [
    'mwu_test',
    'count_groups',
]


# =============================================================================
# Group Counting
# =============================================================================

@parallel_jit
def count_groups(group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    """Count elements in each group.
    
    Args:
        group_ids: Group ID for each element (0, 1, 2, ...)
        n_groups: Total number of groups
    
    Returns:
        counts: Count for each group
    """
    n = len(group_ids)
    assume(n > 0)
    
    counts = np.zeros(n_groups, dtype=np.int64)
    
    for i in range(n):
        g = group_ids[i]
        if g >= 0 and g < n_groups:
            counts[g] += 1
    
    return counts


# =============================================================================
# Normal distribution SF (survival function)
# =============================================================================

from numba import njit

@njit(cache=True, fastmath=True, inline='always')
def _normal_sf(z: float) -> float:
    """Normal distribution survival function P(Z > z).
    
    Uses erfc for numerical stability.
    """
    INV_SQRT2 = 0.7071067811865475  # 1/sqrt(2), inline to avoid closure cost
    return 0.5 * math.erfc(z * INV_SQRT2)


# =============================================================================
# MWU Test for Sparse Matrix (One-vs-All Design)
# =============================================================================

@parallel_jit
def mwu_test(
    csr: CSR,
    group_ids: np.ndarray,
    n_targets: int,
    alternative: int = 0,
    use_continuity: bool = True
) -> tuple:
    """Perform MWU test: reference (group 0) vs all targets (groups 1..n_targets).
    
    Matches scipy.stats.mannwhitneyu algorithm (asymptotic method).
    
    Args:
        csr: CSR sparse matrix (CSRF32 or CSRF64), genes x cells
        group_ids: Group assignment for each column (cell)
                   0 = reference, 1..n_targets = target groups
        n_targets: Number of target groups (excludes reference)
        alternative: Alternative hypothesis
                    0 = two-sided (default)
                    1 = greater (ref > target)
                    -1 = less (ref < target)
        use_continuity: Whether to apply continuity correction (default True)
    
    Returns:
        (u_stats, p_values, log2_fc, auroc):
            Each has shape (n_rows, n_targets)
            
    Notes:
        - U statistic is always U1 (for reference group)
        - AUROC = U / (n1 * n2), ranges from 0 to 1
        - P-values use normal approximation with tie correction
    """
    # Inline magic numbers for numba performance.
    EPS = 1e-9
    SIGMA_MIN = 1e-12
    INF = 1e308

    n_rows = csr.nrows
    
    assume(n_rows > 0)
    assume(n_targets > 0)
    
    # Count elements in each group (sequential, small)
    n_groups = n_targets + 1
    group_counts = count_groups(group_ids, n_groups)
    n_ref = group_counts[0]
    
    assume(n_ref > 0)
    
    # Allocate output arrays: (n_rows, n_targets)
    out_u_stats = np.empty((n_rows, n_targets), dtype=np.float64)
    out_p_values = np.empty((n_rows, n_targets), dtype=np.float64)
    out_log2_fc = np.empty((n_rows, n_targets), dtype=np.float64)
    out_auroc = np.empty((n_rows, n_targets), dtype=np.float64)
    
    n_ref_f = float(n_ref)
    
    row = 0
    for values, col_indices in csr:
        nnz = len(values)
        
        # Partition values by group
        buf_ref = np.empty(nnz, dtype=np.float64)
        n_ref_nz = 0
        sum_ref = 0.0
        
        # Per-target buffers and sums
        buf_tar_all = np.empty((n_targets, nnz), dtype=np.float64)
        n_tar_nz = np.zeros(n_targets, dtype=np.int64)
        sum_tar = np.zeros(n_targets, dtype=np.float64)
        
        for j in range(nnz):
            col_idx = col_indices[j]
            val = float(values[j])
            g = group_ids[col_idx]
            
            if g == 0:
                buf_ref[n_ref_nz] = val
                sum_ref += val
                n_ref_nz += 1
            elif g > 0 and g <= n_targets:
                t = g - 1  # target index
                buf_tar_all[t, n_tar_nz[t]] = val
                sum_tar[t] += val
                n_tar_nz[t] += 1
        
        mean_ref = sum_ref / n_ref_f
        
        # Sort reference buffer once
        buf_ref_slice = buf_ref[:n_ref_nz]
        buf_ref_slice.sort()
        
        # Find negative boundary for reference
        na_neg_ref = 0
        for i in range(n_ref_nz):
            if buf_ref_slice[i] >= 0.0:
                break
            na_neg_ref += 1
        
        # Process each target
        for t in range(n_targets):
            n_tar = group_counts[t + 1]
            
            if unlikely(n_tar == 0):
                out_u_stats[row, t] = 0.0
                out_p_values[row, t] = 1.0
                out_log2_fc[row, t] = 0.0
                out_auroc[row, t] = 0.5
                continue
            
            n_tar_f = float(n_tar)
            n_tar_nz_t = n_tar_nz[t]
            
            # Constants for this target
            N = n_ref_f + n_tar_f
            n1_n2 = n_ref_f * n_tar_f
            half_n1_n1p1 = 0.5 * n_ref_f * (n_ref_f + 1.0)
            mu = 0.5 * n1_n2  # Expected value under null
            
            # Log2 fold change
            mean_tar = sum_tar[t] / n_tar_f
            out_log2_fc[row, t] = np.log2((mean_tar + EPS) / (mean_ref + EPS))
            
            # Sort target buffer
            buf_tar_slice = buf_tar_all[t, :n_tar_nz_t]
            buf_tar_slice.sort()
            
            # Compute rank sum with ties (for reference group = R1)
            R1 = 0.0
            tie_sum = 0.0  # sum of (t^3 - t) for each tie group
            
            n_ref_zeros = n_ref - n_ref_nz
            n_tar_zeros = n_tar - n_tar_nz_t
            total_zeros = n_ref_zeros + n_tar_zeros
            
            # Find negative boundary for target
            na_neg_tar = 0
            for i in range(n_tar_nz_t):
                if buf_tar_slice[i] >= 0.0:
                    break
                na_neg_tar += 1
            
            rank = 1
            p1 = 0
            p2 = 0
            
            # Merge negative values
            while p1 < na_neg_ref or p2 < na_neg_tar:
                v1 = buf_ref_slice[p1] if p1 < na_neg_ref else INF
                v2 = buf_tar_slice[p2] if p2 < na_neg_tar else INF
                val = v1 if v1 < v2 else v2
                
                count1 = 0
                while p1 < na_neg_ref and buf_ref_slice[p1] == val:
                    count1 += 1
                    p1 += 1
                
                count2 = 0
                while p2 < na_neg_tar and buf_tar_slice[p2] == val:
                    count2 += 1
                    p2 += 1
                
                tc = count1 + count2
                avg_rank = float(rank) + float(tc - 1) * 0.5
                R1 += float(count1) * avg_rank
                
                if tc > 1:
                    td = float(tc)
                    tie_sum += td * td * td - td  # t^3 - t
                
                rank += tc
            
            # Handle zeros
            if total_zeros > 0:
                avg_rank = float(rank) + float(total_zeros - 1) * 0.5
                R1 += float(n_ref_zeros) * avg_rank
                
                if total_zeros > 1:
                    tz = float(total_zeros)
                    tie_sum += tz * tz * tz - tz
                
                rank += total_zeros
            
            # Merge positive values
            p1 = na_neg_ref
            p2 = na_neg_tar
            
            while p1 < n_ref_nz or p2 < n_tar_nz_t:
                v1 = buf_ref_slice[p1] if p1 < n_ref_nz else INF
                v2 = buf_tar_slice[p2] if p2 < n_tar_nz_t else INF
                val = v1 if v1 < v2 else v2
                
                count1 = 0
                while p1 < n_ref_nz and buf_ref_slice[p1] == val:
                    count1 += 1
                    p1 += 1
                
                count2 = 0
                while p2 < n_tar_nz_t and buf_tar_slice[p2] == val:
                    count2 += 1
                    p2 += 1
                
                tc = count1 + count2
                avg_rank = float(rank) + float(tc - 1) * 0.5
                R1 += float(count1) * avg_rank
                
                if tc > 1:
                    td = float(tc)
                    tie_sum += td * td * td - td
                
                rank += tc
            
            # Compute U statistics
            # U1 = R1 - n1*(n1+1)/2 (U for reference group)
            # U2 = n1*n2 - U1 (U for target group)
            U1 = R1 - half_n1_n1p1
            U2 = n1_n2 - U1
            
            out_u_stats[row, t] = U1
            
            # Compute variance with tie correction
            # var = (n1*n2/12) * ((N+1) - tie_sum / (N*(N-1)))
            N_Nm1 = N * (N - 1.0)
            if likely(N_Nm1 > EPS):
                tie_term = tie_sum / N_Nm1
                var = (n1_n2 / 12.0) * (N + 1.0 - tie_term)
            else:
                var = 0.0
            
            sigma = np.sqrt(var) if var > 0.0 else 0.0
            
            # Compute p-value based on alternative hypothesis
            if unlikely(sigma <= SIGMA_MIN):
                out_p_values[row, t] = 1.0
            else:
                # Select U statistic based on alternative
                # scipy: for 'greater', use U1's SF; for 'less', use U2's SF
                # for 'two-sided', use max(U1, U2) and multiply by 2
                if alternative == 1:  # greater
                    U = U1
                    factor = 1.0
                elif alternative == -1:  # less
                    U = U2
                    factor = 1.0
                else:  # two-sided (default)
                    U = U1 if U1 > U2 else U2
                    factor = 2.0
                
                # Compute z-score
                # z = (U - mu) / sigma, with optional continuity correction
                z_numer = U - mu
                if use_continuity:
                    # Continuity correction: subtract 0.5 from |U - mu|
                    # This is equivalent to: z = (U - mu - 0.5) / sigma for SF
                    z_numer -= 0.5
                
                z = z_numer / sigma
                
                # P-value = SF(z) * factor
                p_val = _normal_sf(z) * factor
                
                # Clamp to [0, 1]
                if p_val > 1.0:
                    p_val = 1.0
                if p_val < 0.0:
                    p_val = 0.0
                
                out_p_values[row, t] = p_val
            
            # AUROC: U1 / (n1 * n2) is the probability that a random sample from
            # reference has a higher rank than a random sample from target
            if likely(n1_n2 > 0.0):
                out_auroc[row, t] = U1 / n1_n2
            else:
                out_auroc[row, t] = 0.5
        
        row += 1
    
    return out_u_stats, out_p_values, out_log2_fc, out_auroc
