"""Mann-Whitney U Test for Sparse Matrices.

Vectorized implementation of MWU test for CSR sparse matrices.
Computes per-row U-statistics, p-values, log2FC, and AUROC.

Design:
    - group_ids: 0 = reference group, 1/2/3... = target groups
    - One-vs-all: computes ref vs target_i for all targets at once
    - Output shape: (n_rows, n_targets) for multi-target results

Uses normal approximation with tie correction:
    mu = 0.5 * n1 * n2
    var = (n1 * n2 / 12) * (N + 1 - tie_correction)
    z = (|U - mu| - cc) / sd
    p = 2 * normal_sf(z)

Uses project's CSR sparse matrix type.
"""

import numpy as np
from numba import prange
from scipy import special

from optim import parallel_jit, assume, vectorize
from _binding import CSR

# Import for type hints only
import _numba  # noqa: F401 - registers CSR/CSC types

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
# MWU Test for Sparse Matrix (One-vs-All Design)
# =============================================================================

@parallel_jit
def mwu_test(
    csr: CSR,
    group_ids: np.ndarray,
    n_targets: int
) -> tuple:
    """Perform MWU test: reference (group 0) vs all targets (groups 1..n_targets).
    
    Args:
        csr: CSR sparse matrix (CSRF32 or CSRF64), genes x cells
        group_ids: Group assignment for each column (cell)
                   0 = reference, 1..n_targets = target groups
        n_targets: Number of target groups (excludes reference)
    
    Returns:
        (u_stats, p_values, log2_fc, auroc):
            Each has shape (n_rows, n_targets)
    """
    INV_SQRT2 = 0.7071067811865475
    EPS = 1e-9
    SIGMA_MIN = 1e-12
    
    n_rows = csr.nrows
    n_cols = csr.ncols
    
    assume(n_rows > 0)
    assume(n_targets > 0)
    
    # Count elements in each group
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
        # buf_ref for group 0, buf_tar[t] for group t+1
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
            
            if n_tar == 0:
                out_u_stats[row, t] = 0.0
                out_p_values[row, t] = 1.0
                out_log2_fc[row, t] = 0.0
                out_auroc[row, t] = 0.5
                continue
            
            n_tar_f = float(n_tar)
            n_tar_nz_t = n_tar_nz[t]
            
            # Constants for this target
            N = n_ref_f + n_tar_f
            half_n1_n1p1 = 0.5 * n_ref_f * (n_ref_f + 1.0)
            half_n1_n2 = 0.5 * n_ref_f * n_tar_f
            var_base = n_ref_f * n_tar_f / 12.0
            N_p1 = N + 1.0
            N_Nm1 = N * (N - 1.0)
            inv_N_Nm1 = 1.0 / N_Nm1 if N_Nm1 > EPS else 0.0
            
            # Log2 fold change
            mean_tar = sum_tar[t] / n_tar_f
            out_log2_fc[row, t] = np.log2((mean_tar + EPS) / (mean_ref + EPS))
            
            # Sort target buffer
            buf_tar_slice = buf_tar_all[t, :n_tar_nz_t]
            buf_tar_slice.sort()
            
            # Compute rank sum with ties
            R1 = 0.0
            tie_sum = 0.0
            
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
                INF = 1e308
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
                    tie_sum += td * (td * td - 1.0)
                
                rank += tc
            
            # Handle zeros
            if total_zeros > 0:
                avg_rank = float(rank) + float(total_zeros - 1) * 0.5
                R1 += float(n_ref_zeros) * avg_rank
                
                if total_zeros > 1:
                    tz = float(total_zeros)
                    tie_sum += tz * (tz * tz - 1.0)
                
                rank += total_zeros
            
            # Merge positive values
            p1 = na_neg_ref
            p2 = na_neg_tar
            
            while p1 < n_ref_nz or p2 < n_tar_nz_t:
                INF = 1e308
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
                    tie_sum += td * (td * td - 1.0)
                
                rank += tc
            
            # Compute U and p-value
            U = R1 - half_n1_n1p1
            
            tie_term = tie_sum * inv_N_Nm1
            var = var_base * (N_p1 - tie_term)
            sigma = np.sqrt(var) if var > 0.0 else 0.0
            
            out_u_stats[row, t] = U
            
            if sigma <= SIGMA_MIN:
                out_p_values[row, t] = 1.0
            else:
                z_numer = U - half_n1_n2
                
                if z_numer > 0.5:
                    correction = 0.5
                elif z_numer < -0.5:
                    correction = -0.5
                else:
                    correction = -z_numer
                z_numer += correction
                
                z = z_numer / sigma
                p_val = special.erfc(abs(z) * INV_SQRT2)
                out_p_values[row, t] = p_val
            
            # AUROC
            if n_ref_f * n_tar_f > 0.0:
                out_auroc[row, t] = U / (n_ref_f * n_tar_f) + 0.5
            else:
                out_auroc[row, t] = 0.5
        
        row += 1
    
    return out_u_stats, out_p_values, out_log2_fc, out_auroc
