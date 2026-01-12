"""Maximum Mean Discrepancy (MMD) with RBF Kernel.

Vectorized implementation of MMD^2 computation for sparse matrices.
Uses RBF (Gaussian) kernel: k(x, y) = exp(-gamma * ||x - y||^2)

Design:
    - group_ids: 0 = reference group, 1/2/3... = target groups
    - One-vs-all: computes MMD^2 between ref and each target
    - Output shape: (n_rows, n_targets) for multi-target results

For sparse vectors:
    MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]

Uses project's CSR sparse matrix type.
"""

import numpy as np
from numba import prange

from biosparse.optim import parallel_jit, assume, vectorize
from biosparse._binding import CSR

# Import for type hints only
import biosparse._numba  # noqa: F401 - registers CSR/CSC types

__all__ = [
    'mmd_rbf',
]


# =============================================================================
# MMD^2 for Sparse Matrices (One-vs-All Design)
# =============================================================================

@parallel_jit
def mmd_rbf(
    csr: CSR,
    group_ids: np.ndarray,
    n_targets: int,
    gamma: float
) -> np.ndarray:
    """Compute MMD^2 with RBF kernel: ref (group 0) vs all targets.
    
    Optimized with prange for parallel row processing.
    
    Args:
        csr: CSR sparse matrix (features x samples)
        group_ids: Group assignment for each column
                   0 = reference, 1..n_targets = target groups
        n_targets: Number of target groups
        gamma: RBF kernel parameter (typically 1 / (2 * sigma^2))
    
    Returns:
        mmd2: MMD^2 values with shape (n_rows, n_targets)
    """
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
    
    # Allocate output
    out_mmd = np.empty((n_rows, n_targets), dtype=np.float64)
    
    # Parallel row processing
    for row in prange(n_rows):
        values, col_indices = csr.row(row)
        nnz = len(values)
        
        # Partition values by group (thread-local buffers)
        buf_ref = np.empty(nnz, dtype=np.float64)
        n_ref_nz = 0
        
        buf_tar_all = np.empty((n_targets, nnz), dtype=np.float64)
        n_tar_nz = np.zeros(n_targets, dtype=np.int64)
        
        for j in range(nnz):
            col_idx = col_indices[j]
            val = float(values[j])
            g = group_ids[col_idx]
            
            if g == 0:
                buf_ref[n_ref_nz] = val
                n_ref_nz += 1
            elif g > 0 and g <= n_targets:
                t = g - 1
                buf_tar_all[t, n_tar_nz[t]] = val
                n_tar_nz[t] += 1
        
        # Precompute reference self-kernel
        n_ref_zeros = n_ref - n_ref_nz
        
        # Unary sum for ref
        sum_ref_unary = 0.0
        for k in range(n_ref_nz):
            val = buf_ref[k]
            sum_ref_unary += np.exp(-gamma * val * val)
        
        # Self kernel sum for ref
        sum_xx = float(n_ref_zeros * n_ref_zeros)  # k(0,0) pairs
        if n_ref_zeros > 0:
            sum_xx += 2.0 * float(n_ref_zeros) * sum_ref_unary  # k(0,x) pairs
        sum_xx += float(n_ref_nz)  # k(x,x) diagonal
        
        if n_ref_nz > 1:
            off_diag = 0.0
            for k in range(n_ref_nz - 1):
                vi = buf_ref[k]
                for m in range(k + 1, n_ref_nz):
                    diff = vi - buf_ref[m]
                    off_diag += np.exp(-gamma * diff * diff)
            sum_xx += 2.0 * off_diag
        
        inv_Nx2 = 1.0 / float(n_ref * n_ref)
        
        # Process each target
        for t in range(n_targets):
            n_tar = group_counts[t + 1]
            
            if n_tar == 0:
                out_mmd[row, t] = 0.0
                continue
            
            n_tar_nz_t = n_tar_nz[t]
            n_tar_zeros = n_tar - n_tar_nz_t
            
            inv_Ny2 = 1.0 / float(n_tar * n_tar)
            inv_NxNy = 1.0 / float(n_ref * n_tar)
            
            # Trivial case
            if n_ref_nz == 0 and n_tar_nz_t == 0:
                out_mmd[row, t] = 0.0
                continue
            
            # Unary sum for target
            sum_tar_unary = 0.0
            for k in range(n_tar_nz_t):
                val = buf_tar_all[t, k]
                sum_tar_unary += np.exp(-gamma * val * val)
            
            # Self kernel sum for target
            sum_yy = float(n_tar_zeros * n_tar_zeros)
            if n_tar_zeros > 0:
                sum_yy += 2.0 * float(n_tar_zeros) * sum_tar_unary
            sum_yy += float(n_tar_nz_t)
            
            if n_tar_nz_t > 1:
                off_diag = 0.0
                for k in range(n_tar_nz_t - 1):
                    vi = buf_tar_all[t, k]
                    for m in range(k + 1, n_tar_nz_t):
                        diff = vi - buf_tar_all[t, m]
                        off_diag += np.exp(-gamma * diff * diff)
                sum_yy += 2.0 * off_diag
            
            # Cross kernel sum
            sum_xy = float(n_ref_zeros * n_tar_zeros)
            if n_ref_zeros > 0:
                sum_xy += float(n_ref_zeros) * sum_tar_unary
            if n_tar_zeros > 0:
                sum_xy += float(n_tar_zeros) * sum_ref_unary
            
            if n_ref_nz > 0 and n_tar_nz_t > 0:
                cross_sum = 0.0
                for k in range(n_ref_nz):
                    xi = buf_ref[k]
                    for m in range(n_tar_nz_t):
                        diff = xi - buf_tar_all[t, m]
                        cross_sum += np.exp(-gamma * diff * diff)
                sum_xy += cross_sum
            
            # MMD^2
            mmd2 = sum_xx * inv_Nx2 + sum_yy * inv_Ny2 - 2.0 * sum_xy * inv_NxNy
            
            out_mmd[row, t] = mmd2 if mmd2 > 0.0 else 0.0
    
    return out_mmd
