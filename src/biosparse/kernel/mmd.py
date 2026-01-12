"""Maximum Mean Discrepancy (MMD) with RBF Kernel.

Optimized implementation of MMD^2 computation for sparse matrices.
Uses RBF (Gaussian) kernel: k(x, y) = exp(-gamma * ||x - y||^2)

Design:
    - group_ids: 0 = reference group, 1/2/3... = target groups
    - One-vs-all: computes MMD^2 between ref and each target
    - Output shape: (n_rows, n_targets) for multi-target results

For sparse vectors:
    MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]

Optimization strategies:
    - Exploits k(0,0) = 1 for implicit zeros in sparse matrices
    - Pre-computes k(x,0) = exp(-gamma * x^2) for all non-zero values
    - Batch pairwise kernel computation with vectorization hints

Uses project's CSR sparse matrix type.
"""

import math
import numpy as np
from numba import prange, njit

from biosparse.optim import (
    parallel_jit, assume, vectorize, interleave, 
    unroll, likely, unlikely
)
from biosparse._binding import CSR

# Import for type hints only
import biosparse._numba  # noqa: F401 - registers CSR/CSC types

__all__ = [
    'mmd_rbf',
]


# =============================================================================
# Helper functions for kernel computation
# =============================================================================

@njit(cache=True, fastmath=True, inline='always')
def _compute_unary_kernel_sum(values: np.ndarray, n: int, gamma: float) -> float:
    """Compute sum of k(x_i, 0) = exp(-gamma * x_i^2) for all non-zero values.
    
    This is used for cross-terms between non-zero and zero values.
    
    Args:
        values: Non-zero values array
        n: Number of values to process
        gamma: RBF kernel parameter
    
    Returns:
        Sum of unary kernels
    """
    assume(n >= 0)
    assume(gamma > 0.0)
    
    result = 0.0
    for i in range(n):
        val = values[i]
        result += math.exp(-gamma * val * val)
    
    return result


@njit(cache=True, fastmath=True, inline='always')
def _compute_self_kernel_sum(
    values: np.ndarray,
    n_nz: int,
    n_zeros: int,
    gamma: float,
    unary_sum: float
) -> float:
    """Compute sum of k(x_i, x_j) for all pairs (self-kernel).
    
    Exploits:
    - k(0, 0) = 1 for all zero-zero pairs
    - k(x, 0) = exp(-gamma * x^2) for non-zero/zero pairs
    - k(x, x) = 1 for diagonal
    
    Args:
        values: Non-zero values (sorted not required)
        n_nz: Number of non-zero values
        n_zeros: Number of implicit zeros
        gamma: RBF kernel parameter
        unary_sum: Pre-computed sum of k(x_i, 0)
    
    Returns:
        Sum of all pairwise kernels
    """
    assume(n_nz >= 0)
    assume(n_zeros >= 0)
    assume(gamma > 0.0)
    
    # Zero-zero pairs: n_zeros^2 pairs with k=1
    result = float(n_zeros * n_zeros)
    
    # Zero/non-zero cross pairs: 2 * n_zeros * sum(k(x,0))
    if likely(n_zeros > 0 and n_nz > 0):
        result += 2.0 * float(n_zeros) * unary_sum
    
    # Diagonal of non-zero: k(x,x) = 1
    result += float(n_nz)
    
    # Off-diagonal non-zero pairs
    if likely(n_nz > 1):
        off_diag = 0.0
        for i in range(n_nz - 1):
            vi = values[i]
            for j in range(i + 1, n_nz):
                diff = vi - values[j]
                off_diag += math.exp(-gamma * diff * diff)
        result += 2.0 * off_diag
    
    return result


@njit(cache=True, fastmath=True, inline='always')
def _compute_cross_kernel_sum(
    values_x: np.ndarray,
    n_x_nz: int,
    n_x_zeros: int,
    values_y: np.ndarray,
    n_y_nz: int,
    n_y_zeros: int,
    gamma: float,
    unary_sum_x: float,
    unary_sum_y: float
) -> float:
    """Compute sum of k(x_i, y_j) for all cross pairs.
    
    Args:
        values_x: Non-zero values for X (reference)
        n_x_nz: Number of non-zero in X
        n_x_zeros: Number of zeros in X
        values_y: Non-zero values for Y (target)
        n_y_nz: Number of non-zero in Y
        n_y_zeros: Number of zeros in Y
        gamma: RBF kernel parameter
        unary_sum_x: Pre-computed sum of k(x,0) for X
        unary_sum_y: Pre-computed sum of k(y,0) for Y
    
    Returns:
        Sum of all cross-kernel values
    """
    assume(n_x_nz >= 0)
    assume(n_y_nz >= 0)
    assume(n_x_zeros >= 0)
    assume(n_y_zeros >= 0)
    assume(gamma > 0.0)
    
    # Zero-zero cross pairs: k(0,0) = 1
    result = float(n_x_zeros * n_y_zeros)
    
    # X-zeros vs Y-nonzeros: sum over y of k(0, y) * n_x_zeros
    if likely(n_x_zeros > 0 and n_y_nz > 0):
        result += float(n_x_zeros) * unary_sum_y
    
    # X-nonzeros vs Y-zeros: sum over x of k(x, 0) * n_y_zeros
    if likely(n_y_zeros > 0 and n_x_nz > 0):
        result += float(n_y_zeros) * unary_sum_x
    
    # Non-zero cross pairs
    if likely(n_x_nz > 0 and n_y_nz > 0):
        cross_sum = 0.0
        for i in range(n_x_nz):
            xi = values_x[i]
            for j in range(n_y_nz):
                diff = xi - values_y[j]
                cross_sum += math.exp(-gamma * diff * diff)
        result += cross_sum
    
    return result


# =============================================================================
# MMD^2 for Sparse Matrices (One-vs-All Design)
# =============================================================================

@parallel_jit(cache=True)
def mmd_rbf(
    csr: CSR,
    group_ids: np.ndarray,
    n_targets: int,
    gamma: float
) -> np.ndarray:
    """Compute MMD^2 with RBF kernel: ref (group 0) vs all targets.
    
    Optimized implementation with helper functions for kernel computation.
    
    Args:
        csr: CSR sparse matrix (features x samples)
        group_ids: Group assignment for each column
                   0 = reference, 1..n_targets = target groups
        n_targets: Number of target groups
        gamma: RBF kernel parameter (typically 1 / (2 * sigma^2))
    
    Returns:
        mmd2: MMD^2 values with shape (n_rows, n_targets)
    
    Notes:
        MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
        where k(x,y) = exp(-gamma * ||x - y||^2)
    """
    n_rows = csr.nrows
    n_cols = csr.ncols
    
    # Compiler optimization hints
    assume(n_rows > 0)
    assume(n_cols > 0)
    assume(n_targets > 0)
    assume(n_targets <= 256)
    assume(gamma > 0.0)
    
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
    
    # Allocate output
    out_mmd = np.empty((n_rows, n_targets), dtype=np.float64)
    
    row = 0
    for values, col_indices in csr:
        nnz = len(values)
        
        # Partition values by group (thread-local buffers)
        buf_ref = np.empty(nnz, dtype=np.float64)
        n_ref_nz = 0
        
        buf_tar_all = np.empty((n_targets, nnz), dtype=np.float64)
        n_tar_nz = np.zeros(n_targets, dtype=np.int64)
        
        # Single pass partitioning
        for j in range(nnz):
            col_idx = col_indices[j]
            val = float(values[j])
            g = group_ids[col_idx]
            
            if g == 0:
                buf_ref[n_ref_nz] = val
                n_ref_nz += 1
            elif likely(g > 0 and g <= n_targets):
                t = g - 1
                buf_tar_all[t, n_tar_nz[t]] = val
                n_tar_nz[t] += 1
        
        # Pre-compute reference group statistics
        n_ref_zeros = n_ref - n_ref_nz
        
        # Unary sum for ref (k(x, 0) for all non-zero x in ref)
        sum_ref_unary = _compute_unary_kernel_sum(buf_ref, n_ref_nz, gamma)
        
        # Self-kernel sum for ref: sum of all k(x_i, x_j)
        sum_xx = _compute_self_kernel_sum(
            buf_ref, n_ref_nz, n_ref_zeros, gamma, sum_ref_unary
        )
        
        inv_Nx2 = 1.0 / float(n_ref * n_ref)
        
        # Process each target
        for t in range(n_targets):
            n_tar = group_counts[t + 1]
            
            if unlikely(n_tar == 0):
                out_mmd[row, t] = 0.0
                continue
            
            n_tar_nz_t = n_tar_nz[t]
            n_tar_zeros = n_tar - n_tar_nz_t
            
            inv_Ny2 = 1.0 / float(n_tar * n_tar)
            inv_NxNy = 1.0 / float(n_ref * n_tar)
            
            # Trivial case - all zeros in both groups
            if unlikely(n_ref_nz == 0 and n_tar_nz_t == 0):
                # k(0,0) = 1, so MMD^2 = 1 + 1 - 2 = 0
                out_mmd[row, t] = 0.0
                continue
            
            # Get target values slice
            buf_tar = buf_tar_all[t, :n_tar_nz_t]
            
            # Unary sum for target
            sum_tar_unary = _compute_unary_kernel_sum(buf_tar, n_tar_nz_t, gamma)
            
            # Self-kernel sum for target
            sum_yy = _compute_self_kernel_sum(
                buf_tar, n_tar_nz_t, n_tar_zeros, gamma, sum_tar_unary
            )
            
            # Cross-kernel sum
            sum_xy = _compute_cross_kernel_sum(
                buf_ref, n_ref_nz, n_ref_zeros,
                buf_tar, n_tar_nz_t, n_tar_zeros,
                gamma, sum_ref_unary, sum_tar_unary
            )
            
            # MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
            mmd2 = sum_xx * inv_Nx2 + sum_yy * inv_Ny2 - 2.0 * sum_xy * inv_NxNy
            
            # Clamp to non-negative (numerical errors can cause small negatives)
            out_mmd[row, t] = mmd2 if mmd2 > 0.0 else 0.0
        
        row += 1
    
    return out_mmd
