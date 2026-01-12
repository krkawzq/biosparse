"""BioSparse Kernel Module.

High-performance computational kernels for bioinformatics,
implemented using Numba JIT with optimization hints.

Design Pattern (One-vs-All):
    All group-based kernels use the same convention:
    - group_ids: 0 = reference group, 1/2/3... = target groups
    - One-vs-all: computes ref vs target_i for all targets at once
    - Output shape: (n_rows, n_targets) for multi-target results

Quick Start:
    from biosparse.kernel import mann_whitney_u, t_test, mmd, highly_variable_genes

    # Differential expression
    u_stats, pvals, log2fc, auroc = mann_whitney_u(csr, groups, n_targets)
    t_stats, pvals, log2fc = t_test(csr, groups, n_targets)
    
    # MMD distance
    mmd_scores = mmd(csr, groups, n_targets, gamma=1.0)
    
    # HVG selection (multiple flavors)
    indices, mask, means, disps, disps_norm = hvg_seurat(csr, n_top=2000)
    indices, mask, means, vars, vars_norm = hvg_seurat_v3(csr, n_top=2000)
    indices, mask, means, vars, res_vars = hvg_pearson_residuals(csr, n_top=2000)

All sparse matrix kernels accept the project's CSR type directly.
"""

# Submodules
from . import math
from . import hvg as _hvg
from . import mwu as _mwu
from . import mmd as _mmd
from . import ttest as _ttest

# =============================================================================
# Statistical Tests (with elegant aliases)
# =============================================================================

# Mann-Whitney U test
from .mwu import mwu_test, mwu_test_csr_arrays
mann_whitney_u = mwu_test  # Elegant alias

# T-tests
from .ttest import ttest, welch_ttest, student_ttest
t_test = ttest            # Elegant alias
t_test_welch = welch_ttest
t_test_student = student_ttest

# Maximum Mean Discrepancy
from .mmd import mmd_rbf
mmd = mmd_rbf             # Elegant alias

# =============================================================================
# HVG Selection (All Flavors)
# =============================================================================

from .hvg import (
    # Primary API - Flavor-specific
    hvg_seurat,
    hvg_cell_ranger,
    hvg_seurat_v3,
    hvg_pearson_residuals,
    
    # Legacy API
    select_hvg_by_dispersion,
    
    # Building blocks
    compute_dispersion,
    normalize_dispersion,
    select_top_k,
    select_top_k_sorted,
    compute_moments,
    compute_clipped_moments,
)

# Elegant aliases
highly_variable_genes = hvg_seurat  # Default to Seurat flavor
dispersion = compute_dispersion
moments = compute_moments

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Submodules (for advanced usage)
    'math',
    
    # --- Statistical Tests (Primary API) ---
    # Mann-Whitney U
    'mann_whitney_u',       # Elegant name
    'mwu_test',             # Original name
    'mwu_test_csr_arrays',  # Low-level API
    
    # T-tests
    't_test',               # Elegant name (default: Welch)
    't_test_welch',
    't_test_student',
    'ttest',                # Original name
    'welch_ttest',
    'student_ttest',
    
    # MMD
    'mmd',                  # Elegant name
    'mmd_rbf',              # Original name
    
    # --- HVG Selection (Primary API) ---
    # Flavor-specific functions
    'hvg_seurat',           # Seurat flavor (binning + mean/std)
    'hvg_cell_ranger',      # Cell Ranger flavor (percentile + MAD)
    'hvg_seurat_v3',        # Seurat V3 flavor (VST + LOESS)
    'hvg_pearson_residuals', # Pearson residuals flavor
    
    # Legacy/elegant aliases
    'highly_variable_genes',  # Elegant name (defaults to seurat)
    'select_hvg_by_dispersion',
    
    # HVG utilities
    'dispersion',           # Elegant name
    'compute_dispersion',
    'normalize_dispersion',
    'moments',              # Elegant name
    'compute_moments',
    'compute_clipped_moments',
    'select_top_k',
    'select_top_k_sorted',
]
