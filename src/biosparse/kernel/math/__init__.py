"""Mathematical and Statistical Kernels.

This module provides high-performance vectorized mathematical functions
for statistical computing, optimized with Numba JIT.

Strategy:
    - Use Python's math module functions which Numba supports natively
    - Provide Numba-compatible implementations for complex algorithms
    - All functions compile to pure C code via Numba
    - Use optim module for LLVM intrinsics and loop hints

Submodules:
    stats: Basic statistical distribution functions (erfc, normal_sf, etc.)
    tdist: Student's t-distribution functions
    mwu: Mann-Whitney U test utilities
    ttest: T-test utilities
    descriptive: Descriptive statistics (median, MAD, quantile, etc.)
    regression: Regression algorithms (LOESS, weighted polyfit)
"""

# =============================================================================
# Statistical Distribution Functions
# =============================================================================

from ._stats import (
    # Precise (math library based)
    erfc,
    erf,
    normal_cdf,
    normal_sf,
    normal_pdf,
    normal_logcdf,
    normal_logsf,
    
    # Approximate versions (faster)
    erfc_approx,
    normal_sf_approx,
    normal_cdf_approx,
)

# =============================================================================
# Student's t-distribution
# =============================================================================

from ._tdist import (
    stdtr,
    stdtr_sf,
    t_test_pvalue,
    t_test_pvalue_batch,
    t_cdf_two_sided,
    betainc,
)

# =============================================================================
# Mann-Whitney U Test Utilities
# =============================================================================

from ._mwu import (
    mwu_p_value_two_sided,
    mwu_p_value_greater,
    mwu_p_value_less,
    mwu_p_value_two_sided_approx,
    mwu_p_value_greater_approx,
    mwu_p_value_less_approx,
)

# =============================================================================
# T-Test Utilities
# =============================================================================

from ._ttest import (
    welch_test,
    student_test,
    welch_se,
    welch_df,
    pooled_se,
)

# =============================================================================
# Descriptive Statistics
# =============================================================================

from ._descriptive import (
    # Core statistics
    median,
    mad,
    quantile,
    percentile,
    quantiles_batch,
    
    # Sorting utilities
    argsort_full,
    argpartition,
    
    # Binning utilities
    assign_bins_equal_width,
    assign_bins_by_quantiles,
    compute_bin_edges_quantile,
    
    # Group statistics
    group_mean_std,
    group_median_mad,
    
    # Convenience
    zscore_by_bin,
    zscore_by_bin_mad,
)

# =============================================================================
# Regression
# =============================================================================

from ._regression import (
    loess_fit,
    loess_fit_sorted,
    loess_fit_parallel,
    weighted_polyfit_1,
    weighted_polyfit_2,
    tricube_weight,
    compute_vst_clip_values,
    compute_normalized_variance,
)

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Stats - precise
    'erfc',
    'erf',
    'normal_cdf',
    'normal_sf',
    'normal_pdf',
    'normal_logcdf',
    'normal_logsf',
    
    # Stats - approx
    'erfc_approx',
    'normal_sf_approx',
    'normal_cdf_approx',
    
    # T-distribution
    'stdtr',
    'stdtr_sf',
    't_test_pvalue',
    't_test_pvalue_batch',
    't_cdf_two_sided',
    'betainc',
    
    # MWU
    'mwu_p_value_two_sided',
    'mwu_p_value_greater',
    'mwu_p_value_less',
    'mwu_p_value_two_sided_approx',
    'mwu_p_value_greater_approx',
    'mwu_p_value_less_approx',
    
    # T-test
    'welch_test',
    'student_test',
    'welch_se',
    'welch_df',
    'pooled_se',
    
    # Descriptive statistics
    'median',
    'mad',
    'quantile',
    'percentile',
    'quantiles_batch',
    'argsort_full',
    'argpartition',
    'assign_bins_equal_width',
    'assign_bins_by_quantiles',
    'compute_bin_edges_quantile',
    'group_mean_std',
    'group_median_mad',
    'zscore_by_bin',
    'zscore_by_bin_mad',
    
    # Regression
    'loess_fit',
    'loess_fit_sorted',
    'loess_fit_parallel',
    'weighted_polyfit_1',
    'weighted_polyfit_2',
    'tricube_weight',
    'compute_vst_clip_values',
    'compute_normalized_variance',
]
