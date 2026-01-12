"""Mathematical and Statistical Kernels.

This module provides high-performance vectorized mathematical functions
for statistical computing, optimized with Numba JIT.

Strategy:
    - Use scipy functions when they can be JIT-compiled (scipy.special.*)
    - Provide Numba-compatible implementations for functions not supported by Numba
    - Provide approx versions for maximum performance when precision is not critical
    - All functions accept numpy arrays (vectorized)

Submodules:
    stats: Basic statistical distribution functions (erfc, normal_sf, normal_cdf, etc.)
    tdist: Student's t-distribution functions (Numba-compatible)
    mwu: Mann-Whitney U test
    ttest: T-test (Welch and Student)
"""

from ._stats import (
    # Scipy-based (precise)
    erfc,
    erf,
    normal_cdf,
    normal_sf,
    normal_pdf,
    normal_logcdf,
    normal_logsf,
    
    # Approx versions (faster)
    erfc_approx,
    normal_sf_approx,
    normal_cdf_approx,
)

from ._tdist import (
    # Student's t-distribution (Numba-compatible, matches scipy.special.stdtr)
    stdtr,
    stdtr_sf,
    t_test_pvalue,
    t_test_pvalue_batch,
    t_cdf_two_sided,
    betainc,
)

from ._mwu import (
    mwu_p_value_two_sided,
    mwu_p_value_greater,
    mwu_p_value_less,
    
    # Approx versions
    mwu_p_value_two_sided_approx,
    mwu_p_value_greater_approx,
    mwu_p_value_less_approx,
)

from ._ttest import (
    welch_test,
    student_test,
    welch_se,
    welch_df,
    pooled_se,
)

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
    
    # T-distribution (Numba-compatible)
    'stdtr',
    'stdtr_sf',
    't_test_pvalue',
    't_test_pvalue_batch',
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
]
