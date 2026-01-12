"""T-Test Statistical Functions.

Vectorized implementation of Student's t-test and Welch's t-test.
Provides fast p-value computation using normal approximation for large DF.

Functions:
    - welch_test: Welch's t-test (unequal variance)
    - student_test: Student's t-test (pooled variance)
    - welch_se: Welch's standard error
    - welch_df: Welch-Satterthwaite degrees of freedom
    - pooled_se: Pooled standard error
"""

import numpy as np
from numba import prange
from scipy import special

from optim import parallel_jit, assume

__all__ = [
    'welch_test',
    'student_test',
    'welch_se',
    'welch_df',
    'pooled_se',
    'welch_test_approx',
    'student_test_approx',
]


# =============================================================================
# Standard Error and Degrees of Freedom
# =============================================================================

@parallel_jit
def welch_se(
    var1: np.ndarray,
    n1: np.ndarray,
    var2: np.ndarray,
    n2: np.ndarray,
    out: np.ndarray
) -> None:
    """Welch's standard error.
    
    SE = sqrt(var1/n1 + var2/n2)
    
    Args:
        var1: Variances of group 1
        n1: Sample sizes of group 1
        var2: Variances of group 2
        n2: Sample sizes of group 2
        out: Output standard errors
    """
    n = len(var1)
    assume(n > 0)
    assume(len(out) >= n)
    
    for i in prange(n):
        out[i] = np.sqrt(var1[i] / n1[i] + var2[i] / n2[i])


@parallel_jit
def welch_df(
    var1: np.ndarray,
    n1: np.ndarray,
    var2: np.ndarray,
    n2: np.ndarray,
    out: np.ndarray
) -> None:
    """Welch-Satterthwaite degrees of freedom.
    
    df = (v1/n1 + v2/n2)^2 / ((v1/n1)^2/(n1-1) + (v2/n2)^2/(n2-1))
    
    Args:
        var1: Variances of group 1
        n1: Sample sizes of group 1
        var2: Variances of group 2
        n2: Sample sizes of group 2
        out: Output degrees of freedom
    """
    n = len(var1)
    assume(n > 0)
    assume(len(out) >= n)
    
    for i in prange(n):
        v1_n1 = var1[i] / n1[i]
        v2_n2 = var2[i] / n2[i]
        sum_v = v1_n1 + v2_n2
        
        if sum_v < 1e-12:
            out[i] = 1.0
            continue
        
        denom = (v1_n1 * v1_n1) / (n1[i] - 1.0) + (v2_n2 * v2_n2) / (n2[i] - 1.0)
        out[i] = (sum_v * sum_v) / denom


@parallel_jit
def pooled_se(
    var1: np.ndarray,
    n1: np.ndarray,
    var2: np.ndarray,
    n2: np.ndarray,
    out: np.ndarray
) -> None:
    """Pooled standard error for Student's t-test.
    
    SE = sqrt(pooled_var * (1/n1 + 1/n2))
    pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
    
    Args:
        var1: Variances of group 1
        n1: Sample sizes of group 1
        var2: Variances of group 2
        n2: Sample sizes of group 2
        out: Output standard errors
    """
    n = len(var1)
    assume(n > 0)
    assume(len(out) >= n)
    
    for i in prange(n):
        df = n1[i] + n2[i] - 2.0
        if df <= 0.0:
            out[i] = 0.0
            continue
        
        v_pool = ((n1[i] - 1.0) * var1[i] + (n2[i] - 1.0) * var2[i]) / df
        out[i] = np.sqrt(v_pool * (1.0 / n1[i] + 1.0 / n2[i]))


# =============================================================================
# Complete T-Test Functions (Precise)
# =============================================================================

@parallel_jit
def welch_test(
    mean1: np.ndarray,
    var1: np.ndarray,
    n1: np.ndarray,
    mean2: np.ndarray,
    var2: np.ndarray,
    n2: np.ndarray,
    out: np.ndarray
) -> None:
    """Welch's t-test (two-sided p-value, precise).
    
    Welch's t-test does not assume equal variances.
    
    Args:
        mean1: Means of group 1
        var1: Variances of group 1
        n1: Sample sizes of group 1
        mean2: Means of group 2
        var2: Variances of group 2
        n2: Sample sizes of group 2
        out: Output p-values
    """
    INV_SQRT2 = 0.7071067811865475
    
    n = len(mean1)
    assume(n > 0)
    assume(len(out) >= n)
    
    for i in prange(n):
        # Standard error
        v1_n1 = var1[i] / n1[i]
        v2_n2 = var2[i] / n2[i]
        se = np.sqrt(v1_n1 + v2_n2)
        
        if se < 1e-15:
            out[i] = 1.0
            continue
        
        # T-statistic
        t_stat = (mean1[i] - mean2[i]) / se
        
        # Degrees of freedom (Welch-Satterthwaite)
        sum_v = v1_n1 + v2_n2
        if sum_v < 1e-12:
            out[i] = 1.0
            continue
        
        denom = (v1_n1 * v1_n1) / (n1[i] - 1.0) + (v2_n2 * v2_n2) / (n2[i] - 1.0)
        df = (sum_v * sum_v) / denom
        
        # P-value
        abs_t = abs(t_stat)
        
        if df > 30.0:
            # Normal approximation
            sf = 0.5 * special.erfc(abs_t * INV_SQRT2)
            out[i] = 2.0 * sf
        else:
            # Sigmoid heuristic for small DF
            z = abs_t / np.sqrt(df + abs_t * abs_t)
            cdf = 0.5 * (1.0 + z)
            out[i] = 2.0 * (1.0 - cdf)


@parallel_jit
def student_test(
    mean1: np.ndarray,
    var1: np.ndarray,
    n1: np.ndarray,
    mean2: np.ndarray,
    var2: np.ndarray,
    n2: np.ndarray,
    out: np.ndarray
) -> None:
    """Student's t-test (two-sided p-value, precise).
    
    Student's t-test assumes equal variances (uses pooled variance).
    
    Args:
        mean1: Means of group 1
        var1: Variances of group 1
        n1: Sample sizes of group 1
        mean2: Means of group 2
        var2: Variances of group 2
        n2: Sample sizes of group 2
        out: Output p-values
    """
    INV_SQRT2 = 0.7071067811865475
    
    n = len(mean1)
    assume(n > 0)
    assume(len(out) >= n)
    
    for i in prange(n):
        # Degrees of freedom
        df = n1[i] + n2[i] - 2.0
        
        if df <= 0.0:
            out[i] = 1.0
            continue
        
        # Pooled variance and SE
        v_pool = ((n1[i] - 1.0) * var1[i] + (n2[i] - 1.0) * var2[i]) / df
        se = np.sqrt(v_pool * (1.0 / n1[i] + 1.0 / n2[i]))
        
        if se < 1e-15:
            out[i] = 1.0
            continue
        
        # T-statistic
        t_stat = (mean1[i] - mean2[i]) / se
        
        # P-value
        abs_t = abs(t_stat)
        
        if df > 30.0:
            # Normal approximation
            sf = 0.5 * special.erfc(abs_t * INV_SQRT2)
            out[i] = 2.0 * sf
        else:
            # Sigmoid heuristic for small DF
            z = abs_t / np.sqrt(df + abs_t * abs_t)
            cdf = 0.5 * (1.0 + z)
            out[i] = 2.0 * (1.0 - cdf)


# =============================================================================
# Approximate Implementations (faster)
# =============================================================================

@parallel_jit
def welch_test_approx(
    mean1: np.ndarray,
    var1: np.ndarray,
    n1: np.ndarray,
    mean2: np.ndarray,
    var2: np.ndarray,
    n2: np.ndarray,
    out: np.ndarray
) -> None:
    """Welch's t-test (two-sided p-value, approximate).
    
    Uses Abramowitz-Stegun approximation for faster computation.
    
    Args:
        mean1: Means of group 1
        var1: Variances of group 1
        n1: Sample sizes of group 1
        mean2: Means of group 2
        var2: Variances of group 2
        n2: Sample sizes of group 2
        out: Output p-values
    """
    INV_SQRT2 = 0.7071067811865475
    
    n = len(mean1)
    assume(n > 0)
    assume(len(out) >= n)
    
    for i in prange(n):
        # Standard error
        v1_n1 = var1[i] / n1[i]
        v2_n2 = var2[i] / n2[i]
        se = np.sqrt(v1_n1 + v2_n2)
        
        if se < 1e-15:
            out[i] = 1.0
            continue
        
        # T-statistic
        t_stat = (mean1[i] - mean2[i]) / se
        
        # Degrees of freedom
        sum_v = v1_n1 + v2_n2
        if sum_v < 1e-12:
            out[i] = 1.0
            continue
        
        denom = (v1_n1 * v1_n1) / (n1[i] - 1.0) + (v2_n2 * v2_n2) / (n2[i] - 1.0)
        df = (sum_v * sum_v) / denom
        
        # P-value
        abs_t = abs(t_stat)
        
        if df > 30.0:
            # Normal approximation with approx erfc
            arg = abs_t * INV_SQRT2
            ax = abs(arg)
            t = 1.0 / (1.0 + 0.5 * ax)
            
            tau = t * np.exp(
                -ax * ax
                - 1.26551223
                + t * ( 1.00002368
                + t * ( 0.37409196
                + t * ( 0.09678418
                + t * (-0.18628806
                + t * ( 0.27886807
                + t * (-1.13520398
                + t * ( 1.48851587
                + t * (-0.82215223
                + t * ( 0.17087277 )))))))))
            )
            
            r = tau if arg >= 0.0 else 2.0 - tau
            
            if r < 0.0:
                r = 0.0
            elif r > 2.0:
                r = 2.0
            
            sf = 0.5 * r
            out[i] = 2.0 * sf
        else:
            # Sigmoid heuristic for small DF
            z = abs_t / np.sqrt(df + abs_t * abs_t)
            cdf = 0.5 * (1.0 + z)
            out[i] = 2.0 * (1.0 - cdf)


@parallel_jit
def student_test_approx(
    mean1: np.ndarray,
    var1: np.ndarray,
    n1: np.ndarray,
    mean2: np.ndarray,
    var2: np.ndarray,
    n2: np.ndarray,
    out: np.ndarray
) -> None:
    """Student's t-test (two-sided p-value, approximate).
    
    Uses Abramowitz-Stegun approximation for faster computation.
    
    Args:
        mean1: Means of group 1
        var1: Variances of group 1
        n1: Sample sizes of group 1
        mean2: Means of group 2
        var2: Variances of group 2
        n2: Sample sizes of group 2
        out: Output p-values
    """
    INV_SQRT2 = 0.7071067811865475
    
    n = len(mean1)
    assume(n > 0)
    assume(len(out) >= n)
    
    for i in prange(n):
        df = n1[i] + n2[i] - 2.0
        
        if df <= 0.0:
            out[i] = 1.0
            continue
        
        v_pool = ((n1[i] - 1.0) * var1[i] + (n2[i] - 1.0) * var2[i]) / df
        se = np.sqrt(v_pool * (1.0 / n1[i] + 1.0 / n2[i]))
        
        if se < 1e-15:
            out[i] = 1.0
            continue
        
        t_stat = (mean1[i] - mean2[i]) / se
        
        # P-value
        abs_t = abs(t_stat)
        
        if df > 30.0:
            # Normal approximation with approx erfc
            arg = abs_t * INV_SQRT2
            ax = abs(arg)
            t = 1.0 / (1.0 + 0.5 * ax)
            
            tau = t * np.exp(
                -ax * ax
                - 1.26551223
                + t * ( 1.00002368
                + t * ( 0.37409196
                + t * ( 0.09678418
                + t * (-0.18628806
                + t * ( 0.27886807
                + t * (-1.13520398
                + t * ( 1.48851587
                + t * (-0.82215223
                + t * ( 0.17087277 )))))))))
            )
            
            r = tau if arg >= 0.0 else 2.0 - tau
            
            if r < 0.0:
                r = 0.0
            elif r > 2.0:
                r = 2.0
            
            sf = 0.5 * r
            out[i] = 2.0 * sf
        else:
            # Sigmoid heuristic for small DF
            z = abs_t / np.sqrt(df + abs_t * abs_t)
            cdf = 0.5 * (1.0 + z)
            out[i] = 2.0 * (1.0 - cdf)


# =============================================================================
# Convenience Functions (allocating versions)
# =============================================================================

@parallel_jit
def welch_test_new(
    mean1: np.ndarray,
    var1: np.ndarray,
    n1: np.ndarray,
    mean2: np.ndarray,
    var2: np.ndarray,
    n2: np.ndarray
) -> np.ndarray:
    """Allocating version of welch_test."""
    INV_SQRT2 = 0.7071067811865475
    
    n = len(mean1)
    assume(n > 0)
    out = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        # Standard error
        v1_n1 = var1[i] / n1[i]
        v2_n2 = var2[i] / n2[i]
        se = np.sqrt(v1_n1 + v2_n2)
        
        if se < 1e-15:
            out[i] = 1.0
            continue
        
        # T-statistic
        t_stat = (mean1[i] - mean2[i]) / se
        
        # Degrees of freedom
        sum_v = v1_n1 + v2_n2
        if sum_v < 1e-12:
            out[i] = 1.0
            continue
        
        denom = (v1_n1 * v1_n1) / (n1[i] - 1.0) + (v2_n2 * v2_n2) / (n2[i] - 1.0)
        df = (sum_v * sum_v) / denom
        
        # P-value
        abs_t = abs(t_stat)
        
        if df > 30.0:
            sf = 0.5 * special.erfc(abs_t * INV_SQRT2)
            out[i] = 2.0 * sf
        else:
            z = abs_t / np.sqrt(df + abs_t * abs_t)
            cdf = 0.5 * (1.0 + z)
            out[i] = 2.0 * (1.0 - cdf)
    
    return out


@parallel_jit
def student_test_new(
    mean1: np.ndarray,
    var1: np.ndarray,
    n1: np.ndarray,
    mean2: np.ndarray,
    var2: np.ndarray,
    n2: np.ndarray
) -> np.ndarray:
    """Allocating version of student_test."""
    INV_SQRT2 = 0.7071067811865475
    
    n = len(mean1)
    assume(n > 0)
    out = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        df = n1[i] + n2[i] - 2.0
        
        if df <= 0.0:
            out[i] = 1.0
            continue
        
        v_pool = ((n1[i] - 1.0) * var1[i] + (n2[i] - 1.0) * var2[i]) / df
        se = np.sqrt(v_pool * (1.0 / n1[i] + 1.0 / n2[i]))
        
        if se < 1e-15:
            out[i] = 1.0
            continue
        
        t_stat = (mean1[i] - mean2[i]) / se
        
        abs_t = abs(t_stat)
        
        if df > 30.0:
            sf = 0.5 * special.erfc(abs_t * INV_SQRT2)
            out[i] = 2.0 * sf
        else:
            z = abs_t / np.sqrt(df + abs_t * abs_t)
            cdf = 0.5 * (1.0 + z)
            out[i] = 2.0 * (1.0 - cdf)
    
    return out
