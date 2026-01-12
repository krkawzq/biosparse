"""Mann-Whitney U Test.

Vectorized implementation of Mann-Whitney U test p-value computation.
Uses normal approximation with tie correction.

Provides two versions:
    - Precise: Uses math.erfc (Python standard library)
    - Approx: Uses Abramowitz-Stegun approximation (~1e-7 precision)

Formula:
    mu = 0.5 * n1 * n2
    var = (n1 * n2 / 12) * (N + 1 - tie_correction)
    z = (|U - mu| - cc) / sd
    p = 2 * normal_sf(z)  # two-sided
"""

import math
import numpy as np
from numba import prange

from biosparse.optim import parallel_jit, assume

__all__ = [
    'mwu_p_value_two_sided',
    'mwu_p_value_greater',
    'mwu_p_value_less',
    'mwu_p_value_two_sided_approx',
    'mwu_p_value_greater_approx',
    'mwu_p_value_less_approx',
]


# =============================================================================
# Precise Implementations (scipy-based)
# =============================================================================

@parallel_jit
def mwu_p_value_two_sided(
    U: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    tie_sum: np.ndarray,
    cc: float,
    out: np.ndarray
) -> None:
    """Two-sided Mann-Whitney U test p-value (precise).
    
    Args:
        U: U statistic values
        n1: Sample sizes of group 1
        n2: Sample sizes of group 2
        tie_sum: Sum of t^3 - t for each tie group (0 if no ties)
        cc: Continuity correction (typically 0.5)
        out: Output p-values
    """
    INV_SQRT2 = 0.7071067811865475
    
    n = len(U)
    assume(n > 0)
    assume(len(out) >= n)
    
    for i in prange(n):
        # Compute moments
        N = n1[i] + n2[i]
        mu = 0.5 * n1[i] * n2[i]
        denom = N * (N - 1.0)
        base = n1[i] * n2[i] / 12.0
        
        if denom > 1e-9:
            var = base * (N + 1.0 - tie_sum[i] / denom)
        else:
            var = base * (N + 1.0)
        
        if var < 0.0:
            var = 0.0
        
        sd = np.sqrt(var)
        
        if sd <= 0.0:
            out[i] = 1.0
            continue
        
        diff = abs(U[i] - mu) - cc
        if diff < 0.0:
            diff = 0.0
        
        z = diff / sd
        
        # normal_sf using scipy.math.erfc
        sf = 0.5 * math.erfc(z * INV_SQRT2)
        out[i] = 2.0 * sf


@parallel_jit
def mwu_p_value_greater(
    U: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    tie_sum: np.ndarray,
    cc: float,
    out: np.ndarray
) -> None:
    """One-sided Mann-Whitney U test p-value (greater alternative).
    
    Tests if group 1 tends to have larger values than group 2.
    
    Args:
        U: U statistic values
        n1: Sample sizes of group 1
        n2: Sample sizes of group 2
        tie_sum: Sum of t^3 - t for each tie group
        cc: Continuity correction
        out: Output p-values
    """
    INV_SQRT2 = 0.7071067811865475
    
    n = len(U)
    assume(n > 0)
    assume(len(out) >= n)
    
    for i in prange(n):
        # Compute moments
        N = n1[i] + n2[i]
        mu = 0.5 * n1[i] * n2[i]
        denom = N * (N - 1.0)
        base = n1[i] * n2[i] / 12.0
        
        if denom > 1e-9:
            var = base * (N + 1.0 - tie_sum[i] / denom)
        else:
            var = base * (N + 1.0)
        
        if var < 0.0:
            var = 0.0
        
        sd = np.sqrt(var)
        
        if sd <= 0.0:
            out[i] = 0.0 if U[i] > mu else 1.0
            continue
        
        z = (U[i] - mu - cc) / sd
        
        # normal_sf
        out[i] = 0.5 * math.erfc(z * INV_SQRT2)


@parallel_jit
def mwu_p_value_less(
    U: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    tie_sum: np.ndarray,
    cc: float,
    out: np.ndarray
) -> None:
    """One-sided Mann-Whitney U test p-value (less alternative).
    
    Tests if group 1 tends to have smaller values than group 2.
    
    Args:
        U: U statistic values
        n1: Sample sizes of group 1
        n2: Sample sizes of group 2
        tie_sum: Sum of t^3 - t for each tie group
        cc: Continuity correction
        out: Output p-values
    """
    INV_SQRT2 = 0.7071067811865475
    
    n = len(U)
    assume(n > 0)
    assume(len(out) >= n)
    
    for i in prange(n):
        # Compute moments
        N = n1[i] + n2[i]
        mu = 0.5 * n1[i] * n2[i]
        denom = N * (N - 1.0)
        base = n1[i] * n2[i] / 12.0
        
        if denom > 1e-9:
            var = base * (N + 1.0 - tie_sum[i] / denom)
        else:
            var = base * (N + 1.0)
        
        if var < 0.0:
            var = 0.0
        
        sd = np.sqrt(var)
        
        if sd <= 0.0:
            out[i] = 0.0 if U[i] < mu else 1.0
            continue
        
        z = (mu - U[i] - cc) / sd
        
        # normal_sf
        out[i] = 0.5 * math.erfc(z * INV_SQRT2)


# =============================================================================
# Approximate Implementations (faster, ~1e-7 precision)
# =============================================================================

@parallel_jit
def mwu_p_value_two_sided_approx(
    U: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    tie_sum: np.ndarray,
    cc: float,
    out: np.ndarray
) -> None:
    """Two-sided Mann-Whitney U test p-value (approximate).
    
    Uses Abramowitz-Stegun approximation for erfc.
    
    Args:
        U: U statistic values
        n1: Sample sizes of group 1
        n2: Sample sizes of group 2
        tie_sum: Sum of t^3 - t for each tie group
        cc: Continuity correction
        out: Output p-values
    """
    INV_SQRT2 = 0.7071067811865475
    
    n = len(U)
    assume(n > 0)
    assume(len(out) >= n)
    
    for i in prange(n):
        # Compute moments
        N = n1[i] + n2[i]
        mu = 0.5 * n1[i] * n2[i]
        denom = N * (N - 1.0)
        base = n1[i] * n2[i] / 12.0
        
        if denom > 1e-9:
            var = base * (N + 1.0 - tie_sum[i] / denom)
        else:
            var = base * (N + 1.0)
        
        if var < 0.0:
            var = 0.0
        
        sd = np.sqrt(var)
        
        if sd <= 0.0:
            out[i] = 1.0
            continue
        
        diff = abs(U[i] - mu) - cc
        if diff < 0.0:
            diff = 0.0
        
        z = diff / sd
        
        # Approx erfc
        arg = z * INV_SQRT2
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


@parallel_jit
def mwu_p_value_greater_approx(
    U: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    tie_sum: np.ndarray,
    cc: float,
    out: np.ndarray
) -> None:
    """One-sided Mann-Whitney U test p-value, greater (approximate).
    
    Args:
        U: U statistic values
        n1: Sample sizes of group 1
        n2: Sample sizes of group 2
        tie_sum: Sum of t^3 - t for each tie group
        cc: Continuity correction
        out: Output p-values
    """
    INV_SQRT2 = 0.7071067811865475
    
    n = len(U)
    assume(n > 0)
    assume(len(out) >= n)
    
    for i in prange(n):
        # Compute moments
        N = n1[i] + n2[i]
        mu = 0.5 * n1[i] * n2[i]
        denom = N * (N - 1.0)
        base = n1[i] * n2[i] / 12.0
        
        if denom > 1e-9:
            var = base * (N + 1.0 - tie_sum[i] / denom)
        else:
            var = base * (N + 1.0)
        
        if var < 0.0:
            var = 0.0
        
        sd = np.sqrt(var)
        
        if sd <= 0.0:
            out[i] = 0.0 if U[i] > mu else 1.0
            continue
        
        z = (U[i] - mu - cc) / sd
        
        # Approx erfc
        arg = z * INV_SQRT2
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
        
        out[i] = 0.5 * r


@parallel_jit
def mwu_p_value_less_approx(
    U: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    tie_sum: np.ndarray,
    cc: float,
    out: np.ndarray
) -> None:
    """One-sided Mann-Whitney U test p-value, less (approximate).
    
    Args:
        U: U statistic values
        n1: Sample sizes of group 1
        n2: Sample sizes of group 2
        tie_sum: Sum of t^3 - t for each tie group
        cc: Continuity correction
        out: Output p-values
    """
    INV_SQRT2 = 0.7071067811865475
    
    n = len(U)
    assume(n > 0)
    assume(len(out) >= n)
    
    for i in prange(n):
        # Compute moments
        N = n1[i] + n2[i]
        mu = 0.5 * n1[i] * n2[i]
        denom = N * (N - 1.0)
        base = n1[i] * n2[i] / 12.0
        
        if denom > 1e-9:
            var = base * (N + 1.0 - tie_sum[i] / denom)
        else:
            var = base * (N + 1.0)
        
        if var < 0.0:
            var = 0.0
        
        sd = np.sqrt(var)
        
        if sd <= 0.0:
            out[i] = 0.0 if U[i] < mu else 1.0
            continue
        
        z = (mu - U[i] - cc) / sd
        
        # Approx erfc
        arg = z * INV_SQRT2
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
        
        out[i] = 0.5 * r


# =============================================================================
# Convenience Functions (allocating versions)
# =============================================================================

@parallel_jit
def mwu_p_value_two_sided_new(
    U: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    tie_sum: np.ndarray,
    cc: float
) -> np.ndarray:
    """Allocating version of mwu_p_value_two_sided."""
    INV_SQRT2 = 0.7071067811865475
    
    n = len(U)
    assume(n > 0)
    out = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        # Compute moments
        N = n1[i] + n2[i]
        mu = 0.5 * n1[i] * n2[i]
        denom = N * (N - 1.0)
        base = n1[i] * n2[i] / 12.0
        
        if denom > 1e-9:
            var = base * (N + 1.0 - tie_sum[i] / denom)
        else:
            var = base * (N + 1.0)
        
        if var < 0.0:
            var = 0.0
        
        sd = np.sqrt(var)
        
        if sd <= 0.0:
            out[i] = 1.0
            continue
        
        diff = abs(U[i] - mu) - cc
        if diff < 0.0:
            diff = 0.0
        
        z = diff / sd
        
        sf = 0.5 * math.erfc(z * INV_SQRT2)
        out[i] = 2.0 * sf
    
    return out


@parallel_jit
def mwu_p_value_two_sided_approx_new(
    U: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    tie_sum: np.ndarray,
    cc: float
) -> np.ndarray:
    """Allocating version of mwu_p_value_two_sided_approx."""
    INV_SQRT2 = 0.7071067811865475
    
    n = len(U)
    assume(n > 0)
    out = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        # Compute moments
        N = n1[i] + n2[i]
        mu = 0.5 * n1[i] * n2[i]
        denom = N * (N - 1.0)
        base = n1[i] * n2[i] / 12.0
        
        if denom > 1e-9:
            var = base * (N + 1.0 - tie_sum[i] / denom)
        else:
            var = base * (N + 1.0)
        
        if var < 0.0:
            var = 0.0
        
        sd = np.sqrt(var)
        
        if sd <= 0.0:
            out[i] = 1.0
            continue
        
        diff = abs(U[i] - mu) - cc
        if diff < 0.0:
            diff = 0.0
        
        z = diff / sd
        
        # Approx erfc
        arg = z * INV_SQRT2
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
    
    return out
