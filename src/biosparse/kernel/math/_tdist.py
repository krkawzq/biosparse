"""Student's t-distribution functions for Numba.

Implements the t-distribution CDF using the regularized incomplete beta function.
This is compatible with Numba JIT compilation.

The relationship is:
    stdtr(df, t) = 1 - 0.5 * betainc(df/2, 0.5, x)  for t >= 0
    stdtr(df, t) = 0.5 * betainc(df/2, 0.5, x)      for t < 0
    where x = df / (df + t^2)

For two-sided p-value:
    p = 2 * stdtr(df, -|t|) = betainc(df/2, 0.5, x)
    where x = df / (df + t^2)

Optimization:
    - All helper functions use inline='always' for maximum performance
    - Uses assume/likely/unlikely hints for branch prediction
    - Uses fastmath for numerical operations
"""

import math
import numpy as np
from numba import njit, prange

from biosparse.optim import assume, likely, unlikely


# =============================================================================
# Constants
# =============================================================================

_FPMIN = 1e-300  # Minimum float to avoid division by zero
_EPS = 3e-14     # Convergence tolerance for continued fraction
_MAX_ITER = 200  # Maximum iterations for continued fraction


# =============================================================================
# Gamma function (log)
# =============================================================================

@njit(cache=True, fastmath=True, inline='always')
def _lgamma(x: float) -> float:
    """Log-gamma function.
    
    Uses math.lgamma which Numba supports natively.
    """
    return math.lgamma(x)


# =============================================================================
# Beta function (log)
# =============================================================================

@njit(cache=True, fastmath=True, inline='always')
def _lbeta(a: float, b: float) -> float:
    """Log of the beta function B(a, b) = Gamma(a) * Gamma(b) / Gamma(a+b)."""
    return _lgamma(a) + _lgamma(b) - _lgamma(a + b)


# =============================================================================
# Regularized Incomplete Beta Function - Continued Fraction
# =============================================================================

@njit(cache=True, fastmath=True, inline='always')
def _betainc_cf(a: float, b: float, x: float) -> float:
    """Continued fraction expansion for incomplete beta function.
    
    Uses Lentz's algorithm for continued fraction evaluation.
    This computes I_x(a, b) for x < (a+1)/(a+b+2).
    
    Reference: Numerical Recipes, Press et al.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    
    # First step of Lentz's method
    c = 1.0
    d = 1.0 - qab * x / qap
    
    if unlikely(abs(d) < _FPMIN):
        d = _FPMIN
    d = 1.0 / d
    h = d
    
    for m in range(1, _MAX_ITER + 1):
        m_f = float(m)
        m2 = 2.0 * m_f
        
        # Even step
        aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if unlikely(abs(d) < _FPMIN):
            d = _FPMIN
        c = 1.0 + aa / c
        if unlikely(abs(c) < _FPMIN):
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        
        # Odd step
        aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if unlikely(abs(d) < _FPMIN):
            d = _FPMIN
        c = 1.0 + aa / c
        if unlikely(abs(c) < _FPMIN):
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        
        if likely(abs(delta - 1.0) < _EPS):
            break
    
    return h


@njit(cache=True, fastmath=True, inline='always')
def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b).
    
    Args:
        a: First shape parameter (> 0)
        b: Second shape parameter (> 0)
        x: Upper limit of integration (0 <= x <= 1)
    
    Returns:
        The regularized incomplete beta function value.
    """
    # Handle edge cases
    if unlikely(x <= 0.0):
        return 0.0
    if unlikely(x >= 1.0):
        return 1.0
    
    # Assume valid parameters
    assume(a > 0.0)
    assume(b > 0.0)
    
    # For numerical stability, use the symmetry relation:
    # I_x(a, b) = 1 - I_{1-x}(b, a)
    # Choose the continued fraction that converges faster
    
    threshold = (a + 1.0) / (a + b + 2.0)
    
    if likely(x < threshold):
        # Use direct continued fraction (more common case for t-distribution)
        # I_x(a, b) = x^a * (1-x)^b / (a * B(a,b)) * CF
        bt = math.exp(
            a * math.log(x) + b * math.log(1.0 - x) - _lbeta(a, b)
        )
        return bt * _betainc_cf(a, b, x) / a
    else:
        # Use symmetry relation for faster convergence
        bt = math.exp(
            b * math.log(1.0 - x) + a * math.log(x) - _lbeta(a, b)
        )
        return 1.0 - bt * _betainc_cf(b, a, 1.0 - x) / b


# =============================================================================
# Student's t-distribution CDF
# =============================================================================

@njit(cache=True, fastmath=True, inline='always')
def stdtr(df: float, t: float) -> float:
    """Student's t-distribution CDF.
    
    Computes P(T <= t) where T follows a t-distribution with df degrees of freedom.
    
    This is equivalent to scipy.special.stdtr but Numba-compatible.
    
    Args:
        df: Degrees of freedom (> 0)
        t: The value at which to evaluate the CDF
    
    Returns:
        The cumulative distribution function value.
    """
    if unlikely(df <= 0.0):
        return math.nan
    
    assume(df > 0.0)
    
    # x = df / (df + t^2)
    t2 = t * t
    x = df / (df + t2)
    
    # I_x(df/2, 0.5) gives 2 * min(cdf, sf)
    half_df = df * 0.5
    p = 0.5 * betainc(half_df, 0.5, x)
    
    # Adjust based on sign of t
    if likely(t >= 0.0):
        return 1.0 - p
    else:
        return p


@njit(cache=True, fastmath=True, inline='always')
def stdtr_sf(df: float, t: float) -> float:
    """Student's t-distribution survival function (1 - CDF).
    
    Computes P(T > t) = 1 - P(T <= t).
    
    Args:
        df: Degrees of freedom (> 0)
        t: The value at which to evaluate the SF
    
    Returns:
        The survival function value.
    """
    return stdtr(df, -t)


@njit(cache=True, fastmath=True, inline='always')
def t_test_pvalue(t_stat: float, df: float, alternative: int = 0) -> float:
    """Compute p-value for t-test.
    
    Args:
        t_stat: The t-statistic
        df: Degrees of freedom
        alternative: 
            0 = two-sided (default)
            -1 = less (one-sided, left tail)
            1 = greater (one-sided, right tail)
    
    Returns:
        The p-value.
    """
    if unlikely(alternative == -1):  # less
        return stdtr(df, t_stat)
    elif unlikely(alternative == 1):  # greater
        return stdtr_sf(df, t_stat)
    else:  # two-sided (default, most common)
        assume(alternative == 0)
        abs_t = abs(t_stat)
        # p = 2 * P(T > |t|) = 2 * stdtr(df, -|t|)
        return 2.0 * stdtr(df, -abs_t)


# =============================================================================
# Vectorized versions using prange
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def t_test_pvalue_batch(t_stats: np.ndarray, dfs: np.ndarray, alternative: int = 0) -> np.ndarray:
    """Compute p-values for multiple t-tests.
    
    Args:
        t_stats: Array of t-statistics
        dfs: Array of degrees of freedom (same shape as t_stats)
        alternative: 0=two-sided, -1=less, 1=greater
    
    Returns:
        Array of p-values with same shape as input.
    """
    n = len(t_stats)
    assume(n > 0)
    assume(len(dfs) == n)
    
    out = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        out[i] = t_test_pvalue(t_stats[i], dfs[i], alternative)
    
    return out


# =============================================================================
# Additional utility functions
# =============================================================================

@njit(cache=True, fastmath=True, inline='always')
def t_cdf_two_sided(t_stat: float, df: float) -> float:
    """Two-sided t-distribution CDF (p-value).
    
    Convenience function for the most common use case.
    
    Args:
        t_stat: The t-statistic
        df: Degrees of freedom
    
    Returns:
        Two-sided p-value.
    """
    assume(df > 0.0)
    abs_t = abs(t_stat)
    return 2.0 * stdtr(df, -abs_t)
