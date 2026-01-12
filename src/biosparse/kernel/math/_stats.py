"""Statistical Distribution Functions.

Provides vectorized statistical functions using Python's math.erfc/erf
which are supported in Numba nopython mode.

All functions accept numpy arrays and return numpy arrays.
Uses optim module for loop hints and parallelization.
"""

import math
import numpy as np
from numba import prange

# Use our own optimized JIT decorators
from biosparse.optim import parallel_jit, assume, vectorize, interleave, likely, unlikely

__all__ = [
    # Precise (math library based, JIT compatible)
    'erfc',
    'erf',
    'normal_cdf',
    'normal_sf',
    'normal_pdf',
    'normal_logcdf',
    'normal_logsf',
    
    # Approximate (same as precise, kept for API compatibility)
    'erfc_approx',
    'normal_sf_approx',
    'normal_cdf_approx',
]


# =============================================================================
# Precise Implementations (math library based, JIT compatible)
# =============================================================================

@parallel_jit(cache=True, inline='always')
def erfc(x: np.ndarray, out: np.ndarray) -> None:
    """Complementary error function.
    
    Args:
        x: Input array
        out: Output array (same shape as x)
    """
    n = len(x)
    assume(n > 0)
    assume(len(out) >= n)
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        out[i] = math.erfc(x[i])


@parallel_jit(cache=True, inline='always')
def erf(x: np.ndarray, out: np.ndarray) -> None:
    """Error function.
    
    Args:
        x: Input array
        out: Output array (same shape as x)
    """
    n = len(x)
    assume(n > 0)
    assume(len(out) >= n)
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        out[i] = math.erf(x[i])


@parallel_jit(cache=True, inline='always')
def normal_cdf(z: np.ndarray, out: np.ndarray) -> None:
    """Standard normal CDF: P(X <= z).
    
    Args:
        z: Input z-scores
        out: Output probabilities
    """
    INV_SQRT2 = 0.7071067811865475
    n = len(z)
    assume(n > 0)
    assume(len(out) >= n)
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        out[i] = 0.5 * math.erfc(-z[i] * INV_SQRT2)


@parallel_jit(cache=True, inline='always')
def normal_sf(z: np.ndarray, out: np.ndarray) -> None:
    """Standard normal survival function: P(X > z) = 1 - CDF(z).
    
    Args:
        z: Input z-scores
        out: Output probabilities
    """
    INV_SQRT2 = 0.7071067811865475
    n = len(z)
    assume(n > 0)
    assume(len(out) >= n)
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        out[i] = 0.5 * math.erfc(z[i] * INV_SQRT2)


@parallel_jit(cache=True, inline='always')
def normal_pdf(z: np.ndarray, out: np.ndarray) -> None:
    """Standard normal PDF.
    
    Args:
        z: Input z-scores
        out: Output density values
    """
    INV_SQRT_2PI = 0.3989422804014327
    n = len(z)
    assume(n > 0)
    assume(len(out) >= n)
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        out[i] = INV_SQRT_2PI * np.exp(-0.5 * z[i] * z[i])


@parallel_jit(cache=True, inline='always')
def normal_logcdf(z: np.ndarray, out: np.ndarray) -> None:
    """Log of standard normal CDF (numerically stable).
    
    Uses asymptotic expansion for large negative z.
    
    Args:
        z: Input z-scores
        out: Output log probabilities
    """
    INV_SQRT2 = 0.7071067811865475
    LOG_SQRT_2PI = 0.9189385332046727
    n = len(z)
    assume(n > 0)
    assume(len(out) >= n)
    
    vectorize(8)
    for i in prange(n):
        zi = z[i]
        if unlikely(zi < -20.0):
            # Asymptotic expansion for large negative z
            z2 = zi * zi
            out[i] = -0.5 * z2 - LOG_SQRT_2PI - np.log(-zi)
        else:
            cdf = 0.5 * math.erfc(-zi * INV_SQRT2)
            out[i] = np.log(cdf)


@parallel_jit(cache=True, inline='always')
def normal_logsf(z: np.ndarray, out: np.ndarray) -> None:
    """Log of standard normal survival function (numerically stable).
    
    Uses asymptotic expansion for large positive z.
    
    Args:
        z: Input z-scores
        out: Output log probabilities
    """
    INV_SQRT2 = 0.7071067811865475
    LOG_SQRT_2PI = 0.9189385332046727
    n = len(z)
    assume(n > 0)
    assume(len(out) >= n)
    
    vectorize(8)
    for i in prange(n):
        zi = z[i]
        if unlikely(zi > 20.0):
            # Asymptotic expansion for large positive z
            z2 = zi * zi
            out[i] = -0.5 * z2 - LOG_SQRT_2PI - np.log(zi)
        else:
            sf = 0.5 * math.erfc(zi * INV_SQRT2)
            out[i] = np.log(sf)


# =============================================================================
# Approximate Implementations (Abramowitz-Stegun, ~1e-7 precision)
# =============================================================================

@parallel_jit(cache=True, inline='always')
def erfc_approx(x: np.ndarray, out: np.ndarray) -> None:
    """Approximate complementary error function.
    
    Uses Abramowitz-Stegun rational approximation (~1e-7 precision).
    Faster than scipy.special.erfc for large arrays.
    
    Args:
        x: Input array
        out: Output array (same shape as x)
    """
    n = len(x)
    assume(n > 0)
    assume(len(out) >= n)
    
    vectorize(8)
    for i in prange(n):
        xi = x[i]
        ax = abs(xi)
        t = 1.0 / (1.0 + 0.5 * ax)
        
        # Horner's method polynomial evaluation
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
        
        r = tau if xi >= 0.0 else 2.0 - tau
        
        # Clamp to valid range
        if r < 0.0:
            r = 0.0
        elif r > 2.0:
            r = 2.0
        
        out[i] = r


@parallel_jit(cache=True, inline='always')
def normal_sf_approx(z: np.ndarray, out: np.ndarray) -> None:
    """Approximate standard normal survival function P(X > z).
    
    Args:
        z: Input z-scores
        out: Output probabilities
    """
    INV_SQRT2 = 0.7071067811865475
    n = len(z)
    assume(n > 0)
    assume(len(out) >= n)
    
    vectorize(8)
    for i in prange(n):
        zi = z[i]
        arg = zi * INV_SQRT2
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


@parallel_jit(cache=True, inline='always')
def normal_cdf_approx(z: np.ndarray, out: np.ndarray) -> None:
    """Approximate standard normal CDF P(X <= z).
    
    Args:
        z: Input z-scores
        out: Output probabilities
    """
    INV_SQRT2 = 0.7071067811865475
    n = len(z)
    assume(n > 0)
    assume(len(out) >= n)
    
    vectorize(8)
    for i in prange(n):
        zi = z[i]
        arg = -zi * INV_SQRT2
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

@parallel_jit(cache=True, inline='always')
def erfc_new(x: np.ndarray) -> np.ndarray:
    """Allocating version of erfc."""
    n = len(x)
    assume(n > 0)
    out = np.empty(n, dtype=np.float64)
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        out[i] = math.erfc(x[i])
    
    return out


@parallel_jit(cache=True, inline='always')
def normal_sf_new(z: np.ndarray) -> np.ndarray:
    """Allocating version of normal_sf."""
    INV_SQRT2 = 0.7071067811865475
    n = len(z)
    assume(n > 0)
    out = np.empty(n, dtype=np.float64)
    
    vectorize(8)
    interleave(4)
    for i in prange(n):
        out[i] = 0.5 * math.erfc(z[i] * INV_SQRT2)
    
    return out


@parallel_jit
def normal_sf_approx_new(z: np.ndarray) -> np.ndarray:
    """Allocating version of normal_sf_approx."""
    INV_SQRT2 = 0.7071067811865475
    n = len(z)
    assume(n > 0)
    out = np.empty(n, dtype=np.float64)
    
    vectorize(8)
    for i in prange(n):
        zi = z[i]
        arg = zi * INV_SQRT2
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
    
    return out
