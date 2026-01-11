"""SCL Optimization Toolkit for Numba.

This module provides low-level optimization tools for Numba JIT-compiled
functions, including LLVM intrinsics and loop optimization hints.

Quick Start:
    from scl.optim import optimized_jit, assume, likely, vectorize_hint
    
    @optimized_jit(fastmath=True)
    def fast_sum(arr):
        n = len(arr)
        assume(n > 0)
        assume(n % 8 == 0)
        
        vectorize_hint(8)
        total = 0.0
        for i in range(n):
            if likely(arr[i] > 0):
                total += arr[i]
        return total

Available Components:

    JIT Decorators:
        - optimized_jit: Enhanced @njit with loop hint processing
        - fast_jit: Shorthand for @optimized_jit(fastmath=True)
        - parallel_jit: Shorthand for @optimized_jit(parallel=True, fastmath=True)

    LLVM Intrinsics (work with @njit and @optimized_jit):
        - assume(condition): Assume condition is always true
        - likely(condition): Branch is likely taken
        - unlikely(condition): Branch is unlikely taken
        - unreachable(): Mark code as unreachable
        - prefetch_read(ptr, locality): Prefetch for reading
        - prefetch_write(ptr, locality): Prefetch for writing
        - assume_aligned(ptr, align): Assume pointer alignment
        - invariant_start(size, ptr): Mark memory as immutable
        - invariant_end(token, size, ptr): End immutable region

    Loop Hints (require @optimized_jit):
        - vectorize_hint(width): Hint vectorization width
        - no_vectorize(): Disable vectorization
        - unroll_hint(count): Hint unroll count
        - no_unroll(): Disable unrolling
        - interleave_hint(count): Hint interleave count
        - distribute_hint(): Enable loop distribution
        - pipeline_hint(stages): Hint software pipelining

    Utilities:
        - inspect_hints(func): Print loop hints in compiled function
        - get_modified_ir(func): Get IR with loop metadata

Note:
    LLVM intrinsics (assume, likely, etc.) work with both @njit and
    @optimized_jit. Loop hints only take effect with @optimized_jit
    as they require IR post-processing.
"""

# Version info
__version__ = '0.1.0'
__author__ = 'SCL Team'

# =============================================================================
# LLVM Intrinsics
# =============================================================================

from ._intrinsics import (
    # Core optimization hints
    assume,
    likely,
    unlikely,
    unreachable,
    
    # Memory prefetch
    prefetch_read,
    prefetch_write,
    
    # Memory invariant
    invariant_start,
    invariant_end,
    
    # Convenience
    assume_aligned,
)

# =============================================================================
# Loop Optimization Hints
# =============================================================================

from ._loop_hints import (
    # Vectorization
    vectorize_hint,
    no_vectorize,
    
    # Unrolling
    unroll_hint,
    no_unroll,
    
    # Interleaving
    interleave_hint,
    
    # Advanced
    distribute_hint,
    pipeline_hint,
    
    # Convenience
    fast_loop,
)

# =============================================================================
# Enhanced JIT Decorators
# =============================================================================

from ._jit import (
    optimized_jit,
    fast_jit,
    parallel_jit,
    OptimizedDispatcher,
    
    # Debug utilities
    inspect_hints,
    get_modified_ir,
)

# =============================================================================
# IR Processing (for advanced users)
# =============================================================================

from ._ir_processor import (
    IRProcessor,
    LoopHint,
    HintType,
    process_ir,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    '__version__',
    
    # JIT Decorators
    'optimized_jit',
    'fast_jit',
    'parallel_jit',
    'OptimizedDispatcher',
    
    # LLVM Intrinsics
    'assume',
    'likely',
    'unlikely',
    'unreachable',
    'prefetch_read',
    'prefetch_write',
    'invariant_start',
    'invariant_end',
    'assume_aligned',
    
    # Loop Hints
    'vectorize_hint',
    'no_vectorize',
    'unroll_hint',
    'no_unroll',
    'interleave_hint',
    'distribute_hint',
    'pipeline_hint',
    'fast_loop',
    
    # IR Processing
    'IRProcessor',
    'LoopHint',
    'HintType',
    'process_ir',
    
    # Utilities
    'inspect_hints',
    'get_modified_ir',
]


# =============================================================================
# Convenience: Re-export Numba's njit for comparison
# =============================================================================

def njit(*args, **kwargs):
    """Re-export of Numba's njit for convenience.
    
    Use this when you don't need loop hints:
        from scl.optim import njit, assume
        
        @njit
        def my_func(arr):
            assume(len(arr) > 0)
            ...
    """
    from numba import njit as numba_njit
    return numba_njit(*args, **kwargs)
