"""BioSparse Optimization Toolkit for Numba.

This module provides low-level optimization tools for Numba JIT-compiled
functions, including LLVM intrinsics and loop optimization hints.

Quick Start:
    from biosparse.optim import optimized_jit, assume, likely, vectorize
    
    @optimized_jit(fastmath=True)
    def fast_sum(arr):
        n = len(arr)
        assume(n > 0)
        assume(n % 8 == 0)
        
        vectorize(8)
        total = 0.0
        for i in range(n):
            if likely(arr[i] > 0):
                total += arr[i]
        return total

Components:

    JIT Decorators:
        optimized_jit   - Enhanced @njit with loop hint processing
        fast_jit        - @optimized_jit(fastmath=True)
        parallel_jit    - @optimized_jit(parallel=True, fastmath=True)

    LLVM Intrinsics (work with @njit and @optimized_jit):
        assume          - Assume condition is always true
        likely          - Branch is likely taken
        unlikely        - Branch is unlikely taken
        unreachable     - Mark code as unreachable
        prefetch_read   - Prefetch memory for reading
        prefetch_write  - Prefetch memory for writing
        assume_aligned  - Assume pointer alignment
        invariant_start - Mark memory as immutable
        invariant_end   - End immutable region

    Loop Hints (require @optimized_jit):
        vectorize       - Hint vectorization width
        unroll          - Hint unroll count
        interleave      - Hint interleave count
        distribute      - Enable loop distribution
        pipeline        - Hint software pipelining

    Utilities:
        inspect_hints   - Print loop hints in compiled function
        get_modified_ir - Get IR with loop metadata
        set_log_level   - Set logging level
        enable_debug    - Enable debug logging
"""

__version__ = '0.1.0'

# =============================================================================
# Logging
# =============================================================================

from ._logging import (
    logger,
    set_log_level,
    enable_debug,
    disable_logging,
)

# =============================================================================
# LLVM Intrinsics
# =============================================================================

from ._intrinsics import (
    assume,
    likely,
    unlikely,
    unreachable,
    prefetch_read,
    prefetch_write,
    invariant_start,
    invariant_end,
    assume_aligned,
)

# =============================================================================
# Loop Hints
# =============================================================================

from ._loop_hints import (
    vectorize,
    unroll,
    interleave,
    distribute,
    pipeline,
)

# =============================================================================
# JIT Decorators
# =============================================================================

from ._jit import (
    optimized_jit,
    fast_jit,
    parallel_jit,
    OptimizedDispatcher,
    inspect_hints,
    get_modified_ir,
)

# =============================================================================
# IR Processing
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
    
    # JIT
    'optimized_jit',
    'fast_jit',
    'parallel_jit',
    'OptimizedDispatcher',
    
    # Intrinsics
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
    'vectorize',
    'unroll',
    'interleave',
    'distribute',
    'pipeline',
    
    # IR
    'IRProcessor',
    'LoopHint',
    'HintType',
    'process_ir',
    
    # Utilities
    'inspect_hints',
    'get_modified_ir',
    'logger',
    'set_log_level',
    'enable_debug',
    'disable_logging',
]
