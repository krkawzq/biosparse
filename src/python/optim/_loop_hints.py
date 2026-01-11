"""Loop Optimization Hints for Numba.

This module provides intrinsics that insert markers into LLVM IR, which can
be processed by a custom JIT wrapper to add loop metadata for vectorization,
unrolling, and other loop optimizations.

The markers use special global variable accesses that survive LLVM optimization 
passes, allowing post-processing to locate and annotate loops.

Available hints:
    - vectorize_hint(width): Hint loop vectorization with specific width
    - unroll_hint(count): Hint loop unrolling with specific count
    - no_vectorize(): Disable vectorization for the next loop
    - no_unroll(): Disable unrolling for the next loop
    - interleave_hint(count): Hint loop interleaving count

Example:
    from scl.optim import optimized_jit, vectorize_hint, unroll_hint
    
    @optimized_jit
    def process(arr):
        n = len(arr)
        total = 0.0
        
        vectorize_hint(8)  # Next loop should vectorize with width 8
        for i in range(n):
            total += arr[i]
        
        return total

Note:
    These hints require using @optimized_jit instead of @njit to enable
    the IR post-processing that converts markers to LLVM loop metadata.
"""

from numba import types
from numba.core import cgutils
from numba.extending import intrinsic
import llvmlite.ir as lir
import platform


__all__ = [
    'vectorize_hint',
    'unroll_hint',
    'interleave_hint',
    'no_vectorize',
    'no_unroll',
    'distribute_hint',
    'pipeline_hint',
]


# =============================================================================
# Marker Format
# =============================================================================
# 
# Markers are inserted as loads from specially-named global variables:
#   @__SCL_LOOP_<TYPE>_<VALUE>__
#
# Examples:
#   @__SCL_LOOP_VECTORIZE_8__      -> vectorize with width 8
#   @__SCL_LOOP_UNROLL_4__         -> unroll 4 times
#   @__SCL_LOOP_INTERLEAVE_2__     -> interleave count 2
#   @__SCL_LOOP_NO_VECTORIZE__     -> disable vectorization
#   @__SCL_LOOP_NO_UNROLL__        -> disable unrolling
#
# The IR post-processor scans for these markers and adds the corresponding
# LLVM loop metadata to the next loop after each marker.
# =============================================================================

# Check if we're on Windows (no inline asm support in some LLVM builds)
_USE_GLOBAL_MARKERS = platform.system() == 'Windows'


def _insert_marker(builder, marker_name):
    """Insert a marker into the IR.
    
    On Windows, uses a global variable with a special naming convention
    that can be detected in the IR.
    On other platforms, uses inline assembly (more reliable).
    """
    if _USE_GLOBAL_MARKERS:
        # Use a global variable as marker
        # The global name itself carries the marker information
        module = builder.module
        
        # Create or get the global marker variable
        try:
            gv = module.get_global(marker_name)
        except KeyError:
            gv = lir.GlobalVariable(module, lir.IntType(32), marker_name)
            gv.linkage = 'external'
            gv.global_constant = False  # Not constant so it's not optimized away
        
        # Load from the global - this creates a reference in the IR
        # The load instruction will contain the marker name
        load_val = builder.load(gv)
        
        # Use the loaded value in a way that can't be optimized away
        # by calling llvm.assume with a condition involving it
        from numba.core import cgutils
        assume_fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(1)])
        assume_fn = cgutils.get_or_insert_function(builder.module, assume_fnty, 'llvm.assume')
        
        # assume(load_val >= 0) - always true for unsigned, but keeps the load alive
        zero = lir.Constant(lir.IntType(32), 0)
        cond = builder.icmp_signed('>=', load_val, zero)
        builder.call(assume_fn, [cond])
    else:
        # Use inline assembly comment
        asm_ty = lir.FunctionType(lir.VoidType(), [])
        inline_asm = lir.InlineAsm(
            asm_ty,
            f"# {marker_name}",
            "~{{memory}}",
            side_effect=True
        )
        builder.call(inline_asm, [])


# =============================================================================
# Loop Vectorization Hints
# =============================================================================

@intrinsic
def vectorize_hint(typingctx, width_ty):
    """Hint that the next loop should be vectorized with specified width.
    
    This inserts a marker into the IR that the post-processor converts to
    LLVM loop metadata: !{!"llvm.loop.vectorize.width", i32 <width>}
    
    Args:
        width: Vectorization width (e.g., 4, 8, 16)
               Must be a power of 2.
    
    Example:
        @optimized_jit
        def sum_vectorized(arr):
            vectorize_hint(8)  # Use AVX-256 (8 x float32)
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
    
    Note:
        - Requires @optimized_jit decorator for the hint to take effect
        - The actual vectorization depends on CPU capabilities
        - Works best with simple loops and contiguous memory access
    """
    sig = types.void(types.intp)
    
    def codegen(context, builder, sig, args):
        [width] = args
        
        # Extract constant value
        if isinstance(width, lir.Constant):
            width_val = width.constant
        else:
            # Non-constant: try to get the value at codegen time
            width_val = 4  # Default fallback
        
        marker = f"__SCL_LOOP_VECTORIZE_{width_val}__"
        _insert_marker(builder, marker)
        
        return context.get_dummy_value()
    
    return sig, codegen


@intrinsic
def no_vectorize(typingctx):
    """Disable vectorization for the next loop.
    
    This inserts a marker that translates to:
    !{!"llvm.loop.vectorize.enable", i1 false}
    
    Example:
        @optimized_jit
        def process(arr):
            no_vectorize()  # Don't vectorize this loop
            for i in range(len(arr)):
                if arr[i] < 0:
                    arr[i] = 0  # Conditional writes hurt vectorization
    """
    sig = types.void()
    
    def codegen(context, builder, sig, args):
        marker = "__SCL_LOOP_NO_VECTORIZE__"
        _insert_marker(builder, marker)
        
        return context.get_dummy_value()
    
    return sig, codegen


# =============================================================================
# Loop Unrolling Hints
# =============================================================================

@intrinsic
def unroll_hint(typingctx, count_ty):
    """Hint that the next loop should be unrolled.
    
    This inserts a marker that translates to:
    !{!"llvm.loop.unroll.count", i32 <count>}
    
    Args:
        count: Number of times to unroll the loop body.
               Use 0 for full unrolling (if trip count is known).
    
    Example:
        @optimized_jit
        def matrix_mul_4x4(a, b, c):
            unroll_hint(4)  # Fully unroll inner loop
            for i in range(4):
                unroll_hint(4)
                for j in range(4):
                    sum = 0.0
                    unroll_hint(4)
                    for k in range(4):
                        sum += a[i, k] * b[k, j]
                    c[i, j] = sum
    
    Note:
        - Unrolling increases code size
        - Best for small, fixed-size loops
        - count=0 means "unroll completely"
    """
    sig = types.void(types.intp)
    
    def codegen(context, builder, sig, args):
        [count] = args
        
        if isinstance(count, lir.Constant):
            count_val = count.constant
        else:
            count_val = 4  # Default
        
        marker = f"__SCL_LOOP_UNROLL_{count_val}__"
        _insert_marker(builder, marker)
        
        return context.get_dummy_value()
    
    return sig, codegen


@intrinsic
def no_unroll(typingctx):
    """Disable unrolling for the next loop.
    
    This inserts a marker that translates to:
    !{!"llvm.loop.unroll.disable"}
    
    Example:
        @optimized_jit
        def large_loop(arr):
            no_unroll()  # Keep code size small
            for i in range(len(arr)):
                process(arr[i])
    """
    sig = types.void()
    
    def codegen(context, builder, sig, args):
        marker = "__SCL_LOOP_NO_UNROLL__"
        _insert_marker(builder, marker)
        
        return context.get_dummy_value()
    
    return sig, codegen


# =============================================================================
# Loop Interleaving Hints
# =============================================================================

@intrinsic
def interleave_hint(typingctx, count_ty):
    """Hint the interleaving count for the next loop.
    
    Interleaving executes multiple iterations in parallel by replicating
    the loop body. This can help hide latencies and improve ILP.
    
    This inserts a marker that translates to:
    !{!"llvm.loop.interleave.count", i32 <count>}
    
    Args:
        count: Number of iterations to interleave
    
    Example:
        @optimized_jit
        def accumulate(arr):
            interleave_hint(4)  # Use 4 accumulators
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
    """
    sig = types.void(types.intp)
    
    def codegen(context, builder, sig, args):
        [count] = args
        
        if isinstance(count, lir.Constant):
            count_val = count.constant
        else:
            count_val = 2  # Default
        
        marker = f"__SCL_LOOP_INTERLEAVE_{count_val}__"
        _insert_marker(builder, marker)
        
        return context.get_dummy_value()
    
    return sig, codegen


# =============================================================================
# Advanced Loop Hints
# =============================================================================

@intrinsic
def distribute_hint(typingctx):
    """Hint that the next loop should be distributed.
    
    Loop distribution splits a loop with multiple statements into
    separate loops, which can enable further optimizations.
    
    This inserts a marker that translates to:
    !{!"llvm.loop.distribute.enable", i1 true}
    
    Example:
        @optimized_jit
        def process(a, b, c):
            distribute_hint()
            for i in range(len(a)):
                b[i] = a[i] * 2      # Independent
                c[i] = a[i] + 1      # Independent
            # May be split into two loops for better optimization
    """
    sig = types.void()
    
    def codegen(context, builder, sig, args):
        marker = "__SCL_LOOP_DISTRIBUTE__"
        _insert_marker(builder, marker)
        
        return context.get_dummy_value()
    
    return sig, codegen


@intrinsic
def pipeline_hint(typingctx, stages_ty):
    """Hint software pipelining for the next loop.
    
    Software pipelining overlaps operations from different iterations
    to maximize throughput. This is an advanced optimization.
    
    Args:
        stages: Number of pipeline stages (0 for auto)
    
    Example:
        @optimized_jit
        def pipeline_example(arr):
            pipeline_hint(3)
            for i in range(len(arr)):
                # Stage 1: Load
                # Stage 2: Compute
                # Stage 3: Store
                arr[i] = arr[i] * 2 + 1
    """
    sig = types.void(types.intp)
    
    def codegen(context, builder, sig, args):
        [stages] = args
        
        if isinstance(stages, lir.Constant):
            stages_val = stages.constant
        else:
            stages_val = 0
        
        marker = f"__SCL_LOOP_PIPELINE_{stages_val}__"
        _insert_marker(builder, marker)
        
        return context.get_dummy_value()
    
    return sig, codegen


# =============================================================================
# Combined Hints
# =============================================================================

def fast_loop(vectorize=None, unroll=None, interleave=None):
    """Apply multiple loop optimization hints at once.
    
    This is a convenience function that calls multiple hint functions.
    
    Args:
        vectorize: Vectorization width (optional)
        unroll: Unroll count (optional)
        interleave: Interleave count (optional)
    
    Example:
        @optimized_jit
        def optimized(arr):
            fast_loop(vectorize=8, interleave=4)
            for i in range(len(arr)):
                arr[i] *= 2
    
    Note:
        This is a regular Python function, not an intrinsic. It calls
        the individual hint intrinsics. The function itself will be
        inlined by Numba.
    """
    if vectorize is not None:
        vectorize_hint(vectorize)
    if unroll is not None:
        unroll_hint(unroll)
    if interleave is not None:
        interleave_hint(interleave)
