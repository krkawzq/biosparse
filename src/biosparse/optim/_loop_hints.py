"""Loop Optimization Hints for Numba.

This module provides intrinsics that insert markers into LLVM IR, which can
be processed by a custom JIT wrapper to add loop metadata for vectorization,
unrolling, and other loop optimizations.

Available hints:
    - vectorize(width): Hint loop vectorization with specific width
    - unroll(count): Hint loop unrolling with specific count
    - interleave(count): Hint loop interleaving count
    - distribute(): Enable loop distribution
    - pipeline(stages): Hint software pipelining

Example:
    from biosparse.optim import optimized_jit, vectorize, unroll
    
    @optimized_jit
    def process(arr):
        n = len(arr)
        total = 0.0
        
        vectorize(8)  # Next loop should vectorize with width 8
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
    'vectorize',
    'unroll',
    'interleave',
    'distribute',
    'pipeline',
]


# Check if we're on Windows (no inline asm support in some LLVM builds)
# _USE_GLOBAL_MARKERS = platform.system() == 'Windows'
_USE_GLOBAL_MARKERS = True # asm may not support in llvmlite in some versions


def _get_constant_value(ir_value, default):
    """Extract constant value from LLVM IR value.
    
    Args:
        ir_value: LLVM IR value (may be Constant or other)
        default: Default value if not a constant
    
    Returns:
        The constant integer value, or default if not a constant
    """
    if isinstance(ir_value, lir.Constant):
        return ir_value.constant
    return default


def _insert_marker(builder, marker_name):
    """Insert a marker into the IR."""
    if _USE_GLOBAL_MARKERS:
        module = builder.module
        try:
            gv = module.get_global(marker_name)
        except KeyError:
            gv = lir.GlobalVariable(module, lir.IntType(32), marker_name)
            gv.linkage = 'external'
            gv.global_constant = False
        
        load_val = builder.load(gv)
        assume_fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(1)])
        assume_fn = cgutils.get_or_insert_function(builder.module, assume_fnty, 'llvm.assume')
        zero = lir.Constant(lir.IntType(32), 0)
        cond = builder.icmp_signed('>=', load_val, zero)
        builder.call(assume_fn, [cond])
    else:
        asm_ty = lir.FunctionType(lir.VoidType(), [])
        inline_asm = lir.InlineAsm(asm_ty, f"# {marker_name}", "~{{memory}}", side_effect=True)
        builder.call(inline_asm, [])


# =============================================================================
# Loop Hints
# =============================================================================

@intrinsic
def vectorize(typingctx, width_ty):
    """Hint that the next loop should be vectorized.
    
    Args:
        width: Vectorization width (e.g., 4, 8, 16). Must be power of 2.
    
    Example:
        @optimized_jit
        def sum_vec(arr):
            vectorize(8)  # AVX-256 for float32
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
    """
    if not isinstance(width_ty, types.Integer):
        return None
    
    sig = types.void(width_ty)
    
    def codegen(context, builder, sig, args):
        [width] = args
        width_val = _get_constant_value(width, 4)
        _insert_marker(builder, f"__BIOSPARSE_LOOP_VECTORIZE_{width_val}__")
        return context.get_dummy_value()
    
    return sig, codegen


@intrinsic
def unroll(typingctx, count_ty):
    """Hint that the next loop should be unrolled.
    
    Args:
        count: Unroll count. Use 0 for full unrolling.
    
    Example:
        @optimized_jit
        def small_loop():
            unroll(4)
            for i in range(4):
                process(i)
    """
    if not isinstance(count_ty, types.Integer):
        return None
    
    sig = types.void(count_ty)
    
    def codegen(context, builder, sig, args):
        [count] = args
        count_val = _get_constant_value(count, 4)
        _insert_marker(builder, f"__BIOSPARSE_LOOP_UNROLL_{count_val}__")
        return context.get_dummy_value()
    
    return sig, codegen


@intrinsic
def interleave(typingctx, count_ty):
    """Hint loop interleaving count.
    
    Args:
        count: Number of iterations to interleave.
    
    Example:
        @optimized_jit
        def accumulate(arr):
            interleave(4)  # Use 4 accumulators
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
    """
    if not isinstance(count_ty, types.Integer):
        return None
    
    sig = types.void(count_ty)
    
    def codegen(context, builder, sig, args):
        [count] = args
        count_val = _get_constant_value(count, 2)
        _insert_marker(builder, f"__BIOSPARSE_LOOP_INTERLEAVE_{count_val}__")
        return context.get_dummy_value()
    
    return sig, codegen


@intrinsic
def distribute(typingctx):
    """Hint that the next loop should be distributed.
    
    Loop distribution splits a loop into separate loops for independent
    operations, enabling better optimization.
    
    Example:
        @optimized_jit
        def process(a, b, c):
            distribute()
            for i in range(len(a)):
                b[i] = a[i] * 2
                c[i] = a[i] + 1
    """
    sig = types.void()
    
    def codegen(context, builder, sig, args):
        _insert_marker(builder, "__BIOSPARSE_LOOP_DISTRIBUTE__")
        return context.get_dummy_value()
    
    return sig, codegen


@intrinsic
def pipeline(typingctx, stages_ty):
    """Hint software pipelining for the next loop.
    
    Args:
        stages: Number of pipeline stages (0 for auto).
    
    Example:
        @optimized_jit
        def pipelined(arr):
            pipeline(3)
            for i in range(len(arr)):
                arr[i] = arr[i] * 2 + 1
    """
    if not isinstance(stages_ty, types.Integer):
        return None
    
    sig = types.void(stages_ty)
    
    def codegen(context, builder, sig, args):
        [stages] = args
        stages_val = _get_constant_value(stages, 0)
        _insert_marker(builder, f"__BIOSPARSE_LOOP_PIPELINE_{stages_val}__")
        return context.get_dummy_value()
    
    return sig, codegen
