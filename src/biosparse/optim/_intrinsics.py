"""LLVM Intrinsic Functions for Numba.

This module provides low-level LLVM intrinsics that can be used in Numba JIT
compiled functions to guide compiler optimizations.

Available intrinsics:
    - assume(condition): Tell LLVM a condition is always true
    - likely(condition): Hint that a branch is likely to be taken
    - unlikely(condition): Hint that a branch is unlikely to be taken
    - unreachable(): Mark code as unreachable (undefined behavior if reached)
    - prefetch_read(ptr, locality): Prefetch memory for reading
    - prefetch_write(ptr, locality): Prefetch memory for writing

Example:
    from biosparse.optim import assume, likely, prefetch_read
    
    @njit(fastmath=True)
    def optimized_sum(arr):
        n = len(arr)
        assume(n > 0)
        assume(n % 4 == 0)
        
        total = 0.0
        for i in range(n):
            if likely(arr[i] > 0):
                total += arr[i]
        return total
"""

from numba import types
from numba.core import cgutils
from numba.extending import intrinsic
import llvmlite.ir as lir


__all__ = [
    'assume',
    'likely',
    'unlikely',
    'unreachable',
    'prefetch_read',
    'prefetch_write',
    'invariant_start',
    'invariant_end',
]


# =============================================================================
# Core Optimization Hints
# =============================================================================

@intrinsic
def assume(typingctx, condition_ty):
    """Tell LLVM that a condition is always true.
    
    This allows the optimizer to make aggressive assumptions and eliminate
    dead code paths. Use with caution - if the assumption is violated at
    runtime, the behavior is undefined.
    
    Args:
        condition: A boolean expression that is assumed to be true
    
    Example:
        @njit
        def safe_divide(a, b):
            assume(b != 0)  # Promise: b is never zero
            return a / b
    
    Note:
        The assume intrinsic may be optimized away if it doesn't affect
        any optimization decisions. Combine with conditional branches for
        best results.
    """
    if condition_ty != types.boolean:
        return None
    
    sig = types.void(types.boolean)
    
    def codegen(context, builder, sig, args):
        [condition] = args
        
        # declare void @llvm.assume(i1)
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(1)])
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'llvm.assume')
        builder.call(fn, [condition])
        
        return context.get_dummy_value()
    
    return sig, codegen


@intrinsic
def likely(typingctx, condition_ty):
    """Hint that a condition is likely to be true.
    
    This helps the compiler optimize branch prediction and code layout
    by indicating that the true branch is the common case.
    
    Args:
        condition: A boolean condition
    
    Returns:
        The same boolean value (passed through)
    
    Example:
        @njit
        def process(x):
            if likely(x > 0):
                # This path is optimized for
                return fast_path(x)
            else:
                return slow_path(x)
    """
    if condition_ty != types.boolean:
        return None
    
    sig = types.boolean(types.boolean)
    
    def codegen(context, builder, sig, args):
        [condition] = args
        
        # declare i1 @llvm.expect.i1(i1 %val, i1 %expected)
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(1), lir.IntType(1)])
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'llvm.expect.i1')
        
        # Expect the condition to be true
        expected_true = lir.Constant(lir.IntType(1), 1)
        return builder.call(fn, [condition, expected_true])
    
    return sig, codegen


@intrinsic
def unlikely(typingctx, condition_ty):
    """Hint that a condition is unlikely to be true.
    
    This helps the compiler optimize branch prediction and code layout
    by indicating that the false branch is the common case.
    
    Args:
        condition: A boolean condition
    
    Returns:
        The same boolean value (passed through)
    
    Example:
        @njit
        def process(x):
            if unlikely(x < 0):
                # Error handling - rarely executed
                return handle_error(x)
            return normal_path(x)
    """
    if condition_ty != types.boolean:
        return None
    
    sig = types.boolean(types.boolean)
    
    def codegen(context, builder, sig, args):
        [condition] = args
        
        # declare i1 @llvm.expect.i1(i1 %val, i1 %expected)
        fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(1), lir.IntType(1)])
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'llvm.expect.i1')
        
        # Expect the condition to be false
        expected_false = lir.Constant(lir.IntType(1), 0)
        return builder.call(fn, [condition, expected_false])
    
    return sig, codegen


@intrinsic
def unreachable(typingctx):
    """Mark code as unreachable.
    
    This tells LLVM that execution will never reach this point, allowing
    aggressive dead code elimination. If execution does reach this point,
    the behavior is undefined.
    
    Example:
        @njit
        def safe_access(arr, idx):
            if idx < 0 or idx >= len(arr):
                unreachable()  # Caller guarantees valid index
            return arr[idx]
    
    Warning:
        Use with extreme caution! If the unreachable point is actually
        reached, the program may crash or behave unpredictably.
    """
    sig = types.void()
    
    def codegen(context, builder, sig, args):
        builder.unreachable()
        # Note: no return needed as unreachable terminates the block
        return context.get_dummy_value()
    
    return sig, codegen


# =============================================================================
# Memory Prefetch Hints
# =============================================================================

@intrinsic
def prefetch_read(typingctx, ptr_ty, locality_ty):
    """Prefetch memory for reading.
    
    This hints to the CPU that the specified memory location will be read
    soon, allowing it to be loaded into cache ahead of time.
    
    Args:
        ptr: Pointer to the memory to prefetch (use arr.ctypes.data)
        locality: Temporal locality hint (0=none, 1=low, 2=medium, 3=high)
                  Higher values mean the data is more likely to be reused.
    
    Example:
        @njit
        def sum_with_prefetch(arr):
            n = len(arr)
            total = 0.0
            for i in range(n):
                if i + 16 < n:
                    # Prefetch 16 elements ahead
                    prefetch_read(arr[i + 16:].ctypes.data, 3)
                total += arr[i]
            return total
    
    Note:
        This is a hint - the CPU may ignore it. Prefetching too aggressively
        can actually hurt performance by polluting the cache.
    """
    # Accept various pointer types
    if not isinstance(ptr_ty, (types.CPointer, types.voidptr.__class__)):
        if ptr_ty != types.intp and ptr_ty != types.uintp:
            return None
    
    sig = types.void(ptr_ty, types.int32)
    
    def codegen(context, builder, sig, args):
        [ptr, locality] = args
        
        # Cast to i8* if needed
        i8ptr = lir.IntType(8).as_pointer()
        if ptr.type != i8ptr:
            ptr = builder.inttoptr(ptr, i8ptr) if isinstance(ptr.type, lir.IntType) else builder.bitcast(ptr, i8ptr)
        
        # declare void @llvm.prefetch(i8* <address>, i32 <rw>, i32 <locality>, i32 <cache_type>)
        fnty = lir.FunctionType(lir.VoidType(), [
            i8ptr,           # address
            lir.IntType(32), # rw: 0=read, 1=write
            lir.IntType(32), # locality: 0-3
            lir.IntType(32), # cache type: 0=instruction, 1=data
        ])
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'llvm.prefetch.p0i8')
        
        rw_read = lir.Constant(lir.IntType(32), 0)
        cache_data = lir.Constant(lir.IntType(32), 1)
        
        builder.call(fn, [ptr, rw_read, locality, cache_data])
        
        return context.get_dummy_value()
    
    return sig, codegen


@intrinsic
def prefetch_write(typingctx, ptr_ty, locality_ty):
    """Prefetch memory for writing.
    
    Similar to prefetch_read, but hints that the memory will be written to.
    This may cause the CPU to fetch the cache line in exclusive mode.
    
    Args:
        ptr: Pointer to the memory to prefetch
        locality: Temporal locality hint (0-3)
    
    Example:
        @njit
        def fill_with_prefetch(arr, value):
            n = len(arr)
            for i in range(n):
                if i + 16 < n:
                    prefetch_write(arr[i + 16:].ctypes.data, 3)
                arr[i] = value
    """
    if not isinstance(ptr_ty, (types.CPointer, types.voidptr.__class__)):
        if ptr_ty != types.intp and ptr_ty != types.uintp:
            return None
    
    sig = types.void(ptr_ty, types.int32)
    
    def codegen(context, builder, sig, args):
        [ptr, locality] = args
        
        i8ptr = lir.IntType(8).as_pointer()
        if ptr.type != i8ptr:
            ptr = builder.inttoptr(ptr, i8ptr) if isinstance(ptr.type, lir.IntType) else builder.bitcast(ptr, i8ptr)
        
        fnty = lir.FunctionType(lir.VoidType(), [
            i8ptr,
            lir.IntType(32),
            lir.IntType(32),
            lir.IntType(32),
        ])
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'llvm.prefetch.p0i8')
        
        rw_write = lir.Constant(lir.IntType(32), 1)
        cache_data = lir.Constant(lir.IntType(32), 1)
        
        builder.call(fn, [ptr, rw_write, locality, cache_data])
        
        return context.get_dummy_value()
    
    return sig, codegen


# =============================================================================
# Memory Invariant Hints
# =============================================================================

@intrinsic
def invariant_start(typingctx, size_ty, ptr_ty):
    """Mark memory region as invariant (will not change).
    
    This tells LLVM that the memory at the given pointer will not be
    modified, allowing aggressive load hoisting and CSE optimizations.
    
    Args:
        size: Size of the invariant region in bytes (-1 for unknown)
        ptr: Pointer to the start of the invariant region
    
    Returns:
        A token that must be passed to invariant_end()
    
    Example:
        @njit
        def access_readonly(arr):
            # Mark array as read-only for this scope
            token = invariant_start(len(arr) * 8, arr.ctypes.data)
            
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]  # LLVM knows arr won't change
            
            invariant_end(token, len(arr) * 8, arr.ctypes.data)
            return total
    
    Warning:
        Writing to invariant memory is undefined behavior!
    """
    sig = types.voidptr(types.int64, types.voidptr)
    
    def codegen(context, builder, sig, args):
        [size, ptr] = args
        
        i8ptr = lir.IntType(8).as_pointer()
        if ptr.type != i8ptr:
            ptr = builder.bitcast(ptr, i8ptr)
        
        # declare {}* @llvm.invariant.start.p0i8(i64 <size>, i8* <ptr>)
        # Returns a token (represented as i8*)
        fnty = lir.FunctionType(i8ptr, [lir.IntType(64), i8ptr])
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'llvm.invariant.start.p0i8')
        
        return builder.call(fn, [size, ptr])
    
    return sig, codegen


@intrinsic
def invariant_end(typingctx, token_ty, size_ty, ptr_ty):
    """End an invariant memory region.
    
    This marks the end of an invariant region started by invariant_start().
    
    Args:
        token: The token returned by invariant_start()
        size: Size of the region (must match invariant_start)
        ptr: Pointer to the region (must match invariant_start)
    """
    sig = types.void(types.voidptr, types.int64, types.voidptr)
    
    def codegen(context, builder, sig, args):
        [token, size, ptr] = args
        
        i8ptr = lir.IntType(8).as_pointer()
        if token.type != i8ptr:
            token = builder.bitcast(token, i8ptr)
        if ptr.type != i8ptr:
            ptr = builder.bitcast(ptr, i8ptr)
        
        # declare void @llvm.invariant.end.p0i8({}* <start>, i64 <size>, i8* <ptr>)
        fnty = lir.FunctionType(lir.VoidType(), [i8ptr, lir.IntType(64), i8ptr])
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'llvm.invariant.end.p0i8')
        
        builder.call(fn, [token, size, ptr])
        
        return context.get_dummy_value()
    
    return sig, codegen


# =============================================================================
# Convenience Functions
# =============================================================================

@intrinsic
def assume_aligned(typingctx, ptr_ty, align_ty):
    """Assume a pointer is aligned to a specific boundary.
    
    This can enable better vectorization and memory access patterns.
    
    Args:
        ptr: The pointer value
        align: Alignment in bytes (must be a power of 2)
    
    Returns:
        The same pointer (passed through with alignment assumption)
    
    Example:
        @njit
        def process_aligned(arr):
            ptr = arr.ctypes.data
            aligned_ptr = assume_aligned(ptr, 32)  # 32-byte alignment
            # Now LLVM can use AVX instructions more freely
    """
    sig = types.voidptr(ptr_ty, types.int64)
    
    def codegen(context, builder, sig, args):
        [ptr, align] = args
        
        i8ptr = lir.IntType(8).as_pointer()
        if ptr.type != i8ptr:
            ptr_int = builder.ptrtoint(ptr, lir.IntType(64))
            ptr = builder.inttoptr(ptr_int, i8ptr)
        
        # Use assume with alignment check
        # assume(ptr & (align - 1) == 0)
        ptr_int = builder.ptrtoint(ptr, lir.IntType(64))
        align_minus_1 = builder.sub(align, lir.Constant(lir.IntType(64), 1))
        masked = builder.and_(ptr_int, align_minus_1)
        is_aligned = builder.icmp_unsigned('==', masked, lir.Constant(lir.IntType(64), 0))
        
        # Call llvm.assume
        assume_fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(1)])
        assume_fn = cgutils.get_or_insert_function(builder.module, assume_fnty, 'llvm.assume')
        builder.call(assume_fn, [is_aligned])
        
        return ptr
    
    return sig, codegen
