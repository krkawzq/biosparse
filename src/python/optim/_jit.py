"""Enhanced JIT Decorators with Loop Optimization Support.

This module provides JIT decorators that wrap Numba's @njit and add
support for loop optimization hints through IR post-processing.

The main decorator @optimized_jit works like @njit but:
1. Compiles the function with Numba
2. Scans the generated LLVM IR for loop hint markers
3. Adds corresponding LLVM loop metadata
4. Optionally recompiles with the enhanced IR

Usage:
    from biosparse.optim import optimized_jit, vectorize, assume
    
    @optimized_jit
    def fast_sum(arr):
        n = len(arr)
        assume(n > 0)
        
        vectorize(8)
        total = 0.0
        for i in range(n):
            total += arr[i]
        return total

Options:
    @optimized_jit(fastmath=True, parallel=True, process_hints=True)
    def my_func(...):
        ...

Note:
    The loop hints (vectorize, unroll, etc.) only take effect when using
    @optimized_jit. They still work as no-ops with regular @njit.
"""

import warnings
from typing import Callable, Optional, Any, Dict, Union
from numba import njit
from numba.core.dispatcher import Dispatcher

from ._ir_processor import IRProcessor
from ._logging import logger


__all__ = [
    'optimized_jit',
    'fast_jit',
    'parallel_jit',
    'OptimizedDispatcher',
    'inspect_hints',
    'get_modified_ir',
]


# =============================================================================
# Optimized Dispatcher
# =============================================================================

class OptimizedDispatcher:
    """Wrapper around Numba Dispatcher that adds IR post-processing.
    
    This class wraps a compiled Numba function and provides the same
    interface, while also processing loop hints in the generated IR.
    
    The wrapper is designed to have zero overhead after the first call:
    - First call: compile, process hints, then call
    - Subsequent calls: direct dispatch to underlying function
    """
    
    def __init__(
        self, 
        dispatcher: Dispatcher, 
        process_hints: bool = True,
        verbose: bool = False,
        recompile: bool = False,
    ):
        self._dispatcher = dispatcher
        self._process_hints = process_hints
        self._verbose = verbose
        self._recompile = recompile
        self._processed_signatures: set = set()
        self._ir_processor: Optional[IRProcessor] = None
        self._modified_irs: Dict[Any, str] = {}
        self._initialized = False
        self._call_func = dispatcher.__call__
    
    def __call__(self, *args, **kwargs):
        """Call the compiled function."""
        if self._initialized:
            return self._call_func(*args, **kwargs)
        
        result = self._call_func(*args, **kwargs)
        
        if self._process_hints:
            self._lazy_init_processor()
            self._process_new_signatures()
        
        self._initialized = True
        return result
    
    def _lazy_init_processor(self) -> None:
        if self._ir_processor is None:
            self._ir_processor = IRProcessor(verbose=self._verbose)
    
    def _process_new_signatures(self) -> None:
        for sig in self._dispatcher.signatures:
            if sig not in self._processed_signatures:
                self._processed_signatures.add(sig)
                self._process_signature(sig)
    
    def _process_signature(self, sig) -> None:
        try:
            ir = self._dispatcher.inspect_llvm(sig)
            
            if '__BIOSPARSE_LOOP_' not in ir:
                return
            
            logger.debug("Processing hints for signature: %s", sig)
            modified_ir, hints = self._ir_processor.process(ir)
            
            if hints:
                self._modified_irs[sig] = modified_ir
                logger.debug("Applied %d loop hints", len(hints))
                
                if self._recompile:
                    self._do_recompile(sig, modified_ir)
            
        except Exception as e:
            logger.warning("Failed to process IR: %s", e)
    
    def _do_recompile(self, sig, modified_ir: str) -> None:
        warnings.warn(
            "IR recompilation is experimental. "
            "The hints are recorded but the original compiled code is used.",
            RuntimeWarning,
            stacklevel=3
        )
    
    # Dispatcher Interface
    
    @property
    def signatures(self):
        return self._dispatcher.signatures
    
    def inspect_llvm(self, signature=None):
        if signature is None and self._dispatcher.signatures:
            signature = self._dispatcher.signatures[0]
        if signature in self._modified_irs:
            return self._modified_irs[signature]
        return self._dispatcher.inspect_llvm(signature)
    
    def inspect_asm(self, signature=None):
        return self._dispatcher.inspect_asm(signature)
    
    def inspect_types(self, file=None):
        return self._dispatcher.inspect_types(file)
    
    @property
    def py_func(self):
        return self._dispatcher.py_func
    
    @property
    def __name__(self):
        return self._dispatcher.__name__
    
    @property
    def __doc__(self):
        return self._dispatcher.__doc__
    
    def __repr__(self):
        return f"<OptimizedDispatcher({self._dispatcher.__name__})>"
    
    def __getattr__(self, name):
        return getattr(self._dispatcher, name)


# =============================================================================
# JIT Decorators
# =============================================================================

def optimized_jit(
    func: Optional[Callable] = None,
    *,
    process_hints: bool = True,
    verbose: bool = False,
    recompile: bool = False,
    nogil: bool = True,
    cache: bool = False,
    parallel: bool = False,
    fastmath: bool = False,
    locals: Optional[Dict] = None,
    boundscheck: bool = False,
    **numba_options
) -> Union[Callable, OptimizedDispatcher]:
    """Enhanced JIT decorator with loop optimization support.
    
    Args:
        func: Function to compile (when used without parentheses)
        process_hints: Process loop optimization hints (default: True)
        verbose: Enable debug logging (default: False)
        recompile: Experimental: recompile with modified IR (default: False)
        nogil: Release GIL during execution (default: True)
        cache: Cache compiled function to disk (default: False)
        parallel: Enable automatic parallelization (default: False)
        fastmath: Enable fast math optimizations (default: False)
        locals: Dictionary of local variable types
        boundscheck: Enable array bounds checking (default: False)
        **numba_options: Additional Numba options
    
    Returns:
        OptimizedDispatcher wrapping the compiled function
    
    Example:
        @optimized_jit
        def sum_array(arr):
            vectorize(8)
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
    """
    numba_opts = {
        'nogil': nogil,
        'cache': cache,
        'parallel': parallel,
        'fastmath': fastmath,
        'boundscheck': boundscheck,
        **numba_options
    }
    if locals is not None:
        numba_opts['locals'] = locals
    
    def decorator(fn: Callable) -> OptimizedDispatcher:
        dispatcher = njit(**numba_opts)(fn)
        wrapper = OptimizedDispatcher(
            dispatcher,
            process_hints=process_hints,
            verbose=verbose,
            recompile=recompile,
        )
        wrapper.__wrapped__ = fn
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


def fast_jit(func: Optional[Callable] = None, **kwargs) -> Union[Callable, OptimizedDispatcher]:
    """Shorthand for @optimized_jit(fastmath=True)."""
    kwargs.setdefault('fastmath', True)
    return optimized_jit(func, **kwargs)


def parallel_jit(func: Optional[Callable] = None, **kwargs) -> Union[Callable, OptimizedDispatcher]:
    """Shorthand for @optimized_jit(parallel=True, fastmath=True)."""
    kwargs.setdefault('parallel', True)
    kwargs.setdefault('fastmath', True)
    return optimized_jit(func, **kwargs)


# =============================================================================
# Debug Utilities
# =============================================================================

def inspect_hints(func: OptimizedDispatcher) -> None:
    """Print loop hints found in a compiled function."""
    if not isinstance(func, OptimizedDispatcher):
        print("Note: Function is not an OptimizedDispatcher")
        return
    
    if not func.signatures:
        print("Function has not been compiled yet. Call it first.")
        return
    
    processor = IRProcessor(verbose=False)
    
    for sig in func.signatures:
        print(f"\nSignature: {sig}")
        print("-" * 40)
        
        ir = func._dispatcher.inspect_llvm(sig)
        hints = processor.scan_markers(ir)
        
        if hints:
            for hint in hints:
                val_str = f"({hint.value})" if hint.value is not None else ""
                print(f"  {hint.hint_type}{val_str} at line {hint.line_number}")
        else:
            print("  No loop hints found")


def get_modified_ir(func: OptimizedDispatcher, signature=None) -> Optional[str]:
    """Get the modified IR with loop metadata."""
    if not isinstance(func, OptimizedDispatcher):
        return None
    
    if signature is None and func._modified_irs:
        signature = next(iter(func._modified_irs.keys()))
    
    return func._modified_irs.get(signature)
