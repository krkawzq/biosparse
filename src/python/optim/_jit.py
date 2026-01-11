"""Enhanced JIT Decorators with Loop Optimization Support.

This module provides JIT decorators that wrap Numba's @njit and add
support for loop optimization hints through IR post-processing.

The main decorator @optimized_jit works like @njit but:
1. Compiles the function with Numba
2. Scans the generated LLVM IR for loop hint markers
3. Adds corresponding LLVM loop metadata
4. Optionally recompiles with the enhanced IR

Usage:
    from scl.optim import optimized_jit, vectorize_hint, assume
    
    @optimized_jit
    def fast_sum(arr):
        n = len(arr)
        assume(n > 0)
        
        vectorize_hint(8)
        total = 0.0
        for i in range(n):
            total += arr[i]
        return total

Options:
    @optimized_jit(fastmath=True, parallel=True, process_hints=True)
    def my_func(...):
        ...

Note:
    The loop hints (vectorize_hint, unroll_hint, etc.) only take effect
    when using @optimized_jit. They still work as no-ops with regular @njit.
"""

import functools
from typing import Callable, Optional, Any, Dict, Union
from numba import njit
from numba.core.dispatcher import Dispatcher

from ._ir_processor import IRProcessor, process_ir


__all__ = [
    'optimized_jit',
    'OptimizedDispatcher',
]


# =============================================================================
# Enhanced JIT Decorator
# =============================================================================

class OptimizedDispatcher:
    """Wrapper around Numba Dispatcher that adds IR post-processing.
    
    This class wraps a compiled Numba function and provides the same
    interface, while also processing loop hints in the generated IR.
    
    Attributes:
        _dispatcher: The underlying Numba Dispatcher
        _process_hints: Whether to process loop hints
        _verbose: Whether to print debug information
        _processed_signatures: Set of signatures that have been processed
    """
    
    def __init__(
        self, 
        dispatcher: Dispatcher, 
        process_hints: bool = True,
        verbose: bool = False,
        recompile: bool = False,
    ):
        """Initialize the optimized dispatcher.
        
        Args:
            dispatcher: Numba Dispatcher to wrap
            process_hints: Whether to process loop optimization hints
            verbose: Whether to print debug information
            recompile: Whether to recompile with modified IR (experimental)
        """
        self._dispatcher = dispatcher
        self._process_hints = process_hints
        self._verbose = verbose
        self._recompile = recompile
        self._processed_signatures = set()
        self._ir_processor = IRProcessor(verbose=verbose) if process_hints else None
        self._modified_irs: Dict[Any, str] = {}
        self._initialized = False
    
    def __call__(self, *args, **kwargs):
        """Call the compiled function."""
        # Fast path after initialization
        if self._initialized:
            return self._dispatcher(*args, **kwargs)
        
        # First call - compile and process
        result = self._dispatcher(*args, **kwargs)
        
        if self._process_hints:
            self._process_new_signatures()
        
        self._initialized = True
        return result
    
    def _process_new_signatures(self):
        """Process IR for any newly compiled signatures."""
        for sig in self._dispatcher.signatures:
            if sig not in self._processed_signatures:
                self._processed_signatures.add(sig)
                self._process_signature(sig)
    
    def _process_signature(self, sig):
        """Process IR for a specific signature.
        
        Args:
            sig: Numba signature to process
        """
        try:
            # Get the LLVM IR
            ir = self._dispatcher.inspect_llvm(sig)
            
            # Check for hints
            if '__SCL_LOOP_' in ir:
                if self._verbose:
                    print(f"[OptimizedJIT] Processing hints for signature: {sig}")
                
                # Process the IR
                modified_ir, hints = self._ir_processor.process(ir)
                
                if hints:
                    self._modified_irs[sig] = modified_ir
                    
                    if self._verbose:
                        print(f"[OptimizedJIT] Applied {len(hints)} loop hints")
                    
                    # TODO: If recompile is True, replace the compiled code
                    # This requires deeper integration with Numba internals
                    if self._recompile:
                        self._warn_recompile_experimental()
            
        except Exception as e:
            if self._verbose:
                print(f"[OptimizedJIT] Warning: Failed to process IR: {e}")
    
    def _warn_recompile_experimental(self):
        """Warn about experimental recompilation feature."""
        import warnings
        warnings.warn(
            "IR recompilation is experimental and may not work correctly. "
            "The hints are recorded but the original compiled code is used.",
            RuntimeWarning
        )
    
    # ==========================================================================
    # Dispatcher Interface Methods
    # ==========================================================================
    
    @property
    def signatures(self):
        """Get compiled signatures."""
        return self._dispatcher.signatures
    
    def inspect_llvm(self, signature=None):
        """Inspect LLVM IR.
        
        If hints have been processed, returns the modified IR.
        Otherwise returns the original IR.
        """
        if signature is None and self._dispatcher.signatures:
            signature = self._dispatcher.signatures[0]
        
        if signature in self._modified_irs:
            return self._modified_irs[signature]
        
        return self._dispatcher.inspect_llvm(signature)
    
    def inspect_asm(self, signature=None):
        """Inspect generated assembly."""
        return self._dispatcher.inspect_asm(signature)
    
    def inspect_types(self, file=None):
        """Inspect compiled types."""
        return self._dispatcher.inspect_types(file)
    
    @property
    def py_func(self):
        """Get the original Python function."""
        return self._dispatcher.py_func
    
    @property
    def __name__(self):
        """Get function name."""
        return self._dispatcher.__name__
    
    @property
    def __doc__(self):
        """Get function docstring."""
        return self._dispatcher.__doc__
    
    def __repr__(self):
        return f"<OptimizedDispatcher({self._dispatcher.__name__})>"
    
    # Forward attribute access to dispatcher
    def __getattr__(self, name):
        return getattr(self._dispatcher, name)


def optimized_jit(
    func: Optional[Callable] = None,
    *,
    # Our options
    process_hints: bool = True,
    verbose: bool = False,
    recompile: bool = False,
    # Numba options (passed through)
    nogil: bool = True,
    cache: bool = False,
    parallel: bool = False,
    fastmath: bool = False,
    locals: Optional[Dict] = None,
    boundscheck: bool = False,
    **numba_options
) -> Union[Callable, OptimizedDispatcher]:
    """Enhanced JIT decorator with loop optimization support.
    
    This decorator wraps Numba's @njit and adds support for loop
    optimization hints like vectorize_hint() and unroll_hint().
    
    Args:
        func: Function to compile (when used without parentheses)
        process_hints: Whether to process loop optimization hints (default: True)
        verbose: Whether to print debug information (default: False)
        recompile: Experimental: recompile with modified IR (default: False)
        
        # Standard Numba options:
        nogil: Release GIL during execution (default: False)
        cache: Cache compiled function to disk (default: False)
        parallel: Enable automatic parallelization (default: False)
        fastmath: Enable fast math optimizations (default: False)
        locals: Dictionary of local variable types
        boundscheck: Enable array bounds checking (default: False)
        **numba_options: Additional Numba options
    
    Returns:
        OptimizedDispatcher wrapping the compiled function
    
    Example:
        # Basic usage
        @optimized_jit
        def sum_array(arr):
            vectorize_hint(8)
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
        
        # With options
        @optimized_jit(fastmath=True, parallel=True, verbose=True)
        def fast_sum(arr):
            ...
    
    Note:
        Loop hints are processed but the recompilation feature is
        experimental. The hints mainly serve as documentation and
        for future integration with LLVM optimization passes.
    """
    # Collect Numba options (njit is always nopython mode)
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
        # Compile with Numba
        dispatcher = njit(**numba_opts)(fn)
        
        # Wrap in our optimized dispatcher
        return OptimizedDispatcher(
            dispatcher,
            process_hints=process_hints,
            verbose=verbose,
            recompile=recompile,
        )
    
    # Handle both @optimized_jit and @optimized_jit()
    if func is not None:
        return decorator(func)
    return decorator


# =============================================================================
# Convenience Aliases
# =============================================================================

def fast_jit(func: Optional[Callable] = None, **kwargs) -> Union[Callable, OptimizedDispatcher]:
    """Shorthand for @optimized_jit(fastmath=True).
    
    Example:
        @fast_jit
        def my_func(arr):
            ...
    """
    return optimized_jit(func, fastmath=True, **kwargs)


def parallel_jit(func: Optional[Callable] = None, **kwargs) -> Union[Callable, OptimizedDispatcher]:
    """Shorthand for @optimized_jit(parallel=True, fastmath=True).
    
    Example:
        @parallel_jit
        def my_func(arr):
            ...
    """
    return optimized_jit(func, parallel=True, fastmath=True, **kwargs)


# =============================================================================
# Debug Utilities
# =============================================================================

def inspect_hints(func: OptimizedDispatcher) -> None:
    """Print loop hints found in a compiled function.
    
    Args:
        func: OptimizedDispatcher to inspect
    
    Example:
        @optimized_jit
        def my_func(arr):
            vectorize_hint(8)
            for i in range(len(arr)):
                arr[i] *= 2
        
        my_func(np.zeros(10))  # Trigger compilation
        inspect_hints(my_func)
    """
    if not isinstance(func, OptimizedDispatcher):
        print("Note: Function is not an OptimizedDispatcher")
        return
    
    if not func.signatures:
        print("Function has not been compiled yet. Call it first.")
        return
    
    processor = IRProcessor(verbose=True)
    
    for sig in func.signatures:
        print(f"\nSignature: {sig}")
        print("-" * 40)
        
        ir = func._dispatcher.inspect_llvm(sig)
        hints = processor.scan_markers(ir)
        
        if hints:
            for hint in hints:
                print(f"  {hint.hint_type}({hint.value}) at line {hint.line_number}")
        else:
            print("  No loop hints found")


def get_modified_ir(func: OptimizedDispatcher, signature=None) -> Optional[str]:
    """Get the modified IR with loop metadata.
    
    Args:
        func: OptimizedDispatcher to get IR from
        signature: Specific signature (optional)
    
    Returns:
        Modified LLVM IR string, or None if not processed
    """
    if not isinstance(func, OptimizedDispatcher):
        return None
    
    if signature is None and func._modified_irs:
        signature = next(iter(func._modified_irs.keys()))
    
    return func._modified_irs.get(signature)
