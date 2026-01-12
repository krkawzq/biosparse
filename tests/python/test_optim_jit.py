"""Tests for biosparse.optim JIT decorators."""

import pytest
import numpy as np
import time
import warnings

pytest.importorskip("numba")

from numba import njit
from biosparse.optim import optimized_jit, fast_jit, parallel_jit, assume, vectorize


class TestOptimizedJit:
    """Test optimized_jit decorator."""
    
    def test_basic_compilation(self, arr):
        """optimized_jit should compile and run correctly."""
        @optimized_jit
        def sum_arr(arr):
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
        
        result = sum_arr(arr)
        assert np.isclose(result, np.sum(arr))
    
    def test_with_options(self, arr):
        """optimized_jit should accept numba options."""
        @optimized_jit(fastmath=True, cache=False)
        def fast_sum(arr):
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
        
        result = fast_sum(arr)
        assert np.isclose(result, np.sum(arr))
    
    def test_with_intrinsics(self, aligned_arr):
        """optimized_jit should work with intrinsics."""
        @optimized_jit
        def optimized_sum(arr):
            n = len(arr)
            assume(n > 0)
            
            vectorize(8)
            total = 0.0
            for i in range(n):
                total += arr[i]
            return total
        
        result = optimized_sum(aligned_arr)
        assert np.isclose(result, np.sum(aligned_arr))
    
    def test_returns_correct_type(self, arr):
        """optimized_jit should return OptimizedDispatcher."""
        from biosparse.optim import OptimizedDispatcher
        
        @optimized_jit
        def my_func(arr):
            return np.sum(arr)
        
        assert isinstance(my_func, OptimizedDispatcher)
    
    def test_inspect_llvm(self, arr):
        """Should be able to inspect LLVM IR."""
        @optimized_jit
        def my_func(arr):
            return len(arr)
        
        _ = my_func(arr)
        ir = my_func.inspect_llvm()
        assert 'define' in ir  # LLVM function definition


class TestFastJit:
    """Test fast_jit decorator."""
    
    def test_basic(self, arr):
        """fast_jit should compile with fastmath."""
        @fast_jit
        def fast_sum(arr):
            total = 0.0
            for x in arr:
                total += x
            return total
        
        result = fast_sum(arr)
        assert np.isclose(result, np.sum(arr))


class TestParallelJit:
    """Test parallel_jit decorator."""
    
    def test_basic(self, arr):
        """parallel_jit should compile with parallel and fastmath."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            @parallel_jit
            def par_sum(arr):
                total = 0.0
                for i in range(len(arr)):
                    total += arr[i]
                return total
            
            result = par_sum(arr)
            assert np.isclose(result, np.sum(arr))


class TestPerformance:
    """Test performance characteristics."""
    
    def test_minimal_overhead(self, arr):
        """optimized_jit should have minimal overhead vs njit."""
        @njit
        def njit_sum(arr):
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
        
        @optimized_jit
        def opt_sum(arr):
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
        
        # Warmup
        _ = njit_sum(arr)
        _ = opt_sum(arr)
        
        # Benchmark
        iters = 1000
        
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = njit_sum(arr)
        t_njit = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = opt_sum(arr)
        t_opt = time.perf_counter() - t0
        
        # Allow up to 50% overhead (generous for test stability)
        overhead = (t_opt - t_njit) / t_njit
        assert overhead < 0.5, f"Overhead too high: {overhead*100:.1f}%"
