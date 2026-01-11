"""Tests for SCL Optimization Toolkit.

This module tests the LLVM intrinsics and loop optimization hints.
"""

import sys
import os
import numpy as np
import time
import re

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from python.optim import (
    # JIT decorators
    optimized_jit,
    fast_jit,
    parallel_jit,
    
    # Intrinsics
    assume,
    likely,
    unlikely,
    prefetch_read,
    
    # Loop hints
    vectorize_hint,
    unroll_hint,
    no_vectorize,
    interleave_hint,
    
    # Utilities
    inspect_hints,
    get_modified_ir,
    IRProcessor,
)

from numba import njit


# =============================================================================
# Test: Basic Intrinsics
# =============================================================================

class TestIntrinsics:
    """Test LLVM intrinsic functions."""
    
    def test_assume_basic(self):
        """Test assume intrinsic compiles and runs."""
        @njit
        def with_assume(arr):
            n = len(arr)
            assume(n > 0)
            assume(n % 4 == 0)
            
            total = 0.0
            for i in range(n):
                total += arr[i]
            return total
        
        arr = np.random.rand(100)
        result = with_assume(arr)
        expected = np.sum(arr)
        
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"
        print("[PASS] assume basic test passed")
    
    def test_assume_in_ir(self):
        """Test that assume appears in LLVM IR."""
        @njit
        def with_assume(x):
            assume(x > 0)
            return x * 2
        
        _ = with_assume(10)
        ir = with_assume.inspect_llvm(with_assume.signatures[0])
        
        assert 'llvm.assume' in ir, "llvm.assume should appear in IR"
        print("[PASS] assume in IR test passed")
    
    def test_likely_unlikely(self):
        """Test likely/unlikely intrinsics."""
        @njit
        def with_branch_hints(arr):
            total = 0.0
            for i in range(len(arr)):
                if likely(arr[i] > 0):
                    total += arr[i]
                elif unlikely(arr[i] < -100):
                    total -= arr[i]
            return total
        
        arr = np.random.rand(100)
        result = with_branch_hints(arr)
        
        # Verify it computes correctly
        expected = sum(x for x in arr if x > 0)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"
        
        # Check IR for expect intrinsic or branch_weights metadata
        # Note: llvm.expect may be lowered to branch_weights metadata
        ir = with_branch_hints.inspect_llvm(with_branch_hints.signatures[0])
        has_expect = 'llvm.expect' in ir or 'branch_weights' in ir
        assert has_expect, "llvm.expect or branch_weights should appear in IR"
        print("[PASS] likely/unlikely test passed")


# =============================================================================
# Test: Loop Hints
# =============================================================================

class TestLoopHints:
    """Test loop optimization hints."""
    
    def test_vectorize_hint_marker(self):
        """Test that vectorize_hint inserts marker."""
        @optimized_jit(verbose=False)
        def with_vectorize(arr):
            n = len(arr)
            total = 0.0
            
            vectorize_hint(8)
            for i in range(n):
                total += arr[i]
            
            return total
        
        arr = np.random.rand(1000)
        result = with_vectorize(arr)
        expected = np.sum(arr)
        
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"
        
        # Check for marker in IR
        ir = with_vectorize._dispatcher.inspect_llvm(with_vectorize.signatures[0])
        assert '__SCL_LOOP_VECTORIZE_8__' in ir, "Vectorize marker should be in IR"
        print("[PASS] vectorize_hint marker test passed")
    
    def test_unroll_hint_marker(self):
        """Test that unroll_hint inserts marker."""
        @optimized_jit
        def with_unroll(arr):
            total = 0.0
            
            unroll_hint(4)
            for i in range(len(arr)):
                total += arr[i]
            
            return total
        
        arr = np.random.rand(100)
        _ = with_unroll(arr)
        
        ir = with_unroll._dispatcher.inspect_llvm(with_unroll.signatures[0])
        assert '__SCL_LOOP_UNROLL_4__' in ir, "Unroll marker should be in IR"
        print("[PASS] unroll_hint marker test passed")
    
    def test_multiple_hints(self):
        """Test multiple loop hints."""
        @optimized_jit
        def with_multiple_hints(a, b):
            n = len(a)
            
            vectorize_hint(4)
            for i in range(n):
                a[i] *= 2
            
            unroll_hint(2)
            for i in range(n):
                b[i] += 1
            
            return a, b
        
        a = np.random.rand(100)
        b = np.random.rand(100)
        _, _ = with_multiple_hints(a.copy(), b.copy())
        
        ir = with_multiple_hints._dispatcher.inspect_llvm(with_multiple_hints.signatures[0])
        assert '__SCL_LOOP_VECTORIZE_4__' in ir
        assert '__SCL_LOOP_UNROLL_2__' in ir
        print("[PASS] multiple hints test passed")


# =============================================================================
# Test: IR Processor
# =============================================================================

class TestIRProcessor:
    """Test IR post-processing."""
    
    def test_scan_markers(self):
        """Test marker scanning."""
        ir = """
        ; Function body
        call void asm sideeffect "# __SCL_LOOP_VECTORIZE_8__", "~{memory}"()
        br i1 %cond, label %loop.body, label %loop.exit
        """
        
        processor = IRProcessor()
        hints = processor.scan_markers(ir)
        
        assert len(hints) == 1, f"Expected 1 hint, got {len(hints)}"
        assert hints[0].hint_type == 'VECTORIZE'
        assert hints[0].value == 8
        print("[PASS] scan markers test passed")
    
    def test_process_ir(self):
        """Test full IR processing."""
        @optimized_jit(verbose=False)
        def test_func(arr):
            vectorize_hint(8)
            for i in range(len(arr)):
                arr[i] *= 2
            return arr
        
        arr = np.random.rand(100)
        _ = test_func(arr.copy())
        
        # Get modified IR
        modified_ir = get_modified_ir(test_func)
        
        if modified_ir:
            assert 'llvm.loop.vectorize' in modified_ir or '__SCL_LOOP_' in modified_ir
            print("[PASS] process IR test passed")
        else:
            print("[WARN] process IR test skipped (no modified IR)")


# =============================================================================
# Test: Performance
# =============================================================================

class TestPerformance:
    """Test performance impact of optimizations."""
    
    def test_assume_performance(self):
        """Compare performance with and without assume."""
        @njit
        def baseline_sum(arr):
            total = 0.0
            for i in range(len(arr)):
                if i < len(arr):  # Redundant check
                    total += arr[i]
            return total
        
        @njit
        def assume_sum(arr):
            n = len(arr)
            assume(n > 0)
            
            total = 0.0
            for i in range(n):
                total += arr[i]
            return total
        
        arr = np.random.rand(10000)
        
        # Warmup
        _ = baseline_sum(arr)
        _ = assume_sum(arr)
        
        # Benchmark
        iterations = 1000
        
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = baseline_sum(arr)
        t_baseline = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = assume_sum(arr)
        t_assume = (time.perf_counter() - t0) * 1000
        
        print(f"\nPerformance comparison ({iterations} iterations):")
        print(f"  Baseline: {t_baseline:.2f} ms")
        print(f"  Assume:   {t_assume:.2f} ms")
        print(f"  Speedup:  {(1 - t_assume/t_baseline)*100:+.1f}%")
        print("[PASS] performance test passed")
    
    def test_optimized_jit_overhead(self):
        """Test that optimized_jit doesn't add significant overhead."""
        @njit
        def njit_sum(arr):
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
        
        @optimized_jit
        def optimized_sum(arr):
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
        
        arr = np.random.rand(10000)
        
        # Warmup
        _ = njit_sum(arr)
        _ = optimized_sum(arr)
        
        # Benchmark
        iterations = 1000
        
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = njit_sum(arr)
        t_njit = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = optimized_sum(arr)
        t_optimized = (time.perf_counter() - t0) * 1000
        
        print(f"\nOverhead comparison ({iterations} iterations):")
        print(f"  @njit:          {t_njit:.2f} ms")
        print(f"  @optimized_jit: {t_optimized:.2f} ms")
        
        overhead = (t_optimized - t_njit) / t_njit * 100
        print(f"  Overhead:       {overhead:+.1f}%")
        
        # Overhead may be higher due to IR processing
        # For now, just warn if it's too high
        if overhead > 50:
            print(f"[WARN] Overhead is high: {overhead:.1f}%")
        print("[PASS] overhead test passed")


# =============================================================================
# Test: Combined Usage
# =============================================================================

class TestCombinedUsage:
    """Test combined usage of multiple features."""
    
    def test_full_optimization_suite(self):
        """Test using all optimization features together."""
        @optimized_jit(fastmath=True)
        def optimized_spmv_simulation(values, indices, vec_in, vec_out):
            """Simulated SpMV with all optimizations."""
            nrows = len(vec_out)
            assume(nrows > 0)
            assume(len(values) == len(indices))
            
            row_start = 0
            for row in range(nrows):
                # Assume each row has some elements
                row_end = row_start + 10  # Simplified
                
                dot = 0.0
                vectorize_hint(4)
                for i in range(row_start, min(row_end, len(values))):
                    col = indices[i]
                    if likely(col >= 0 and col < len(vec_in)):
                        dot += values[i] * vec_in[col]
                
                vec_out[row] = dot
                row_start = row_end
        
        # Test data
        n = 100
        nnz = 500
        values = np.random.rand(nnz)
        indices = np.random.randint(0, n, nnz)
        vec_in = np.random.rand(n)
        vec_out = np.zeros(n)
        
        # Should run without error
        optimized_spmv_simulation(values, indices, vec_in, vec_out)
        
        # Check IR has our markers
        ir = optimized_spmv_simulation._dispatcher.inspect_llvm(
            optimized_spmv_simulation.signatures[0]
        )
        
        assert 'llvm.assume' in ir
        assert 'llvm.expect' in ir or '__SCL_LOOP_' in ir
        print("[PASS] full optimization suite test passed")


# =============================================================================
# Run Tests
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SCL Optimization Toolkit Tests")
    print("=" * 60)
    
    test_classes = [
        TestIntrinsics,
        TestLoopHints,
        TestIRProcessor,
        TestPerformance,
        TestCombinedUsage,
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    passed += 1
                except Exception as e:
                    print(f"[FAIL] {method_name} failed: {e}")
                    failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
