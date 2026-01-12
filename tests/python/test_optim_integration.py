"""Integration tests for biosparse.optim."""

import pytest
import numpy as np

pytest.importorskip("numba")

from biosparse.optim import (
    optimized_jit, fast_jit,
    assume, likely,
    vectorize, unroll,
)


class TestCombinedUsage:
    """Test combining multiple features."""
    
    def test_full_optimization_suite(self):
        """All optimization features should work together."""
        @optimized_jit(fastmath=True)
        def optimized_spmv(values, indices, vec_in, vec_out):
            nrows = len(vec_out)
            assume(nrows > 0)
            assume(len(values) == len(indices))
            
            row_start = 0
            for row in range(nrows):
                row_end = row_start + 10
                
                dot = 0.0
                vectorize(4)
                for i in range(row_start, min(row_end, len(values))):
                    col = indices[i]
                    if likely(col >= 0 and col < len(vec_in)):
                        dot += values[i] * vec_in[col]
                
                vec_out[row] = dot
                row_start = row_end
        
        n = 100
        nnz = 500
        values = np.random.rand(nnz)
        indices = np.random.randint(0, n, nnz)
        vec_in = np.random.rand(n)
        vec_out = np.zeros(n)
        
        optimized_spmv(values, indices, vec_in, vec_out)
        
        ir = optimized_spmv._dispatcher.inspect_llvm(
            optimized_spmv.signatures[0]
        )
        assert 'llvm.assume' in ir
    
    def test_nested_loops(self):
        """Hints should work with nested loops."""
        @optimized_jit
        def matmul_hints(a, b, c):
            n = len(c)
            
            for i in range(n):
                unroll(4)
                for j in range(n):
                    vectorize(4)
                    for k in range(n):
                        c[i, j] += a[i, k] * b[k, j]
        
        n = 16
        a = np.random.rand(n, n)
        b = np.random.rand(n, n)
        c = np.zeros((n, n))
        
        matmul_hints(a, b, c)
        expected = a @ b
        assert np.allclose(c, expected)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_array(self):
        """Should handle empty arrays."""
        @optimized_jit
        def sum_empty(arr):
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
        
        result = sum_empty(np.array([]))
        assert result == 0.0
    
    def test_single_element(self):
        """Should handle single element arrays."""
        @optimized_jit
        def sum_single(arr):
            vectorize(4)
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
        
        result = sum_single(np.array([42.0]))
        assert result == 42.0
    
    def test_different_dtypes(self):
        """Should work with different data types."""
        @optimized_jit
        def sum_int(arr):
            total = 0
            for i in range(len(arr)):
                total += arr[i]
            return total
        
        arr_int32 = np.array([1, 2, 3, 4], dtype=np.int32)
        arr_int64 = np.array([1, 2, 3, 4], dtype=np.int64)
        
        assert sum_int(arr_int32) == 10
        assert sum_int(arr_int64) == 10
    
    def test_2d_array(self):
        """Should work with 2D arrays."""
        @optimized_jit
        def sum_2d(arr):
            total = 0.0
            for i in range(arr.shape[0]):
                vectorize(4)
                for j in range(arr.shape[1]):
                    total += arr[i, j]
            return total
        
        arr = np.random.rand(10, 10)
        result = sum_2d(arr)
        assert np.isclose(result, np.sum(arr))
    
    def test_conditional_loops(self):
        """Should handle loops with complex conditions."""
        @optimized_jit
        def conditional_sum(arr, threshold):
            total = 0.0
            count = 0
            
            for i in range(len(arr)):
                if likely(arr[i] > threshold):
                    total += arr[i]
                    count += 1
            
            return total, count
        
        arr = np.random.rand(100)
        total, count = conditional_sum(arr, 0.5)
        
        expected_mask = arr > 0.5
        assert np.isclose(total, np.sum(arr[expected_mask]))
        assert count == np.sum(expected_mask)
