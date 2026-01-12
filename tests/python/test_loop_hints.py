"""Tests for loop optimization hints."""

import pytest
import numpy as np

from optim import optimized_jit, vectorize, unroll, interleave


class TestVectorize:
    """Test vectorize hint."""
    
    def test_inserts_marker(self, arr):
        """vectorize() should insert marker in IR."""
        @optimized_jit
        def with_vec(arr):
            vectorize(8)
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
        
        result = with_vec(arr)
        assert np.isclose(result, np.sum(arr))
        
        ir = with_vec._dispatcher.inspect_llvm(with_vec.signatures[0])
        assert '__BIOSPARSE_LOOP_VECTORIZE_8__' in ir
    
    def test_different_widths(self, small_arr):
        """vectorize() should work with different widths."""
        for width in [4, 8, 16]:
            @optimized_jit
            def vec_loop(arr, w=width):
                vectorize(w)
                total = 0.0
                for i in range(len(arr)):
                    total += arr[i]
                return total
            
            result = vec_loop(small_arr)
            assert np.isclose(result, np.sum(small_arr))


class TestUnroll:
    """Test unroll hint."""
    
    def test_inserts_marker(self, small_arr):
        """unroll() should insert marker in IR."""
        @optimized_jit
        def with_unroll(arr):
            unroll(4)
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
        
        _ = with_unroll(small_arr)
        ir = with_unroll._dispatcher.inspect_llvm(with_unroll.signatures[0])
        assert '__BIOSPARSE_LOOP_UNROLL_4__' in ir


class TestInterleave:
    """Test interleave hint."""
    
    def test_inserts_marker(self, small_arr):
        """interleave() should insert marker in IR."""
        @optimized_jit
        def with_interleave(arr):
            interleave(4)
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total
        
        _ = with_interleave(small_arr)
        ir = with_interleave._dispatcher.inspect_llvm(with_interleave.signatures[0])
        assert '__BIOSPARSE_LOOP_INTERLEAVE_4__' in ir


class TestMultipleHints:
    """Test combining multiple hints."""
    
    def test_multiple_loops(self, small_arr):
        """Multiple hints should work on different loops."""
        @optimized_jit
        def multi_hints(a, b):
            vectorize(4)
            for i in range(len(a)):
                a[i] *= 2
            
            unroll(2)
            for i in range(len(b)):
                b[i] += 1
            
            return a, b
        
        a = small_arr.copy()
        b = small_arr.copy()
        _, _ = multi_hints(a, b)
        
        ir = multi_hints._dispatcher.inspect_llvm(multi_hints.signatures[0])
        assert '__BIOSPARSE_LOOP_VECTORIZE_4__' in ir
        assert '__BIOSPARSE_LOOP_UNROLL_2__' in ir
