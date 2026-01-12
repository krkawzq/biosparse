"""Tests for biosparse.optim intrinsics."""

import pytest
import numpy as np

pytest.importorskip("numba")

from numba import njit
from biosparse.optim import assume, likely, unlikely


class TestAssume:
    """Test assume intrinsic."""
    
    def test_compiles_and_runs(self):
        """assume() should compile and execute correctly."""
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
        assert np.isclose(result, np.sum(arr))
    
    def test_in_llvm_ir(self):
        """assume() should generate llvm.assume in IR."""
        @njit
        def with_assume(x):
            assume(x > 0)
            return x * 2
        
        _ = with_assume(10)
        ir = with_assume.inspect_llvm(with_assume.signatures[0])
        assert 'llvm.assume' in ir
    
    def test_multiple_assumes(self):
        """Multiple assume() calls should work."""
        @njit
        def multi_assume(arr):
            n = len(arr)
            assume(n > 0)
            assume(n < 10000)
            assume(n % 2 == 0)
            return n
        
        arr = np.zeros(100)
        result = multi_assume(arr)
        assert result == 100


class TestLikelyUnlikely:
    """Test likely/unlikely intrinsics."""
    
    def test_likely_basic(self, small_arr):
        """likely() should compile and return correct value."""
        @njit
        def with_likely(arr):
            total = 0.0
            for x in arr:
                if likely(x > 0):
                    total += x
            return total
        
        result = with_likely(small_arr)
        expected = sum(x for x in small_arr if x > 0)
        assert np.isclose(result, expected)
    
    def test_unlikely_basic(self, small_arr):
        """unlikely() should compile and return correct value."""
        @njit
        def with_unlikely(arr):
            total = 0.0
            for x in arr:
                if unlikely(x < 0):
                    total -= x
                else:
                    total += x
            return total
        
        result = with_unlikely(small_arr)
        expected = np.sum(small_arr)
        assert np.isclose(result, expected)
    
    def test_generates_expect_or_weights(self):
        """likely/unlikely should generate llvm.expect or branch_weights."""
        @njit
        def with_hints(arr):
            total = 0.0
            for x in arr:
                if likely(x > 0):
                    total += x
            return total
        
        arr = np.random.rand(10)
        _ = with_hints(arr)
        ir = with_hints.inspect_llvm(with_hints.signatures[0])
        assert 'llvm.expect' in ir or 'branch_weights' in ir
    
    def test_nested_conditions(self, small_arr):
        """Nested likely/unlikely should work."""
        @njit
        def nested(arr):
            count = 0
            for x in arr:
                if likely(x > 0):
                    if unlikely(x > 0.99):
                        count += 10
                    else:
                        count += 1
            return count
        
        result = nested(small_arr)
        assert result >= 0
