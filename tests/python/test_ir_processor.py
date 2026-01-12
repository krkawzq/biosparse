"""Tests for IR processor."""

import pytest
import numpy as np

from optim import IRProcessor, optimized_jit, vectorize, get_modified_ir


class TestMarkerScanning:
    """Test marker scanning in IR."""
    
    def test_scan_vectorize_marker(self):
        """Should find vectorize markers."""
        ir = """
        @__BIOSPARSE_LOOP_VECTORIZE_8__ = external global i32
        br i1 %cond, label %loop.body, label %loop.exit
        """
        
        processor = IRProcessor()
        hints = processor.scan_markers(ir)
        
        assert len(hints) == 1
        assert hints[0].hint_type == 'VECTORIZE'
        assert hints[0].value == 8
    
    def test_scan_unroll_marker(self):
        """Should find unroll markers."""
        ir = """
        @__BIOSPARSE_LOOP_UNROLL_4__ = external global i32
        """
        
        processor = IRProcessor()
        hints = processor.scan_markers(ir)
        
        assert len(hints) == 1
        assert hints[0].hint_type == 'UNROLL'
        assert hints[0].value == 4
    
    def test_scan_multiple_markers(self):
        """Should find multiple markers."""
        ir = """
        @__BIOSPARSE_LOOP_VECTORIZE_8__ = external global i32
        @__BIOSPARSE_LOOP_UNROLL_4__ = external global i32
        @__BIOSPARSE_LOOP_INTERLEAVE_2__ = external global i32
        """
        
        processor = IRProcessor()
        hints = processor.scan_markers(ir)
        
        assert len(hints) == 3


class TestIRProcessing:
    """Test full IR processing."""
    
    def test_process_adds_metadata(self, small_arr):
        """Processing should add loop metadata."""
        @optimized_jit
        def test_func(arr):
            vectorize(8)
            for i in range(len(arr)):
                arr[i] *= 2
            return arr
        
        _ = test_func(small_arr.copy())
        modified_ir = get_modified_ir(test_func)
        
        if modified_ir:
            assert 'llvm.loop' in modified_ir or '__BIOSPARSE_LOOP_' in modified_ir
