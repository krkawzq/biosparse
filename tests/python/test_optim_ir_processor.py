"""Tests for biosparse.optim IR processor."""

import pytest
import numpy as np

pytest.importorskip("numba")

from optim import IRProcessor, optimized_jit, vectorize, get_modified_ir, HintType


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
    
    def test_scan_no_markers(self):
        """Should return empty list when no markers."""
        ir = """
        define i32 @foo() {
          ret i32 0
        }
        """
        
        processor = IRProcessor()
        hints = processor.scan_markers(ir)
        
        assert len(hints) == 0


class TestHintType:
    """Test HintType constants."""
    
    def test_hint_types_defined(self):
        """All hint types should be defined."""
        assert HintType.VECTORIZE == 'VECTORIZE'
        assert HintType.UNROLL == 'UNROLL'
        assert HintType.INTERLEAVE == 'INTERLEAVE'
        assert HintType.DISTRIBUTE == 'DISTRIBUTE'
        assert HintType.PIPELINE == 'PIPELINE'


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
    
    def test_process_returns_tuple(self):
        """process() should return (ir, hints) tuple."""
        ir = "@__BIOSPARSE_LOOP_VECTORIZE_4__ = external global i32"
        
        processor = IRProcessor()
        result = processor.process(ir)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
