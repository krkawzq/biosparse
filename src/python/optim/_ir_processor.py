"""LLVM IR Post-Processor for Loop Optimization Hints.

This module provides functionality to scan compiled LLVM IR for loop
optimization markers and add the corresponding LLVM loop metadata.

The processor:
1. Scans IR for __BIOSPARSE_LOOP_*__ markers (inserted by loop hint intrinsics)
2. Identifies the next loop after each marker
3. Adds appropriate LLVM loop metadata to the loop's branch instruction
4. Optionally recompiles the modified IR

Architecture:
    User Code with loop hints
           |
           v
    Numba @njit compilation
           |
           v
    LLVM IR with markers
           |
           v
    IRProcessor.process()  <-- This module
           |
           v
    LLVM IR with loop metadata
           |
           v
    llvmlite optimization & codegen
           |
           v
    Optimized machine code
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from llvmlite import binding as llvm

from ._logging import logger


__all__ = [
    'IRProcessor',
    'LoopHint',
    'process_ir',
    'HintType',
]


# =============================================================================
# Data Structures
# =============================================================================

class HintType:
    """Enumeration of supported hint types."""
    VECTORIZE = 'VECTORIZE'
    NO_VECTORIZE = 'NO_VECTORIZE'
    UNROLL = 'UNROLL'
    NO_UNROLL = 'NO_UNROLL'
    INTERLEAVE = 'INTERLEAVE'
    DISTRIBUTE = 'DISTRIBUTE'
    PIPELINE = 'PIPELINE'


@dataclass(slots=True)
class LoopHint:
    """Represents a parsed loop optimization hint."""
    hint_type: str
    value: Optional[int]
    line_number: int
    
    def to_metadata(self, md_id: int) -> str:
        """Generate LLVM metadata for this hint.
        
        Args:
            md_id: Base metadata ID to use
        
        Returns:
            LLVM metadata string
        """
        if self.hint_type == HintType.VECTORIZE:
            return f"""!{md_id} = distinct !{{!{md_id}, !{md_id + 1}, !{md_id + 2}}}
!{md_id + 1} = !{{"llvm.loop.vectorize.enable", i1 true}}
!{md_id + 2} = !{{"llvm.loop.vectorize.width", i32 {self.value}}}"""
        
        elif self.hint_type == HintType.NO_VECTORIZE:
            return f"""!{md_id} = distinct !{{!{md_id}, !{md_id + 1}}}
!{md_id + 1} = !{{"llvm.loop.vectorize.enable", i1 false}}"""
        
        elif self.hint_type == HintType.UNROLL:
            if self.value == 0:
                return f"""!{md_id} = distinct !{{!{md_id}, !{md_id + 1}}}
!{md_id + 1} = !{{"llvm.loop.unroll.full"}}"""
            else:
                return f"""!{md_id} = distinct !{{!{md_id}, !{md_id + 1}, !{md_id + 2}}}
!{md_id + 1} = !{{"llvm.loop.unroll.enable", i1 true}}
!{md_id + 2} = !{{"llvm.loop.unroll.count", i32 {self.value}}}"""
        
        elif self.hint_type == HintType.NO_UNROLL:
            return f"""!{md_id} = distinct !{{!{md_id}, !{md_id + 1}}}
!{md_id + 1} = !{{"llvm.loop.unroll.disable"}}"""
        
        elif self.hint_type == HintType.INTERLEAVE:
            return f"""!{md_id} = distinct !{{!{md_id}, !{md_id + 1}}}
!{md_id + 1} = !{{"llvm.loop.interleave.count", i32 {self.value}}}"""
        
        elif self.hint_type == HintType.DISTRIBUTE:
            return f"""!{md_id} = distinct !{{!{md_id}, !{md_id + 1}}}
!{md_id + 1} = !{{"llvm.loop.distribute.enable", i1 true}}"""
        
        elif self.hint_type == HintType.PIPELINE:
            stages = self.value if self.value else 0
            return f"""!{md_id} = distinct !{{!{md_id}, !{md_id + 1}}}
!{md_id + 1} = !{{"llvm.loop.pipeline.initiationinterval", i32 {stages}}}"""
        
        return ""


# =============================================================================
# IR Processor
# =============================================================================

class IRProcessor:
    """Processes LLVM IR to add loop optimization metadata.
    
    This class scans compiled LLVM IR for loop hint markers and adds
    the corresponding LLVM loop metadata to enable optimizations.
    
    Example:
        processor = IRProcessor()
        modified_ir, hints = processor.process(original_ir)
    """
    
    # Regex patterns (compiled once for efficiency)
    _MARKER_PATTERN = re.compile(r'(?:#\s*)?__BIOSPARSE_LOOP_(\w+?)(?:_(\d+))?__')
    _LABEL_PATTERN = re.compile(r'^(\w+):')
    _BRANCH_PATTERN = re.compile(r'br\s+i1\s+%\w+,\s+label\s+%(\w+),\s+label\s+%(\w+)')
    _SIMPLE_BRANCH_PATTERN = re.compile(r'br\s+i1\s+')
    
    __slots__ = ('verbose', 'metadata_start_id', '_current_md_id')
    
    def __init__(self, verbose: bool = False, metadata_start_id: int = 10000):
        """Initialize the IR processor.
        
        Args:
            verbose: If True, enable debug logging
            metadata_start_id: Starting ID for generated metadata nodes
        """
        self.verbose = verbose
        self.metadata_start_id = metadata_start_id
        self._current_md_id = metadata_start_id
    
    def process(self, ir: str) -> Tuple[str, List[LoopHint]]:
        """Process IR and add loop metadata.
        
        Args:
            ir: LLVM IR string to process
        
        Returns:
            Tuple of (modified_ir, list of applied hints)
        """
        self._current_md_id = self.metadata_start_id
        
        # Scan for markers
        hints = self.scan_markers(ir)
        
        if not hints:
            logger.debug("No loop hints found in IR")
            return ir, []
        
        logger.debug("Found %d loop hints", len(hints))
        for hint in hints:
            logger.debug("  %s(%s) at line %d", hint.hint_type, hint.value, hint.line_number)
        
        # Associate hints with loops
        associations = self._associate_hints_with_loops(ir, hints)
        logger.debug("Associated %d hints with loops", len(associations))
        
        # Insert metadata
        modified_ir = self._insert_metadata(ir, associations)
        
        return modified_ir, hints
    
    def scan_markers(self, ir: str) -> List[LoopHint]:
        """Scan IR for loop hint markers.
        
        Args:
            ir: LLVM IR string to scan
        
        Returns:
            List of LoopHint objects found in the IR
        """
        hints = []
        
        for line_num, line in enumerate(ir.split('\n')):
            match = self._MARKER_PATTERN.search(line)
            if match:
                hint_type = match.group(1)
                value_str = match.group(2)
                value = int(value_str) if value_str else None
                hints.append(LoopHint(hint_type, value, line_num))
        
        return hints
    
    def _associate_hints_with_loops(
        self, 
        ir: str, 
        hints: List[LoopHint]
    ) -> List[Tuple[LoopHint, int]]:
        """Associate each hint with the next loop's branch instruction."""
        lines = ir.split('\n')
        associations = []
        
        for hint in hints:
            branch_line = self._find_next_loop_branch(lines, hint.line_number)
            if branch_line is not None:
                associations.append((hint, branch_line))
            else:
                logger.debug("No loop found after hint at line %d", hint.line_number)
        
        return associations
    
    def _find_next_loop_branch(self, lines: List[str], start_line: int) -> Optional[int]:
        """Find the next loop-like branch instruction after a given line."""
        # Track labels in the search range
        scope_labels = set()
        
        # Search window: up to 100 lines after marker
        end_line = min(start_line + 100, len(lines))
        
        for i in range(start_line + 1, end_line):
            line = lines[i]
            
            # Track label definitions
            label_match = self._LABEL_PATTERN.match(line)
            if label_match:
                scope_labels.add(label_match.group(1))
            
            # Look for conditional branch (loop backedge)
            branch_match = self._BRANCH_PATTERN.search(line)
            if branch_match:
                target1, target2 = branch_match.group(1), branch_match.group(2)
                # Check if either target is a backedge
                if target1 in scope_labels or target2 in scope_labels:
                    return i
        
        # Fallback: return next conditional branch
        for i in range(start_line + 1, min(start_line + 50, len(lines))):
            if self._SIMPLE_BRANCH_PATTERN.search(lines[i]):
                return i
        
        return None
    
    def _insert_metadata(
        self, 
        ir: str, 
        associations: List[Tuple[LoopHint, int]]
    ) -> str:
        """Insert loop metadata into the IR."""
        if not associations:
            return ir
        
        lines = ir.split('\n')
        metadata_blocks = []
        
        for hint, branch_line in associations:
            md_id = self._get_next_md_id()
            metadata = hint.to_metadata(md_id)
            
            if metadata:
                metadata_blocks.append(metadata)
                
                # Add !llvm.loop reference to branch instruction
                if branch_line < len(lines):
                    line = lines[branch_line]
                    if '!llvm.loop' not in line:
                        lines[branch_line] = f"{line.rstrip()}, !llvm.loop !{md_id}"
        
        modified_ir = '\n'.join(lines)
        
        if metadata_blocks:
            modified_ir += '\n\n; BioSparse Loop Optimization Metadata\n'
            modified_ir += '\n'.join(metadata_blocks)
        
        return modified_ir
    
    def _get_next_md_id(self) -> int:
        """Get the next available metadata ID."""
        md_id = self._current_md_id
        self._current_md_id += 10
        return md_id
    
    def remove_markers(self, ir: str) -> str:
        """Remove marker instructions from IR."""
        pattern = re.compile(
            r'call void asm sideeffect "# __BIOSPARSE_LOOP_\w+(?:_\d+)?__".*?\n',
            re.MULTILINE
        )
        return pattern.sub('', ir)


# =============================================================================
# Convenience Functions
# =============================================================================

def process_ir(ir: str, verbose: bool = False) -> str:
    """Process LLVM IR and add loop optimization metadata.
    
    Args:
        ir: LLVM IR string to process
        verbose: If True, enable debug logging
    
    Returns:
        Modified IR with loop metadata
    """
    processor = IRProcessor(verbose=verbose)
    modified_ir, _ = processor.process(ir)
    return modified_ir


# =============================================================================
# LLVM Compilation Helpers
# =============================================================================

def compile_modified_ir(ir: str, opt_level: int = 3) -> llvm.ModuleRef:
    """Compile modified IR using llvmlite.
    
    Args:
        ir: LLVM IR string to compile
        opt_level: Optimization level (0-3)
    
    Returns:
        Compiled LLVM module
    
    Raises:
        RuntimeError: If IR parsing or verification fails
    """
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    
    try:
        mod = llvm.parse_assembly(ir)
        mod.verify()
    except Exception as e:
        raise RuntimeError(f"Failed to parse/verify IR: {e}") from e
    
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine(opt=opt_level)
    
    pmb = llvm.create_pass_manager_builder()
    pmb.opt_level = opt_level
    pmb.loop_vectorize = True
    pmb.slp_vectorize = True
    
    pm = llvm.create_module_pass_manager()
    pmb.populate(pm)
    pm.run(mod)
    
    return mod


def get_function_pointer(
    module: llvm.ModuleRef, 
    func_name: str,
    target_machine: Optional[llvm.TargetMachine] = None
) -> int:
    """Get function pointer from compiled module.
    
    Args:
        module: Compiled LLVM module
        func_name: Name of the function
        target_machine: Optional target machine
    
    Returns:
        Function pointer as integer
    """
    if target_machine is None:
        target = llvm.Target.from_default_triple()
        target_machine = target.create_target_machine()
    
    engine = llvm.create_mcjit_compiler(module, target_machine)
    return engine.get_function_address(func_name)
