"""LLVM IR Post-Processor for Loop Optimization Hints.

This module provides functionality to scan compiled LLVM IR for loop
optimization markers and add the corresponding LLVM loop metadata.

The processor:
1. Scans IR for __SCL_LOOP_*__ markers (inserted by loop hint intrinsics)
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
    LLVM IR with inline asm markers
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
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from llvmlite import binding as llvm


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


@dataclass
class LoopHint:
    """Represents a parsed loop optimization hint."""
    hint_type: str
    value: Optional[int]
    line_number: int  # Line in IR where marker was found
    
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
                # Full unroll
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
        
        else:
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
        modified_ir = processor.process(original_ir)
        
        # Or with more control:
        processor = IRProcessor(verbose=True)
        hints = processor.scan_markers(original_ir)
        loops = processor.find_loops(original_ir)
        modified_ir = processor.apply_hints(original_ir, hints, loops)
    """
    
    # Regex patterns for parsing IR
    # Matches both inline asm markers and global variable markers
    MARKER_PATTERN = re.compile(
        r'(?:#\s*)?__SCL_LOOP_(\w+?)(?:_(\d+))?__'
    )
    
    # Pattern to find loop headers (phi nodes with loop-like structure)
    LOOP_HEADER_PATTERN = re.compile(
        r'^(\w+):\s*;\s*preds\s*=.*\n'
        r'(?:.*\n)*?'
        r'.*phi\s+',
        re.MULTILINE
    )
    
    # Pattern to find branch instructions (potential loop backedges)
    BRANCH_PATTERN = re.compile(
        r'br\s+(?:i1\s+%\w+,\s+)?label\s+%(\w+)(?:,\s+label\s+%(\w+))?',
        re.MULTILINE
    )
    
    # Pattern to find the end of a basic block (terminator instruction)
    TERMINATOR_PATTERN = re.compile(
        r'(br\s+.*|ret\s+.*|switch\s+.*|unreachable)$',
        re.MULTILINE
    )
    
    def __init__(self, verbose: bool = False, metadata_start_id: int = 10000):
        """Initialize the IR processor.
        
        Args:
            verbose: If True, print debug information during processing
            metadata_start_id: Starting ID for generated metadata nodes
        """
        self.verbose = verbose
        self.metadata_start_id = metadata_start_id
        self._current_md_id = metadata_start_id
    
    def process(self, ir: str) -> Tuple[str, List[LoopHint]]:
        """Process IR and add loop metadata.
        
        This is the main entry point. It scans for markers, finds loops,
        and applies the appropriate metadata.
        
        Args:
            ir: LLVM IR string to process
        
        Returns:
            Tuple of (modified_ir, list of applied hints)
        """
        # Reset metadata ID counter
        self._current_md_id = self.metadata_start_id
        
        # Step 1: Scan for markers
        hints = self.scan_markers(ir)
        
        if not hints:
            if self.verbose:
                print("[IRProcessor] No loop hints found in IR")
            return ir, []
        
        if self.verbose:
            print(f"[IRProcessor] Found {len(hints)} loop hints:")
            for hint in hints:
                print(f"  - {hint.hint_type}({hint.value}) at line {hint.line_number}")
        
        # Step 2: Find loops after each marker
        loop_associations = self._associate_hints_with_loops(ir, hints)
        
        if self.verbose:
            print(f"[IRProcessor] Associated {len(loop_associations)} hints with loops")
        
        # Step 3: Generate and insert metadata
        modified_ir = self._insert_metadata(ir, loop_associations)
        
        return modified_ir, hints
    
    def scan_markers(self, ir: str) -> List[LoopHint]:
        """Scan IR for loop hint markers.
        
        Args:
            ir: LLVM IR string to scan
        
        Returns:
            List of LoopHint objects found in the IR
        """
        hints = []
        lines = ir.split('\n')
        
        for line_num, line in enumerate(lines):
            match = self.MARKER_PATTERN.search(line)
            if match:
                hint_type = match.group(1)
                value_str = match.group(2)
                value = int(value_str) if value_str else None
                
                hints.append(LoopHint(
                    hint_type=hint_type,
                    value=value,
                    line_number=line_num
                ))
        
        return hints
    
    def _associate_hints_with_loops(
        self, 
        ir: str, 
        hints: List[LoopHint]
    ) -> List[Tuple[LoopHint, int]]:
        """Associate each hint with the next loop's branch instruction.
        
        Args:
            ir: LLVM IR string
            hints: List of hints to associate
        
        Returns:
            List of (hint, branch_line_number) tuples
        """
        lines = ir.split('\n')
        associations = []
        
        for hint in hints:
            # Find the next branch instruction after this hint
            # that looks like a loop backedge
            branch_line = self._find_next_loop_branch(lines, hint.line_number)
            
            if branch_line is not None:
                associations.append((hint, branch_line))
            elif self.verbose:
                print(f"[IRProcessor] Warning: No loop found after hint at line {hint.line_number}")
        
        return associations
    
    def _find_next_loop_branch(self, lines: List[str], start_line: int) -> Optional[int]:
        """Find the next loop-like branch instruction after a given line.
        
        A loop branch is typically a conditional branch where one target
        is a label that appears before the branch (forming a backedge).
        
        Args:
            lines: List of IR lines
            start_line: Line number to start searching from
        
        Returns:
            Line number of the branch instruction, or None if not found
        """
        # Collect all label definitions before we search
        labels_before = set()
        current_label = None
        
        for i, line in enumerate(lines):
            # Track label definitions
            label_match = re.match(r'^(\w+):', line)
            if label_match:
                current_label = label_match.group(1)
                if i < start_line:
                    labels_before.add(current_label)
        
        # Now search for branch instructions after the marker
        in_scope = False
        scope_labels = set()
        
        for i in range(start_line + 1, min(start_line + 100, len(lines))):
            line = lines[i]
            
            # Track labels we pass
            label_match = re.match(r'^(\w+):', line)
            if label_match:
                scope_labels.add(label_match.group(1))
            
            # Look for branch instructions
            branch_match = re.search(r'br\s+i1\s+%\w+,\s+label\s+%(\w+),\s+label\s+%(\w+)', line)
            if branch_match:
                target1 = branch_match.group(1)
                target2 = branch_match.group(2)
                
                # Check if either target is a backedge (jumps to a label we passed)
                if target1 in scope_labels or target2 in scope_labels:
                    return i
        
        # Fallback: return the next conditional branch
        for i in range(start_line + 1, min(start_line + 50, len(lines))):
            line = lines[i]
            if re.search(r'br\s+i1\s+', line):
                return i
        
        return None
    
    def _insert_metadata(
        self, 
        ir: str, 
        associations: List[Tuple[LoopHint, int]]
    ) -> str:
        """Insert loop metadata into the IR.
        
        Args:
            ir: Original LLVM IR string
            associations: List of (hint, branch_line) tuples
        
        Returns:
            Modified IR with loop metadata
        """
        if not associations:
            return ir
        
        lines = ir.split('\n')
        metadata_blocks = []
        
        # Process each association
        for hint, branch_line in associations:
            md_id = self._get_next_md_id()
            
            # Generate metadata
            metadata = hint.to_metadata(md_id)
            if metadata:
                metadata_blocks.append(metadata)
                
                # Add !llvm.loop reference to the branch instruction
                if branch_line < len(lines):
                    line = lines[branch_line]
                    # Check if already has metadata
                    if '!llvm.loop' not in line:
                        # Add metadata reference before any existing metadata
                        if line.rstrip().endswith(')'):
                            # Function call - add after
                            lines[branch_line] = f"{line}, !llvm.loop !{md_id}"
                        else:
                            # Regular instruction
                            lines[branch_line] = f"{line.rstrip()}, !llvm.loop !{md_id}"
        
        # Reconstruct IR
        modified_ir = '\n'.join(lines)
        
        # Append metadata at the end
        if metadata_blocks:
            modified_ir += '\n\n; SCL Loop Optimization Metadata\n'
            modified_ir += '\n'.join(metadata_blocks)
        
        return modified_ir
    
    def _get_next_md_id(self) -> int:
        """Get the next available metadata ID."""
        md_id = self._current_md_id
        self._current_md_id += 10  # Reserve space for sub-metadata
        return md_id
    
    def remove_markers(self, ir: str) -> str:
        """Remove marker inline assembly from IR.
        
        After processing, the markers are no longer needed and can be
        removed to clean up the IR.
        
        Args:
            ir: LLVM IR string
        
        Returns:
            IR with marker instructions removed
        """
        # Pattern to match the entire inline asm call with marker
        pattern = re.compile(
            r'call void asm sideeffect "# __SCL_LOOP_\w+(?:_\d+)?__".*?\n',
            re.MULTILINE
        )
        return pattern.sub('', ir)


# =============================================================================
# Convenience Function
# =============================================================================

def process_ir(ir: str, verbose: bool = False) -> str:
    """Process LLVM IR and add loop optimization metadata.
    
    This is a convenience function that creates an IRProcessor and
    processes the given IR.
    
    Args:
        ir: LLVM IR string to process
        verbose: If True, print debug information
    
    Returns:
        Modified IR with loop metadata
    
    Example:
        original_ir = func.inspect_llvm(sig)
        optimized_ir = process_ir(original_ir)
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
    # Initialize LLVM (safe to call multiple times)
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    
    try:
        # Parse IR
        mod = llvm.parse_assembly(ir)
        mod.verify()
    except Exception as e:
        raise RuntimeError(f"Failed to parse/verify IR: {e}")
    
    # Create target machine
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine(opt=opt_level)
    
    # Create pass manager and run optimizations
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
        target_machine: Optional target machine (creates default if None)
    
    Returns:
        Function pointer as integer
    """
    if target_machine is None:
        target = llvm.Target.from_default_triple()
        target_machine = target.create_target_machine()
    
    # Create execution engine
    engine = llvm.create_mcjit_compiler(module, target_machine)
    
    # Get function address
    return engine.get_function_address(func_name)
