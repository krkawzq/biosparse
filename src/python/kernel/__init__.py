"""BioSparse Kernel Module.

High-performance computational kernels for bioinformatics,
implemented using Numba JIT with optimization hints.

Design Pattern (One-vs-All):
    All group-based kernels use the same convention:
    - group_ids: 0 = reference group, 1/2/3... = target groups
    - One-vs-all: computes ref vs target_i for all targets at once
    - Output shape: (n_rows, n_targets) for multi-target results

Submodules:
    math: Statistical and mathematical functions
    hvg: Highly Variable Gene selection
    mwu: Mann-Whitney U test (ref vs targets)
    mmd: Maximum Mean Discrepancy (ref vs targets)
    ttest: T-test Welch and Student (ref vs targets)

All sparse matrix kernels accept the project's CSR type directly.
"""

from . import math
from . import hvg
from . import mwu
from . import mmd
from . import ttest

__all__ = [
    'math',
    'hvg',
    'mwu',
    'mmd',
    'ttest',
]
