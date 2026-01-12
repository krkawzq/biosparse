#!/usr/bin/env python
"""Benchmark: biosparse MWU vs Scanpy Wilcoxon Core Operators.

直接对比核心算子性能（不包含封装层开销）:
1. biosparse.kernel.mwu.mwu_test (Numba JIT on custom CSR)
2. scanpy 核心 wilcoxon 算子 (rankdata + tiecorrect + rank_sum)

Usage:
    python benchmarks/benchmark_mwu_vs_scanpy.py
    python benchmarks/benchmark_mwu_vs_scanpy.py --quick
    python benchmarks/benchmark_mwu_vs_scanpy.py --large
"""

import argparse
import time
import warnings
import numpy as np
import numba
from numba import njit, prange
from scipy import stats

warnings.filterwarnings('ignore')

# Check dependencies
try:
    import scipy.sparse as sp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("ERROR: scipy is required")

try:
    from biosparse._binding import CSRF64
    import biosparse._numba
    from biosparse.kernel.mwu import mwu_test
    BIOSPARSE_AVAILABLE = True
except ImportError:
    BIOSPARSE_AVAILABLE = False
    print("ERROR: biosparse is required")


# =============================================================================
# Scanpy Core Operators (extracted from scanpy/tools/_rank_genes_groups.py)
# =============================================================================

@njit(parallel=True, cache=True)
def scanpy_rankdata(data: np.ndarray) -> np.ndarray:
    """Scanpy's parallelized rankdata - ranks columns independently.
    
    Input: (n_cells, n_genes) - each column is a gene
    Output: (n_cells, n_genes) - ranks for each gene
    """
    ranked = np.empty(data.shape, dtype=np.float64)
    for j in prange(data.shape[1]):
        arr = np.ravel(data[:, j])
        sorter = np.argsort(arr)
        
        arr_sorted = arr[sorter]
        obs = np.concatenate((np.array([True]), arr_sorted[1:] != arr_sorted[:-1]))
        
        dense = np.empty(obs.size, dtype=np.int64)
        dense[sorter] = obs.cumsum()
        
        # cumulative counts of each unique value
        count = np.concatenate((np.flatnonzero(obs), np.array([len(obs)])))
        ranked[:, j] = 0.5 * (count[dense] + count[dense - 1] + 1)
    
    return ranked


@njit(parallel=True, cache=True)
def scanpy_tiecorrect(rankvals: np.ndarray) -> np.ndarray:
    """Scanpy's parallelized tiecorrect.
    
    Input: (n_cells, n_genes) - rank values
    Output: (n_genes,) - tie correction coefficients
    """
    tc = np.ones(rankvals.shape[1], dtype=np.float64)
    for j in prange(rankvals.shape[1]):
        arr = np.sort(np.ravel(rankvals[:, j]))
        idx = np.flatnonzero(
            np.concatenate((np.array([True]), arr[1:] != arr[:-1], np.array([True])))
        )
        cnt = np.diff(idx).astype(np.float64)
        
        size = np.float64(arr.size)
        if size >= 2:
            tc[j] = 1.0 - (cnt**3 - cnt).sum() / (size**3 - size)
    
    return tc


def scanpy_wilcoxon_core(X: np.ndarray, mask_group: np.ndarray, tie_correct: bool = True):
    """Scanpy's core wilcoxon test implementation.
    
    Args:
        X: Dense matrix (n_cells, n_genes) - cells x genes format
        mask_group: Boolean mask for target group
        tie_correct: Whether to apply tie correction
    
    Returns:
        scores: Z-scores for each gene
        pvals: P-values for each gene
    """
    n_cells, n_genes = X.shape
    n_active = np.count_nonzero(mask_group)
    n_rest = n_cells - n_active
    
    # Compute ranks for all cells
    ranks = scanpy_rankdata(X)
    
    # Compute tie correction if needed
    if tie_correct:
        tc_coef = scanpy_tiecorrect(ranks)
    else:
        tc_coef = 1.0
    
    # Compute rank sum for target group
    rank_sum = ranks[mask_group, :].sum(axis=0)
    
    # Compute z-scores
    # Expected value under null: n_active * (n_cells + 1) / 2
    expected = n_active * (n_cells + 1) / 2.0
    
    # Standard deviation with tie correction
    std_dev = np.sqrt(tc_coef * n_active * n_rest * (n_cells + 1) / 12.0)
    
    # Avoid division by zero
    std_dev = np.where(std_dev < 1e-15, 1e-15, std_dev)
    
    scores = (rank_sum - expected) / std_dev
    scores = np.where(np.isnan(scores), 0, scores)
    
    # Two-sided p-values
    pvals = 2 * stats.distributions.norm.sf(np.abs(scores))
    
    return scores, pvals


def scanpy_wilcoxon_multigroup(X: np.ndarray, group_ids: np.ndarray, n_groups: int, tie_correct: bool = True):
    """Scanpy's wilcoxon for multiple groups (one-vs-rest).
    
    Args:
        X: Dense matrix (n_cells, n_genes)
        group_ids: Group ID for each cell (0, 1, 2, ...)
        n_groups: Total number of groups
        tie_correct: Whether to apply tie correction
    
    Returns:
        all_scores: (n_genes, n_groups) Z-scores
        all_pvals: (n_genes, n_groups) P-values
    """
    n_cells, n_genes = X.shape
    all_scores = np.zeros((n_genes, n_groups), dtype=np.float64)
    all_pvals = np.zeros((n_genes, n_groups), dtype=np.float64)
    
    # Compute ranks once for all cells (shared across groups)
    ranks = scanpy_rankdata(X)
    
    for g in range(n_groups):
        mask_group = group_ids == g
        n_active = np.count_nonzero(mask_group)
        n_rest = n_cells - n_active
        
        if n_active == 0 or n_rest == 0:
            all_pvals[:, g] = 1.0
            continue
        
        # Tie correction
        if tie_correct:
            tc_coef = scanpy_tiecorrect(ranks)
        else:
            tc_coef = 1.0
        
        # Rank sum for this group
        rank_sum = ranks[mask_group, :].sum(axis=0)
        
        # Expected and std
        expected = n_active * (n_cells + 1) / 2.0
        std_dev = np.sqrt(tc_coef * n_active * n_rest * (n_cells + 1) / 12.0)
        std_dev = np.where(std_dev < 1e-15, 1e-15, std_dev)
        
        scores = (rank_sum - expected) / std_dev
        scores = np.where(np.isnan(scores), 0, scores)
        pvals = 2 * stats.distributions.norm.sf(np.abs(scores))
        
        all_scores[:, g] = scores
        all_pvals[:, g] = pvals
    
    return all_scores, all_pvals


# =============================================================================
# Utilities
# =============================================================================

def format_time(seconds: float) -> str:
    """Format time with appropriate unit."""
    if seconds < 0.001:
        return f"{seconds*1e6:.1f}us"
    elif seconds < 1:
        return f"{seconds*1e3:.2f}ms"
    else:
        return f"{seconds:.2f}s"


def create_test_data(
    n_genes: int, 
    n_cells: int, 
    density: float,
    n_groups: int,
    seed: int = 42
):
    """Create test data for both biosparse and scanpy.
    
    Returns:
        scipy_csr: scipy sparse matrix (genes x cells) for biosparse
        bio_csr: biosparse CSR matrix (genes x cells)
        dense_cells_genes: dense matrix (cells x genes) for scanpy
        group_ids: numpy array of group assignments
    """
    np.random.seed(seed)
    
    # Create sparse matrix (genes x cells)
    scipy_csr = sp.random(n_genes, n_cells, density=density, format='csr', dtype=np.float64)
    
    # Set realistic values (log-normalized counts)
    scipy_csr.data[:] = np.abs(np.random.randn(len(scipy_csr.data))) * 2
    
    # Create balanced group assignments
    group_ids = np.zeros(n_cells, dtype=np.int64)
    cells_per_group = n_cells // n_groups
    for g in range(n_groups):
        start = g * cells_per_group
        end = start + cells_per_group if g < n_groups - 1 else n_cells
        group_ids[start:end] = g
    np.random.shuffle(group_ids)
    
    # Create biosparse matrix (genes x cells)
    bio_csr = CSRF64.from_scipy(scipy_csr) if BIOSPARSE_AVAILABLE else None
    
    # Create dense matrix (cells x genes) for scanpy
    dense_cells_genes = scipy_csr.T.toarray()  # transpose to cells x genes
    
    return scipy_csr, bio_csr, dense_cells_genes, group_ids


def warmup_jit():
    """Warmup JIT compilation with small data."""
    print("Warming up JIT compilation...")
    
    # Warmup biosparse
    if BIOSPARSE_AVAILABLE:
        scipy_small = sp.random(100, 50, density=0.1, format='csr', dtype=np.float64)
        scipy_small.data[:] = np.abs(np.random.randn(len(scipy_small.data)))
        bio_small = CSRF64.from_scipy(scipy_small)
        group_ids_small = np.array([0]*25 + [1]*25, dtype=np.int64)
        _ = mwu_test(bio_small, group_ids_small, n_targets=1)
    
    # Warmup scanpy operators
    dense_small = np.random.randn(50, 100).astype(np.float64)
    _ = scanpy_rankdata(dense_small)
    _ = scanpy_tiecorrect(dense_small)
    
    print("JIT warmup complete.\n")


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_biosparse_mwu(bio_csr, group_ids: np.ndarray, n_targets: int, n_runs: int = 5):
    """Benchmark biosparse MWU test."""
    # Warmup run
    _ = mwu_test(bio_csr, group_ids, n_targets)
    
    start = time.perf_counter()
    for _ in range(n_runs):
        u_stats, p_values, log2_fc, auroc = mwu_test(bio_csr, group_ids, n_targets)
    elapsed = time.perf_counter() - start
    
    return elapsed / n_runs, (u_stats, p_values, log2_fc, auroc)


def benchmark_scanpy_wilcoxon(dense_mat: np.ndarray, group_ids: np.ndarray, n_groups: int, n_runs: int = 5):
    """Benchmark scanpy wilcoxon core operators."""
    # Warmup run
    _ = scanpy_wilcoxon_multigroup(dense_mat, group_ids, n_groups, tie_correct=True)
    
    start = time.perf_counter()
    for _ in range(n_runs):
        scores, pvals = scanpy_wilcoxon_multigroup(dense_mat, group_ids, n_groups, tie_correct=True)
    elapsed = time.perf_counter() - start
    
    return elapsed / n_runs, (scores, pvals)


def benchmark_scanpy_rankdata_only(dense_mat: np.ndarray, n_runs: int = 5):
    """Benchmark just the rankdata step."""
    # Warmup
    _ = scanpy_rankdata(dense_mat)
    
    start = time.perf_counter()
    for _ in range(n_runs):
        ranks = scanpy_rankdata(dense_mat)
    elapsed = time.perf_counter() - start
    
    return elapsed / n_runs


def benchmark_scipy_mwu(dense_mat: np.ndarray, group_ids: np.ndarray, n_groups: int, n_sample: int = 50):
    """Benchmark scipy MWU (sampled, as full is too slow)."""
    # dense_mat is (cells x genes), we need (genes x cells) for per-gene comparison
    dense_genes_cells = dense_mat.T
    n_genes = min(n_sample, dense_genes_cells.shape[0])
    
    ref_mask = group_ids == 0
    ref_data = dense_genes_cells[:n_genes, ref_mask]
    
    start = time.perf_counter()
    for t in range(1, n_groups):
        tar_mask = group_ids == t
        tar_data = dense_genes_cells[:n_genes, tar_mask]
        
        for g in range(n_genes):
            u, p = stats.mannwhitneyu(ref_data[g], tar_data[g], alternative='two-sided')
    elapsed = time.perf_counter() - start
    
    return elapsed, n_genes


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(
    n_genes: int,
    n_cells: int,
    n_groups: int,
    density: float,
    n_runs: int = 5
):
    """Run full benchmark comparison."""
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {n_genes} genes x {n_cells} cells, {n_groups} groups, {density:.0%} density")
    print(f"{'='*80}")
    
    # Create test data
    print("\nCreating test data...")
    scipy_csr, bio_csr, dense_cells_genes, group_ids = create_test_data(
        n_genes, n_cells, density, n_groups
    )
    n_targets = n_groups - 1
    
    print(f"  biosparse CSR: {n_genes} genes x {n_cells} cells (genes x cells)")
    print(f"  scanpy dense:  {n_cells} cells x {n_genes} genes (cells x genes)")
    print(f"  NNZ: {scipy_csr.nnz:,} ({scipy_csr.nnz / (n_genes * n_cells) * 100:.1f}%)")
    
    results = {}
    
    # Biosparse MWU benchmark
    if BIOSPARSE_AVAILABLE:
        print(f"\nRunning biosparse MWU ({n_runs} runs)...")
        bio_time, bio_result = benchmark_biosparse_mwu(bio_csr, group_ids, n_targets, n_runs)
        results['biosparse'] = bio_time
        print(f"  biosparse mwu_test: {format_time(bio_time)}")
    
    # Scanpy wilcoxon core benchmark
    print(f"\nRunning scanpy wilcoxon core ({n_runs} runs)...")
    scanpy_time, scanpy_result = benchmark_scanpy_wilcoxon(dense_cells_genes, group_ids, n_groups, n_runs)
    results['scanpy'] = scanpy_time
    print(f"  scanpy wilcoxon:    {format_time(scanpy_time)}")
    
    # Breakdown: rankdata only
    print(f"\nRunning scanpy rankdata only ({n_runs} runs)...")
    rankdata_time = benchmark_scanpy_rankdata_only(dense_cells_genes, n_runs)
    results['scanpy_rankdata'] = rankdata_time
    print(f"  scanpy rankdata:    {format_time(rankdata_time)}")
    
    # Scipy MWU (sampled for reference)
    print(f"\nRunning scipy MWU (sampled 50 genes, for reference)...")
    scipy_time, n_sampled = benchmark_scipy_mwu(dense_cells_genes, group_ids, n_groups, n_sample=50)
    scipy_time_scaled = scipy_time * (n_genes / n_sampled)
    results['scipy_scaled'] = scipy_time_scaled
    print(f"  scipy (50 genes):   {format_time(scipy_time)}")
    print(f"  scipy (scaled):     {format_time(scipy_time_scaled)}")
    
    # Summary
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    if 'biosparse' in results and 'scanpy' in results:
        speedup_vs_scanpy = results['scanpy'] / results['biosparse']
        print(f"\n  biosparse vs scanpy core: {speedup_vs_scanpy:.2f}x {'faster' if speedup_vs_scanpy > 1 else 'slower'}")
    
    if 'biosparse' in results and 'scipy_scaled' in results:
        speedup_vs_scipy = results['scipy_scaled'] / results['biosparse']
        print(f"  biosparse vs scipy:       {speedup_vs_scipy:.1f}x faster (estimated)")
    
    # Additional breakdown
    print(f"\n  Time breakdown:")
    print(f"    biosparse total:    {format_time(results.get('biosparse', 0))}")
    print(f"    scanpy total:       {format_time(results.get('scanpy', 0))}")
    print(f"    scanpy rankdata:    {format_time(results.get('scanpy_rankdata', 0))} ({results.get('scanpy_rankdata', 0)/results.get('scanpy', 1)*100:.0f}% of scanpy)")
    
    return results


def run_scaling_benchmark(quick: bool = False, large: bool = False):
    """Run scaling benchmarks across different sizes."""
    
    warmup_jit()
    
    if quick:
        configs = [
            (10000, 5000, 2, 0.2),
            (20000, 5000, 3, 0.2),
        ]
        n_runs = 3
    elif large:
        configs = [
            (15000, 2000, 3, 0.2),
            (25000, 5000, 5, 0.2),
            (25000, 10000, 5, 0.2),
            (25000, 15000, 5, 0.2),
        ]
        n_runs = 3
    else:
        configs = [
            # (n_genes, n_cells, n_groups, density)
            (50000, 10000, 5, 0.2),
            (50000, 20000, 5, 0.2),
            (100000, 50000, 10, 0.2),
            (200000, 80000, 20, 0.2),
        ]
        n_runs = 3
    
    print("\n" + "="*100)
    print("SCALING BENCHMARK: biosparse MWU vs Scanpy Wilcoxon Core")
    print("="*100)
    
    all_results = []
    
    for n_genes, n_cells, n_groups, density in configs:
        try:
            results = run_benchmark(n_genes, n_cells, n_groups, density, n_runs)
            
            speedup = results.get('scanpy', 0) / results.get('biosparse', 1) if results.get('biosparse') else 0
            
            all_results.append({
                'n_genes': n_genes,
                'n_cells': n_cells,
                'n_groups': n_groups,
                'density': density,
                **results,
                'speedup': speedup
            })
        except Exception as e:
            print(f"\nError with {n_genes}x{n_cells}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall summary table
    print(f"\n{'='*100}")
    print("OVERALL SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"\n{'Genes':<10} {'Cells':<10} {'Groups':<8} {'Density':<10} {'biosparse':<12} {'scanpy':<12} {'Speedup':<10}")
    print("-" * 80)
    
    for r in all_results:
        print(f"{r['n_genes']:<10} {r['n_cells']:<10} {r['n_groups']:<8} {r['density']:<10.0%} "
              f"{format_time(r.get('biosparse', 0)):<12} "
              f"{format_time(r.get('scanpy', 0)):<12} "
              f"{r['speedup']:.2f}x")
    
    speedups = [r['speedup'] for r in all_results if r.get('speedup', 0) > 0]
    if speedups:
        print(f"\n{'='*80}")
        print(f"biosparse MWU vs Scanpy Wilcoxon Core:")
        print(f"  Average speedup: {np.mean(speedups):.2f}x")
        print(f"  Median speedup:  {np.median(speedups):.2f}x")
        print(f"  Range:           {min(speedups):.2f}x - {max(speedups):.2f}x")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="MWU vs Scanpy Wilcoxon Core Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode with smaller matrices")
    parser.add_argument("--large", action="store_true", help="Large-scale benchmark")
    parser.add_argument("--genes", type=int, default=None, help="Number of genes")
    parser.add_argument("--cells", type=int, default=None, help="Number of cells")
    parser.add_argument("--groups", type=int, default=3, help="Number of groups")
    parser.add_argument("--density", type=float, default=0.1, help="Sparsity density")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    args = parser.parse_args()
    
    print("="*80)
    print("MWU vs SCANPY WILCOXON CORE BENCHMARK")
    print("="*80)
    print(f"scipy available:     {SCIPY_AVAILABLE}")
    print(f"biosparse available: {BIOSPARSE_AVAILABLE}")
    
    if not SCIPY_AVAILABLE or not BIOSPARSE_AVAILABLE:
        print("\nERROR: Required dependencies not available")
        return
    
    if args.genes and args.cells:
        # Single benchmark
        warmup_jit()
        run_benchmark(args.genes, args.cells, args.groups, args.density, args.runs)
    else:
        # Scaling benchmark
        run_scaling_benchmark(quick=args.quick, large=args.large)


if __name__ == "__main__":
    main()
