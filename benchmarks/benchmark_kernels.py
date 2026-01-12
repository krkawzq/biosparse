#!/usr/bin/env python
"""Benchmark: Statistical Kernel Performance.

Detailed benchmarks for biosparse kernel functions vs scipy equivalents.

Scenarios tested:
1. Multi-group differential expression (2, 5, 10, 20 groups)
2. Different sparsity levels (1%, 5%, 15%, 30%)
3. Different matrix sizes (small, medium, large)

Usage:
    python benchmarks/benchmark_kernels.py
    python benchmarks/benchmark_kernels.py --kernel ttest
    python benchmarks/benchmark_kernels.py --quick
"""

import argparse
import time
import warnings
from typing import Callable, List, Tuple, Dict, Any
import numpy as np

warnings.filterwarnings('ignore')

# Check dependencies
try:
    import scipy.sparse as sp
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    from biosparse._binding import CSRF64
    import biosparse._numba
    from biosparse.kernel.ttest import ttest, welch_ttest, student_ttest
    from biosparse.kernel.mwu import mwu_test
    from biosparse.kernel.hvg import compute_moments, compute_dispersion, select_hvg_by_dispersion
    BIOSPARSE_AVAILABLE = True
except ImportError:
    BIOSPARSE_AVAILABLE = False


# =============================================================================
# Utilities
# =============================================================================

def timeit(func: Callable, n_runs: int = 5, warmup: int = 1) -> Tuple[float, Any]:
    """Time a function and return (avg_time, last_result)."""
    result = None
    for _ in range(warmup):
        result = func()
    
    start = time.perf_counter()
    for _ in range(n_runs):
        result = func()
    elapsed = time.perf_counter() - start
    
    return elapsed / n_runs, result


def create_test_data(
    n_genes: int, 
    n_cells: int, 
    density: float,
    n_groups: int,
    seed: int = 42
) -> Tuple[sp.csr_matrix, "CSRF64", np.ndarray]:
    """Create test data with balanced groups."""
    np.random.seed(seed)
    
    # Create sparse matrix
    scipy_mat = sp.random(n_genes, n_cells, density=density, format='csr', dtype=np.float64)
    
    # Set realistic values (log-normalized counts)
    scipy_mat.data[:] = np.abs(np.random.randn(len(scipy_mat.data))) * 2
    
    # Create balanced group assignments
    group_ids = np.zeros(n_cells, dtype=np.int64)
    cells_per_group = n_cells // n_groups
    for g in range(n_groups):
        start = g * cells_per_group
        end = start + cells_per_group if g < n_groups - 1 else n_cells
        group_ids[start:end] = g
    np.random.shuffle(group_ids)
    
    # Create biosparse matrix
    bio_mat = CSRF64.from_scipy(scipy_mat) if BIOSPARSE_AVAILABLE else None
    
    return scipy_mat, bio_mat, group_ids


def format_time(seconds: float) -> str:
    """Format time with appropriate unit."""
    if seconds < 0.001:
        return f"{seconds*1e6:.1f}μs"
    elif seconds < 1:
        return f"{seconds*1e3:.2f}ms"
    else:
        return f"{seconds:.2f}s"


# =============================================================================
# T-Test Benchmarks
# =============================================================================

class TTestBenchmark:
    """T-test performance benchmarks."""
    
    @staticmethod
    def scipy_ttest(dense_mat: np.ndarray, group_ids: np.ndarray, n_groups: int) -> Tuple[np.ndarray, np.ndarray]:
        """Scipy Welch's t-test (vectorized where possible)."""
        n_genes = dense_mat.shape[0]
        n_targets = n_groups - 1
        
        t_stats = np.zeros((n_genes, n_targets))
        p_values = np.zeros((n_genes, n_targets))
        
        ref_mask = group_ids == 0
        ref_data = dense_mat[:, ref_mask]
        
        for t in range(n_targets):
            tar_mask = group_ids == (t + 1)
            tar_data = dense_mat[:, tar_mask]
            
            # Vectorized t-test (scipy doesn't support axis parameter for ttest_ind)
            # So we compute manually
            n1 = ref_data.shape[1]
            n2 = tar_data.shape[1]
            
            mean1 = ref_data.mean(axis=1)
            mean2 = tar_data.mean(axis=1)
            var1 = ref_data.var(axis=1, ddof=1)
            var2 = tar_data.var(axis=1, ddof=1)
            
            # Welch's t-statistic
            se = np.sqrt(var1/n1 + var2/n2)
            se = np.where(se < 1e-15, 1e-15, se)
            t_stat = (mean2 - mean1) / se
            
            # Welch-Satterthwaite degrees of freedom
            v1_n1 = var1 / n1
            v2_n2 = var2 / n2
            df = (v1_n1 + v2_n2)**2 / (v1_n1**2/(n1-1) + v2_n2**2/(n2-1))
            df = np.where(np.isnan(df) | (df < 1), 1, df)
            
            # P-values (two-sided)
            p_val = 2 * stats.t.sf(np.abs(t_stat), df)
            
            t_stats[:, t] = t_stat
            p_values[:, t] = p_val
        
        return t_stats, p_values
    
    @staticmethod
    def run(
        n_genes: int,
        n_cells: int,
        n_groups: int,
        density: float,
        n_runs: int = 5
    ) -> Dict[str, Any]:
        """Run t-test benchmark."""
        scipy_mat, bio_mat, group_ids = create_test_data(n_genes, n_cells, density, n_groups)
        dense_mat = scipy_mat.toarray()
        n_targets = n_groups - 1
        
        # Scipy benchmark
        scipy_time, scipy_result = timeit(
            lambda: TTestBenchmark.scipy_ttest(dense_mat, group_ids, n_groups),
            n_runs
        )
        
        # Biosparse benchmark
        if BIOSPARSE_AVAILABLE:
            bio_time, bio_result = timeit(
                lambda: ttest(bio_mat, group_ids, n_targets, use_welch=True),
                n_runs
            )
            
            # Verify correlation
            valid = ~(np.isnan(bio_result[0]) | np.isnan(scipy_result[0]))
            if valid.sum() > 10:
                corr = np.corrcoef(bio_result[0][valid].flatten(), scipy_result[0][valid].flatten())[0, 1]
            else:
                corr = np.nan
        else:
            bio_time = float('inf')
            corr = np.nan
        
        speedup = scipy_time / bio_time if bio_time > 0 else 0
        
        return {
            "kernel": "ttest",
            "n_genes": n_genes,
            "n_cells": n_cells,
            "n_groups": n_groups,
            "density": density,
            "scipy_time": scipy_time,
            "biosparse_time": bio_time,
            "speedup": speedup,
            "correlation": corr
        }


# =============================================================================
# MWU Benchmarks
# =============================================================================

class MWUBenchmark:
    """Mann-Whitney U test performance benchmarks."""
    
    @staticmethod
    def scipy_mwu_sample(dense_mat: np.ndarray, group_ids: np.ndarray, n_groups: int, n_sample: int = 100) -> float:
        """Sample-based scipy MWU (full is too slow)."""
        n_genes = min(n_sample, dense_mat.shape[0])
        n_targets = n_groups - 1
        
        ref_mask = group_ids == 0
        ref_data = dense_mat[:n_genes, ref_mask]
        
        u_stats = []
        for t in range(n_targets):
            tar_mask = group_ids == (t + 1)
            tar_data = dense_mat[:n_genes, tar_mask]
            
            for g in range(n_genes):
                u, _ = stats.mannwhitneyu(ref_data[g], tar_data[g], alternative='two-sided')
                u_stats.append(u)
        
        return np.array(u_stats)
    
    @staticmethod
    def run(
        n_genes: int,
        n_cells: int,
        n_groups: int,
        density: float,
        n_runs: int = 3
    ) -> Dict[str, Any]:
        """Run MWU benchmark."""
        scipy_mat, bio_mat, group_ids = create_test_data(n_genes, n_cells, density, n_groups)
        dense_mat = scipy_mat.toarray()
        n_targets = n_groups - 1
        
        # Scipy benchmark (sampled)
        n_sample = min(100, n_genes)
        scipy_time, _ = timeit(
            lambda: MWUBenchmark.scipy_mwu_sample(dense_mat, group_ids, n_groups, n_sample),
            n_runs
        )
        # Scale to full size
        scipy_time_scaled = scipy_time * (n_genes / n_sample)
        
        # Biosparse benchmark
        if BIOSPARSE_AVAILABLE:
            bio_time, bio_result = timeit(
                lambda: mwu_test(bio_mat, group_ids, n_targets),
                n_runs
            )
        else:
            bio_time = float('inf')
        
        speedup = scipy_time_scaled / bio_time if bio_time > 0 else 0
        
        return {
            "kernel": "mwu",
            "n_genes": n_genes,
            "n_cells": n_cells,
            "n_groups": n_groups,
            "density": density,
            "scipy_time": scipy_time_scaled,
            "biosparse_time": bio_time,
            "speedup": speedup,
            "note": f"scipy sampled {n_sample} genes, time scaled"
        }


# =============================================================================
# HVG Benchmarks
# =============================================================================

class HVGBenchmark:
    """HVG (Highly Variable Genes) performance benchmarks."""
    
    @staticmethod
    def scipy_hvg(dense_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scipy/numpy HVG computation."""
        means = dense_mat.mean(axis=1)
        vars_ = dense_mat.var(axis=1, ddof=1)
        dispersions = np.where(means > 1e-12, vars_ / means, 0)
        return means, vars_, dispersions
    
    @staticmethod
    def run(
        n_genes: int,
        n_cells: int,
        density: float,
        n_runs: int = 5
    ) -> Dict[str, Any]:
        """Run HVG benchmark."""
        scipy_mat, bio_mat, _ = create_test_data(n_genes, n_cells, density, n_groups=2)
        dense_mat = scipy_mat.toarray()
        
        # Scipy benchmark
        scipy_time, scipy_result = timeit(
            lambda: HVGBenchmark.scipy_hvg(dense_mat),
            n_runs
        )
        
        # Biosparse benchmark
        if BIOSPARSE_AVAILABLE:
            bio_time, bio_result = timeit(
                lambda: compute_moments(bio_mat, ddof=1),
                n_runs
            )
            
            # Verify
            means_close = np.allclose(bio_result[0], scipy_result[0], rtol=1e-10)
            vars_close = np.allclose(bio_result[1], scipy_result[1], rtol=1e-10)
            correct = means_close and vars_close
        else:
            bio_time = float('inf')
            correct = False
        
        speedup = scipy_time / bio_time if bio_time > 0 else 0
        
        return {
            "kernel": "hvg_moments",
            "n_genes": n_genes,
            "n_cells": n_cells,
            "density": density,
            "scipy_time": scipy_time,
            "biosparse_time": bio_time,
            "speedup": speedup,
            "correct": correct
        }


# =============================================================================
# Comprehensive Benchmark Suite
# =============================================================================

def run_scaling_benchmark(kernel: str, quick: bool = False):
    """Run scaling benchmarks for a specific kernel."""
    
    print(f"\n{'='*80}")
    print(f"SCALING BENCHMARK: {kernel.upper()}")
    print(f"{'='*80}")
    
    # Each config is (n_genes, n_cells, n_groups, density)
    if quick:
        configs = [
            (10000, 5000, 2, 0.2),
            (10000, 5000, 3, 0.2),
            (20000, 5000, 2, 0.2),
            (20000, 5000, 3, 0.2),
        ]
    else:
        configs = [
            (15000, 2000, 3, 0.2),
            (25000, 5000, 5, 0.2),
            (25000, 10000, 5, 0.2),
            (25000, 15000, 5, 0.2),
            (50000, 10000, 5, 0.2),
            (50000, 20000, 5, 0.2),
            (100000, 50000, 10, 0.2),
            (200000, 80000, 20, 0.2),
        ]
    
    results = []
    
    if kernel == "ttest":
        # Test scaling with groups
        print("\n--- Scaling with Number of Groups ---")
        print(f"{'Groups':<8} {'Genes':<8} {'Cells':<8} {'Density':<10} {'scipy':<12} {'biosparse':<12} {'Speedup':<10}")
        print("-" * 80)
        
        for n_genes, n_cells, n_groups, density in configs:
            result = TTestBenchmark.run(n_genes, n_cells, n_groups, density)
            results.append(result)
            
            print(f"{n_groups:<8} {n_genes:<8} {n_cells:<8} {density:<10.1%} "
                  f"{format_time(result['scipy_time']):<12} "
                  f"{format_time(result['biosparse_time']):<12} "
                  f"{result['speedup']:<10.2f}x")
    
    elif kernel == "mwu":
        print("\n--- MWU Scaling ---")
        print(f"{'Groups':<8} {'Genes':<8} {'Cells':<8} {'Density':<10} {'scipy*':<12} {'biosparse':<12} {'Speedup':<10}")
        print("-" * 80)
        
        for n_genes, n_cells, n_groups, density in configs:
            result = MWUBenchmark.run(n_genes, n_cells, n_groups, density)
            results.append(result)
            
            print(f"{n_groups:<8} {n_genes:<8} {n_cells:<8} {density:<10.1%} "
                  f"{format_time(result['scipy_time']):<12} "
                  f"{format_time(result['biosparse_time']):<12} "
                  f"{result['speedup']:<10.2f}x")
        
        print("\n* scipy time is extrapolated from 100-gene sample")
    
    elif kernel == "hvg":
        print("\n--- HVG Moments Scaling ---")
        print(f"{'Genes':<10} {'Cells':<10} {'Density':<10} {'scipy':<12} {'biosparse':<12} {'Speedup':<10} {'Correct':<8}")
        print("-" * 80)
        
        for n_genes, n_cells, n_groups, density in configs:
            # HVG doesn't use n_groups, but we keep the unified config format
            result = HVGBenchmark.run(n_genes, n_cells, density)
            results.append(result)
            
            print(f"{n_genes:<10} {n_cells:<10} {density:<10.1%} "
                  f"{format_time(result['scipy_time']):<12} "
                  f"{format_time(result['biosparse_time']):<12} "
                  f"{result['speedup']:<10.2f}x "
                  f"{'✓' if result.get('correct', False) else '✗':<8}")
    
    return results


def run_all_benchmarks(quick: bool = False):
    """Run all kernel benchmarks."""
    all_results = []
    
    for kernel in ["ttest", "mwu", "hvg"]:
        results = run_scaling_benchmark(kernel, quick)
        all_results.extend(results)
    
    # Summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    for kernel in ["ttest", "mwu", "hvg"]:
        kernel_results = [r for r in all_results if r.get("kernel", "").startswith(kernel)]
        if kernel_results:
            speedups = [r["speedup"] for r in kernel_results if r["speedup"] > 0 and r["speedup"] != float('inf')]
            if speedups:
                print(f"\n{kernel.upper()}:")
                print(f"  Average speedup: {np.mean(speedups):.1f}x")
                print(f"  Median speedup:  {np.median(speedups):.1f}x")
                print(f"  Range:           {min(speedups):.1f}x - {max(speedups):.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Kernel Performance Benchmarks")
    parser.add_argument("--kernel", choices=["ttest", "mwu", "hvg", "all"], default="all",
                        help="Which kernel to benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode with smaller matrices")
    args = parser.parse_args()
    
    print("="*80)
    print("KERNEL PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"scipy available: {SCIPY_AVAILABLE}")
    print(f"numba available: {NUMBA_AVAILABLE}")
    print(f"biosparse available: {BIOSPARSE_AVAILABLE}")
    
    if not SCIPY_AVAILABLE:
        print("ERROR: scipy is required")
        return
    
    if not BIOSPARSE_AVAILABLE:
        print("ERROR: biosparse is required")
        return
    
    if args.kernel == "all":
        run_all_benchmarks(args.quick)
    else:
        run_scaling_benchmark(args.kernel, args.quick)


if __name__ == "__main__":
    main()
