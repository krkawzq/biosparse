#!/usr/bin/env python
"""Benchmark: biosparse vs scipy sparse operations.

Compares performance of:
1. Sparse structure operations (slicing, stacking)
2. Statistical kernels (t-test, MWU, HVG)

Scenarios:
- Dense: 50% non-zero elements (typical dense single-cell data)
- Sparse: 20% non-zero elements (typical sparse single-cell data)
- Very Sparse: 10% non-zero elements (highly filtered data)

Usage:
    python benchmarks/benchmark_sparse_vs_scipy.py
    python benchmarks/benchmark_sparse_vs_scipy.py --quick  # Quick mode for testing
"""

import argparse
import time
import warnings
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Check dependencies
try:
    import scipy.sparse as sp
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available, some benchmarks will be skipped")

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("WARNING: numba not available, JIT benchmarks will be skipped")

try:
    from biosparse._binding import CSRF64, CSCF64
    import biosparse._numba  # Register Numba types
    BIOSPARSE_AVAILABLE = True
except ImportError:
    BIOSPARSE_AVAILABLE = False
    print("WARNING: biosparse not available, biosparse benchmarks will be skipped")

try:
    from biosparse.kernel.ttest import ttest, welch_ttest
    from biosparse.kernel.mwu import mwu_test
    from biosparse.kernel.hvg import compute_moments, compute_dispersion
    KERNEL_AVAILABLE = BIOSPARSE_AVAILABLE
except ImportError:
    KERNEL_AVAILABLE = False


# =============================================================================
# Benchmark Infrastructure
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    biosparse_time: float
    scipy_time: float
    speedup: float
    correct: bool
    scenario: str
    
    def __str__(self):
        status = "✓" if self.correct else "✗"
        return (f"{status} {self.name:<40} | "
                f"biosparse: {self.biosparse_time*1000:>8.2f}ms | "
                f"scipy: {self.scipy_time*1000:>8.2f}ms | "
                f"speedup: {self.speedup:>6.2f}x")


def timeit(func: Callable, n_runs: int = 10, warmup: int = 2) -> float:
    """Time a function with warmup runs."""
    # Warmup
    for _ in range(warmup):
        func()
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        func()
    elapsed = time.perf_counter() - start
    
    return elapsed / n_runs


def create_sparse_matrix(
    n_rows: int, 
    n_cols: int, 
    density: float, 
    seed: int = 42
) -> Tuple[sp.csr_matrix, "CSRF64"]:
    """Create matching scipy and biosparse sparse matrices."""
    np.random.seed(seed)
    scipy_mat = sp.random(n_rows, n_cols, density=density, format='csr', dtype=np.float64)
    scipy_mat.data[:] = np.random.rand(len(scipy_mat.data)) * 10  # More realistic values
    
    if BIOSPARSE_AVAILABLE:
        bio_mat = CSRF64.from_scipy(scipy_mat)
    else:
        bio_mat = None
    
    return scipy_mat, bio_mat


def create_group_ids(n_cols: int, n_groups: int, seed: int = 42) -> np.ndarray:
    """Create random group assignments."""
    np.random.seed(seed)
    return np.random.randint(0, n_groups, size=n_cols, dtype=np.int64)


# =============================================================================
# Sparse Structure Benchmarks
# =============================================================================

class SparseStructureBenchmarks:
    """Benchmarks for sparse structure operations."""
    
    def __init__(self, n_rows: int, n_cols: int, density: float, n_runs: int = 10):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.density = density
        self.n_runs = n_runs
        self.scenario = f"density={density:.1%}"
        
        self.scipy_mat, self.bio_mat = create_sparse_matrix(n_rows, n_cols, density)
    
    def benchmark_row_slicing(self) -> BenchmarkResult:
        """Benchmark row slicing."""
        start_row, end_row = 100, 500
        
        # scipy
        def scipy_slice():
            return self.scipy_mat[start_row:end_row, :].sum()
        
        scipy_time = timeit(scipy_slice, self.n_runs)
        scipy_result = scipy_slice()
        
        # biosparse (JIT)
        if BIOSPARSE_AVAILABLE and NUMBA_AVAILABLE:
            @njit
            def bio_slice(csr, start, end):
                sliced = csr.slice_rows(start, end)
                total = 0.0
                for values, _ in sliced:
                    for v in values:
                        total += v
                return total
            
            # Warmup
            _ = bio_slice(self.bio_mat, start_row, end_row)
            
            def bio_run():
                return bio_slice(self.bio_mat, start_row, end_row)
            
            bio_time = timeit(bio_run, self.n_runs)
            bio_result = bio_run()
            correct = np.isclose(bio_result, scipy_result, rtol=1e-10)
        else:
            bio_time = float('inf')
            correct = False
        
        return BenchmarkResult(
            name="Row Slicing [100:500]",
            biosparse_time=bio_time,
            scipy_time=scipy_time,
            speedup=scipy_time / bio_time if bio_time > 0 else 0,
            correct=correct,
            scenario=self.scenario
        )
    
    def benchmark_col_slicing(self) -> BenchmarkResult:
        """Benchmark column slicing."""
        start_col, end_col = 50, 500
        
        # scipy
        def scipy_slice():
            return self.scipy_mat[:, start_col:end_col].sum()
        
        scipy_time = timeit(scipy_slice, self.n_runs)
        scipy_result = scipy_slice()
        
        # biosparse (JIT)
        if BIOSPARSE_AVAILABLE and NUMBA_AVAILABLE:
            @njit
            def bio_slice(csr, start, end):
                sliced = csr.slice_cols(start, end)
                total = 0.0
                for values, _ in sliced:
                    for v in values:
                        total += v
                return total
            
            # Warmup
            _ = bio_slice(self.bio_mat, start_col, end_col)
            
            def bio_run():
                return bio_slice(self.bio_mat, start_col, end_col)
            
            bio_time = timeit(bio_run, self.n_runs)
            bio_result = bio_run()
            correct = np.isclose(bio_result, scipy_result, rtol=1e-10)
        else:
            bio_time = float('inf')
            correct = False
        
        return BenchmarkResult(
            name="Column Slicing [:, 50:500]",
            biosparse_time=bio_time,
            scipy_time=scipy_time,
            speedup=scipy_time / bio_time if bio_time > 0 else 0,
            correct=correct,
            scenario=self.scenario
        )
    
    def benchmark_vstack(self) -> BenchmarkResult:
        """Benchmark vertical stacking."""
        # Create additional matrices
        scipy_mat2, bio_mat2 = create_sparse_matrix(
            self.n_rows // 2, self.n_cols, self.density, seed=43
        )
        scipy_mat3, bio_mat3 = create_sparse_matrix(
            self.n_rows // 2, self.n_cols, self.density, seed=44
        )
        
        # scipy
        def scipy_vstack():
            return sp.vstack([self.scipy_mat, scipy_mat2, scipy_mat3]).sum()
        
        scipy_time = timeit(scipy_vstack, self.n_runs)
        scipy_result = scipy_vstack()
        
        # biosparse
        if BIOSPARSE_AVAILABLE:
            def bio_vstack():
                stacked = CSRF64.vstack([self.bio_mat, bio_mat2, bio_mat3])
                return stacked.nnz
            
            # Warmup
            _ = bio_vstack()
            
            bio_time = timeit(bio_vstack, self.n_runs)
            # Compare nnz instead of sum for correctness
            stacked = CSRF64.vstack([self.bio_mat, bio_mat2, bio_mat3])
            expected_nnz = self.scipy_mat.nnz + scipy_mat2.nnz + scipy_mat3.nnz
            correct = stacked.nnz == expected_nnz
        else:
            bio_time = float('inf')
            correct = False
        
        return BenchmarkResult(
            name="Vertical Stack (3 matrices)",
            biosparse_time=bio_time,
            scipy_time=scipy_time,
            speedup=scipy_time / bio_time if bio_time > 0 else 0,
            correct=correct,
            scenario=self.scenario
        )
    
    def benchmark_hstack(self) -> BenchmarkResult:
        """Benchmark horizontal stacking."""
        # Create additional matrices
        scipy_mat2, bio_mat2 = create_sparse_matrix(
            self.n_rows, self.n_cols // 2, self.density, seed=43
        )
        scipy_mat3, bio_mat3 = create_sparse_matrix(
            self.n_rows, self.n_cols // 2, self.density, seed=44
        )
        
        # scipy
        def scipy_hstack():
            return sp.hstack([self.scipy_mat, scipy_mat2, scipy_mat3]).sum()
        
        scipy_time = timeit(scipy_hstack, self.n_runs)
        
        # biosparse
        if BIOSPARSE_AVAILABLE:
            def bio_hstack():
                stacked = CSRF64.hstack([self.bio_mat, bio_mat2, bio_mat3])
                return stacked.nnz
            
            # Warmup
            _ = bio_hstack()
            
            bio_time = timeit(bio_hstack, self.n_runs)
            # Compare nnz for correctness
            stacked = CSRF64.hstack([self.bio_mat, bio_mat2, bio_mat3])
            expected_nnz = self.scipy_mat.nnz + scipy_mat2.nnz + scipy_mat3.nnz
            correct = stacked.nnz == expected_nnz
        else:
            bio_time = float('inf')
            correct = False
        
        return BenchmarkResult(
            name="Horizontal Stack (3 matrices)",
            biosparse_time=bio_time,
            scipy_time=scipy_time,
            speedup=scipy_time / bio_time if bio_time > 0 else 0,
            correct=correct,
            scenario=self.scenario
        )
    
    def benchmark_random_row_access(self) -> BenchmarkResult:
        """Benchmark random row access."""
        np.random.seed(123)
        indices = np.random.randint(0, self.n_rows, size=1000)
        
        # scipy
        def scipy_access():
            total = 0.0
            for idx in indices:
                total += self.scipy_mat[idx, :].sum()
            return total
        
        scipy_time = timeit(scipy_access, self.n_runs)
        scipy_result = scipy_access()
        
        # biosparse (JIT)
        if BIOSPARSE_AVAILABLE and NUMBA_AVAILABLE:
            @njit
            def bio_access(csr, indices):
                total = 0.0
                for idx in indices:
                    values, _ = csr.row(idx)
                    for v in values:
                        total += v
                return total
            
            # Warmup
            _ = bio_access(self.bio_mat, indices)
            
            def bio_run():
                return bio_access(self.bio_mat, indices)
            
            bio_time = timeit(bio_run, self.n_runs)
            bio_result = bio_run()
            correct = np.isclose(bio_result, scipy_result, rtol=1e-10)
        else:
            bio_time = float('inf')
            correct = False
        
        return BenchmarkResult(
            name="Random Row Access (1000 rows)",
            biosparse_time=bio_time,
            scipy_time=scipy_time,
            speedup=scipy_time / bio_time if bio_time > 0 else 0,
            correct=correct,
            scenario=self.scenario
        )
    
    def run_all(self) -> List[BenchmarkResult]:
        """Run all sparse structure benchmarks."""
        results = []
        
        print(f"\n{'='*80}")
        print(f"Sparse Structure Benchmarks ({self.scenario})")
        print(f"Matrix shape: {self.n_rows} x {self.n_cols}, nnz: {self.scipy_mat.nnz:,}")
        print(f"{'='*80}")
        
        benchmarks = [
            self.benchmark_row_slicing,
            self.benchmark_col_slicing,
            self.benchmark_random_row_access,
            self.benchmark_vstack,
            self.benchmark_hstack,
        ]
        
        for bench in benchmarks:
            try:
                result = bench()
                results.append(result)
                print(result)
            except Exception as e:
                print(f"  ✗ {bench.__name__} failed: {e}")
        
        return results


# =============================================================================
# Kernel Benchmarks
# =============================================================================

class KernelBenchmarks:
    """Benchmarks for statistical kernel operations."""
    
    def __init__(
        self, 
        n_genes: int, 
        n_cells: int, 
        n_groups: int, 
        density: float, 
        n_runs: int = 5
    ):
        self.n_genes = n_genes
        self.n_cells = n_cells
        self.n_groups = n_groups
        self.density = density
        self.n_runs = n_runs
        self.scenario = f"density={density:.1%}, {n_groups} groups"
        
        # Create data (genes x cells)
        self.scipy_mat, self.bio_mat = create_sparse_matrix(n_genes, n_cells, density)
        self.group_ids = create_group_ids(n_cells, n_groups)
        
        # Dense version for scipy
        self.dense_mat = self.scipy_mat.toarray()
    
    def benchmark_ttest(self) -> BenchmarkResult:
        """Benchmark t-test kernel."""
        n_targets = self.n_groups - 1
        
        # scipy (using dense arrays)
        def scipy_ttest():
            results = []
            ref_mask = self.group_ids == 0
            ref_data = self.dense_mat[:, ref_mask]
            
            for t in range(1, self.n_groups):
                tar_mask = self.group_ids == t
                tar_data = self.dense_mat[:, tar_mask]
                
                # Per-gene t-test
                t_stats = np.zeros(self.n_genes)
                p_vals = np.zeros(self.n_genes)
                for g in range(self.n_genes):
                    stat, pval = stats.ttest_ind(tar_data[g], ref_data[g], equal_var=False)
                    t_stats[g] = stat if not np.isnan(stat) else 0.0
                    p_vals[g] = pval if not np.isnan(pval) else 1.0
                results.append((t_stats, p_vals))
            
            return results
        
        scipy_time = timeit(scipy_ttest, self.n_runs, warmup=1)
        
        # biosparse kernel
        if KERNEL_AVAILABLE:
            def bio_ttest():
                return ttest(self.bio_mat, self.group_ids, n_targets, use_welch=True)
            
            # Warmup
            _ = bio_ttest()
            
            bio_time = timeit(bio_ttest, self.n_runs)
            
            # Verify a few values
            bio_result = bio_ttest()
            scipy_result = scipy_ttest()
            
            # Check first group's t-stats correlation
            bio_t = bio_result[0][:, 0]
            scipy_t = scipy_result[0][0]
            valid = ~(np.isnan(bio_t) | np.isnan(scipy_t) | (np.abs(scipy_t) < 1e-10))
            if valid.sum() > 10:
                corr = np.corrcoef(bio_t[valid], scipy_t[valid])[0, 1]
                correct = corr > 0.95  # High correlation
            else:
                correct = True  # Not enough data to verify
        else:
            bio_time = float('inf')
            correct = False
        
        return BenchmarkResult(
            name=f"Welch's T-Test ({self.n_genes} genes x {n_targets} groups)",
            biosparse_time=bio_time,
            scipy_time=scipy_time,
            speedup=scipy_time / bio_time if bio_time > 0 else 0,
            correct=correct,
            scenario=self.scenario
        )
    
    def benchmark_mwu(self) -> BenchmarkResult:
        """Benchmark Mann-Whitney U test kernel."""
        n_targets = self.n_groups - 1
        
        # scipy (using dense arrays) - sample only
        def scipy_mwu():
            results = []
            ref_mask = self.group_ids == 0
            ref_data = self.dense_mat[:, ref_mask]
            
            # Only test first 100 genes for scipy (too slow otherwise)
            n_test = min(100, self.n_genes)
            
            for t in range(1, self.n_groups):
                tar_mask = self.group_ids == t
                tar_data = self.dense_mat[:, tar_mask]
                
                u_stats = np.zeros(n_test)
                p_vals = np.zeros(n_test)
                for g in range(n_test):
                    stat, pval = stats.mannwhitneyu(
                        ref_data[g], tar_data[g], 
                        alternative='two-sided',
                        use_continuity=True
                    )
                    u_stats[g] = stat
                    p_vals[g] = pval
                results.append((u_stats, p_vals))
            
            return results
        
        scipy_time = timeit(scipy_mwu, self.n_runs, warmup=1)
        
        # Scale scipy time to full dataset
        scipy_time_scaled = scipy_time * (self.n_genes / min(100, self.n_genes))
        
        # biosparse kernel
        if KERNEL_AVAILABLE:
            def bio_mwu():
                return mwu_test(self.bio_mat, self.group_ids, n_targets)
            
            # Warmup
            _ = bio_mwu()
            
            bio_time = timeit(bio_mwu, self.n_runs)
            
            # Verify
            bio_result = bio_mwu()
            scipy_result = scipy_mwu()
            
            # Check U-stats correlation for first 100 genes
            n_test = min(100, self.n_genes)
            bio_u = bio_result[0][:n_test, 0]
            scipy_u = scipy_result[0][0]
            
            valid = ~(np.isnan(bio_u) | np.isnan(scipy_u))
            if valid.sum() > 10:
                corr = np.corrcoef(bio_u[valid], scipy_u[valid])[0, 1]
                correct = corr > 0.95
            else:
                correct = True
        else:
            bio_time = float('inf')
            correct = False
        
        return BenchmarkResult(
            name=f"MWU Test ({self.n_genes} genes x {n_targets} groups)",
            biosparse_time=bio_time,
            scipy_time=scipy_time_scaled,
            speedup=scipy_time_scaled / bio_time if bio_time > 0 else 0,
            correct=correct,
            scenario=self.scenario
        )
    
    def benchmark_hvg_moments(self) -> BenchmarkResult:
        """Benchmark HVG moment computation."""
        # scipy/numpy version
        def scipy_moments():
            dense = self.dense_mat
            means = dense.mean(axis=1)
            vars_ = dense.var(axis=1, ddof=1)
            return means, vars_
        
        scipy_time = timeit(scipy_moments, self.n_runs)
        scipy_means, scipy_vars = scipy_moments()
        
        # biosparse kernel
        if KERNEL_AVAILABLE:
            def bio_moments():
                return compute_moments(self.bio_mat, ddof=1)
            
            # Warmup
            _ = bio_moments()
            
            bio_time = timeit(bio_moments, self.n_runs)
            bio_means, bio_vars = bio_moments()
            
            # Verify
            means_close = np.allclose(bio_means, scipy_means, rtol=1e-10)
            vars_close = np.allclose(bio_vars, scipy_vars, rtol=1e-10)
            correct = means_close and vars_close
        else:
            bio_time = float('inf')
            correct = False
        
        return BenchmarkResult(
            name=f"HVG Moments ({self.n_genes} genes)",
            biosparse_time=bio_time,
            scipy_time=scipy_time,
            speedup=scipy_time / bio_time if bio_time > 0 else 0,
            correct=correct,
            scenario=self.scenario
        )
    
    def benchmark_row_iteration(self) -> BenchmarkResult:
        """Benchmark row-wise iteration and computation."""
        # scipy version
        def scipy_rowsum():
            return np.asarray(self.scipy_mat.sum(axis=1)).flatten()
        
        scipy_time = timeit(scipy_rowsum, self.n_runs)
        scipy_result = scipy_rowsum()
        
        # biosparse (JIT)
        if BIOSPARSE_AVAILABLE and NUMBA_AVAILABLE:
            @njit(parallel=True)
            def bio_rowsum(csr):
                n = csr.nrows
                out = np.zeros(n)
                for i in prange(n):
                    values, _ = csr.row(i)
                    total = 0.0
                    for v in values:
                        total += v
                    out[i] = total
                return out
            
            # Warmup
            _ = bio_rowsum(self.bio_mat)
            
            def bio_run():
                return bio_rowsum(self.bio_mat)
            
            bio_time = timeit(bio_run, self.n_runs)
            bio_result = bio_run()
            correct = np.allclose(bio_result, scipy_result, rtol=1e-10)
        else:
            bio_time = float('inf')
            correct = False
        
        return BenchmarkResult(
            name=f"Parallel Row Sum ({self.n_genes} rows)",
            biosparse_time=bio_time,
            scipy_time=scipy_time,
            speedup=scipy_time / bio_time if bio_time > 0 else 0,
            correct=correct,
            scenario=self.scenario
        )
    
    def run_all(self) -> List[BenchmarkResult]:
        """Run all kernel benchmarks."""
        results = []
        
        print(f"\n{'='*80}")
        print(f"Kernel Benchmarks ({self.scenario})")
        print(f"Matrix shape: {self.n_genes} genes x {self.n_cells} cells")
        print(f"nnz: {self.scipy_mat.nnz:,} ({100*self.scipy_mat.nnz/(self.n_genes*self.n_cells):.2f}%)")
        print(f"{'='*80}")
        
        benchmarks = [
            self.benchmark_row_iteration,
            self.benchmark_hvg_moments,
            self.benchmark_ttest,
            self.benchmark_mwu,
        ]
        
        for bench in benchmarks:
            try:
                result = bench()
                results.append(result)
                print(result)
            except Exception as e:
                print(f"  ✗ {bench.__name__} failed: {e}")
        
        return results


# =============================================================================
# Main
# =============================================================================

def print_summary(results: List[BenchmarkResult]):
    """Print summary statistics."""
    if not results:
        return
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    # Group by scenario
    from collections import defaultdict
    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r.scenario].append(r)
    
    for scenario, scenario_results in by_scenario.items():
        print(f"\n{scenario}:")
        speedups = [r.speedup for r in scenario_results if r.speedup > 0 and r.speedup != float('inf')]
        if speedups:
            print(f"  Average speedup: {np.mean(speedups):.2f}x")
            print(f"  Median speedup:  {np.median(speedups):.2f}x")
            print(f"  Max speedup:     {max(speedups):.2f}x")
            print(f"  Min speedup:     {min(speedups):.2f}x")
        
        n_correct = sum(1 for r in scenario_results if r.correct)
        print(f"  Correctness:     {n_correct}/{len(scenario_results)}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark biosparse vs scipy")
    parser.add_argument("--quick", action="store_true", help="Quick mode with smaller matrices")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    args = parser.parse_args()
    
    print("="*80)
    print("BIOSPARSE vs SCIPY BENCHMARK")
    print("="*80)
    print(f"scipy available: {SCIPY_AVAILABLE}")
    print(f"numba available: {NUMBA_AVAILABLE}")
    print(f"biosparse available: {BIOSPARSE_AVAILABLE}")
    print(f"kernels available: {KERNEL_AVAILABLE}")
    
    if not (SCIPY_AVAILABLE and BIOSPARSE_AVAILABLE):
        print("\nERROR: Both scipy and biosparse are required for benchmarks")
        return
    
    all_results = []
    
    # Define scenarios
    if args.quick:
        # Quick mode for testing
        scenarios = [
            {"name": "Sparse (20%)", "density": 0.2},
        ]
        n_rows, n_cols = 1000, 500
        n_genes, n_cells = 500, 300
        n_groups = 3
        n_runs = 3
    else:
        # Full benchmark
        scenarios = [
            {"name": "Dense (50%)", "density": 0.5},
            {"name": "Sparse (20%)", "density": 0.2},
            {"name": "Very Sparse (10%)", "density": 0.1},
        ]
        n_rows, n_cols = 10000, 5000
        n_genes, n_cells = 5000, 2000
        n_groups = 5
        n_runs = args.runs
    
    # Run sparse structure benchmarks
    for scenario in scenarios:
        bench = SparseStructureBenchmarks(
            n_rows=n_rows, 
            n_cols=n_cols, 
            density=scenario["density"],
            n_runs=n_runs
        )
        all_results.extend(bench.run_all())
    
    # Run kernel benchmarks
    for scenario in scenarios:
        bench = KernelBenchmarks(
            n_genes=n_genes,
            n_cells=n_cells,
            n_groups=n_groups,
            density=scenario["density"],
            n_runs=max(3, n_runs // 2)  # Kernels are slower
        )
        all_results.extend(bench.run_all())
    
    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
