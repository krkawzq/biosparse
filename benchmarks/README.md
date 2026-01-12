# Benchmarks

Performance comparison benchmarks between biosparse and scipy.

## Files

| File | Description |
|------|-------------|
| `benchmark_sparse_vs_scipy.py` | Comprehensive test: sparse operations + statistical kernels |
| `benchmark_kernels.py` | Detailed kernel benchmarks: multi-group scenarios, different scales |

## Usage

### Quick Test

```bash
# Comprehensive test (quick mode)
python benchmarks/benchmark_sparse_vs_scipy.py --quick

# Kernel test (quick mode)
python benchmarks/benchmark_kernels.py --quick
```

### Full Test

```bash
# Comprehensive test (full mode, ~5-10 minutes)
python benchmarks/benchmark_sparse_vs_scipy.py

# Kernel test (full mode)
python benchmarks/benchmark_kernels.py

# Test specific kernel
python benchmarks/benchmark_kernels.py --kernel ttest
python benchmarks/benchmark_kernels.py --kernel mwu
python benchmarks/benchmark_kernels.py --kernel hvg
```

## Test Scenarios

### Sparsity Levels

| Scenario | Density | Description |
|----------|---------|-------------|
| Dense | 15% | Typical dense single-cell data |
| Sparse | 5% | Typical sparse single-cell data |
| Very Sparse | 1% | Highly filtered data |

### Matrix Sizes

| Size | Genes | Cells | Description |
|------|-------|-------|-------------|
| Small | 1,000 | 500 | Small test |
| Medium | 5,000 | 2,000 | Medium dataset |
| Large | 10,000 | 5,000 | Large dataset |

### Group Scenarios

- 2 groups: Simple A vs B comparison
- 5 groups: Multi-celltype comparison
- 10 groups: Complex clustering analysis

## Test Content

### 1. Sparse Structure Operations

- **Row Slicing**: `csr[100:500, :]`
- **Column Slicing**: `csr[:, 50:500]`
- **Random Access**: Random row access
- **VStack**: Vertical stacking of matrices
- **HStack**: Horizontal stacking of matrices

### 2. Statistical Kernels

- **T-Test**: Welch's t-test (multi-group differential expression)
- **MWU**: Mann-Whitney U test (non-parametric test)
- **HVG**: Highly variable gene selection (mean/variance computation)

## Expected Results

Based on design goals, biosparse should significantly outperform scipy in:

1. **JIT-compiled iteration**: 10-100x speedup
2. **Multi-group statistical tests**: 10-50x speedup (avoiding Python loops)
3. **Zero-copy slicing**: 2-5x speedup
4. **Parallelized kernels**: Linear scaling with CPU cores

## Example Output

```
================================================================================
KERNEL PERFORMANCE BENCHMARK
================================================================================
scipy available: True
numba available: True
biosparse available: True

--- T-Test Scaling with Number of Groups ---
Groups   Genes    Cells    Density    scipy        biosparse    Speedup   
--------------------------------------------------------------------------------
2        5000     2000     5.0%       1.23s        45.67ms      26.93x
5        5000     2000     5.0%       3.45s        52.34ms      65.91x
10       5000     2000     5.0%       6.78s        61.23ms      110.73x
```

## Notes

1. **First run**: JIT compilation requires warmup, first run will be slower
2. **Memory**: Large matrix tests require sufficient memory (8GB+ recommended)
3. **scipy MWU**: Since scipy doesn't support vectorized MWU, tests use sampling with extrapolation
4. **Correctness verification**: All tests verify correlation/consistency with scipy results
