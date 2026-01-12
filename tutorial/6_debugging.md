# Chapter 6: Debugging and Verification

## Inspecting Loop Hints

### Using `inspect_hints()`

```python
from biosparse.optim import optimized_jit, vectorize, interleave, inspect_hints
import numpy as np

@optimized_jit
def my_optimized_func(arr):
    n = len(arr)
    total = 0.0
    
    vectorize(8)
    interleave(4)
    for i in range(n):
        total += arr[i]
    
    return total

# First, compile the function by calling it
arr = np.array([1.0, 2.0, 3.0, 4.0])
result = my_optimized_func(arr)

# Now inspect the hints
inspect_hints(my_optimized_func)
```

Output:
```
Signature: (float64[::1],)
----------------------------------------
  VECTORIZE(8) at line 7
  INTERLEAVE(4) at line 8
```

### Using `get_modified_ir()`

```python
from biosparse.optim import optimized_jit, vectorize, get_modified_ir

@optimized_jit
def simple_sum(arr):
    n = len(arr)
    total = 0.0
    
    vectorize(8)
    for i in range(n):
        total += arr[i]
    
    return total

# Compile
simple_sum(np.array([1.0, 2.0]))

# Get the modified IR with loop metadata
ir = get_modified_ir(simple_sum)
print(ir)
```

Look for lines like:
```llvm
br i1 %cond, label %loop.body, label %loop.end, !llvm.loop !10000

; BioSparse Loop Optimization Metadata
!10000 = distinct !{!10000, !10001, !10002}
!10001 = !{"llvm.loop.vectorize.enable", i1 true}
!10002 = !{"llvm.loop.vectorize.width", i32 8}
```

## Checking for Python Calls in Hot Paths

A common performance problem is **Python API calls** leaking into JIT-compiled code. This happens when:
- Using unsupported Python features
- Forgetting to import Numba types
- Boxing/unboxing happening in inner loops

### Detecting Python Calls in IR

```python
def check_for_python_calls(func):
    """Check if a JIT function has Python API calls in hot path."""
    
    # Get LLVM IR
    if hasattr(func, '_dispatcher'):
        dispatcher = func._dispatcher
    else:
        dispatcher = func
    
    if not dispatcher.signatures:
        print("Function not compiled yet. Call it first.")
        return
    
    sig = dispatcher.signatures[0]
    ir = dispatcher.inspect_llvm(sig)
    
    # Look for Python API calls
    python_calls = [
        'PyObject',
        'PyErr',
        'Py_DECREF',
        'Py_INCREF',
        'PySequence',
        'PyList',
        'PyDict',
        'nrt_',  # Numba Runtime (memory management)
    ]
    
    issues = []
    for i, line in enumerate(ir.split('\n')):
        for call in python_calls:
            if call in line and 'declare' not in line:
                issues.append((i, line.strip()))
    
    if issues:
        print(f"⚠️  Found {len(issues)} potential Python calls:")
        for line_no, line in issues[:10]:  # Show first 10
            print(f"  Line {line_no}: {line[:80]}...")
    else:
        print("✓ No Python API calls detected in hot path")
    
    return issues
```

### Example: Finding Performance Issues

```python
from biosparse.optim import parallel_jit, assume
import numpy as np

@parallel_jit
def suspicious_func(csr):
    """This function might have performance issues."""
    n_rows = csr.nrows
    result = np.empty(n_rows, dtype=np.float64)
    
    for row in range(n_rows):
        values, _ = csr.row_to_numpy(row)
        result[row] = np.sum(values)  # np.sum might not inline!
    
    return result

# Compile and check
suspicious_func(test_csr)
issues = check_for_python_calls(suspicious_func)
```

### Fixed Version

```python
@parallel_jit
def fixed_func(csr):
    """Fixed: Manual sum instead of np.sum."""
    n_rows = csr.nrows
    assume(n_rows > 0)
    
    result = np.empty(n_rows, dtype=np.float64)
    
    for row in prange(n_rows):
        values, _ = csr.row_to_numpy(row)
        
        total = 0.0
        for v in values:
            total += v
        
        result[row] = total
    
    return result
```

## Verifying Correctness

### Pattern: Reference Implementation

Always maintain a simple, correct reference implementation:

```python
def ttest_reference(dense_matrix, group_ids):
    """Reference implementation using scipy (known correct)."""
    import scipy.stats as stats
    
    n_rows = dense_matrix.shape[0]
    t_stats = np.empty(n_rows)
    p_values = np.empty(n_rows)
    
    for i in range(n_rows):
        row = dense_matrix[i, :]
        ref = row[group_ids == 0]
        tar = row[group_ids == 1]
        t, p = stats.ttest_ind(ref, tar, equal_var=False)
        t_stats[i] = t if not np.isnan(t) else 0.0
        p_values[i] = p if not np.isnan(p) else 1.0
    
    return t_stats, p_values

def test_ttest_correctness():
    """Verify optimized implementation matches reference."""
    # Create test data
    np.random.seed(42)
    dense = np.random.randn(100, 200)
    csr = CSRF64.from_scipy(sp.csr_matrix(dense))
    group_ids = np.array([0] * 100 + [1] * 100, dtype=np.int32)
    
    # Reference
    t_ref, p_ref = ttest_reference(dense, group_ids)
    
    # Optimized
    t_opt, p_opt = ttest_optimized(csr, group_ids, 2)
    
    # Compare
    np.testing.assert_allclose(t_opt, t_ref, rtol=1e-10)
    np.testing.assert_allclose(p_opt, p_ref, rtol=1e-6)
    print("✓ Optimized implementation matches reference")
```

### Pattern: Property-Based Testing

```python
import pytest
from hypothesis import given, strategies as st

@given(
    n_rows=st.integers(1, 100),
    n_cols=st.integers(1, 100),
    density=st.floats(0.01, 0.5),
)
def test_row_sums_property(n_rows, n_cols, density):
    """Property: row sums should match dense computation."""
    # Generate random sparse matrix
    dense = np.random.randn(n_rows, n_cols)
    mask = np.random.random((n_rows, n_cols)) < density
    dense[~mask] = 0
    
    csr = CSRF64.from_scipy(sp.csr_matrix(dense))
    
    # Optimized
    result = row_sums_optimized(csr)
    
    # Reference
    expected = dense.sum(axis=1)
    
    np.testing.assert_allclose(result, expected, rtol=1e-10)
```

### Pattern: Numerical Stability Tests

```python
def test_numerical_stability():
    """Test with extreme values."""
    
    # Very small values
    small = np.array([[1e-300, 1e-300], [1e-300, 0]])
    csr_small = CSRF64.from_scipy(sp.csr_matrix(small))
    result = my_kernel(csr_small)
    assert np.all(np.isfinite(result))
    
    # Very large values
    large = np.array([[1e300, 1e300], [1e300, 0]])
    csr_large = CSRF64.from_scipy(sp.csr_matrix(large))
    result = my_kernel(csr_large)
    assert np.all(np.isfinite(result))
    
    # Mixed magnitudes
    mixed = np.array([[1e-100, 1e100], [1e100, 1e-100]])
    csr_mixed = CSRF64.from_scipy(sp.csr_matrix(mixed))
    result = my_kernel(csr_mixed)
    assert np.all(np.isfinite(result))
```

## Performance Benchmarking

### Basic Timing

```python
import time

def benchmark(func, *args, warmup=3, runs=10):
    """Benchmark a function with warmup."""
    # Warmup (JIT compilation, cache warming)
    for _ in range(warmup):
        func(*args)
    
    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'median': np.median(times),
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'times': times,
    }
```

### Comparative Benchmarking

```python
def compare_implementations(implementations, *args, **kwargs):
    """Compare multiple implementations."""
    print(f"{'Implementation':<30} {'Median (ms)':<15} {'Speedup':<10}")
    print("-" * 55)
    
    results = {}
    baseline_time = None
    
    for name, func in implementations.items():
        stats = benchmark(func, *args, **kwargs)
        results[name] = stats
        
        if baseline_time is None:
            baseline_time = stats['median']
            speedup = 1.0
        else:
            speedup = baseline_time / stats['median']
        
        print(f"{name:<30} {stats['median']*1000:<15.2f} {speedup:<10.1f}x")
    
    return results

# Usage
implementations = {
    'naive_python': naive_implementation,
    'basic_numba': numba_v1,
    'parallel_numba': numba_v2,
    'fully_optimized': numba_v5,
}

results = compare_implementations(
    implementations,
    test_csr, group_ids, n_groups,
    warmup=3, runs=10
)
```

### Scaling Analysis

```python
def scaling_analysis(func, sizes, create_data):
    """Analyze how performance scales with input size."""
    print(f"{'Size':<15} {'Time (ms)':<15} {'Throughput (M/s)':<15}")
    print("-" * 45)
    
    for size in sizes:
        data = create_data(size)
        stats = benchmark(func, *data, warmup=2, runs=5)
        
        time_ms = stats['median'] * 1000
        throughput = size / stats['median'] / 1e6
        
        print(f"{size:<15} {time_ms:<15.2f} {throughput:<15.2f}")

# Usage
sizes = [1000, 5000, 10000, 50000, 100000]

def create_test_data(n):
    dense = np.random.randn(n, 1000)
    csr = CSRF64.from_scipy(sp.csr_matrix(dense))
    group_ids = np.array([0] * 500 + [1] * 500, dtype=np.int32)
    return (csr, group_ids, 2)

scaling_analysis(optimized_kernel, sizes, create_test_data)
```

## Debugging Tips

### Tip 1: Isolate the Problem

```python
# If full function fails, test components:

@njit
def test_row_access(csr):
    """Test: Can we access rows?"""
    values, indices = csr.row_to_numpy(0)
    return len(values)

@njit
def test_iteration(csr):
    """Test: Does iteration work?"""
    count = 0
    for values, _ in csr:
        count += 1
    return count

@njit
def test_prange(csr):
    """Test: Does prange work?"""
    n = csr.nrows
    result = np.empty(n)
    for i in prange(n):
        result[i] = float(i)
    return result
```

### Tip 2: Print Intermediate Values

```python
# During development, add print debugging:

@njit
def debug_kernel(csr):
    n_rows = csr.nrows
    print("n_rows =", n_rows)  # Numba supports basic print
    
    for row in range(min(3, n_rows)):  # Just first 3 rows
        values, indices = csr.row_to_numpy(row)
        print("row", row, "nnz =", len(values))
        if len(values) > 0:
            print("  first value =", values[0])
    
    return 0
```

### Tip 3: Check Types

```python
from numba import typeof

def debug_types(csr, group_ids):
    """Print Numba types for debugging."""
    print(f"csr type: {typeof(csr)}")
    print(f"group_ids type: {typeof(group_ids)}")
    print(f"group_ids dtype: {group_ids.dtype}")
```

### Tip 4: Use Numba's Verbose Mode

```python
import numba
import logging

# Enable detailed Numba logging
numba.config.DEBUG = True
numba.config.DEBUG_TYPEINFER = True

# Or set environment variable:
# NUMBA_DEBUG=1 python script.py
```

## Summary: Debugging Workflow

1. **Verify correctness** against reference implementation
2. **Check for Python calls** in hot paths
3. **Inspect loop hints** to ensure they're applied
4. **Benchmark** to measure actual improvement
5. **Test edge cases** to ensure robustness
6. **Profile** to find remaining bottlenecks

---

**Next Chapter**: [Chapter 7: biosparse Sparse Matrix Features](./7_sparse_features.md)
