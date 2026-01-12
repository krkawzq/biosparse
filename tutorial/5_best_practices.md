# Chapter 5: Best Practices and Pitfalls

## Critical Pitfalls

### Pitfall 1: Race Conditions with prange

**The most common bug** when using `prange` is writing to shared memory from multiple threads.

```python
# ❌ DEADLY BUG: Race condition
@parallel_jit
def count_by_group_WRONG(group_ids, n_groups):
    counts = np.zeros(n_groups, dtype=np.int64)
    
    for i in prange(len(group_ids)):  # Multiple threads
        g = group_ids[i]
        counts[g] += 1  # ALL threads write to SAME location!
    
    return counts
```

**What happens**:
1. Thread A reads `counts[0] = 5`
2. Thread B reads `counts[0] = 5`
3. Thread A writes `counts[0] = 6`
4. Thread B writes `counts[0] = 6`
5. Result: `counts[0] = 6`, but should be `7`

**Solutions**:

```python
# ✅ Solution 1: Use sequential loop
@fast_jit
def count_by_group_v1(group_ids, n_groups):
    counts = np.zeros(n_groups, dtype=np.int64)
    
    for i in range(len(group_ids)):  # Sequential, not prange
        g = group_ids[i]
        if g >= 0 and g < n_groups:
            counts[g] += 1
    
    return counts

# ✅ Solution 2: Per-thread accumulation (advanced)
@parallel_jit
def count_by_group_v2(group_ids, n_groups, n_threads=8):
    # Each thread gets its own row
    local_counts = np.zeros((n_threads, n_groups), dtype=np.int64)
    n = len(group_ids)
    
    for t in prange(n_threads):
        start = t * n // n_threads
        end = (t + 1) * n // n_threads
        
        for i in range(start, end):
            g = group_ids[i]
            if g >= 0 and g < n_groups:
                local_counts[t, g] += 1  # Each thread writes own row
    
    # Reduce (sequential)
    counts = np.zeros(n_groups, dtype=np.int64)
    for t in range(n_threads):
        for g in range(n_groups):
            counts[g] += local_counts[t, g]
    
    return counts
```

**Rule of thumb**: If multiple iterations of a `prange` loop might write to the same memory location, you have a race condition.

### Pitfall 2: Cache Invalidation with `cache=True`

When using `cache=True`, Numba caches compiled functions to disk. This can cause **stale code** issues.

```python
# First version
@parallel_jit(cache=True)
def my_func(x):
    return x + 1

# Later, you modify it:
@parallel_jit(cache=True)
def my_func(x):
    return x + 2  # Changed!

# Problem: Numba might still use the cached version!
```

**Solutions**:

```python
# Solution 1: Clear cache manually
import shutil
import numba

cache_dir = numba.config.CACHE_DIR
shutil.rmtree(cache_dir, ignore_errors=True)

# Solution 2: Disable cache during development
@parallel_jit(cache=False)  # Use False during development
def my_func(x):
    return x + 2
```

**biosparse fix**: The `@optimized_jit` decorator automatically skips IR processing for cached code.

### Pitfall 3: Iterator + prange Incompatibility

**Iterators cannot be parallelized**:

```python
# ❌ WRONG: Iterator prevents parallelization
@parallel_jit
def wrong(csr):
    result = 0.0
    for values, indices in csr:  # This is sequential!
        for v in values:
            result += v
    return result

# ✅ CORRECT: Use indexed access
@parallel_jit
def correct(csr):
    result = np.zeros(csr.nrows, dtype=np.float64)
    
    for row in prange(csr.nrows):  # This parallelizes!
        values, _ = csr.row_to_numpy(row)
        row_sum = 0.0
        for v in values:
            row_sum += v
        result[row] = row_sum
    
    return np.sum(result)
```

### Pitfall 4: Boolean Types in Numba Structs

Numba's `prange` has issues with `boolean` fields in custom structs (LLVM type mismatch).

```python
# This can cause: TypeError: Invalid store of i1 to i8

# Solution: Use uint8 instead of boolean in data models
class MyModel(StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', types.voidptr),
            # ('flag', types.boolean),  # ❌ Can cause issues
            ('flag', types.uint8),       # ✅ Use uint8 instead
        ]
        super().__init__(dmm, fe_type, members)
```

### Pitfall 5: Forgetting assume Preconditions

If your function has `assume()` statements, callers **must** satisfy those conditions:

```python
@parallel_jit
def process_matrix(csr):
    assume(csr.nrows > 0)  # PRECONDITION
    assume(csr.ncols > 0)  # PRECONDITION
    # ...

# ❌ WRONG: Calling with empty matrix
empty_csr = create_empty_csr()
process_matrix(empty_csr)  # UNDEFINED BEHAVIOR!

# ✅ CORRECT: Validate before calling
if csr.nrows > 0 and csr.ncols > 0:
    process_matrix(csr)
else:
    handle_empty_case()
```

## Best Practices

### Practice 1: Layer Your Optimizations

Apply optimizations in order of impact:

```python
# Level 1: Algorithm (most impact)
# - Choose sparse vs dense
# - Choose O(n) vs O(n²) algorithm

# Level 2: Parallelization
@parallel_jit  # Multi-threaded

# Level 3: Assumptions
assume(n > 0)
assume(n <= MAX_SIZE)

# Level 4: Branch hints
if likely(common_case):
    ...

# Level 5: Loop hints (least impact, fine-tuning)
vectorize(8)
interleave(4)
```

### Practice 2: Document Preconditions

```python
@parallel_jit
def welch_ttest(csr, group_ids, n_groups):
    """
    Compute Welch's t-test for each row.
    
    Preconditions (enforced via assume):
        - csr.nrows > 0: Matrix has at least one row
        - csr.ncols > 0: Matrix has at least one column
        - len(group_ids) == csr.ncols: One group ID per column
        - n_groups >= 2: At least reference + one target
        - all(0 <= g < n_groups for g in group_ids): Valid group IDs
    
    Args:
        csr: Sparse matrix (genes × cells)
        group_ids: Group assignment per cell
        n_groups: Total number of groups
    
    Returns:
        t_stats: T-statistics per row
        p_values: P-values per row
    """
    # Enforce preconditions
    assume(csr.nrows > 0)
    assume(csr.ncols > 0)
    assume(len(group_ids) >= csr.ncols)
    assume(n_groups >= 2)
    # ...
```

### Practice 3: Use Appropriate Decorator

| Decorator | Use When |
|-----------|----------|
| `@fast_jit` | Sequential computation, no parallelism needed |
| `@parallel_jit` | Outer loop can be parallelized with `prange` |
| `@fast_jit(cache=True)` | Stable code, want faster startup |
| `@parallel_jit(cache=True)` | Stable parallel code |

```python
# Sequential helper (called from parallel code)
@fast_jit(cache=True, inline='always')
def compute_variance(values, mean, n):
    """Inline helper - always inlined into caller."""
    sq_sum = 0.0
    for v in values:
        sq_sum += (v - mean) ** 2
    return sq_sum / n

# Main parallel kernel
@parallel_jit(cache=True)
def row_variances(csr):
    """Main entry point - parallel over rows."""
    for row in prange(csr.nrows):
        values, _ = csr.row_to_numpy(row)
        mean = np.mean(values)
        var = compute_variance(values, mean, len(values))
        # ...
```

### Practice 4: Minimize Memory Allocation in Hot Loops

```python
# ❌ BAD: Allocating inside parallel loop
@parallel_jit
def bad_allocation(csr, n_groups):
    for row in prange(csr.nrows):
        temp = np.zeros(n_groups)  # Allocation per iteration!
        # ...

# ✅ GOOD: Pre-allocate outside loop
@parallel_jit
def good_allocation(csr, n_groups):
    n_rows = csr.nrows
    # One allocation for all rows
    temp_all = np.zeros((n_rows, n_groups), dtype=np.float64)
    
    for row in prange(n_rows):
        temp = temp_all[row, :]  # View, no allocation
        # ...
```

### Practice 5: Use `likely`/`unlikely` Consistently

```python
# Pattern: Error/validation at start with unlikely
@parallel_jit
def robust_kernel(arr, threshold):
    n = len(arr)
    
    # Validation (should rarely fail)
    if unlikely(n == 0):
        return np.empty(0, dtype=np.float64)
    
    assume(n > 0)
    result = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        val = arr[i]
        
        # Common case processing
        if likely(val >= threshold):
            result[i] = process_normal(val)
        # Edge case
        elif unlikely(val < 0):
            result[i] = 0.0  # Handle negative
        else:
            result[i] = process_small(val)
    
    return result
```

### Practice 6: Test with Edge Cases

```python
import pytest

def test_kernel_edge_cases():
    """Test that assumptions don't cause crashes with edge cases."""
    
    # Empty input (if supported)
    # result = my_kernel(empty_csr)  # Should not crash
    
    # Single row
    single_row = create_csr(nrows=1, ncols=100)
    result = my_kernel(single_row)  # Should work
    
    # Single column
    single_col = create_csr(nrows=100, ncols=1)
    result = my_kernel(single_col)  # Should work
    
    # All zeros
    all_zeros = create_empty_csr(nrows=100, ncols=100)
    result = my_kernel(all_zeros)  # Should handle gracefully
    
    # Dense (no sparsity)
    dense = create_dense_csr(nrows=10, ncols=10)
    result = my_kernel(dense)  # Should work
```

## Performance Anti-Patterns

### Anti-Pattern 1: Over-Optimization

```python
# ❌ TOO MUCH: Hints everywhere
@parallel_jit
def over_optimized(arr):
    n = len(arr)
    assume(n > 0)
    assume(n < 1000000)
    assume(n % 2 == 0)
    assume(n % 4 == 0)
    assume(n % 8 == 0)  # Really? Must be multiple of 8?
    
    total = 0.0
    vectorize(16)
    interleave(8)
    unroll(4)  # Too many hints - confuses optimizer
    for i in range(n):
        if likely(arr[i] > 0):  # Is this really likely?
            total += arr[i]
    
    return total

# ✅ BALANCED: Minimal effective hints
@parallel_jit
def balanced(arr):
    n = len(arr)
    assume(n > 0)
    
    total = 0.0
    vectorize(8)
    for i in range(n):
        total += arr[i]
    
    return total
```

### Anti-Pattern 2: Wrong Vectorization Width

```python
# ❌ WRONG: float32 data with float64 width
arr_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

@parallel_jit
def wrong_width(arr):
    vectorize(4)  # Width 4 is for float64, not float32!
    for i in range(len(arr)):
        ...

# ✅ CORRECT: Match width to data type
@parallel_jit
def correct_width_f32(arr):
    vectorize(8)  # 8 x float32 = 256 bits (AVX)
    for i in range(len(arr)):
        ...

@parallel_jit
def correct_width_f64(arr):
    vectorize(4)  # 4 x float64 = 256 bits (AVX)
    for i in range(len(arr)):
        ...
```

### Anti-Pattern 3: Premature Optimization

```python
# ❌ DON'T: Optimize before profiling
@parallel_jit
def premature():
    # Added all hints without measuring
    assume(...)
    vectorize(...)
    interleave(...)
    # But the actual bottleneck is I/O, not computation!

# ✅ DO: Profile first
import cProfile

def find_bottleneck():
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = my_function(data)
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')
```

## Summary Checklist

Before deploying optimized code:

- [ ] No race conditions with `prange`
- [ ] All `assume()` preconditions documented
- [ ] Edge cases tested
- [ ] Performance measured (not assumed)
- [ ] `cache=True` tested with fresh cache
- [ ] Correctness verified against reference implementation

---

**Next Chapter**: [Chapter 6: Debugging and Verification](./6_debugging.md)
