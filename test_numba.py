"""Complete Numba integration test suite for SCL sparse matrices.

This test suite verifies that all sparse matrix operations work correctly
in Numba JIT-compiled functions.
"""

import sys
import numpy as np
import scipy.sparse as sp
from numba import njit

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, 'src')
from python._binding._sparse import CSRF64, CSRF32, CSCF64, CSCF32, is_numba_available


def test_basic_properties():
    """Test basic property access in JIT."""
    print("=" * 60)
    print("Test 1: Basic Properties")
    print("=" * 60)
    
    mat = sp.csr_matrix([[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 5.0]])
    csr = CSRF64.from_scipy(mat)
    print(f"Python: shape={csr.shape}, nnz={csr.nnz}, density={csr.density:.3f}")
    
    @njit
    def test_props(csr):
        return (csr.nrows, csr.ncols, csr.nnz, csr.shape, 
                csr.density, csr.sparsity, csr.is_empty, csr.is_zero)
    
    result = test_props(csr)
    print(f"JIT: nrows={result[0]}, ncols={result[1]}, nnz={result[2]}, shape={result[3]}")
    print(f"     density={result[4]:.3f}, sparsity={result[5]:.3f}")
    print(f"     is_empty={result[6]}, is_zero={result[7]}")
    
    assert result[0] == 3
    assert result[1] == 3
    assert result[2] == 5
    assert result[3] == (3, 3)
    print("✓ Basic properties test passed")


def test_row_access():
    """Test row access methods in JIT."""
    print("\n" + "=" * 60)
    print("Test 2: Row Access")
    print("=" * 60)
    
    mat = sp.csr_matrix([[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 5.0]])
    csr = CSRF64.from_scipy(mat)
    
    @njit
    def get_row_data(csr, row_idx):
        values, indices = csr.row_to_numpy(row_idx)
        return values.copy(), indices.copy(), csr.row_len(row_idx)
    
    for i in range(3):
        vals, idxs, length = get_row_data(csr, i)
        py_vals, py_idxs = csr.row_to_numpy(i)
        print(f"Row {i}: len={length}, values={vals}, indices={idxs}")
        assert len(vals) == length
        assert np.allclose(vals, py_vals)
        assert np.array_equal(idxs, py_idxs)
    
    print("✓ Row access test passed")


def test_iterator():
    """Test iterator in JIT."""
    print("\n" + "=" * 60)
    print("Test 3: Iterator")
    print("=" * 60)
    
    mat = sp.csr_matrix([[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 5.0]])
    csr = CSRF64.from_scipy(mat)
    
    @njit
    def sum_all_values(csr):
        total = 0.0
        count = 0
        for values, indices in csr:
            for v in values:
                total += v
                count += 1
        return total, count
    
    total, count = sum_all_values(csr)
    print(f"Sum via iterator: {total}, count: {count}")
    assert abs(total - 15.0) < 1e-10
    assert count == 5
    print("✓ Iterator test passed")


def test_len():
    """Test len() builtin."""
    print("\n" + "=" * 60)
    print("Test 4: len() Builtin")
    print("=" * 60)
    
    mat = sp.random(100, 50, density=0.1, format='csr', dtype='float64')
    csr = CSRF64.from_scipy(mat)
    
    @njit
    def use_len(csr):
        n = len(csr)
        total = 0.0
        for i in range(n):
            vals, _ = csr.row_to_numpy(i)
            total += vals.sum()
        return total, n
    
    total, n = use_len(csr)
    print(f"len(csr) = {n}, sum = {total:.6f}")
    assert n == 100
    print("✓ len() test passed")


def test_slicing():
    """Test slice operations in JIT."""
    print("\n" + "=" * 60)
    print("Test 5: Slicing Operations")
    print("=" * 60)
    
    mat = sp.random(100, 80, density=0.1, format='csr', dtype='float64')
    csr = CSRF64.from_scipy(mat)
    
    @njit
    def test_slices(csr):
        # Row slice
        sub1 = csr.slice_rows(20, 40)
        
        # Column slice
        sub2 = csr.slice_cols(10, 30)
        
        # Get shapes
        return sub1.shape, sub2.shape
    
    shape1, shape2 = test_slices(csr)
    print(f"Row slice [20:40]: shape={shape1}")
    print(f"Col slice [10:30]: shape={shape2}")
    
    assert shape1 == (20, 80)
    assert shape2 == (100, 20)
    print("✓ Slicing test passed")


def test_getitem_syntax():
    """Test __getitem__ slice syntax."""
    print("\n" + "=" * 60)
    print("Test 6: __getitem__ Syntax")
    print("=" * 60)
    
    mat = sp.random(100, 80, density=0.1, format='csr', dtype='float64')
    csr = CSRF64.from_scipy(mat)
    
    @njit
    def test_slice_syntax(csr):
        # Single slice
        sub1 = csr[10:30]
        
        # Tuple slice
        sub2 = csr[10:30, 5:15]
        
        return sub1.shape, sub2.shape
    
    shape1, shape2 = test_slice_syntax(csr)
    print(f"csr[10:30]: shape={shape1}")
    print(f"csr[10:30, 5:15]: shape={shape2}")
    
    assert shape1 == (20, 80)
    assert shape2 == (20, 10)
    print("✓ __getitem__ syntax test passed")


def test_stacking():
    """Test hstack and vstack in JIT."""
    print("\n" + "=" * 60)
    print("Test 7: Stack Operations")
    print("=" * 60)
    
    mat1 = sp.random(50, 100, density=0.1, format='csr', dtype='float64')
    mat2 = sp.random(30, 100, density=0.1, format='csr', dtype='float64')
    mat3 = sp.random(50, 60, density=0.1, format='csr', dtype='float64')
    
    csr1 = CSRF64.from_scipy(mat1)
    csr2 = CSRF64.from_scipy(mat2)
    csr3 = CSRF64.from_scipy(mat3)
    
    @njit
    def test_vstack(csr1, csr2):
        # Note: We'll need to pass arrays, not use class method directly
        # For now, skip class method testing in JIT
        return csr1.shape, csr2.shape
    
    # Test in Python mode for now
    vstacked = CSRF64.vstack([csr1, csr2])
    print(f"vstack: {csr1.shape} + {csr2.shape} = {vstacked.shape}")
    assert vstacked.shape == (80, 100)
    
    hstacked = CSRF64.hstack([csr1, csr3])
    print(f"hstack: {csr1.shape} + {csr3.shape} = {hstacked.shape}")
    assert hstacked.shape == (50, 160)
    
    print("✓ Stack operations test passed (Python mode)")


def test_conversions():
    """Test conversion operations in JIT."""
    print("\n" + "=" * 60)
    print("Test 8: Conversions")
    print("=" * 60)
    
    mat = sp.random(50, 40, density=0.1, format='csr', dtype='float64')
    csr = CSRF64.from_scipy(mat)
    
    @njit
    def test_to_dense(csr):
        dense = csr.to_dense()
        return dense.shape, dense.sum()
    
    @njit
    def test_to_coo(csr):
        rows, cols, data = csr.to_coo()
        return len(rows), len(cols), len(data), data.sum()
    
    @njit
    def test_to_csc(csr):
        csc = csr.to_csc()
        return csc.shape, csc.nnz
    
    @njit
    def test_clone(csr):
        csr2 = csr.clone()
        return csr2.shape, csr2.nnz
    
    # Test to_dense
    shape, total = test_to_dense(csr)
    print(f"to_dense: shape={shape}, sum={total:.6f}")
    assert shape == (50, 40)
    
    # Test to_coo
    nr, nc, nd, data_sum = test_to_coo(csr)
    print(f"to_coo: rows={nr}, cols={nc}, data={nd}, sum={data_sum:.6f}")
    assert nr == nc == nd == csr.nnz
    
    # Test to_csc
    csc_shape, csc_nnz = test_to_csc(csr)
    print(f"to_csc: shape={csc_shape}, nnz={csc_nnz}")
    assert csc_shape == (50, 40)
    
    # Test clone
    clone_shape, clone_nnz = test_clone(csr)
    print(f"clone: shape={clone_shape}, nnz={clone_nnz}")
    assert clone_shape == (50, 40)
    
    print("✓ Conversions test passed")


def test_validation():
    """Test validation methods in JIT."""
    print("\n" + "=" * 60)
    print("Test 9: Validation Methods")
    print("=" * 60)
    
    mat = sp.random(50, 40, density=0.1, format='csr', dtype='float64')
    mat.sort_indices()
    csr = CSRF64.from_scipy(mat)
    
    @njit
    def test_validation(csr):
        is_valid = csr.is_valid
        is_sorted = csr.is_sorted
        indices_ok = csr.indices_in_bounds
        validated = csr.validate()
        return is_valid, is_sorted, indices_ok, validated
    
    is_valid, is_sorted, indices_ok, validated = test_validation(csr)
    print(f"is_valid={is_valid}, is_sorted={is_sorted}")
    print(f"indices_in_bounds={indices_ok}, validate={validated}")
    
    assert is_valid
    assert is_sorted
    assert indices_ok
    assert validated
    print("✓ Validation test passed")


def test_performance():
    """Performance comparison: Python vs JIT."""
    print("\n" + "=" * 60)
    print("Test 10: Performance Comparison")
    print("=" * 60)
    
    mat = sp.random(1000, 1000, density=0.01, format='csr', dtype='float64')
    csr = CSRF64.from_scipy(mat)
    print(f"Matrix: shape={csr.shape}, nnz={csr.nnz}")
    
    # Python version
    def python_sum(csr):
        total = 0.0
        for i in range(csr.nrows):
            vals, _ = csr.row_to_numpy(i)
            total += vals.sum()
        return total
    
    # JIT version
    @njit
    def jit_sum_explicit(csr):
        total = 0.0
        for i in range(csr.nrows):
            vals, _ = csr.row_to_numpy(i)
            total += vals.sum()
        return total
    
    @njit
    def jit_sum_iterator(csr):
        total = 0.0
        for vals, _ in csr:
            total += vals.sum()
        return total
    
    # Warm up JIT
    _ = jit_sum_explicit(csr)
    _ = jit_sum_iterator(csr)
    
    # Time Python
    import time
    
    start = time.perf_counter()
    py_result = python_sum(csr)
    py_time = time.perf_counter() - start
    
    # Time JIT (explicit loop)
    start = time.perf_counter()
    jit_result1 = jit_sum_explicit(csr)
    jit_time1 = time.perf_counter() - start
    
    # Time JIT (iterator)
    start = time.perf_counter()
    jit_result2 = jit_sum_iterator(csr)
    jit_time2 = time.perf_counter() - start
    
    print(f"Python sum: {py_result:.6f} in {py_time*1000:.2f} ms")
    print(f"JIT sum (explicit): {jit_result1:.6f} in {jit_time1*1000:.2f} ms")
    print(f"JIT sum (iterator): {jit_result2:.6f} in {jit_time2*1000:.2f} ms")
    print(f"Speedup: {py_time/jit_time1:.1f}x (explicit), {py_time/jit_time2:.1f}x (iterator)")
    
    assert abs(py_result - jit_result1) < 1e-6
    assert abs(py_result - jit_result2) < 1e-6
    print("✓ Performance test passed")


def test_float32():
    """Test float32 type support."""
    print("\n" + "=" * 60)
    print("Test 11: Float32 Support")
    print("=" * 60)
    
    mat = sp.random(100, 80, density=0.1, format='csr', dtype='float32')
    csr = CSRF32.from_scipy(mat)
    
    @njit
    def test_f32(csr):
        total = 0.0
        for vals, _ in csr:
            total += vals.sum()
        return total, csr.shape, csr.nnz
    
    total, shape, nnz = test_f32(csr)
    print(f"CSRF32: shape={shape}, nnz={nnz}, sum={total:.6f}")
    
    assert shape == (100, 80)
    print("✓ Float32 test passed")


def test_csc():
    """Test CSC type support."""
    print("\n" + "=" * 60)
    print("Test 12: CSC Support")
    print("=" * 60)
    
    mat = sp.random(50, 60, density=0.1, format='csc', dtype='float64')
    csc = CSCF64.from_scipy(mat)
    
    @njit
    def test_csc_ops(csc):
        total = 0.0
        for vals, _ in csc:
            total += vals.sum()
        return total, csc.shape, len(csc)
    
    total, shape, ncols = test_csc_ops(csc)
    print(f"CSCF64: shape={shape}, ncols={ncols}, sum={total:.6f}")
    
    assert shape == (50, 60)
    assert ncols == 60
    print("✓ CSC test passed")


def main():
    """Run all Numba integration tests."""
    print("\n" + "#" * 60)
    print("# SCL-Core Numba Integration Tests")
    print("#" * 60)
    print(f"\nNumba available: {is_numba_available()}")
    
    if not is_numba_available():
        print("ERROR: Numba extensions not loaded!")
        return 1
    
    try:
        test_basic_properties()
        test_row_access()
        test_iterator()
        test_len()
        test_slicing()
        test_getitem_syntax()
        test_stacking()
        test_conversions()
        test_validation()
        test_performance()
        test_float32()
        test_csc()
        
        print("\n" + "=" * 60)
        print("🎉 All Numba tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
