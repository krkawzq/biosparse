"""
检查 Numba 稀疏矩阵编译的 IR 和汇编代码
验证是否真正优化和内联
"""

import sys
sys.path.insert(0, 'src')

from python._binding._sparse import CSRF64
import scipy.sparse as sp
import numpy as np
from numba import njit

# ============================================================
# 测试用例 1: 基本属性访问
# ============================================================

@njit
def test_basic_properties(csr):
    """测试基本属性访问是否被优化"""
    n = csr.nrows
    m = csr.ncols
    nnz = csr.nnz
    return n * m + nnz


# ============================================================
# 测试用例 2: 行数据访问
# ============================================================

@njit
def test_row_access(csr, row_idx):
    """测试单行访问是否高效"""
    values, indices = csr.row_to_numpy(row_idx)
    total = 0.0
    for i in range(len(values)):
        total += values[i]
    return total


# ============================================================
# 测试用例 3: 迭代器
# ============================================================

@njit
def test_iterator(csr):
    """测试迭代器是否真正零开销"""
    total = 0.0
    for values, indices in csr:
        for i in range(len(values)):
            total += values[i]
    return total


# ============================================================
# 测试用例 4: SpMV（稀疏矩阵向量乘法）
# ============================================================

@njit
def test_spmv(csr, vec):
    """测试 SpMV 的编译效率"""
    result = np.zeros(csr.nrows, dtype=np.float64)
    for row_idx in range(csr.nrows):
        values, indices = csr.row_to_numpy(row_idx)
        dot = 0.0
        for i in range(len(values)):
            dot += values[i] * vec[indices[i]]
        result[row_idx] = dot
    return result


# ============================================================
# 测试用例 5: 复杂循环
# ============================================================

@njit
def test_complex_loop(csr):
    """测试复杂循环中的优化"""
    n = csr.nrows
    row_sums = np.zeros(n, dtype=np.float64)
    row_maxs = np.zeros(n, dtype=np.float64)

    for i in range(n):
        values, indices = csr.row_to_numpy(i)
        if len(values) > 0:
            row_sums[i] = np.sum(values)
            row_maxs[i] = np.max(values)

    return row_sums, row_maxs


# ============================================================
# 主程序：检查 IR 和汇编
# ============================================================

def inspect_function(func, args, name):
    """检查函数的 LLVM IR 和汇编代码"""
    print("\n" + "="*70)
    print(f"Function: {name}")
    print("="*70)

    # 触发编译
    _ = func(*args)

    # 获取签名
    sigs = list(func.signatures)
    if not sigs:
        print("Warning: No signatures found!")
        return

    sig = sigs[0]
    print(f"\nSignature: {sig}")

    # LLVM IR
    print("\n" + "-"*70)
    print("LLVM IR (optimized):")
    print("-"*70)
    ir = func.inspect_llvm(signature=sig)

    # 打印关键部分
    lines = ir.split('\n')

    # 查找主函数
    in_main_func = False
    func_lines = []
    for line in lines:
        if f'define' in line and 'wrapper' not in line.lower():
            in_main_func = True
        if in_main_func:
            func_lines.append(line)
            if line.strip() == '}':
                break

    # 打印前 100 行或整个函数
    for line in func_lines[:100]:
        print(line)

    if len(func_lines) > 100:
        print(f"... (truncated, total {len(func_lines)} lines)")

    # 统计指令
    print("\n" + "-"*70)
    print("IR Statistics:")
    print("-"*70)

    call_count = sum(1 for line in func_lines if 'call' in line)
    load_count = sum(1 for line in func_lines if 'load' in line)
    store_count = sum(1 for line in func_lines if 'store' in line)
    getelementptr_count = sum(1 for line in func_lines if 'getelementptr' in line)

    print(f"  Total lines:      {len(func_lines)}")
    print(f"  Function calls:   {call_count}")
    print(f"  Load instructions: {load_count}")
    print(f"  Store instructions: {store_count}")
    print(f"  GEP instructions:  {getelementptr_count}")

    # 汇编代码
    print("\n" + "-"*70)
    print("Assembly (first 50 lines):")
    print("-"*70)
    asm = func.inspect_asm(signature=sig)
    asm_lines = asm.split('\n')
    for line in asm_lines[:50]:
        print(line)

    if len(asm_lines) > 50:
        print(f"... (truncated, total {len(asm_lines)} lines)")

    print("\n" + "-"*70)
    print("Assembly Statistics:")
    print("-"*70)
    print(f"  Total instructions: {len(asm_lines)}")

    # 查找是否有外部调用
    external_calls = [line for line in asm_lines if 'call' in line.lower()]
    if external_calls:
        print(f"  External calls: {len(external_calls)}")
        for call in external_calls[:5]:
            print(f"    {call.strip()}")
        if len(external_calls) > 5:
            print(f"    ... and {len(external_calls) - 5} more")
    else:
        print("  No external calls found (good!)")


def main():
    print("############################################################")
    print("# Numba Sparse Matrix IR Inspection")
    print("############################################################")

    # 创建测试矩阵
    mat = sp.random(10, 8, density=0.3, format='csr', dtype=np.float64)
    csr = CSRF64.from_scipy(mat)
    vec = np.random.rand(8)

    print(f"\nTest matrix: {mat.shape[0]} x {mat.shape[1]}, nnz={mat.nnz}")

    # 测试 1: 基本属性
    inspect_function(
        test_basic_properties,
        (csr,),
        "test_basic_properties"
    )

    # 测试 2: 行访问
    inspect_function(
        test_row_access,
        (csr, 0),
        "test_row_access"
    )

    # 测试 3: 迭代器
    inspect_function(
        test_iterator,
        (csr,),
        "test_iterator"
    )

    # 测试 4: SpMV
    inspect_function(
        test_spmv,
        (csr, vec),
        "test_spmv"
    )

    # 测试 5: 复杂循环
    inspect_function(
        test_complex_loop,
        (csr,),
        "test_complex_loop"
    )

    print("\n" + "="*70)
    print("Inspection complete!")
    print("="*70)
    print("\nKey things to look for in the IR:")
    print("  1. Minimal function calls (most should be inlined)")
    print("  2. Direct memory access via GEP (getelementptr)")
    print("  3. Simple loops without overhead")
    print("  4. No boxing/unboxing in hot loops")
    print("\nIn the assembly:")
    print("  1. Tight loops with vector operations")
    print("  2. Minimal function calls")
    print("  3. Direct pointer arithmetic")


if __name__ == "__main__":
    main()
