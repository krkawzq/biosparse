"""
展示assume真正起作用的场景

关键：assume必须影响后续的优化决策，否则会被优化掉
"""

import sys
sys.path.insert(0, 'src')

from python._binding._sparse import CSRF64
import scipy.sparse as sp
import numpy as np
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic
import llvmlite.ir as lir
import time


@intrinsic
def assume(typingctx, condition_ty):
    sig = types.void(types.boolean)
    def codegen(context, builder, sig, args):
        [condition] = args
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(1)])
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'llvm.assume')
        builder.call(fn, [condition])
        return context.get_dummy_value()
    return sig, codegen


# ============================================================================
# 有效的assume用法：影响分支消除
# ============================================================================

@njit
def baseline_with_branch(arr, threshold):
    """基础版本：有条件分支"""
    total = 0.0
    for i in range(len(arr)):
        val = arr[i]
        # 这个分支会被编译
        if val > threshold:
            total += val * val
        else:
            total += val
    return total


@njit
def optimized_with_assume(arr, threshold):
    """优化版本：assume消除分支"""
    # 假设：所有值都大于threshold
    total = 0.0
    for i in range(len(arr)):
        val = arr[i]

        # 告诉LLVM：这个条件总是真
        assume(val > threshold)

        # 现在LLVM可以消除这个分支
        if val > threshold:
            total += val * val
        else:
            total += val  # 这个分支永远不会执行

    return total


# ============================================================================
# 有效的assume用法：循环优化
# ============================================================================

@njit
def sum_baseline(arr):
    """基础版本"""
    n = len(arr)
    total = 0.0

    for i in range(n):
        total += arr[i]

    return total


@njit
def sum_with_assume(arr):
    """假设数组长度是8的倍数，启用更好的展开"""
    n = len(arr)

    # 假设：长度是8的倍数
    assume(n % 8 == 0)
    assume(n > 0)

    total = 0.0
    for i in range(n):
        total += arr[i]

    return total


# ============================================================================
# 测试
# ============================================================================

def test_branch_elimination():
    """测试分支消除"""
    print("=" * 70)
    print("测试1: 分支消除优化")
    print("=" * 70)

    # 创建所有值都大于0的数组
    arr = np.random.rand(10000) + 1.0  # 所有值都在[1, 2]之间
    threshold = 0.5

    # 预热
    r1 = baseline_with_branch(arr, threshold)
    r2 = optimized_with_assume(arr, threshold)

    print(f"结果一致: {np.isclose(r1, r2)}")

    # 性能测试
    iterations = 1000

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = baseline_with_branch(arr, threshold)
    t_baseline = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = optimized_with_assume(arr, threshold)
    t_assume = (time.perf_counter() - t0) * 1000

    print(f"\n基础版本 (有分支): {t_baseline:.2f} ms")
    print(f"Assume优化:        {t_assume:.2f} ms")
    print(f"性能提升:          {(1 - t_assume/t_baseline)*100:.1f}%")

    # IR检查
    ir_baseline = baseline_with_branch.inspect_llvm(baseline_with_branch.signatures[0])
    ir_assume = optimized_with_assume.inspect_llvm(optimized_with_assume.signatures[0])

    # 统计分支指令
    import re
    br_baseline = len(re.findall(r'\bbr\b', ir_baseline))
    br_assume = len(re.findall(r'\bbr\b', ir_assume))

    print(f"\nIR分支指令数:")
    print(f"  基础版本: {br_baseline}")
    print(f"  Assume:   {br_assume}")


def test_loop_optimization():
    """测试循环优化"""
    print("\n" + "=" * 70)
    print("测试2: 循环优化（8的倍数）")
    print("=" * 70)

    # 创建长度是8的倍数的数组
    arr = np.random.rand(8000)

    # 预热
    r1 = sum_baseline(arr)
    r2 = sum_with_assume(arr)

    print(f"结果一致: {np.isclose(r1, r2)}")

    # 性能测试
    iterations = 1000

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = sum_baseline(arr)
    t_baseline = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = sum_with_assume(arr)
    t_assume = (time.perf_counter() - t0) * 1000

    print(f"\n基础版本: {t_baseline:.2f} ms")
    print(f"Assume:   {t_assume:.2f} ms")
    print(f"提升:     {(1 - t_assume/t_baseline)*100:.1f}%")


# ============================================================================
# 其他优化选项
# ============================================================================

def test_other_options():
    """测试其他优化选项"""
    print("\n" + "=" * 70)
    print("测试3: 其他优化选项")
    print("=" * 70)

    @njit
    def baseline(arr):
        total = 0.0
        for x in arr:
            total += x * x
        return total

    @njit(fastmath=True)
    def with_fastmath(arr):
        total = 0.0
        for x in arr:
            total += x * x
        return total

    @njit(boundscheck=False)
    def no_boundscheck(arr, indices):
        total = 0.0
        for i in indices:
            total += arr[i]
        return total

    arr = np.random.rand(10000)
    indices = np.arange(10000, dtype=np.int64)

    # 预热
    _ = baseline(arr)
    _ = with_fastmath(arr)
    _ = no_boundscheck(arr, indices)

    iterations = 1000

    # 基础
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = baseline(arr)
    t_baseline = (time.perf_counter() - t0) * 1000

    # fastmath
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = with_fastmath(arr)
    t_fastmath = (time.perf_counter() - t0) * 1000

    # no boundscheck
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = no_boundscheck(arr, indices)
    t_nobounds = (time.perf_counter() - t0) * 1000

    print(f"基础版本:           {t_baseline:.2f} ms")
    print(f"fastmath=True:      {t_fastmath:.2f} ms  (提升 {(1-t_fastmath/t_baseline)*100:.1f}%)")
    print(f"boundscheck=False:  {t_nobounds:.2f} ms  (提升 {(1-t_nobounds/t_baseline)*100:.1f}%)")


if __name__ == "__main__":
    test_branch_elimination()
    test_loop_optimization()
    test_other_options()

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
Assume的有效使用场景:

1. 消除分支
   - assume(condition) + if (condition) -> 分支消除
   - 适用于你知道条件总是真的情况

2. 循环优化
   - assume(n % 8 == 0) -> 完美展开
   - 适用于已知大小或对齐的数据

3. 组合使用
   - assume + fastmath + boundscheck=False
   - 可以获得最大性能提升

注意：
- assume可能被LLVM优化掉如果它不影响任何优化决策
- 最好的方法是直接使用 fastmath/boundscheck 等选项
- assume适合非常特定的优化场景
""")
