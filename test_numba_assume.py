"""
测试Numba中的assume功能用于激进优化

LLVM的assume可以告诉编译器某些条件总是为真，从而启用更激进的优化。
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


# ============================================================================
# 方法1: 直接使用LLVM的assume intrinsic
# ============================================================================

@intrinsic
def assume(typingctx, condition_ty):
    """调用LLVM的llvm.assume intrinsic

    告诉LLVM编译器某个条件总是为真，可以基于此进行优化。

    Example:
        assume(x > 0)  # 告诉编译器x总是大于0
        assume(len(array) == 100)  # 告诉编译器数组长度总是100
    """
    sig = types.void(types.boolean)

    def codegen(context, builder, sig, args):
        [condition] = args

        # 获取或插入llvm.assume函数
        # void @llvm.assume(i1 %cond)
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(1)])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "llvm.assume")

        # 调用llvm.assume
        builder.call(fn, [condition])

        return context.get_dummy_value()

    return sig, codegen


@intrinsic
def assume_aligned(typingctx, ptr_ty, alignment_ty):
    """告诉LLVM指针是对齐的

    这可以启用向量化优化。

    Example:
        assume_aligned(data.ctypes.data, 32)  # 告诉LLVM data按32字节对齐
    """
    sig = types.voidptr(types.voidptr, types.intp)

    def codegen(context, builder, sig, args):
        [ptr, alignment] = args

        # 使用llvm.assume + ptrtoint/icmp来实现对齐假设
        # 更简单的方法：直接设置load/store的对齐属性
        # 这里我们返回带有对齐假设的指针

        # ptr_int = ptrtoint ptr to i64
        ptr_as_int = builder.ptrtoint(ptr, lir.IntType(64))

        # mask = alignment - 1
        mask = builder.sub(alignment, lir.Constant(lir.IntType(64), 1))

        # ptr_int & mask == 0
        masked = builder.and_(ptr_as_int, mask)
        is_aligned = builder.icmp_unsigned('==', masked, lir.Constant(lir.IntType(64), 0))

        # llvm.assume(is_aligned)
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(1)])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "llvm.assume")
        builder.call(fn, [is_aligned])

        return ptr

    return sig, codegen


# ============================================================================
# 方法2: 使用Numba的literally + 静态分支消除
# ============================================================================

from numba import literally

@njit
def optimized_with_literal_size(arr):
    """使用literally强制编译时常量

    这可以启用循环展开等优化
    """
    # literally强制size在编译时已知
    size = literally(100)

    total = 0.0
    for i in range(size):  # LLVM知道这是100次迭代
        if i < len(arr):
            total += arr[i]

    return total


# ============================================================================
# 示例1: 使用assume优化稀疏矩阵访问
# ============================================================================

@njit
def spmv_with_assume(csr, vec):
    """带有assume优化的SpMV

    我们假设：
    1. 矩阵已排序
    2. 索引在范围内
    3. 无NaN/Inf值
    """
    n = csr.nrows
    result = np.zeros(n, dtype=np.float64)

    for row in range(n):
        values, indices = csr.row_to_numpy(row)

        # Assume: 行长度合理（不会超大）
        row_len = len(values)
        assume(row_len >= 0)
        assume(row_len < 10000)  # 假设每行最多10000个非零元素

        dot = 0.0
        for i in range(row_len):
            col = indices[i]

            # Assume: 索引在有效范围内（消除边界检查）
            assume(col >= 0)
            assume(col < len(vec))

            val = values[i]

            # Assume: 值是有限的（消除NaN/Inf检查）
            # assume(val == val)  # NaN != NaN
            # assume(val != np.inf)
            # assume(val != -np.inf)

            dot += val * vec[col]

        result[row] = dot

    return result


# ============================================================================
# 示例2: 使用assume优化循环
# ============================================================================

@njit
def sum_with_assume(arr):
    """带有assume的求和

    告诉编译器数组长度是8的倍数，可以启用更好的向量化
    """
    n = len(arr)

    # Assume: 长度是8的倍数
    assume(n % 8 == 0)

    # Assume: 长度不会太大
    assume(n > 0)
    assume(n < 1000000)

    total = 0.0
    for i in range(n):
        # Assume: 值在合理范围内
        val = arr[i]
        assume(val >= -1e100)
        assume(val <= 1e100)
        total += val

    return total


# ============================================================================
# 示例3: 使用assume优化分支
# ============================================================================

@njit
def search_with_assume(csr, target_value):
    """带有assume的搜索

    假设矩阵是稀疏的，大部分行为空
    """
    count = 0

    for row in range(csr.nrows):
        values, indices = csr.row_to_numpy(row)
        row_len = len(values)

        # Assume: 大部分行很短（启用预测分支优化）
        assume(row_len < 20)

        for i in range(row_len):
            if values[i] == target_value:
                count += 1

    return count


# ============================================================================
# 测试和对比
# ============================================================================

def test_assume_optimization():
    """测试assume是否真的带来优化"""
    print("=" * 70)
    print("测试Numba的assume优化")
    print("=" * 70)

    # 创建测试数据
    mat = sp.random(1000, 800, density=0.01, format='csr', dtype=np.float64)
    csr = CSRF64.from_scipy(mat)
    vec = np.random.rand(800)

    print(f"\n测试矩阵: {mat.shape[0]} x {mat.shape[1]}, nnz={mat.nnz}")

    # ========================================
    # 测试1: 基础版本 vs assume版本
    # ========================================

    @njit
    def spmv_baseline(csr, vec):
        """基础版本（无assume）"""
        n = csr.nrows
        result = np.zeros(n, dtype=np.float64)
        for row in range(n):
            values, indices = csr.row_to_numpy(row)
            dot = 0.0
            for i in range(len(values)):
                dot += values[i] * vec[indices[i]]
            result[row] = dot
        return result

    # 触发编译
    r1 = spmv_baseline(csr, vec)
    r2 = spmv_with_assume(csr, vec)

    # 检查结果一致性
    print(f"\n结果一致性检查: {np.allclose(r1, r2)}")

    # 比较IR
    import time

    t0 = time.perf_counter()
    for _ in range(100):
        _ = spmv_baseline(csr, vec)
    t_baseline = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(100):
        _ = spmv_with_assume(csr, vec)
    t_assume = (time.perf_counter() - t0) * 1000

    print(f"\n性能对比 (100次迭代):")
    print(f"  Baseline: {t_baseline:.2f} ms")
    print(f"  With assume: {t_assume:.2f} ms")
    print(f"  提升: {(t_baseline - t_assume) / t_baseline * 100:.1f}%")

    # ========================================
    # 检查生成的IR
    # ========================================

    print("\n" + "=" * 70)
    print("检查生成的LLVM IR")
    print("=" * 70)

    ir_baseline = spmv_baseline.inspect_llvm(spmv_baseline.signatures[0])
    ir_assume = spmv_with_assume.inspect_llvm(spmv_with_assume.signatures[0])

    # 统计assume调用
    import re
    assume_calls = len(re.findall(r'llvm\.assume', ir_assume))

    print(f"\n基础版本IR行数: {len(ir_baseline.split(chr(10)))}")
    print(f"Assume版本IR行数: {len(ir_assume.split(chr(10)))}")
    print(f"llvm.assume调用次数: {assume_calls}")

    if assume_calls > 0:
        print("\n[OK] LLVM assume已被插入到IR中！")
        print("LLVM优化器可以利用这些假设进行更激进的优化。")
    else:
        print("\n[WARNING] 未找到llvm.assume调用")

    # 检查优化差异
    baseline_lines = ir_baseline.split('\n')
    assume_lines = ir_assume.split('\n')

    # 提取主函数
    def extract_main_func(lines):
        result = []
        in_main = False
        depth = 0
        for line in lines:
            if 'define' in line and 'cpython' not in line:
                in_main = True
                depth = 0
            if in_main:
                result.append(line)
                if '{' in line:
                    depth += line.count('{')
                if '}' in line:
                    depth -= line.count('}')
                    if depth <= 0:
                        break
        return result

    main_baseline = extract_main_func(baseline_lines)
    main_assume = extract_main_func(assume_lines)

    print(f"\n主函数IR行数对比:")
    print(f"  Baseline: {len(main_baseline)}")
    print(f"  Assume: {len(main_assume)}")

    # 查找assume调用的位置
    print(f"\nAssume调用位置:")
    for i, line in enumerate(main_assume):
        if 'llvm.assume' in line:
            print(f"  Line {i}: {line.strip()[:80]}")


# ============================================================================
# 展示assume在IR中的效果
# ============================================================================

def show_assume_in_ir():
    """展示assume如何出现在LLVM IR中"""
    print("\n" + "=" * 70)
    print("Assume在LLVM IR中的表现")
    print("=" * 70)

    @njit
    def simple_assume_example(x):
        # 假设x总是正数
        assume(x > 0)

        # 基于这个假设，LLVM可以优化掉一些检查
        if x > 0:
            return 1.0 / x  # 不需要检查除零
        else:
            return 0.0  # 这个分支永远不会执行

    # 触发编译
    _ = simple_assume_example(5.0)

    # 获取IR
    ir = simple_assume_example.inspect_llvm(simple_assume_example.signatures[0])

    # 查找assume和相关代码
    lines = ir.split('\n')

    print("\n相关IR片段:")
    for i, line in enumerate(lines):
        if 'assume' in line.lower() or ('fcmp' in line and i < len(lines) - 5):
            # 打印上下文
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            for j in range(start, end):
                prefix = ">>> " if j == i else "    "
                print(f"{prefix}{lines[j]}")
            print()


if __name__ == "__main__":
    test_assume_optimization()
    show_assume_in_ir()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
Numba中使用assume的方法:

1. 自定义intrinsic (@intrinsic装饰器)
   - 直接调用llvm.assume
   - 完全控制LLVM IR生成

2. 使用literally
   - 强制编译时常量
   - 启用常量折叠和循环展开

3. 常见用例:
   - 消除边界检查: assume(0 <= idx < length)
   - 对齐假设: assume_aligned(ptr, 32)
   - 值范围假设: assume(val >= min && val <= max)
   - 分支预测: assume(condition)  # 告诉编译器这个条件通常为真

4. 注意事项:
   - assume是未定义行为的来源！如果假设不成立，程序可能崩溃
   - 只在你100%确定的情况下使用
   - 主要用于性能关键代码
   - 始终在release前进行充分测试

5. 其他激进优化选项:
   - fastmath=True (启用不安全的浮点优化)
   - inline='always' (强制内联)
   - boundscheck=False (禁用边界检查)
   - 使用LLVM metadata (tbaa, noalias等)
""")
