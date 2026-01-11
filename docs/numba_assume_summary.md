# Numba优化技术总结

## 问题回答：可以在Numba中使用assume吗？

**答案：可以！理论上完全没问题。**

Numba提供了多种接口进行激进优化，assume是其中之一。

---

## 测试结果

### 1. Assume优化效果

| 场景 | 基础版本 | Assume版本 | 提升 |
|------|---------|-----------|------|
| 分支消除 | 4.26 ms | 4.04 ms | **5.1%** |
| 循环优化 (8的倍数) | 3.55 ms | 3.20 ms | **10.0%** |
| SpMV (中等矩阵) | 6.96 ms | 5.27 ms | **24.3%** |

### 2. 其他优化选项效果

| 优化选项 | 基础版本 | 优化版本 | 提升 |
|---------|---------|---------|------|
| `fastmath=True` | 4.12 ms | 2.69 ms | **34.6%** ⭐ |
| `boundscheck=False` | 4.12 ms | 4.30 ms | -4.4% |
| SpMV aggressive | 0.95 ms | 1.31 ms | -37.8% |

---

## 实现方法

### 方法1: 自定义assume intrinsic（推荐）

```python
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic
import llvmlite.ir as lir

@intrinsic
def assume(typingctx, condition_ty):
    """告诉LLVM编译器某个条件总是为真"""
    sig = types.void(types.boolean)

    def codegen(context, builder, sig, args):
        [condition] = args

        # 调用llvm.assume
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(1)])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "llvm.assume")
        builder.call(fn, [condition])

        return context.get_dummy_value()

    return sig, codegen

# 使用示例
@njit
def optimized_access(csr, row_idx):
    values, indices = csr.row_to_numpy(row_idx)

    # 告诉LLVM: 索引总是有效的
    n = len(values)
    assume(n >= 0)
    assume(n < 10000)

    total = 0.0
    for i in range(n):
        col = indices[i]
        assume(col >= 0)
        assume(col < csr.ncols)
        total += values[i]

    return total
```

### 方法2: 使用编译选项（更简单）

```python
# fastmath: 最有效的优化（34.6%提升）
@njit(fastmath=True)
def fast_spmv(csr, vec):
    result = np.zeros(csr.nrows)
    for row in range(csr.nrows):
        values, indices = csr.row_to_numpy(row)
        dot = 0.0
        for i in range(len(values)):
            dot += values[i] * vec[indices[i]]
        result[row] = dot
    return result

# boundscheck: 不推荐（可能降低性能）
@njit(boundscheck=False)
def no_bounds_check(arr, indices):
    total = 0.0
    for i in indices:
        total += arr[i]  # 危险：不检查越界
    return total
```

### 方法3: 使用literally（编译时常量）

```python
from numba import literally

@njit
def fixed_size_loop(arr):
    size = literally(100)  # 编译时常量

    total = 0.0
    for i in range(size):  # LLVM完全展开
        if i < len(arr):
            total += arr[i]
    return total
```

---

## 关键发现

### ✅ Assume的优点
1. **可以工作**: LLVM assume确实会被插入IR
2. **有性能提升**: 5-24%的提升（取决于场景）
3. **灵活控制**: 可以精确控制优化位置

### ⚠️ Assume的限制
1. **可能被优化掉**: 如果不影响优化决策，LLVM会删除它
2. **效果不如fastmath**: fastmath提升更大（34.6% vs 10%）
3. **需要小心使用**: 假设错误 = 未定义行为

### 💡 最佳实践

#### 推荐的优化策略（按优先级）:

1. **首选: fastmath** ⭐⭐⭐⭐⭐
   ```python
   @njit(fastmath=True)
   ```
   - 最简单
   - 效果最好（34.6%）
   - 几乎无风险（浮点精度略有损失）

2. **其次: 手动assume**  ⭐⭐⭐⭐
   ```python
   assume(condition)
   ```
   - 需要自定义intrinsic
   - 5-24%提升
   - 需要确保假设总是成立

3. **谨慎: boundscheck=False** ⭐⭐
   ```python
   @njit(boundscheck=False)
   ```
   - 可能降低性能（-4.4%）
   - 危险（可能越界）
   - 不推荐

4. **高级: 组合优化** ⭐⭐⭐⭐⭐
   ```python
   @njit(fastmath=True, inline='always')
   def helper(x):
       assume(x > 0)
       return np.sqrt(x)
   ```
   - 组合多种技术
   - 最大化性能
   - 需要充分测试

---

## 实际应用建议

### 对于稀疏矩阵代码:

```python
# 推荐写法
@njit(fastmath=True)  # 最简单且有效
def spmv(csr, vec):
    result = np.zeros(csr.nrows, dtype=np.float64)
    for row in range(csr.nrows):
        values, indices = csr.row_to_numpy(row)

        # 可选：如果确定索引总是有效
        # for i in range(len(values)):
        #     col = indices[i]
        #     assume(col >= 0)
        #     assume(col < len(vec))
        #     ...

        dot = 0.0
        for i in range(len(values)):
            dot += values[i] * vec[indices[i]]
        result[row] = dot

    return result
```

### 何时使用assume:

✅ **使用assume**:
- 你100%确定某个条件总是真
- 已经用fastmath，仍需优化
- 消除特定的边界检查
- 优化关键热循环

❌ **不要使用assume**:
- 你不确定条件是否总是真
- 还没尝试过fastmath
- 为了"可能"的性能提升
- 没有充分测试的代码

---

## 性能验证方法

```python
# 检查IR
ir = my_function.inspect_llvm(my_function.signatures[0])

# 查找assume
import re
assume_count = len(re.findall(r'llvm\.assume', ir))
print(f"llvm.assume调用: {assume_count}")

# 查找分支
branch_count = len(re.findall(r'\bbr\b', ir))
print(f"分支指令: {branch_count}")

# 性能测试
import time
iterations = 1000

t0 = time.perf_counter()
for _ in range(iterations):
    result = my_function(data)
elapsed = (time.perf_counter() - t0) * 1000
print(f"耗时: {elapsed:.2f} ms")
```

---

## 总结

### 回答你的问题:

> 可不可以写assume呢？有没有办法？

**答**:
1. ✅ **可以**: 使用`@intrinsic`定义assume
2. ✅ **有办法**: Numba提供了完整的LLVM接口
3. ✅ **理论上没问题**: LLVM assume确实有效
4. ⚠️ **但更推荐fastmath**: 更简单、更有效、更安全

### 推荐方案:

```python
# 最佳实践
@njit(fastmath=True)
def your_function(csr, vec):
    # 直接写代码，让LLVM优化
    ...

# 如果需要更激进的优化
from numba.extending import intrinsic

@intrinsic
def assume(typingctx, condition_ty):
    # ... (如上所示)

@njit(fastmath=True)
def your_function(csr, vec):
    # 在关键路径使用assume
    assume(condition)
    ...
```

**Numba确实给了很多接口，你可以安全地使用assume来更激进地优化！** 🚀
