# SCL Numba 集成文档

## 概述

SCL-Core 提供了完整的 Numba JIT 支持，使您能够在高性能的 `nopython` 模式下操作稀疏矩阵。所有 CSR/CSC 操作都经过优化，可以在 JIT 编译的函数中无缝使用。

## 快速开始

```python
from scl import CSRF64
from numba import njit
import scipy.sparse as sp
import numpy as np

# 创建稀疏矩阵
mat = sp.random(1000, 1000, density=0.01, format='csr')
csr = CSRF64.from_scipy(mat)

# 在 JIT 函数中使用
@njit
def process(csr):
    total = 0.0
    for values, indices in csr:
        # values 和 indices 是 NumPy 数组视图（零拷贝）
        total += values.sum()
    return total

result = process(csr)
```

## 功能特性

### 1. 属性访问

所有基础属性都可以在 JIT 中高效访问：

```python
@njit
def get_info(csr):
    nrows = csr.nrows          # 行数
    ncols = csr.ncols          # 列数
    nnz = csr.nnz              # 非零元素数
    shape = csr.shape          # (nrows, ncols) 元组
    density = csr.density      # 密度
    sparsity = csr.sparsity    # 稀疏度
    is_empty = csr.is_empty    # 是否为空
    is_zero = csr.is_zero      # 是否全零
    return shape, density
```

### 2. 行/列访问

快速访问单行或单列的数据：

```python
@njit
def process_rows(csr):
    for i in range(csr.nrows):
        values, indices = csr.row_to_numpy(i)
        # values: 该行的非零值数组
        # indices: 对应的列索引数组
        # 都是零拷贝的内存视图
        row_sum = values.sum()
```

### 3. 迭代器支持

支持两种迭代方式：

```python
# 方式 1: 隐式迭代器（推荐）
@njit
def iter_implicit(csr):
    for values, indices in csr:
        # 自动迭代每一行
        pass

# 方式 2: 显式循环
@njit
def iter_explicit(csr):
    for i in range(len(csr)):
        values, indices = csr.row_to_numpy(i)
        pass
```

### 4. 切片操作

完整的切片语法支持：

```python
@njit
def slicing_examples(csr):
    # 行切片
    sub1 = csr[10:20]           # 第 10-19 行
    sub2 = csr[10:20, :]        # 同上
    
    # 列切片
    sub3 = csr[:, 5:15]         # 第 5-14 列
    
    # 组合切片
    sub4 = csr[10:20, 5:15]     # 子矩阵
    
    # 使用方法
    sub5 = csr.slice_rows(10, 20)
    sub6 = csr.slice_cols(5, 15)
    
    return sub4.shape
```

### 5. 堆叠操作

```python
# 注意：类方法在 JIT 中有限制，建议在 Python 端调用
# 或者使用下面的包装方式

# Python 端
csr1 = CSRF64.from_scipy(mat1)
csr2 = CSRF64.from_scipy(mat2)
vstacked = CSRF64.vstack([csr1, csr2])
hstacked = CSRF64.hstack([csr1, csr2])

# JIT 中可以接收和处理堆叠结果
@njit
def process_stacked(vstacked):
    return vstacked.shape, vstacked.nnz
```

### 6. 格式转换

```python
@njit
def convert_formats(csr):
    # CSR -> Dense
    dense = csr.to_dense()      # numpy array
    
    # CSR -> COO
    rows, cols, data = csr.to_coo()
    
    # CSR -> CSC
    csc = csr.to_csc()
    
    # CSC -> CSR
    csr_back = csc.to_csr()
    
    return dense.shape
```

### 7. 克隆

```python
@njit
def deep_copy(csr):
    csr2 = csr.clone()
    # csr2 是完全独立的副本
    return csr2
```

### 8. 验证和排序

```python
@njit
def check_validity(csr):
    # 检查结构是否有效
    if not csr.is_valid:
        return False
    
    # 检查索引是否排序
    if not csr.is_sorted:
        # 原地排序（修改原矩阵）
        csr.ensure_sorted()
    
    # 完整验证
    return csr.validate()
```

## 性能优化建议

### 1. 避免重复创建对象

```python
# 不好：每次迭代都创建新对象
@njit
def bad(matrices):
    for mat in matrices:
        sub = mat[10:20, :]  # 每次都分配新对象
        process(sub)

# 好：重用对象
@njit
def good(mat):
    sub = mat[10:20, :]  # 只创建一次
    for vals, idxs in sub:
        process(vals, idxs)
```

### 2. 使用迭代器

迭代器是零拷贝的，非常高效：

```python
@njit
def efficient_sum(csr):
    total = 0.0
    for values, _ in csr:  # 零拷贝迭代
        total += values.sum()
    return total
```

### 3. 预先分配输出数组

```python
@njit
def with_prealloc(csr):
    # 预先分配输出
    result = np.zeros(csr.ncols)
    
    for values, indices in csr:
        for j, val in zip(indices, values):
            result[j] += val
    
    return result
```

## 架构说明

### 模块结构

```
src/python/_numba/
├── __init__.py          # 入口和注册
├── _types.py            # 类型定义
├── _models.py           # 数据模型
├── _boxing.py           # Python <-> Numba 转换
├── _ffi.py              # FFI 函数封装
├── _overloads.py        # 基础方法重载
├── _iterators.py        # 迭代器实现
├── _operators.py        # 切片和堆叠
├── _conversions.py      # 格式转换
└── _validation.py       # 验证和排序
```

### 内存管理

**Python 对象 -> JIT**：
- Python 拥有数据所有权
- JIT 只借用指针（零拷贝）
- 在 `unbox` 时调用 `_prepare_numba_pointers()` 准备指针数组

**JIT 创建新对象**：
- 通过 FFI 调用 Rust 函数创建新 handle
- 使用 Numba NRT (Runtime) 管理生命周期
- 在 `box` 时将所有权转移给 Python

**对象销毁**：
- Python 端通过 `__del__` 调用 FFI 的 `_free` 函数
- JIT 端通过 NRT 的 MemInfo 自动管理

### 类型系统

```
Python 类型          Numba 类型         底层表示
─────────────────────────────────────────────────────
CSRF64           ->  CSRFloat64Type  -> CSRModel
CSRF32           ->  CSRFloat32Type  -> CSRModel
CSCF64           ->  CSCFloat64Type  -> CSCModel
CSCF32           ->  CSCFloat32Type  -> CSCModel
```

CSRModel 结构：
```c
struct CSRModel {
    void* handle;              // Rust CSR 对象句柄
    MemInfo* meminfo;          // NRT 内存信息（JIT 创建时）
    int64_t nrows;             // 缓存的行数
    int64_t ncols;             // 缓存的列数
    int64_t nnz;               // 缓存的非零元素数
    double** values_ptrs;      // 行值指针数组
    int64_t** indices_ptrs;    // 行索引指针数组
    size_t* row_lens;          // 行长度数组
    bool owns_data;            // 所有权标记
}
```

## 限制和注意事项

1. **类方法限制**：`hstack` 和 `vstack` 作为类方法在 JIT 中不能直接调用，建议在 Python 端预处理

2. **Span 对象**：`row_values()` 返回的 Span 对象在 Numba 中暂不支持，请使用 `row_to_numpy()`

3. **字符串参数**：`to_dense(order='C')` 的 `order` 参数需要是编译时常量

4. **错误处理**：FFI 调用失败会抛出 RuntimeError，在 JIT 中异常处理有限

## 示例：完整的 JIT 算法

```python
from scl import CSRF64
from numba import njit
import numpy as np

@njit
def sparse_matvec(csr, vec):
    """稀疏矩阵-向量乘法"""
    result = np.zeros(csr.nrows)
    
    for i in range(csr.nrows):
        values, indices = csr.row_to_numpy(i)
        row_sum = 0.0
        for j in range(len(values)):
            col_idx = indices[j]
            row_sum += values[j] * vec[col_idx]
        result[i] = row_sum
    
    return result

@njit
def sparse_norm(csr):
    """计算 Frobenius 范数"""
    total = 0.0
    for values, _ in csr:
        for v in values:
            total += v * v
    return np.sqrt(total)

@njit
def sparse_filter(csr, threshold):
    """过滤小于阈值的元素（需要创建新矩阵）"""
    # 这种操作需要在 Python 端实现
    # 或使用 COO 格式重建
    pass
```

## 性能对比

在典型场景下，Numba JIT 版本相比纯 Python 实现：

- **属性访问**：10-20x 加速
- **行迭代**：5-10x 加速（取决于矩阵大小）
- **数值计算**：50-100x 加速（如求和、范数等）
- **切片操作**：1-2x 加速（主要是 FFI 开销）

## 调试技巧

### 1. 查看生成的 LLVM IR

```python
@njit(debug=True)
def my_func(csr):
    pass

# 查看生成的代码
print(my_func.inspect_llvm())
```

### 2. 禁用 JIT 进行调试

```python
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

# 现在 @njit 装饰器不会编译代码
```

### 3. 检查类型推断

```python
from numba import typeof

csr = CSRF64.from_scipy(mat)
print(typeof(csr))  # 应该显示 CSR[float64]
```

## 故障排除

### 问题 1: "Failed to register CFFI module"

**解决方案**：确保 CFFI 库已正确加载
```python
from src.python._binding._cffi import lib
print(lib)  # 应该显示 CFFI 模块
```

### 问题 2: "Invalid use of getiter"

**原因**：迭代器类型未正确注册

**解决方案**：确保导入了 `_numba` 模块
```python
from src.python._binding._sparse import is_numba_available
print(is_numba_available())  # 应该返回 True
```

### 问题 3: 性能没有提升

**可能原因**：
1. 函数太简单，JIT 开销大于收益
2. 频繁创建新对象（应重用对象）
3. 未进行预热编译（第一次调用会触发编译）

**解决方案**：
```python
# 预热 JIT
@njit
def my_func(csr):
    pass

# 先用小数据调用一次
small_mat = sp.random(10, 10, density=0.1, format='csr')
small_csr = CSRF64.from_scipy(small_mat)
my_func(small_csr)  # 触发编译

# 再用大数据
big_csr = CSRF64.from_scipy(big_mat)
result = my_func(big_csr)  # 现在很快
```

## API 参考

### CSRType / CSCType

#### 属性
- `nrows`: int64 - 行数
- `ncols`: int64 - 列数
- `nnz`: int64 - 非零元素数
- `shape`: (int64, int64) - 形状元组
- `density`: float64 - 密度
- `sparsity`: float64 - 稀疏度
- `is_empty`: bool - 是否为空
- `is_zero`: bool - 是否全零
- `is_valid`: bool - 结构是否有效
- `is_sorted`: bool - 索引是否已排序
- `indices_in_bounds`: bool - 索引是否在界内

#### 方法

**行访问** (CSR):
- `row_to_numpy(i: int, copy: bool = False) -> (Array, Array)` - 获取行数据
- `row_len(i: int) -> int` - 获取行长度

**列访问** (CSC):
- `col_to_numpy(j: int, copy: bool = False) -> (Array, Array)` - 获取列数据
- `col_len(j: int) -> int` - 获取列长度

**切片**:
- `slice_rows(start: int, end: int) -> CSR/CSC` - 行切片
- `slice_cols(start: int, end: int) -> CSR/CSC` - 列切片
- `__getitem__(key)` - 切片语法支持

**转换**:
- `to_dense(order: str = 'C') -> Array` - 转换为密集数组
- `to_coo() -> (Array, Array, Array)` - 转换为 COO 格式
- `to_csc() -> CSC` / `to_csr() -> CSR` - 格式互转
- `clone() -> CSR/CSC` - 深拷贝

**验证**:
- `validate() -> bool` - 完整验证
- `ensure_sorted()` - 原地排序
- `ensure_sorted_checked() -> bool` - 按需排序

**迭代**:
- `__iter__()` - 返回迭代器
- `__len__()` - 返回行数/列数

## 内部实现细节

### Unbox 流程

1. 从 Python 对象提取 `handle_as_int`
2. 提取维度信息 (`nrows`, `ncols`, `nnz`)
3. 调用 `_prepare_numba_pointers()` 获取指针数组
4. 从 NumPy 数组中提取底层指针
5. 构造 CSRModel 结构体

### Box 流程

1. 从 CSRModel 提取 handle
2. 确定 Python 类型 (CSRF32/CSRF64)
3. 动态导入 Python 模块
4. 调用 `_from_handle(handle, owns_handle)` 创建 Python 对象
5. 根据 `owns_data` 标志决定所有权转移

### FFI 调用

所有 FFI 函数都通过 `@intrinsic` 封装：

```python
@intrinsic
def _ffi_csr_f64_rows(typingctx, handle_ty):
    sig = types.int64(types.voidptr)
    
    def codegen(context, builder, sig, args):
        [handle] = args
        fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f64_rows")
        return builder.call(fn, [handle])
    
    return sig, codegen
```

这使得 FFI 调用在 LLVM 层面完全内联，没有 Python 调用开销。

## 扩展开发

如果需要添加新的方法支持：

1. **在 `_ffi.py` 中添加 FFI intrinsic**
2. **在相应模块中添加 overload**（`_overloads.py`, `_operators.py` 等）
3. **更新测试**（`test_numba.py`）
4. **更新文档**

示例：添加新方法 `row_sum(i)`

```python
# 1. _ffi.py
@intrinsic
def _ffi_csr_f64_row_sum(typingctx, handle_ty, row_ty):
    sig = types.float64(types.voidptr, types.int64)
    def codegen(context, builder, sig, args):
        [handle, row] = args
        fnty = lir.FunctionType(lir.DoubleType(), 
                               [lir.IntType(8).as_pointer(), lir.IntType(64)])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "csr_f64_row_sum")
        return builder.call(fn, [handle, row])
    return sig, codegen

# 2. _overloads.py
@overload_method(CSRType, 'row_sum')
def csr_row_sum_impl(csr, row_idx):
    if csr.dtype == types.float64:
        def impl(csr, row_idx):
            return _ffi_csr_f64_row_sum(csr.handle, row_idx)
        return impl
```

## 贡献指南

欢迎贡献更多的 Numba 集成功能！请遵循以下原则：

1. **保持零拷贝**：尽量使用内存视图而非拷贝
2. **完整的类型支持**：同时支持 float32 和 float64
3. **一致的 API**：JIT 和 Python 行为应该一致
4. **详细的测试**：每个新功能都需要测试
5. **清晰的文档**：说明用法和限制

## 许可证

本项目遵循 MIT 许可证。
