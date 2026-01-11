"""SCL-Core Python 绑定测试脚本。"""

import sys
import numpy as np
import scipy.sparse as sp

# 修复 Windows 控制台编码问题
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from src.python import CSRF64, CSCF64, CSRF32

def test_basic():
    """测试基本功能。"""
    print("=" * 60)
    print("测试 1: 基本功能")
    print("=" * 60)
    
    # 创建 scipy 稀疏矩阵
    scipy_mat = sp.random(1000, 500, density=0.01, format='csr', dtype='float64')
    print(f"scipy 矩阵: shape={scipy_mat.shape}, nnz={scipy_mat.nnz}")
    
    # 转换为 CSRF64
    csr = CSRF64.from_scipy(scipy_mat)
    print(f"CSRF64: shape={csr.shape}, nnz={csr.nnz}")
    print(f"  density={csr.density:.4f}, sparsity={csr.sparsity:.4f}")
    print(f"  is_valid={csr.is_valid}, is_sorted={csr.is_sorted}")
    
    # 验证数据一致性
    back = csr.to_scipy()
    diff = np.abs(scipy_mat - back).sum()
    print(f"  转换误差: {diff}")
    assert diff < 1e-10, "数据不一致！"
    print("✓ 基本功能测试通过")


def test_slicing():
    """测试切片操作。"""
    print("\n" + "=" * 60)
    print("测试 2: 切片操作")
    print("=" * 60)
    
    # 创建测试矩阵
    scipy_mat = sp.random(100, 80, density=0.1, format='csr', dtype='float64')
    csr = CSRF64.from_scipy(scipy_mat)
    
    # 行切片
    row_slice = csr[20:40, :]
    print(f"行切片 [20:40, :]: shape={row_slice.shape}")
    assert row_slice.shape == (20, 80), "行切片形状错误"
    
    # 列切片
    col_slice = csr[:, 10:30]
    print(f"列切片 [:, 10:30]: shape={col_slice.shape}")
    assert col_slice.shape == (100, 20), "列切片形状错误"
    
    # 行列组合切片
    sub = csr[20:40, 10:30]
    print(f"组合切片 [20:40, 10:30]: shape={sub.shape}")
    assert sub.shape == (20, 20), "组合切片形状错误"
    
    # 与 scipy 结果对比
    scipy_sub = scipy_mat[20:40, 10:30]
    our_sub = sub.to_scipy()
    diff = np.abs(scipy_sub - our_sub).sum()
    print(f"  与 scipy 切片误差: {diff}")
    assert diff < 1e-10, "切片数据不一致！"
    
    print("✓ 切片操作测试通过")


def test_stacking():
    """测试堆叠操作。"""
    print("\n" + "=" * 60)
    print("测试 3: 堆叠操作")
    print("=" * 60)
    
    # 创建测试矩阵
    mat1 = sp.random(50, 100, density=0.1, format='csr', dtype='float64')
    mat2 = sp.random(30, 100, density=0.1, format='csr', dtype='float64')
    
    csr1 = CSRF64.from_scipy(mat1)
    csr2 = CSRF64.from_scipy(mat2)
    
    # 垂直堆叠
    vstacked = CSRF64.vstack([csr1, csr2])
    print(f"vstack: {csr1.shape} + {csr2.shape} = {vstacked.shape}")
    assert vstacked.shape == (80, 100), "vstack 形状错误"
    
    # 与 scipy 结果对比
    scipy_vstacked = sp.vstack([mat1, mat2])
    diff = np.abs(scipy_vstacked - vstacked.to_scipy()).sum()
    print(f"  与 scipy vstack 误差: {diff}")
    assert diff < 1e-10, "vstack 数据不一致！"
    
    # 水平堆叠
    mat3 = sp.random(50, 60, density=0.1, format='csr', dtype='float64')
    csr3 = CSRF64.from_scipy(mat3)
    
    hstacked = CSRF64.hstack([csr1, csr3])
    print(f"hstack: {csr1.shape} + {csr3.shape} = {hstacked.shape}")
    assert hstacked.shape == (50, 160), "hstack 形状错误"
    
    # 与 scipy 结果对比
    scipy_hstacked = sp.hstack([mat1, mat3])
    diff = np.abs(scipy_hstacked - hstacked.to_scipy()).sum()
    print(f"  与 scipy hstack 误差: {diff}")
    assert diff < 1e-10, "hstack 数据不一致！"
    
    print("✓ 堆叠操作测试通过")


def test_conversion():
    """测试格式转换。"""
    print("\n" + "=" * 60)
    print("测试 4: 格式转换")
    print("=" * 60)
    
    # 创建测试矩阵
    scipy_csr = sp.random(100, 80, density=0.05, format='csr', dtype='float64')
    csr = CSRF64.from_scipy(scipy_csr)
    
    # CSR -> CSC
    csc = csr.to_csc()
    print(f"CSR -> CSC: shape={csc.shape}, nnz={csc.nnz}")
    assert csc.shape == csr.shape, "CSC 形状错误"
    assert csc.nnz == csr.nnz, "CSC nnz 错误"
    
    # CSC -> CSR
    csr_back = csc.to_csr()
    print(f"CSC -> CSR: shape={csr_back.shape}, nnz={csr_back.nnz}")
    
    # 验证转换一致性
    diff = np.abs(csr.to_scipy() - csr_back.to_scipy()).sum()
    print(f"  CSR -> CSC -> CSR 误差: {diff}")
    assert diff < 1e-10, "转换不一致！"
    
    # CSR -> Dense
    dense = csr.to_dense()
    print(f"CSR -> Dense: shape={dense.shape}, dtype={dense.dtype}")
    diff = np.abs(scipy_csr.toarray() - dense).sum()
    print(f"  与 scipy toarray 误差: {diff}")
    assert diff < 1e-10, "Dense 转换不一致！"
    
    # CSR -> COO
    row_idx, col_idx, data = csr.to_coo()
    print(f"CSR -> COO: nnz={len(data)}")
    assert len(row_idx) == csr.nnz, "COO 长度错误"
    
    print("✓ 格式转换测试通过")


def test_mask_slicing():
    """测试掩码切片。"""
    print("\n" + "=" * 60)
    print("测试 5: 掩码切片")
    print("=" * 60)
    
    # 创建测试矩阵
    scipy_mat = sp.random(100, 80, density=0.1, format='csr', dtype='float64')
    csr = CSRF64.from_scipy(scipy_mat)
    
    # 行掩码切片
    row_mask = np.zeros(100, dtype=bool)
    row_mask[10:30] = True
    row_mask[50:60] = True
    
    masked = csr.slice_rows_mask(row_mask)
    expected_rows = row_mask.sum()
    print(f"行掩码切片: 选中 {expected_rows} 行, 结果 shape={masked.shape}")
    assert masked.shape[0] == expected_rows, "行掩码切片行数错误"
    assert masked.shape[1] == 80, "行掩码切片列数错误"
    
    # 列掩码切片
    col_mask = np.zeros(80, dtype=bool)
    col_mask[5:25] = True
    
    col_masked = csr.slice_cols_mask(col_mask)
    expected_cols = col_mask.sum()
    print(f"列掩码切片: 选中 {expected_cols} 列, 结果 shape={col_masked.shape}")
    assert col_masked.shape[0] == 100, "列掩码切片行数错误"
    assert col_masked.shape[1] == expected_cols, "列掩码切片列数错误"
    
    print("✓ 掩码切片测试通过")


def test_f32():
    """测试 float32 类型。"""
    print("\n" + "=" * 60)
    print("测试 6: float32 类型")
    print("=" * 60)
    
    # 创建 float32 矩阵
    scipy_mat = sp.random(100, 80, density=0.1, format='csr', dtype='float32')
    csr = CSRF32.from_scipy(scipy_mat)
    
    print(f"CSRF32: shape={csr.shape}, nnz={csr.nnz}")
    print(f"  density={csr.density:.4f}")
    
    # 验证数据
    back = csr.to_scipy()
    diff = np.abs(scipy_mat - back).sum()
    print(f"  转换误差: {diff}")
    assert diff < 1e-5, "float32 数据不一致！"
    
    print("✓ float32 类型测试通过")


def test_row_access():
    """测试行访问。"""
    print("\n" + "=" * 60)
    print("测试 7: 行访问")
    print("=" * 60)
    
    # 创建小矩阵便于验证
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    row = np.array([0, 0, 1, 2, 2])
    col = np.array([0, 2, 1, 0, 2])
    scipy_mat = sp.csr_matrix((data, (row, col)), shape=(3, 3), dtype='float64')
    
    csr = CSRF64.from_scipy(scipy_mat)
    print(f"矩阵 shape={csr.shape}, nnz={csr.nnz}")
    
    # 检查每行
    for i in range(3):
        values, indices = csr.row_to_numpy(i)
        row_len = csr.row_len(i)
        print(f"  行 {i}: len={row_len}, values={values}, indices={indices}")
        assert len(values) == row_len, f"行 {i} 长度不匹配"
    
    print("✓ 行访问测试通过")


def main():
    """运行所有测试。"""
    print("\n" + "#" * 60)
    print("# SCL-Core Python 绑定测试")
    print("#" * 60)
    
    try:
        test_basic()
        test_slicing()
        test_stacking()
        test_conversion()
        test_mask_slicing()
        test_f32()
        test_row_access()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
