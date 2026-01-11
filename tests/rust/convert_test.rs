//! 稀疏矩阵格式转换测试
//!
//! 测试内容：
//! - scipy CSR/CSC 转换
//! - CSR ↔ CSC 互转
//! - COO 格式转换
//! - Dense 格式转换
//! - LIL 格式转换

use scl_core::convert::{
    csc_from_csr, csc_from_dense, csc_from_scipy_coo_copy, csc_from_scipy_csc_copy,
    csc_from_scipy_csr_copy, csr_from_csc, csr_from_dense, csr_from_lil, csr_from_scipy_coo_copy,
    csr_from_scipy_csc_copy, csr_from_scipy_csr_copy, csr_to_coo, csr_to_dense, AllocStrategy,
    ConvertError, DenseLayout,
};
// CSC 和 CSR 类型通过函数返回值自动推断

// =============================================================================
// scipy CSR 转换测试
// =============================================================================

#[test]
fn test_csr_from_scipy_csr_copy_basic() {
    // scipy CSR 格式:
    // data = [1, 2, 3, 4, 5]
    // indices = [0, 2, 2, 0, 3]
    // indptr = [0, 2, 3, 5]
    // 表示 3x4 矩阵:
    // [1, 0, 2, 0]
    // [0, 0, 3, 0]
    // [4, 0, 0, 5]

    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let indices = vec![0i64, 2, 2, 0, 3];
    let indptr = vec![0i64, 2, 3, 5];

    let csr = csr_from_scipy_csr_copy::<f64, i64, 32>(
        3,
        4,
        &data,
        &indices,
        &indptr,
        AllocStrategy::Auto,
    )
    .unwrap();

    assert_eq!(csr.nrows(), 3);
    assert_eq!(csr.ncols(), 4);
    assert_eq!(csr.nnz(), 5);

    // 验证行数据
    assert_eq!(csr.row_values(0), &[1.0, 2.0]);
    assert_eq!(csr.row_indices(0), &[0, 2]);

    assert_eq!(csr.row_values(1), &[3.0]);
    assert_eq!(csr.row_indices(1), &[2]);

    assert_eq!(csr.row_values(2), &[4.0, 5.0]);
    assert_eq!(csr.row_indices(2), &[0, 3]);
}

#[test]
fn test_csr_from_scipy_csr_empty() {
    let data: Vec<f64> = vec![];
    let indices: Vec<i64> = vec![];
    let indptr = vec![0i64, 0, 0, 0];

    let csr = csr_from_scipy_csr_copy::<f64, i64, 32>(
        3,
        4,
        &data,
        &indices,
        &indptr,
        AllocStrategy::Auto,
    )
    .unwrap();

    assert_eq!(csr.nrows(), 3);
    assert_eq!(csr.ncols(), 4);
    assert_eq!(csr.nnz(), 0);
    assert!(csr.is_zero());
}

#[test]
fn test_csr_from_scipy_csr_with_empty_rows() {
    // 第 1 行为空
    let data = vec![1.0f64, 2.0, 3.0];
    let indices = vec![0i64, 0, 1];
    let indptr = vec![0i64, 1, 1, 3];

    let csr = csr_from_scipy_csr_copy::<f64, i64, 32>(
        3,
        3,
        &data,
        &indices,
        &indptr,
        AllocStrategy::Auto,
    )
    .unwrap();

    assert_eq!(csr.row_nnz(0), 1);
    assert_eq!(csr.row_nnz(1), 0);
    assert_eq!(csr.row_nnz(2), 2);
}

#[test]
fn test_csr_from_scipy_csr_invalid_indptr() {
    let data = vec![1.0f64];
    let indices = vec![0i64];
    let indptr = vec![0i64, 1]; // 应该有 rows + 1 = 4 个元素

    let result = csr_from_scipy_csr_copy::<f64, i64, 32>(
        3,
        3,
        &data,
        &indices,
        &indptr,
        AllocStrategy::Auto,
    );
    assert!(matches!(result, Err(ConvertError::InvalidIndptr)));
}

#[test]
fn test_csr_from_scipy_csr_length_mismatch() {
    let data = vec![1.0f64, 2.0];
    let indices = vec![0i64]; // 长度不匹配
    let indptr = vec![0i64, 2, 2, 2];

    let result = csr_from_scipy_csr_copy::<f64, i64, 32>(
        3,
        3,
        &data,
        &indices,
        &indptr,
        AllocStrategy::Auto,
    );
    assert!(matches!(result, Err(ConvertError::LengthMismatch)));
}

// =============================================================================
// scipy CSC 转换测试
// =============================================================================

#[test]
fn test_csc_from_scipy_csc_copy_basic() {
    // scipy CSC 格式（按列存储）
    let data = vec![1.0f64, 4.0, 2.0, 3.0];
    let indices = vec![0i64, 2, 0, 1];
    let indptr = vec![0i64, 2, 2, 4];

    let csc = csc_from_scipy_csc_copy::<f64, i64, 32>(
        3,
        3,
        &data,
        &indices,
        &indptr,
        AllocStrategy::Auto,
    )
    .unwrap();

    assert_eq!(csc.nrows(), 3);
    assert_eq!(csc.ncols(), 3);
    assert_eq!(csc.nnz(), 4);

    assert_eq!(csc.col_values(0), &[1.0, 4.0]);
    assert_eq!(csc.col_indices(0), &[0, 2]);

    assert_eq!(csc.col_nnz(1), 0);

    assert_eq!(csc.col_values(2), &[2.0, 3.0]);
    assert_eq!(csc.col_indices(2), &[0, 1]);
}

// =============================================================================
// CSR ↔ CSC 互转测试
// =============================================================================

#[test]
fn test_csc_from_csr() {
    // 创建 CSR
    let data = vec![1.0f64, 2.0, 3.0, 4.0];
    let indices = vec![0i64, 1, 0, 2];
    let indptr = vec![0i64, 2, 4];

    let csr = csr_from_scipy_csr_copy::<f64, i64, 32>(
        2,
        3,
        &data,
        &indices,
        &indptr,
        AllocStrategy::Auto,
    )
    .unwrap();

    // 转换为 CSC
    let csc = csc_from_csr::<f64, i64, 32>(&csr, AllocStrategy::Auto).unwrap();

    assert_eq!(csc.nrows(), 2);
    assert_eq!(csc.ncols(), 3);
    assert_eq!(csc.nnz(), 4);

    // 验证：CSC 的列应该包含对应行的元素
    // 列 0: 行 0 的值 1.0, 行 1 的值 3.0
    let (vals, _idxs) = csc.col(0);
    assert_eq!(vals.len(), 2);
    // 排序后应该是 [1.0, 3.0] 对应行 [0, 1]
    assert!(vals.contains(&1.0) && vals.contains(&3.0));
}

#[test]
fn test_csr_from_csc() {
    // 创建 CSC
    let data = vec![1.0f64, 3.0, 2.0, 4.0];
    let indices = vec![0i64, 1, 0, 1];
    let indptr = vec![0i64, 2, 4];

    let csc = csc_from_scipy_csc_copy::<f64, i64, 32>(
        2,
        2,
        &data,
        &indices,
        &indptr,
        AllocStrategy::Auto,
    )
    .unwrap();

    // 转换为 CSR
    let csr = csr_from_csc::<f64, i64, 32>(&csc, AllocStrategy::Auto).unwrap();

    assert_eq!(csr.nrows(), 2);
    assert_eq!(csr.ncols(), 2);
    assert_eq!(csr.nnz(), 4);
}

#[test]
fn test_csr_csc_roundtrip() {
    // 创建原始 CSR
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let indices = vec![0i64, 2, 1, 0, 2];
    let indptr = vec![0i64, 2, 3, 5];

    let csr_original = csr_from_scipy_csr_copy::<f64, i64, 32>(
        3,
        3,
        &data,
        &indices,
        &indptr,
        AllocStrategy::Auto,
    )
    .unwrap();

    // CSR -> CSC -> CSR
    let csc = csc_from_csr::<f64, i64, 32>(&csr_original, AllocStrategy::Auto).unwrap();
    let csr_back = csr_from_csc::<f64, i64, 32>(&csc, AllocStrategy::Auto).unwrap();

    // 验证维度和 nnz
    assert_eq!(csr_back.nrows(), csr_original.nrows());
    assert_eq!(csr_back.ncols(), csr_original.ncols());
    assert_eq!(csr_back.nnz(), csr_original.nnz());

    // 验证每行数据一致（排序后比较）
    for i in 0..3 {
        let mut orig_pairs: Vec<_> = csr_original
            .row_values(i)
            .iter()
            .zip(csr_original.row_indices(i).iter())
            .map(|(&v, &idx)| (idx, v))
            .collect();
        orig_pairs.sort_by_key(|p| p.0);

        let mut back_pairs: Vec<_> = csr_back
            .row_values(i)
            .iter()
            .zip(csr_back.row_indices(i).iter())
            .map(|(&v, &idx)| (idx, v))
            .collect();
        back_pairs.sort_by_key(|p| p.0);

        assert_eq!(orig_pairs, back_pairs);
    }
}

// =============================================================================
// COO 格式转换测试
// =============================================================================

#[test]
fn test_csr_from_scipy_coo() {
    // COO 格式
    let row_indices = vec![0i64, 0, 1, 2, 2];
    let col_indices = vec![0i64, 2, 1, 0, 2];
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];

    let csr = csr_from_scipy_coo_copy::<f64, i64, 32>(
        3,
        3,
        &row_indices,
        &col_indices,
        &data,
        AllocStrategy::Auto,
    )
    .unwrap();

    assert_eq!(csr.nrows(), 3);
    assert_eq!(csr.ncols(), 3);
    assert_eq!(csr.nnz(), 5);

    // 验证行 0
    let (vals, _idxs) = csr.row(0);
    assert_eq!(vals.len(), 2);
    // 应该包含列 0 的值 1.0 和列 2 的值 2.0
}

#[test]
fn test_csc_from_scipy_coo() {
    let row_indices = vec![0i64, 1, 0, 1];
    let col_indices = vec![0i64, 0, 1, 1];
    let data = vec![1.0f64, 2.0, 3.0, 4.0];

    let csc = csc_from_scipy_coo_copy::<f64, i64, 32>(
        2,
        2,
        &row_indices,
        &col_indices,
        &data,
        AllocStrategy::Auto,
    )
    .unwrap();

    assert_eq!(csc.nrows(), 2);
    assert_eq!(csc.ncols(), 2);
    assert_eq!(csc.nnz(), 4);
}

#[test]
fn test_csr_to_coo() {
    let data = vec![1.0f64, 2.0, 3.0];
    let indices = vec![0i64, 2, 1];
    let indptr = vec![0i64, 1, 2, 3];

    let csr = csr_from_scipy_csr_copy::<f64, i64, 32>(
        3,
        3,
        &data,
        &indices,
        &indptr,
        AllocStrategy::Auto,
    )
    .unwrap();

    let nnz = csr.nnz() as usize;
    let mut out_rows = vec![0i64; nnz];
    let mut out_cols = vec![0i64; nnz];
    let mut out_data = vec![0.0f64; nnz];

    csr_to_coo(&csr, &mut out_rows, &mut out_cols, &mut out_data).unwrap();

    // 验证输出
    assert_eq!(out_rows, vec![0, 1, 2]);
    assert_eq!(out_cols, vec![0, 2, 1]);
    assert_eq!(out_data, vec![1.0, 2.0, 3.0]);
}

// =============================================================================
// Dense 格式转换测试
// =============================================================================

#[test]
fn test_csr_from_dense_row_major() {
    // 3x4 矩阵（行主序）:
    // [1, 0, 2, 0]
    // [0, 0, 0, 0]
    // [3, 0, 0, 4]
    let dense = vec![
        1.0f64, 0.0, 2.0, 0.0, // 行 0
        0.0, 0.0, 0.0, 0.0, // 行 1
        3.0, 0.0, 0.0, 4.0, // 行 2
    ];

    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 3, 4, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nrows(), 3);
    assert_eq!(csr.ncols(), 4);
    assert_eq!(csr.nnz(), 4);

    assert_eq!(csr.row_values(0), &[1.0, 2.0]);
    assert_eq!(csr.row_indices(0), &[0, 2]);

    assert_eq!(csr.row_nnz(1), 0);

    assert_eq!(csr.row_values(2), &[3.0, 4.0]);
    assert_eq!(csr.row_indices(2), &[0, 3]);
}

#[test]
fn test_csr_from_dense_col_major() {
    // 2x3 矩阵（列主序，内存布局不同）:
    // [1, 0, 3]
    // [2, 0, 4]
    let dense = vec![
        1.0f64, 2.0, // 列 0
        0.0, 0.0, // 列 1
        3.0, 4.0, // 列 2
    ];

    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 2, 3, DenseLayout::ColMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nrows(), 2);
    assert_eq!(csr.ncols(), 3);
    assert_eq!(csr.nnz(), 4);

    // 排序后验证
    assert!(csr.is_sorted());
}

#[test]
fn test_csr_to_dense_row_major() {
    let data = vec![1.0f64, 2.0, 3.0, 4.0];
    let indices = vec![0i64, 2, 1, 3];
    let indptr = vec![0i64, 2, 4];

    let csr = csr_from_scipy_csr_copy::<f64, i64, 32>(
        2,
        4,
        &data,
        &indices,
        &indptr,
        AllocStrategy::Auto,
    )
    .unwrap();

    let mut out = vec![0.0f64; 8];
    csr_to_dense(&csr, &mut out, DenseLayout::RowMajor).unwrap();

    // 期望:
    // [1, 0, 2, 0]
    // [0, 3, 0, 4]
    assert_eq!(out, vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0]);
}

#[test]
fn test_csr_to_dense_col_major() {
    let data = vec![1.0f64, 2.0, 3.0];
    let indices = vec![0i64, 1, 0];
    let indptr = vec![0i64, 2, 3];

    let csr = csr_from_scipy_csr_copy::<f64, i64, 32>(
        2,
        2,
        &data,
        &indices,
        &indptr,
        AllocStrategy::Auto,
    )
    .unwrap();

    let mut out = vec![0.0f64; 4];
    csr_to_dense(&csr, &mut out, DenseLayout::ColMajor).unwrap();

    // 期望 (列主序):
    // 列 0: [1, 3], 列 1: [2, 0]
    // 内存布局: [1, 3, 2, 0]
    assert_eq!(out, vec![1.0, 3.0, 2.0, 0.0]);
}

#[test]
fn test_csc_from_dense_col_major() {
    let dense = vec![
        1.0f64, 2.0, // 列 0
        0.0, 0.0, // 列 1
        3.0, 4.0, // 列 2
    ];

    let csc =
        csc_from_dense::<f64, i64, 32>(&dense, 2, 3, DenseLayout::ColMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csc.nrows(), 2);
    assert_eq!(csc.ncols(), 3);
    assert_eq!(csc.nnz(), 4);

    assert_eq!(csc.col_values(0), &[1.0, 2.0]);
    assert_eq!(csc.col_indices(0), &[0, 1]);

    assert_eq!(csc.col_nnz(1), 0);
}

#[test]
fn test_dense_conversion_roundtrip() {
    // 创建 dense -> CSR -> dense
    let original = vec![
        1.0f64, 0.0, 2.0, //
        3.0, 4.0, 0.0, //
        0.0, 5.0, 6.0, //
    ];

    let csr =
        csr_from_dense::<f64, i64, 32>(&original, 3, 3, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    let mut recovered = vec![0.0f64; 9];
    csr_to_dense(&csr, &mut recovered, DenseLayout::RowMajor).unwrap();

    assert_eq!(original, recovered);
}

// =============================================================================
// LIL 格式转换测试
// =============================================================================

#[test]
fn test_csr_from_lil() {
    let row_indices: Vec<&[i64]> = vec![&[0, 2], &[], &[1, 2]];
    let row_values: Vec<&[f64]> = vec![&[1.0, 2.0], &[], &[3.0, 4.0]];

    let csr =
        csr_from_lil::<f64, i64, 32>(3, 3, &row_indices, &row_values, AllocStrategy::Auto).unwrap();

    assert_eq!(csr.nrows(), 3);
    assert_eq!(csr.ncols(), 3);
    assert_eq!(csr.nnz(), 4);

    assert_eq!(csr.row_values(0), &[1.0, 2.0]);
    assert_eq!(csr.row_indices(0), &[0, 2]);

    assert_eq!(csr.row_nnz(1), 0);

    assert_eq!(csr.row_values(2), &[3.0, 4.0]);
    assert_eq!(csr.row_indices(2), &[1, 2]);
}

#[test]
fn test_csr_from_lil_dimension_mismatch() {
    let row_indices: Vec<&[i64]> = vec![&[0], &[1]]; // 只有 2 行
    let row_values: Vec<&[f64]> = vec![&[1.0], &[2.0]];

    let result = csr_from_lil::<f64, i64, 32>(3, 3, &row_indices, &row_values, AllocStrategy::Auto);
    assert!(matches!(result, Err(ConvertError::DimensionMismatch)));
}

#[test]
fn test_csr_from_lil_length_mismatch() {
    let row_indices: Vec<&[i64]> = vec![&[0, 1], &[0]]; // 第 0 行有 2 个索引
    let row_values: Vec<&[f64]> = vec![&[1.0], &[2.0]]; // 第 0 行只有 1 个值

    let result = csr_from_lil::<f64, i64, 32>(2, 3, &row_indices, &row_values, AllocStrategy::Auto);
    assert!(matches!(result, Err(ConvertError::LengthMismatch)));
}

// =============================================================================
// scipy 交叉转换测试
// =============================================================================

#[test]
fn test_csc_from_scipy_csr() {
    // CSR 格式的输入
    let data = vec![1.0f64, 2.0, 3.0, 4.0];
    let col_indices = vec![0i64, 1, 0, 2];
    let row_indptr = vec![0i64, 2, 4];

    let csc = csc_from_scipy_csr_copy::<f64, i64, 32>(
        2,
        3,
        &data,
        &col_indices,
        &row_indptr,
        AllocStrategy::Auto,
    )
    .unwrap();

    assert_eq!(csc.nrows(), 2);
    assert_eq!(csc.ncols(), 3);
    assert_eq!(csc.nnz(), 4);

    // 列 0 应该包含行 0 的值 1.0 和行 1 的值 3.0
    assert_eq!(csc.col_nnz(0), 2);
}

#[test]
fn test_csr_from_scipy_csc() {
    // CSC 格式的输入
    let data = vec![1.0f64, 3.0, 2.0, 4.0];
    let row_indices = vec![0i64, 1, 0, 1];
    let col_indptr = vec![0i64, 2, 4];

    let csr = csr_from_scipy_csc_copy::<f64, i64, 32>(
        2,
        2,
        &data,
        &row_indices,
        &col_indptr,
        AllocStrategy::Auto,
    )
    .unwrap();

    assert_eq!(csr.nrows(), 2);
    assert_eq!(csr.ncols(), 2);
    assert_eq!(csr.nnz(), 4);
}

// =============================================================================
// 边界情况测试
// =============================================================================

#[test]
fn test_convert_empty_matrix() {
    let dense: Vec<f64> = vec![0.0; 9];

    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 3, 3, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nnz(), 0);
    assert!(csr.is_zero());
}

#[test]
fn test_convert_single_element() {
    let dense = vec![42.0f64];

    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 1, 1, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nrows(), 1);
    assert_eq!(csr.ncols(), 1);
    assert_eq!(csr.nnz(), 1);
    assert_eq!(csr.row_values(0), &[42.0]);
}

#[test]
fn test_convert_all_zeros_dense() {
    let dense = vec![0.0f64; 12];

    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 3, 4, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nnz(), 0);

    for i in 0..3 {
        assert_eq!(csr.row_nnz(i), 0);
    }
}

#[test]
fn test_convert_dimension_mismatch() {
    let dense = vec![1.0f64; 10]; // 10 个元素

    let result =
        csr_from_dense::<f64, i64, 32>(&dense, 3, 4, DenseLayout::RowMajor, AllocStrategy::Auto);
    // 3 * 4 = 12 != 10

    assert!(matches!(result, Err(ConvertError::DimensionMismatch)));
}

#[test]
fn test_convert_large_sparse_matrix() {
    // 创建一个 100x100 的稀疏单位矩阵
    let mut dense = vec![0.0f64; 10000];
    for i in 0..100 {
        dense[i * 100 + i] = 1.0;
    }

    let csr = csr_from_dense::<f64, i64, 32>(
        &dense,
        100,
        100,
        DenseLayout::RowMajor,
        AllocStrategy::Auto,
    )
    .unwrap();

    assert_eq!(csr.nrows(), 100);
    assert_eq!(csr.ncols(), 100);
    assert_eq!(csr.nnz(), 100);

    // 验证对角元素
    for i in 0..100 {
        assert_eq!(csr.row_nnz(i), 1);
        assert_eq!(csr.row_values(i), &[1.0]);
        assert_eq!(csr.row_indices(i), &[i]);
    }
}
