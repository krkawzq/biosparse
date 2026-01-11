//! CSR/CSC 稀疏矩阵测试
//!
//! 测试内容：
//! - 基础构造和属性访问
//! - 行/列访问
//! - 迭代器
//! - 验证和排序
//! - Clone 和深拷贝

use std::ptr::NonNull;

use scl_core::span::{Span, SpanFlags};
use scl_core::sparse::{SparseIndex, CSC, CSR};

// =============================================================================
// 辅助函数
// =============================================================================

/// 创建空 Span（用于空行/列）
fn empty_span<T>() -> Span<T> {
    let ptr = NonNull::dangling();
    unsafe { Span::from_raw_parts_unchecked(ptr, 0, SpanFlags::VIEW) }
}

/// 创建测试用 CSR 矩阵
fn make_test_csr<const ALIGN: usize>(
    rows: usize,
    cols: usize,
    row_data: &[&[f64]],
    row_indices: &[&[i64]],
) -> CSR<f64, i64> {
    assert_eq!(row_data.len(), rows);
    assert_eq!(row_indices.len(), rows);

    let mut values = Vec::with_capacity(rows);
    let mut indices = Vec::with_capacity(rows);

    for i in 0..rows {
        if row_data[i].is_empty() {
            values.push(empty_span());
            indices.push(empty_span());
        } else {
            values.push(Span::copy_from::<ALIGN>(row_data[i]).unwrap());
            indices.push(Span::copy_from::<ALIGN>(row_indices[i]).unwrap());
        }
    }

    unsafe { CSR::from_raw_parts(values, indices, rows as i64, cols as i64) }
}

/// 创建测试用 CSC 矩阵
fn make_test_csc<const ALIGN: usize>(
    rows: usize,
    cols: usize,
    col_data: &[&[f64]],
    col_indices: &[&[i64]],
) -> CSC<f64, i64> {
    assert_eq!(col_data.len(), cols);
    assert_eq!(col_indices.len(), cols);

    let mut values = Vec::with_capacity(cols);
    let mut indices = Vec::with_capacity(cols);

    for j in 0..cols {
        if col_data[j].is_empty() {
            values.push(empty_span());
            indices.push(empty_span());
        } else {
            values.push(Span::copy_from::<ALIGN>(col_data[j]).unwrap());
            indices.push(Span::copy_from::<ALIGN>(col_indices[j]).unwrap());
        }
    }

    unsafe { CSC::from_raw_parts(values, indices, rows as i64, cols as i64) }
}

// =============================================================================
// CSR 基础测试
// =============================================================================

#[test]
fn test_csr_new_empty() {
    let csr: CSR<f64, i64> = CSR::new(5, 10);
    assert_eq!(csr.nrows(), 5);
    assert_eq!(csr.ncols(), 10);
    assert_eq!(csr.nnz(), 0);
    assert!(csr.is_zero());
    assert!(!csr.is_empty());
}

#[test]
fn test_csr_empty_matrix() {
    let csr: CSR<f64, i64> = CSR::new(0, 0);
    assert!(csr.is_empty());
    assert!(csr.is_zero());
}

#[test]
fn test_csr_basic_construction() {
    // 创建一个 3x4 矩阵:
    // [1.0, 2.0, 0.0, 0.0]
    // [0.0, 0.0, 3.0, 0.0]
    // [4.0, 0.0, 0.0, 5.0]
    let csr = make_test_csr::<32>(
        3,
        4,
        &[&[1.0, 2.0], &[3.0], &[4.0, 5.0]],
        &[&[0, 1], &[2], &[0, 3]],
    );

    assert_eq!(csr.nrows(), 3);
    assert_eq!(csr.ncols(), 4);
    assert_eq!(csr.shape(), (3, 4));
    assert_eq!(csr.nnz(), 5);
    assert!(!csr.is_empty());
    assert!(!csr.is_zero());
}

#[test]
fn test_csr_row_access() {
    let csr = make_test_csr::<32>(
        3,
        4,
        &[&[1.0, 2.0], &[3.0], &[4.0, 5.0]],
        &[&[0, 1], &[2], &[0, 3]],
    );

    // 第 0 行
    assert_eq!(csr.row_values(0), &[1.0, 2.0]);
    assert_eq!(csr.row_indices(0), &[0, 1]);
    assert_eq!(csr.row_nnz(0), 2);

    // 第 1 行
    assert_eq!(csr.row_values(1), &[3.0]);
    assert_eq!(csr.row_indices(1), &[2]);
    assert_eq!(csr.row_nnz(1), 1);

    // 第 2 行
    let (vals, idxs) = csr.row(2);
    assert_eq!(vals, &[4.0, 5.0]);
    assert_eq!(idxs, &[0, 3]);
}

#[test]
fn test_csr_row_access_unchecked() {
    let csr = make_test_csr::<32>(
        3,
        4,
        &[&[1.0, 2.0], &[3.0], &[4.0, 5.0]],
        &[&[0, 1], &[2], &[0, 3]],
    );

    unsafe {
        assert_eq!(csr.row_values_unchecked(0), &[1.0, 2.0]);
        assert_eq!(csr.row_indices_unchecked(1), &[2]);
        assert_eq!(csr.row_nnz_unchecked(2), 2);
    }
}

#[test]
fn test_csr_with_empty_rows() {
    // 带空行的矩阵
    let csr = make_test_csr::<32>(
        4,
        3,
        &[&[1.0], &[], &[2.0, 3.0], &[]],
        &[&[0], &[], &[1, 2], &[]],
    );

    assert_eq!(csr.nrows(), 4);
    assert_eq!(csr.nnz(), 3);

    assert_eq!(csr.row_nnz(0), 1);
    assert_eq!(csr.row_nnz(1), 0);
    assert_eq!(csr.row_nnz(2), 2);
    assert_eq!(csr.row_nnz(3), 0);

    // 空行应该返回空切片
    assert!(csr.row_values(1).is_empty());
    assert!(csr.row_indices(3).is_empty());
}

#[test]
fn test_csr_sparsity_density() {
    // 3x4 矩阵，5 个非零元素
    // 密度 = 5 / 12 ≈ 0.4167
    // 稀疏度 = 7 / 12 ≈ 0.5833
    let csr = make_test_csr::<32>(
        3,
        4,
        &[&[1.0, 2.0], &[3.0], &[4.0, 5.0]],
        &[&[0, 1], &[2], &[0, 3]],
    );

    let density = csr.density();
    let sparsity = csr.sparsity();

    assert!((density - 5.0 / 12.0).abs() < 1e-10);
    assert!((sparsity - 7.0 / 12.0).abs() < 1e-10);
    assert!((density + sparsity - 1.0).abs() < 1e-10);
}

#[test]
fn test_csr_iterators() {
    let csr = make_test_csr::<32>(3, 4, &[&[1.0, 2.0], &[3.0], &[4.0]], &[&[0, 1], &[2], &[0]]);

    // iter_row_values
    let values: Vec<&[f64]> = csr.iter_row_values().collect();
    assert_eq!(values.len(), 3);
    assert_eq!(values[0], &[1.0, 2.0]);
    assert_eq!(values[1], &[3.0]);
    assert_eq!(values[2], &[4.0]);

    // iter_row_indices
    let indices: Vec<&[i64]> = csr.iter_row_indices().collect();
    assert_eq!(indices.len(), 3);
    assert_eq!(indices[0], &[0, 1]);

    // iter_rows
    let rows: Vec<(&[f64], &[i64])> = csr.iter_rows().collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], (&[1.0, 2.0][..], &[0i64, 1][..]));
}

// =============================================================================
// CSR 验证测试
// =============================================================================

#[test]
fn test_csr_is_valid() {
    // 有效的 CSR
    let csr = make_test_csr::<32>(
        3,
        4,
        &[&[1.0, 2.0], &[3.0], &[4.0, 5.0]],
        &[&[0, 1], &[2], &[0, 3]],
    );
    assert!(csr.is_valid());
}

#[test]
fn test_csr_indices_in_bounds() {
    // 索引在界内
    let csr = make_test_csr::<32>(3, 4, &[&[1.0], &[2.0], &[3.0]], &[&[0], &[2], &[3]]);
    assert!(csr.indices_in_bounds());

    // 索引越界（列索引 = 4，但 cols = 4，所以有效索引是 0-3）
    let csr_invalid = make_test_csr::<32>(3, 4, &[&[1.0], &[2.0], &[3.0]], &[&[0], &[2], &[4]]);
    assert!(!csr_invalid.indices_in_bounds());
}

#[test]
fn test_csr_is_sorted() {
    // 已排序
    let csr = make_test_csr::<32>(
        2,
        5,
        &[&[1.0, 2.0, 3.0], &[4.0, 5.0]],
        &[&[0, 2, 4], &[1, 3]],
    );
    assert!(csr.is_sorted());

    // 未排序
    let csr_unsorted = make_test_csr::<32>(
        2,
        5,
        &[&[1.0, 2.0, 3.0], &[4.0, 5.0]],
        &[&[2, 0, 4], &[3, 1]],
    );
    assert!(!csr_unsorted.is_sorted());
}

#[test]
fn test_csr_ensure_sorted() {
    // 创建未排序的 CSR
    let mut csr = make_test_csr::<32>(
        2,
        5,
        &[&[2.0, 1.0, 3.0], &[5.0, 4.0]],
        &[&[2, 0, 4], &[3, 1]],
    );

    assert!(!csr.is_sorted());

    // 排序
    csr.ensure_sorted();

    assert!(csr.is_sorted());

    // 验证排序后的数据
    assert_eq!(csr.row_indices(0), &[0, 2, 4]);
    assert_eq!(csr.row_values(0), &[1.0, 2.0, 3.0]);
    assert_eq!(csr.row_indices(1), &[1, 3]);
    assert_eq!(csr.row_values(1), &[4.0, 5.0]);
}

#[test]
fn test_csr_validate() {
    let csr = make_test_csr::<32>(3, 4, &[&[1.0, 2.0], &[3.0], &[4.0]], &[&[0, 2], &[1], &[3]]);
    assert!(csr.validate());
}

// =============================================================================
// CSR Clone 测试
// =============================================================================

#[test]
fn test_csr_clone() {
    let csr1 = make_test_csr::<32>(2, 3, &[&[1.0, 2.0], &[3.0]], &[&[0, 1], &[2]]);
    let csr2 = csr1.clone();

    assert_eq!(csr1.nrows(), csr2.nrows());
    assert_eq!(csr1.ncols(), csr2.ncols());
    assert_eq!(csr1.nnz(), csr2.nnz());

    // Clone 会共享底层 Storage（引用计数增加）
    assert_eq!(csr1.row_values(0), csr2.row_values(0));
}

// =============================================================================
// CSC 基础测试
// =============================================================================

#[test]
fn test_csc_new_empty() {
    let csc: CSC<f64, i64> = CSC::new(5, 10);
    assert_eq!(csc.nrows(), 5);
    assert_eq!(csc.ncols(), 10);
    assert_eq!(csc.nnz(), 0);
    assert!(csc.is_zero());
}

#[test]
fn test_csc_basic_construction() {
    // 创建一个 3x4 矩阵（按列存储）:
    // [1.0, 0.0, 4.0, 0.0]
    // [2.0, 0.0, 0.0, 5.0]
    // [0.0, 3.0, 0.0, 6.0]
    let csc = make_test_csc::<32>(
        3,
        4,
        &[&[1.0, 2.0], &[3.0], &[4.0], &[5.0, 6.0]],
        &[&[0, 1], &[2], &[0], &[1, 2]],
    );

    assert_eq!(csc.nrows(), 3);
    assert_eq!(csc.ncols(), 4);
    assert_eq!(csc.shape(), (3, 4));
    assert_eq!(csc.nnz(), 6);
}

#[test]
fn test_csc_col_access() {
    let csc = make_test_csc::<32>(
        3,
        4,
        &[&[1.0, 2.0], &[3.0], &[4.0], &[5.0, 6.0]],
        &[&[0, 1], &[2], &[0], &[1, 2]],
    );

    // 第 0 列
    assert_eq!(csc.col_values(0), &[1.0, 2.0]);
    assert_eq!(csc.col_indices(0), &[0, 1]);
    assert_eq!(csc.col_nnz(0), 2);

    // 第 1 列
    assert_eq!(csc.col_values(1), &[3.0]);
    assert_eq!(csc.col_indices(1), &[2]);

    // 第 3 列
    let (vals, idxs) = csc.col(3);
    assert_eq!(vals, &[5.0, 6.0]);
    assert_eq!(idxs, &[1, 2]);
}

#[test]
fn test_csc_with_empty_cols() {
    let csc = make_test_csc::<32>(3, 4, &[&[1.0], &[], &[2.0], &[]], &[&[0], &[], &[1], &[]]);

    assert_eq!(csc.ncols(), 4);
    assert_eq!(csc.nnz(), 2);

    assert_eq!(csc.col_nnz(0), 1);
    assert_eq!(csc.col_nnz(1), 0);
    assert_eq!(csc.col_nnz(2), 1);
    assert_eq!(csc.col_nnz(3), 0);
}

#[test]
fn test_csc_iterators() {
    let csc = make_test_csc::<32>(3, 3, &[&[1.0], &[2.0, 3.0], &[4.0]], &[&[0], &[1, 2], &[0]]);

    let cols: Vec<(&[f64], &[i64])> = csc.iter_cols().collect();
    assert_eq!(cols.len(), 3);
    assert_eq!(cols[0].0, &[1.0]);
    assert_eq!(cols[1].0, &[2.0, 3.0]);
}

// =============================================================================
// CSC 验证测试
// =============================================================================

#[test]
fn test_csc_is_valid() {
    let csc = make_test_csc::<32>(3, 3, &[&[1.0], &[2.0], &[3.0]], &[&[0], &[1], &[2]]);
    assert!(csc.is_valid());
}

#[test]
fn test_csc_is_sorted() {
    let csc = make_test_csc::<32>(3, 2, &[&[1.0, 2.0], &[3.0, 4.0]], &[&[0, 2], &[1, 2]]);
    assert!(csc.is_sorted());

    let csc_unsorted = make_test_csc::<32>(3, 2, &[&[1.0, 2.0], &[3.0, 4.0]], &[&[2, 0], &[2, 1]]);
    assert!(!csc_unsorted.is_sorted());
}

#[test]
fn test_csc_ensure_sorted() {
    let mut csc = make_test_csc::<32>(3, 2, &[&[2.0, 1.0], &[4.0, 3.0]], &[&[2, 0], &[2, 1]]);

    assert!(!csc.is_sorted());
    csc.ensure_sorted();
    assert!(csc.is_sorted());

    assert_eq!(csc.col_indices(0), &[0, 2]);
    assert_eq!(csc.col_values(0), &[1.0, 2.0]);
}

// =============================================================================
// SparseIndex trait 测试
// =============================================================================

#[test]
fn test_sparse_index_i32() {
    assert_eq!(i32::ZERO, 0);
    assert_eq!(i32::ONE, 1);
    assert_eq!(i32::MAX, i32::MAX);
    assert_eq!(i32::from_usize(42), 42i32);
    assert_eq!(42i32.to_usize(), 42usize);
}

#[test]
fn test_sparse_index_i64() {
    assert_eq!(i64::ZERO, 0);
    assert_eq!(i64::ONE, 1);
    assert_eq!(i64::from_usize(100), 100i64);
    assert_eq!(100i64.to_usize(), 100usize);
}

#[test]
fn test_sparse_index_usize() {
    assert_eq!(usize::ZERO, 0);
    assert_eq!(usize::ONE, 1);
    assert_eq!(usize::from_usize(999), 999);
    assert_eq!(999usize.to_usize(), 999);
}

// =============================================================================
// 边界情况测试
// =============================================================================

#[test]
fn test_csr_single_element() {
    let csr = make_test_csr::<32>(1, 1, &[&[42.0]], &[&[0]]);
    assert_eq!(csr.nrows(), 1);
    assert_eq!(csr.ncols(), 1);
    assert_eq!(csr.nnz(), 1);
    assert_eq!(csr.row_values(0), &[42.0]);
}

#[test]
fn test_csr_all_empty_rows() {
    let csr = make_test_csr::<32>(3, 3, &[&[], &[], &[]], &[&[], &[], &[]]);
    assert_eq!(csr.nnz(), 0);
    assert!(csr.is_zero());
    assert!(csr.is_sorted()); // 空矩阵认为是已排序的
}

#[test]
fn test_csc_single_element() {
    let csc = make_test_csc::<32>(1, 1, &[&[42.0]], &[&[0]]);
    assert_eq!(csc.nrows(), 1);
    assert_eq!(csc.ncols(), 1);
    assert_eq!(csc.nnz(), 1);
}

#[test]
fn test_large_matrix() {
    // 创建一个较大的对角矩阵
    let n = 100;
    let values: Vec<Vec<f64>> = (0..n).map(|i| vec![(i + 1) as f64]).collect();
    let indices: Vec<Vec<i64>> = (0..n).map(|i| vec![i as i64]).collect();

    let values_refs: Vec<&[f64]> = values.iter().map(|v| v.as_slice()).collect();
    let indices_refs: Vec<&[i64]> = indices.iter().map(|v| v.as_slice()).collect();

    let csr = make_test_csr::<32>(n, n, &values_refs, &indices_refs);

    assert_eq!(csr.nrows(), n as i64);
    assert_eq!(csr.ncols(), n as i64);
    assert_eq!(csr.nnz(), n as i64);
    assert!(csr.is_sorted());
    assert!(csr.is_valid());

    // 验证对角元素
    for i in 0..n {
        assert_eq!(csr.row_values(i as i64), &[(i + 1) as f64]);
        assert_eq!(csr.row_indices(i as i64), &[i as i64]);
    }
}
