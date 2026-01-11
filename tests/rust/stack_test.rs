//! 稀疏矩阵堆叠操作测试
//!
//! 测试内容：
//! - CSR vstack（垂直堆叠）
//! - CSR hstack（水平堆叠）
//! - CSC vstack
//! - CSC hstack
//! - 边界情况和错误处理

use std::ptr::NonNull;

use scl_core::convert::AllocStrategy;
use scl_core::span::{Span, SpanFlags};
use scl_core::sparse::{CSC, CSR};

// =============================================================================
// 辅助函数
// =============================================================================

/// 创建空 Span
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
// CSR vstack 测试
// =============================================================================

#[test]
fn test_csr_vstack_basic() {
    // 创建两个 CSR 矩阵
    // A: 2x3
    // [1, 2, 0]
    // [0, 3, 0]
    let a = make_test_csr::<32>(2, 3, &[&[1.0, 2.0], &[3.0]], &[&[0, 1], &[1]]);

    // B: 2x3
    // [0, 0, 4]
    // [5, 6, 0]
    let b = make_test_csr::<32>(2, 3, &[&[4.0], &[5.0, 6.0]], &[&[2], &[0, 1]]);

    // vstack
    let vstacked = CSR::vstack(&[&a, &b]).unwrap();

    assert_eq!(vstacked.nrows(), 4);
    assert_eq!(vstacked.ncols(), 3);
    assert_eq!(vstacked.nnz(), 6);

    // 验证数据
    assert_eq!(vstacked.row_values(0), &[1.0, 2.0]);
    assert_eq!(vstacked.row_indices(0), &[0, 1]);
    assert_eq!(vstacked.row_values(1), &[3.0]);
    assert_eq!(vstacked.row_indices(1), &[1]);
    assert_eq!(vstacked.row_values(2), &[4.0]);
    assert_eq!(vstacked.row_indices(2), &[2]);
    assert_eq!(vstacked.row_values(3), &[5.0, 6.0]);
    assert_eq!(vstacked.row_indices(3), &[0, 1]);
}

#[test]
fn test_csr_vstack_single_matrix() {
    let a = make_test_csr::<32>(2, 3, &[&[1.0, 2.0], &[3.0]], &[&[0, 1], &[2]]);

    let vstacked = CSR::vstack(&[&a]).unwrap();

    assert_eq!(vstacked.nrows(), 2);
    assert_eq!(vstacked.ncols(), 3);
    assert_eq!(vstacked.nnz(), 3);
}

#[test]
fn test_csr_vstack_many_matrices() {
    // 创建多个小矩阵
    let matrices: Vec<CSR<f64, i64>> = (0..10)
        .map(|i| make_test_csr::<32>(1, 5, &[&[(i + 1) as f64]], &[&[(i % 5) as i64]]))
        .collect();

    let refs: Vec<&CSR<f64, i64>> = matrices.iter().collect();
    let vstacked = CSR::vstack(&refs).unwrap();

    assert_eq!(vstacked.nrows(), 10);
    assert_eq!(vstacked.ncols(), 5);
    assert_eq!(vstacked.nnz(), 10);
}

#[test]
fn test_csr_vstack_with_empty_rows() {
    // A: 带空行
    let a = make_test_csr::<32>(3, 3, &[&[1.0], &[], &[2.0]], &[&[0], &[], &[2]]);

    // B: 带空行
    let b = make_test_csr::<32>(2, 3, &[&[], &[3.0]], &[&[], &[1]]);

    let vstacked = CSR::vstack(&[&a, &b]).unwrap();

    assert_eq!(vstacked.nrows(), 5);
    assert_eq!(vstacked.ncols(), 3);
    assert_eq!(vstacked.nnz(), 3);

    // 验证空行保持
    assert_eq!(vstacked.row_nnz(1), 0);
    assert_eq!(vstacked.row_nnz(3), 0);
}

#[test]
fn test_csr_vstack_empty() {
    let result = CSR::<f64, i64>::vstack(&[]);
    assert!(result.is_err());
}

#[test]
fn test_csr_vstack_dimension_mismatch() {
    let a = make_test_csr::<32>(2, 3, &[&[1.0], &[2.0]], &[&[0], &[1]]);
    let b = make_test_csr::<32>(2, 4, &[&[3.0], &[4.0]], &[&[0], &[1]]); // 不同列数

    let result = CSR::vstack(&[&a, &b]);
    assert!(result.is_err());
}

#[test]
fn test_csr_vstack_zero_matrix() {
    // 零矩阵
    let a = make_test_csr::<32>(2, 3, &[&[], &[]], &[&[], &[]]);
    let b = make_test_csr::<32>(1, 3, &[&[1.0]], &[&[0]]);

    let vstacked = CSR::vstack(&[&a, &b]).unwrap();

    assert_eq!(vstacked.nrows(), 3);
    assert_eq!(vstacked.nnz(), 1);
}

// =============================================================================
// CSR hstack 测试
// =============================================================================

#[test]
fn test_csr_hstack_basic() {
    // A: 2x2
    // [1, 0]
    // [2, 3]
    let a = make_test_csr::<32>(2, 2, &[&[1.0], &[2.0, 3.0]], &[&[0], &[0, 1]]);

    // B: 2x3
    // [0, 4, 0]
    // [5, 0, 6]
    let b = make_test_csr::<32>(2, 3, &[&[4.0], &[5.0, 6.0]], &[&[1], &[0, 2]]);

    // hstack
    let hstacked = CSR::hstack::<32>(&[&a, &b], AllocStrategy::Auto).unwrap();

    assert_eq!(hstacked.nrows(), 2);
    assert_eq!(hstacked.ncols(), 5);
    assert_eq!(hstacked.nnz(), 6);

    // 验证第一行：[1, 0, 0, 4, 0]
    assert_eq!(hstacked.row_values(0), &[1.0, 4.0]);
    assert_eq!(hstacked.row_indices(0), &[0, 3]); // 0 和 2+1=3

    // 验证第二行：[2, 3, 5, 0, 6]
    assert_eq!(hstacked.row_values(1), &[2.0, 3.0, 5.0, 6.0]);
    assert_eq!(hstacked.row_indices(1), &[0, 1, 2, 4]); // 0, 1, 2+0=2, 2+2=4
}

#[test]
fn test_csr_hstack_single_matrix() {
    let a = make_test_csr::<32>(2, 3, &[&[1.0, 2.0], &[3.0]], &[&[0, 1], &[2]]);

    let hstacked = CSR::hstack::<32>(&[&a], AllocStrategy::Auto).unwrap();

    assert_eq!(hstacked.nrows(), 2);
    assert_eq!(hstacked.ncols(), 3);
    assert_eq!(hstacked.nnz(), 3);
}

#[test]
fn test_csr_hstack_many_matrices() {
    // 多个列向量
    let matrices: Vec<CSR<f64, i64>> = (0..5)
        .map(|i| {
            make_test_csr::<32>(
                3,
                1,
                &[&[(i + 1) as f64], &[], &[(i + 10) as f64]],
                &[&[0], &[], &[0]],
            )
        })
        .collect();

    let refs: Vec<&CSR<f64, i64>> = matrices.iter().collect();
    let hstacked = CSR::hstack::<32>(&refs, AllocStrategy::Auto).unwrap();

    assert_eq!(hstacked.nrows(), 3);
    assert_eq!(hstacked.ncols(), 5);
    assert_eq!(hstacked.nnz(), 10);
}

#[test]
fn test_csr_hstack_with_empty_rows() {
    let a = make_test_csr::<32>(3, 2, &[&[1.0], &[], &[2.0]], &[&[0], &[], &[1]]);
    let b = make_test_csr::<32>(3, 2, &[&[3.0], &[], &[4.0]], &[&[1], &[], &[0]]);

    let hstacked = CSR::hstack::<32>(&[&a, &b], AllocStrategy::Auto).unwrap();

    assert_eq!(hstacked.nrows(), 3);
    assert_eq!(hstacked.ncols(), 4);
    assert_eq!(hstacked.nnz(), 4);

    // 空行保持为空
    assert_eq!(hstacked.row_nnz(1), 0);
}

#[test]
fn test_csr_hstack_empty() {
    let result = CSR::<f64, i64>::hstack::<32>(&[], AllocStrategy::Auto);
    assert!(result.is_err());
}

#[test]
fn test_csr_hstack_dimension_mismatch() {
    let a = make_test_csr::<32>(2, 3, &[&[1.0], &[2.0]], &[&[0], &[1]]);
    let b = make_test_csr::<32>(3, 3, &[&[3.0], &[4.0], &[5.0]], &[&[0], &[1], &[2]]); // 不同行数

    let result = CSR::hstack::<32>(&[&a, &b], AllocStrategy::Auto);
    assert!(result.is_err());
}

// =============================================================================
// CSC vstack 测试
// =============================================================================

#[test]
fn test_csc_vstack_basic() {
    // A: 2x3
    let a = make_test_csc::<32>(2, 3, &[&[1.0, 2.0], &[3.0], &[]], &[&[0, 1], &[0], &[]]);

    // B: 2x3
    let b = make_test_csc::<32>(2, 3, &[&[4.0], &[5.0, 6.0], &[7.0]], &[&[0], &[0, 1], &[1]]);

    let vstacked = CSC::vstack::<32>(&[&a, &b], AllocStrategy::Auto).unwrap();

    assert_eq!(vstacked.nrows(), 4);
    assert_eq!(vstacked.ncols(), 3);
    assert_eq!(vstacked.nnz(), 7);

    // 验证列 0：原 A 的列 0 有 [0, 1]，原 B 的列 0 有 [0] -> [0, 1, 2]
    let (vals, idxs) = vstacked.col(0);
    assert_eq!(vals, &[1.0, 2.0, 4.0]);
    assert_eq!(idxs, &[0, 1, 2]);
}

#[test]
fn test_csc_vstack_single_matrix() {
    let a = make_test_csc::<32>(2, 3, &[&[1.0], &[2.0], &[3.0]], &[&[0], &[1], &[0]]);

    let vstacked = CSC::vstack::<32>(&[&a], AllocStrategy::Auto).unwrap();

    assert_eq!(vstacked.nrows(), 2);
    assert_eq!(vstacked.ncols(), 3);
    assert_eq!(vstacked.nnz(), 3);
}

#[test]
fn test_csc_vstack_empty() {
    let result = CSC::<f64, i64>::vstack::<32>(&[], AllocStrategy::Auto);
    assert!(result.is_err());
}

#[test]
fn test_csc_vstack_dimension_mismatch() {
    let a = make_test_csc::<32>(2, 3, &[&[1.0], &[], &[]], &[&[0], &[], &[]]);
    let b = make_test_csc::<32>(2, 4, &[&[2.0], &[], &[], &[]], &[&[0], &[], &[], &[]]); // 不同列数

    let result = CSC::vstack::<32>(&[&a, &b], AllocStrategy::Auto);
    assert!(result.is_err());
}

// =============================================================================
// CSC hstack 测试
// =============================================================================

#[test]
fn test_csc_hstack_basic() {
    // A: 3x2
    let a = make_test_csc::<32>(3, 2, &[&[1.0, 2.0], &[3.0]], &[&[0, 1], &[2]]);

    // B: 3x2
    let b = make_test_csc::<32>(3, 2, &[&[4.0], &[5.0, 6.0]], &[&[0], &[1, 2]]);

    let hstacked = CSC::hstack(&[&a, &b]).unwrap();

    assert_eq!(hstacked.nrows(), 3);
    assert_eq!(hstacked.ncols(), 4);
    assert_eq!(hstacked.nnz(), 6);

    // 验证列
    assert_eq!(hstacked.col_values(0), &[1.0, 2.0]);
    assert_eq!(hstacked.col_values(1), &[3.0]);
    assert_eq!(hstacked.col_values(2), &[4.0]);
    assert_eq!(hstacked.col_values(3), &[5.0, 6.0]);
}

#[test]
fn test_csc_hstack_single_matrix() {
    let a = make_test_csc::<32>(3, 2, &[&[1.0], &[2.0]], &[&[0], &[1]]);

    let hstacked = CSC::hstack(&[&a]).unwrap();

    assert_eq!(hstacked.nrows(), 3);
    assert_eq!(hstacked.ncols(), 2);
    assert_eq!(hstacked.nnz(), 2);
}

#[test]
fn test_csc_hstack_many_matrices() {
    let matrices: Vec<CSC<f64, i64>> = (0..5)
        .map(|i| make_test_csc::<32>(3, 1, &[&[(i + 1) as f64]], &[&[0]]))
        .collect();

    let refs: Vec<&CSC<f64, i64>> = matrices.iter().collect();
    let hstacked = CSC::hstack(&refs).unwrap();

    assert_eq!(hstacked.nrows(), 3);
    assert_eq!(hstacked.ncols(), 5);
    assert_eq!(hstacked.nnz(), 5);
}

#[test]
fn test_csc_hstack_empty() {
    let result = CSC::<f64, i64>::hstack(&[]);
    assert!(result.is_err());
}

#[test]
fn test_csc_hstack_dimension_mismatch() {
    let a = make_test_csc::<32>(3, 2, &[&[1.0], &[]], &[&[0], &[]]);
    let b = make_test_csc::<32>(4, 2, &[&[2.0], &[]], &[&[0], &[]]); // 不同行数

    let result = CSC::hstack(&[&a, &b]);
    assert!(result.is_err());
}

// =============================================================================
// 边界情况测试
// =============================================================================

#[test]
fn test_stack_all_zero_matrices() {
    let a = make_test_csr::<32>(2, 3, &[&[], &[]], &[&[], &[]]);
    let b = make_test_csr::<32>(2, 3, &[&[], &[]], &[&[], &[]]);

    let vstacked = CSR::vstack(&[&a, &b]).unwrap();
    assert_eq!(vstacked.nrows(), 4);
    assert_eq!(vstacked.nnz(), 0);
    assert!(vstacked.is_zero());

    let hstacked = CSR::hstack::<32>(&[&a, &b], AllocStrategy::Auto).unwrap();
    assert_eq!(hstacked.ncols(), 6);
    assert_eq!(hstacked.nnz(), 0);
    assert!(hstacked.is_zero());
}

#[test]
fn test_stack_single_row_matrices() {
    let a = make_test_csr::<32>(1, 3, &[&[1.0, 2.0]], &[&[0, 2]]);
    let b = make_test_csr::<32>(1, 3, &[&[3.0]], &[&[1]]);
    let c = make_test_csr::<32>(1, 3, &[&[4.0, 5.0, 6.0]], &[&[0, 1, 2]]);

    let vstacked = CSR::vstack(&[&a, &b, &c]).unwrap();
    assert_eq!(vstacked.nrows(), 3);
    assert_eq!(vstacked.ncols(), 3);
    assert_eq!(vstacked.nnz(), 6);
}

#[test]
fn test_stack_single_column_matrices() {
    let a = make_test_csr::<32>(3, 1, &[&[1.0], &[2.0], &[]], &[&[0], &[0], &[]]);
    let b = make_test_csr::<32>(3, 1, &[&[], &[3.0], &[4.0]], &[&[], &[0], &[0]]);

    let hstacked = CSR::hstack::<32>(&[&a, &b], AllocStrategy::Auto).unwrap();
    assert_eq!(hstacked.nrows(), 3);
    assert_eq!(hstacked.ncols(), 2);
    assert_eq!(hstacked.nnz(), 4);
}

#[test]
fn test_stack_large_matrices() {
    // 创建较大的矩阵
    let n = 100;
    let values: Vec<Vec<f64>> = (0..n).map(|i| vec![(i + 1) as f64]).collect();
    let indices: Vec<Vec<i64>> = (0..n).map(|i| vec![i as i64]).collect();

    let values_refs: Vec<&[f64]> = values.iter().map(|v| v.as_slice()).collect();
    let indices_refs: Vec<&[i64]> = indices.iter().map(|v| v.as_slice()).collect();

    let a = make_test_csr::<32>(n, n, &values_refs, &indices_refs);
    let b = make_test_csr::<32>(n, n, &values_refs, &indices_refs);

    let vstacked = CSR::vstack(&[&a, &b]).unwrap();
    assert_eq!(vstacked.nrows() as usize, n * 2);
    assert_eq!(vstacked.nnz() as usize, n * 2);
}

#[test]
fn test_stack_preserves_sorting() {
    let a = make_test_csr::<32>(2, 5, &[&[1.0, 2.0], &[3.0, 4.0]], &[&[0, 2], &[1, 4]]);
    let b = make_test_csr::<32>(2, 5, &[&[5.0], &[6.0, 7.0]], &[&[3], &[0, 2]]);

    assert!(a.is_sorted());
    assert!(b.is_sorted());

    let vstacked = CSR::vstack(&[&a, &b]).unwrap();
    assert!(vstacked.is_sorted());

    let hstacked = CSR::hstack::<32>(&[&a, &b], AllocStrategy::Auto).unwrap();
    assert!(hstacked.is_sorted());
}

// =============================================================================
// 数据正确性验证测试
// =============================================================================

#[test]
fn test_vstack_data_correctness() {
    // 精确验证数据
    let a = make_test_csr::<32>(2, 4, &[&[1.0, 2.0, 3.0], &[4.0]], &[&[0, 1, 3], &[2]]);
    let b = make_test_csr::<32>(
        2,
        4,
        &[&[5.0, 6.0], &[7.0, 8.0, 9.0]],
        &[&[1, 2], &[0, 2, 3]],
    );

    let vstacked = CSR::vstack(&[&a, &b]).unwrap();

    // 行 0: [1, 2, 0, 3]
    assert_eq!(vstacked.row_values(0), &[1.0, 2.0, 3.0]);
    assert_eq!(vstacked.row_indices(0), &[0, 1, 3]);

    // 行 1: [0, 0, 4, 0]
    assert_eq!(vstacked.row_values(1), &[4.0]);
    assert_eq!(vstacked.row_indices(1), &[2]);

    // 行 2: [0, 5, 6, 0]
    assert_eq!(vstacked.row_values(2), &[5.0, 6.0]);
    assert_eq!(vstacked.row_indices(2), &[1, 2]);

    // 行 3: [7, 0, 8, 9]
    assert_eq!(vstacked.row_values(3), &[7.0, 8.0, 9.0]);
    assert_eq!(vstacked.row_indices(3), &[0, 2, 3]);
}

#[test]
fn test_hstack_data_correctness() {
    // 精确验证数据
    let a = make_test_csr::<32>(3, 2, &[&[1.0, 2.0], &[3.0], &[]], &[&[0, 1], &[0], &[]]);
    let b = make_test_csr::<32>(3, 3, &[&[4.0], &[5.0, 6.0], &[7.0]], &[&[2], &[0, 1], &[1]]);

    let hstacked = CSR::hstack::<32>(&[&a, &b], AllocStrategy::Auto).unwrap();

    // 行 0: [1, 2] + [0, 0, 4] = [1, 2, 0, 0, 4]
    assert_eq!(hstacked.row_values(0), &[1.0, 2.0, 4.0]);
    assert_eq!(hstacked.row_indices(0), &[0, 1, 4]); // 2+2=4

    // 行 1: [3, 0] + [5, 6, 0] = [3, 0, 5, 6, 0]
    assert_eq!(hstacked.row_values(1), &[3.0, 5.0, 6.0]);
    assert_eq!(hstacked.row_indices(1), &[0, 2, 3]); // 0, 2+0=2, 2+1=3

    // 行 2: [0, 0] + [0, 7, 0] = [0, 0, 0, 7, 0]
    assert_eq!(hstacked.row_values(2), &[7.0]);
    assert_eq!(hstacked.row_indices(2), &[3]); // 2+1=3
}

// =============================================================================
// 三个及以上矩阵堆叠测试
// =============================================================================

#[test]
fn test_vstack_three_matrices() {
    let a = make_test_csr::<32>(1, 3, &[&[1.0]], &[&[0]]);
    let b = make_test_csr::<32>(2, 3, &[&[2.0], &[3.0]], &[&[1], &[2]]);
    let c = make_test_csr::<32>(1, 3, &[&[4.0, 5.0]], &[&[0, 2]]);

    let vstacked = CSR::vstack(&[&a, &b, &c]).unwrap();

    assert_eq!(vstacked.nrows(), 4);
    assert_eq!(vstacked.ncols(), 3);
    assert_eq!(vstacked.nnz(), 5);

    assert_eq!(vstacked.row_values(0), &[1.0]);
    assert_eq!(vstacked.row_values(1), &[2.0]);
    assert_eq!(vstacked.row_values(2), &[3.0]);
    assert_eq!(vstacked.row_values(3), &[4.0, 5.0]);
}

#[test]
fn test_hstack_three_matrices() {
    let a = make_test_csr::<32>(2, 1, &[&[1.0], &[2.0]], &[&[0], &[0]]);
    let b = make_test_csr::<32>(2, 2, &[&[3.0], &[4.0, 5.0]], &[&[1], &[0, 1]]);
    let c = make_test_csr::<32>(2, 1, &[&[], &[6.0]], &[&[], &[0]]);

    let hstacked = CSR::hstack::<32>(&[&a, &b, &c], AllocStrategy::Auto).unwrap();

    assert_eq!(hstacked.nrows(), 2);
    assert_eq!(hstacked.ncols(), 4);
    assert_eq!(hstacked.nnz(), 6);
}
