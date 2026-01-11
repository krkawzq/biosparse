//! 稀疏矩阵切片操作测试
//!
//! 测试内容：
//! - CSR 行切片（主轴，零拷贝）
//! - CSR 列切片（副轴，需要过滤）
//! - CSC 列切片（主轴，零拷贝）
//! - CSC 行切片（副轴，需要过滤）
//! - 掩码切片
//! - 边界情况

use std::ptr::NonNull;

use scl_core::slice::SliceError;
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
// CSR 行切片测试（主轴，零拷贝）
// =============================================================================

#[test]
fn test_csr_slice_rows_basic() {
    // 5x4 矩阵
    let csr = make_test_csr::<32>(
        5,
        4,
        &[&[1.0, 2.0], &[3.0], &[4.0, 5.0], &[], &[6.0]],
        &[&[0, 1], &[2], &[0, 3], &[], &[1]],
    );

    // 切片行 [1, 4)
    let sliced = csr.slice_rows(1, 4).unwrap();

    assert_eq!(sliced.nrows(), 3);
    assert_eq!(sliced.ncols(), 4);

    // 原第 1 行 -> 新第 0 行
    assert_eq!(sliced.row_values(0), &[3.0]);
    assert_eq!(sliced.row_indices(0), &[2]);

    // 原第 2 行 -> 新第 1 行
    assert_eq!(sliced.row_values(1), &[4.0, 5.0]);
    assert_eq!(sliced.row_indices(1), &[0, 3]);

    // 原第 3 行 -> 新第 2 行（空）
    assert_eq!(sliced.row_nnz(2), 0);
}

#[test]
fn test_csr_slice_rows_single_row() {
    let csr = make_test_csr::<32>(3, 4, &[&[1.0], &[2.0, 3.0], &[4.0]], &[&[0], &[1, 2], &[3]]);

    let sliced = csr.slice_rows(1, 2).unwrap();

    assert_eq!(sliced.nrows(), 1);
    assert_eq!(sliced.row_values(0), &[2.0, 3.0]);
}

#[test]
fn test_csr_slice_rows_all() {
    let csr = make_test_csr::<32>(3, 4, &[&[1.0], &[2.0], &[3.0]], &[&[0], &[1], &[2]]);

    let sliced = csr.slice_rows(0, 3).unwrap();

    assert_eq!(sliced.nrows(), 3);
    assert_eq!(sliced.nnz(), 3);
}

#[test]
fn test_csr_slice_rows_empty_range() {
    let csr = make_test_csr::<32>(3, 4, &[&[1.0], &[2.0], &[3.0]], &[&[0], &[1], &[2]]);

    let sliced = csr.slice_rows(2, 2).unwrap();

    assert_eq!(sliced.nrows(), 0);
    assert_eq!(sliced.nnz(), 0);
}

#[test]
fn test_csr_slice_rows_out_of_bounds() {
    let csr = make_test_csr::<32>(3, 4, &[&[1.0], &[], &[]], &[&[0], &[], &[]]);

    let result = csr.slice_rows(0, 5); // 超出范围
    assert!(matches!(result, Err(SliceError::OutOfBounds { .. })));

    let result = csr.slice_rows(2, 1); // start > end
    assert!(matches!(result, Err(SliceError::OutOfBounds { .. })));
}

#[test]
fn test_csr_slice_rows_mask_basic() {
    let csr = make_test_csr::<32>(
        5,
        4,
        &[&[1.0], &[2.0], &[3.0], &[4.0], &[5.0]],
        &[&[0], &[1], &[2], &[3], &[0]],
    );

    // 选择行 0, 2, 4
    let mask = vec![true, false, true, false, true];
    let sliced = csr.slice_rows_mask(&mask).unwrap();

    assert_eq!(sliced.nrows(), 3);
    assert_eq!(sliced.nnz(), 3);

    assert_eq!(sliced.row_values(0), &[1.0]);
    assert_eq!(sliced.row_values(1), &[3.0]);
    assert_eq!(sliced.row_values(2), &[5.0]);
}

#[test]
fn test_csr_slice_rows_mask_none_selected() {
    let csr = make_test_csr::<32>(3, 3, &[&[1.0], &[2.0], &[3.0]], &[&[0], &[1], &[2]]);

    let mask = vec![false, false, false];
    let sliced = csr.slice_rows_mask(&mask).unwrap();

    assert_eq!(sliced.nrows(), 0);
    assert_eq!(sliced.nnz(), 0);
}

#[test]
fn test_csr_slice_rows_mask_all_selected() {
    let csr = make_test_csr::<32>(3, 3, &[&[1.0], &[2.0], &[3.0]], &[&[0], &[1], &[2]]);

    let mask = vec![true, true, true];
    let sliced = csr.slice_rows_mask(&mask).unwrap();

    assert_eq!(sliced.nrows(), 3);
    assert_eq!(sliced.nnz(), 3);
}

#[test]
fn test_csr_slice_rows_mask_length_mismatch() {
    let csr = make_test_csr::<32>(3, 3, &[&[1.0], &[], &[]], &[&[0], &[], &[]]);

    let mask = vec![true, false]; // 长度不匹配
    let result = csr.slice_rows_mask(&mask);

    assert!(matches!(result, Err(SliceError::MaskLengthMismatch { .. })));
}

// =============================================================================
// CSR 列切片测试（副轴，需要过滤）
// =============================================================================

#[test]
fn test_csr_slice_cols_basic() {
    // 3x5 矩阵（索引已排序）
    let csr = make_test_csr::<32>(
        3,
        5,
        &[&[1.0, 2.0, 3.0], &[4.0, 5.0], &[6.0, 7.0, 8.0]],
        &[&[0, 2, 4], &[1, 3], &[0, 2, 4]],
    );

    // 切片列 [1, 4)，即列 1, 2, 3
    let sliced = csr.slice_cols::<32>(1, 4).unwrap();

    assert_eq!(sliced.nrows(), 3);
    assert_eq!(sliced.ncols(), 3);

    // 行 0: 原列 2 -> 新列 1
    assert_eq!(sliced.row_values(0), &[2.0]);
    assert_eq!(sliced.row_indices(0), &[1]); // 2 - 1 = 1

    // 行 1: 原列 1, 3 -> 新列 0, 2
    assert_eq!(sliced.row_values(1), &[4.0, 5.0]);
    assert_eq!(sliced.row_indices(1), &[0, 2]);

    // 行 2: 原列 2 -> 新列 1
    assert_eq!(sliced.row_values(2), &[7.0]);
    assert_eq!(sliced.row_indices(2), &[1]);
}

#[test]
fn test_csr_slice_cols_empty_result() {
    let csr = make_test_csr::<32>(
        2,
        5,
        &[&[1.0], &[2.0]],
        &[&[0], &[4]], // 只有列 0 和 4
    );

    // 切片列 [1, 4)，原数据中这个范围没有元素
    let sliced = csr.slice_cols::<32>(1, 4).unwrap();

    assert_eq!(sliced.nrows(), 2);
    assert_eq!(sliced.ncols(), 3);
    assert_eq!(sliced.nnz(), 0);
}

#[test]
fn test_csr_slice_cols_out_of_bounds() {
    let csr = make_test_csr::<32>(2, 4, &[&[1.0], &[]], &[&[0], &[]]);

    let result = csr.slice_cols::<32>(0, 5);
    assert!(matches!(result, Err(SliceError::OutOfBounds { .. })));

    let result = csr.slice_cols::<32>(3, 2);
    assert!(matches!(result, Err(SliceError::OutOfBounds { .. })));
}

#[test]
fn test_csr_slice_cols_mask_basic() {
    let csr = make_test_csr::<32>(
        2,
        5,
        &[&[1.0, 2.0, 3.0, 4.0, 5.0], &[6.0, 7.0]],
        &[&[0, 1, 2, 3, 4], &[1, 3]],
    );

    // 选择列 0, 2, 4
    let mask = vec![true, false, true, false, true];
    let sliced = csr.slice_cols_mask::<32>(&mask).unwrap();

    assert_eq!(sliced.nrows(), 2);
    assert_eq!(sliced.ncols(), 3);

    // 行 0: 原列 0, 2, 4 -> 新列 0, 1, 2
    assert_eq!(sliced.row_values(0), &[1.0, 3.0, 5.0]);
    assert_eq!(sliced.row_indices(0), &[0, 1, 2]);

    // 行 1: 原列 1, 3 都不在 mask 中
    assert_eq!(sliced.row_nnz(1), 0);
}

// =============================================================================
// CSC 列切片测试（主轴，零拷贝）
// =============================================================================

#[test]
fn test_csc_slice_cols_basic() {
    let csc = make_test_csc::<32>(
        4,
        5,
        &[&[1.0], &[2.0, 3.0], &[4.0], &[], &[5.0, 6.0]],
        &[&[0], &[1, 2], &[3], &[], &[0, 3]],
    );

    // 切片列 [1, 4)
    let sliced = csc.slice_cols(1, 4).unwrap();

    assert_eq!(sliced.nrows(), 4);
    assert_eq!(sliced.ncols(), 3);

    // 原列 1 -> 新列 0
    assert_eq!(sliced.col_values(0), &[2.0, 3.0]);
    assert_eq!(sliced.col_indices(0), &[1, 2]);

    // 原列 2 -> 新列 1
    assert_eq!(sliced.col_values(1), &[4.0]);

    // 原列 3 -> 新列 2（空）
    assert_eq!(sliced.col_nnz(2), 0);
}

#[test]
fn test_csc_slice_cols_mask() {
    let csc = make_test_csc::<32>(
        3,
        4,
        &[&[1.0], &[2.0], &[3.0], &[4.0]],
        &[&[0], &[1], &[2], &[0]],
    );

    let mask = vec![true, false, true, true];
    let sliced = csc.slice_cols_mask(&mask).unwrap();

    assert_eq!(sliced.ncols(), 3);
    assert_eq!(sliced.col_values(0), &[1.0]); // 原列 0
    assert_eq!(sliced.col_values(1), &[3.0]); // 原列 2
    assert_eq!(sliced.col_values(2), &[4.0]); // 原列 3
}

// =============================================================================
// CSC 行切片测试（副轴，需要过滤）
// =============================================================================

#[test]
fn test_csc_slice_rows_basic() {
    let csc = make_test_csc::<32>(
        5,
        3,
        &[&[1.0, 2.0, 3.0], &[4.0, 5.0], &[6.0]],
        &[&[0, 2, 4], &[1, 3], &[2]], // 索引已排序
    );

    // 切片行 [1, 4)
    let sliced = csc.slice_rows::<32>(1, 4).unwrap();

    assert_eq!(sliced.nrows(), 3);
    assert_eq!(sliced.ncols(), 3);

    // 列 0: 原行 2, 4 在范围内，但只有行 2
    // 原索引 2 -> 新索引 2 - 1 = 1
    assert_eq!(sliced.col_values(0), &[2.0]);
    assert_eq!(sliced.col_indices(0), &[1]);

    // 列 1: 原行 1, 3 都在范围内
    // 原索引 1, 3 -> 新索引 0, 2
    assert_eq!(sliced.col_values(1), &[4.0, 5.0]);
    assert_eq!(sliced.col_indices(1), &[0, 2]);

    // 列 2: 原行 2 在范围内
    assert_eq!(sliced.col_values(2), &[6.0]);
    assert_eq!(sliced.col_indices(2), &[1]);
}

#[test]
fn test_csc_slice_rows_mask() {
    let csc = make_test_csc::<32>(
        4,
        2,
        &[&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0]],
        &[&[0, 1, 2, 3], &[1, 3]],
    );

    // 选择行 0, 2
    let mask = vec![true, false, true, false];
    let sliced = csc.slice_rows_mask::<32>(&mask).unwrap();

    assert_eq!(sliced.nrows(), 2);
    assert_eq!(sliced.ncols(), 2);

    // 列 0: 原行 0, 2 -> 新行 0, 1
    assert_eq!(sliced.col_values(0), &[1.0, 3.0]);
    assert_eq!(sliced.col_indices(0), &[0, 1]);

    // 列 1: 原行 1, 3 都不在 mask 中
    assert_eq!(sliced.col_nnz(1), 0);
}

// =============================================================================
// 组合切片测试
// =============================================================================

#[test]
fn test_csr_row_then_col_slice() {
    let csr = make_test_csr::<32>(
        5,
        6,
        &[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0],
            &[6.0, 7.0, 8.0],
            &[9.0],
            &[10.0, 11.0],
        ],
        &[&[0, 2, 4], &[1, 3], &[0, 2, 5], &[1], &[3, 5]],
    );

    // 先行切片 [1, 4)
    let row_sliced = csr.slice_rows(1, 4).unwrap();
    assert_eq!(row_sliced.nrows(), 3);

    // 再列切片 [1, 5)
    let result = row_sliced.slice_cols::<32>(1, 5).unwrap();
    assert_eq!(result.nrows(), 3);
    assert_eq!(result.ncols(), 4);
}

// =============================================================================
// 边界情况测试
// =============================================================================

#[test]
fn test_slice_empty_matrix() {
    let csr: CSR<f64, i64> = CSR::new(0, 0);

    let result = csr.slice_rows(0, 0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().nrows(), 0);
}

#[test]
fn test_slice_single_element() {
    let csr = make_test_csr::<32>(1, 1, &[&[42.0]], &[&[0]]);

    let sliced = csr.slice_rows(0, 1).unwrap();
    assert_eq!(sliced.nrows(), 1);
    assert_eq!(sliced.ncols(), 1);
    assert_eq!(sliced.row_values(0), &[42.0]);
}

#[test]
fn test_slice_preserves_data() {
    // 验证切片后数据正确
    let csr = make_test_csr::<32>(
        4,
        4,
        &[&[1.0, 2.0], &[3.0, 4.0, 5.0], &[6.0], &[7.0, 8.0]],
        &[&[0, 2], &[0, 1, 3], &[2], &[1, 3]],
    );

    // 行切片
    let row_sliced = csr.slice_rows(1, 3).unwrap();
    assert_eq!(row_sliced.row_values(0), &[3.0, 4.0, 5.0]);
    assert_eq!(row_sliced.row_indices(0), &[0, 1, 3]);
    assert_eq!(row_sliced.row_values(1), &[6.0]);
    assert_eq!(row_sliced.row_indices(1), &[2]);
}

#[test]
fn test_slice_large_matrix() {
    // 创建较大的对角矩阵
    let n = 100;
    let values: Vec<Vec<f64>> = (0..n).map(|i| vec![(i + 1) as f64]).collect();
    let indices: Vec<Vec<i64>> = (0..n).map(|i| vec![i as i64]).collect();

    let values_refs: Vec<&[f64]> = values.iter().map(|v| v.as_slice()).collect();
    let indices_refs: Vec<&[i64]> = indices.iter().map(|v| v.as_slice()).collect();

    let csr = make_test_csr::<32>(n, n, &values_refs, &indices_refs);

    // 行切片 [25, 75)
    let sliced = csr.slice_rows(25, 75).unwrap();
    assert_eq!(sliced.nrows(), 50);
    assert_eq!(sliced.nnz(), 50);

    // 验证第一个和最后一个元素
    assert_eq!(sliced.row_values(0), &[26.0]); // 原行 25
    assert_eq!(sliced.row_indices(0), &[25]);

    assert_eq!(sliced.row_values(49), &[75.0]); // 原行 74
    assert_eq!(sliced.row_indices(49), &[74]);
}

// =============================================================================
// 零拷贝验证测试
// =============================================================================

#[test]
fn test_row_slice_is_zero_copy() {
    let csr = make_test_csr::<32>(
        3,
        4,
        &[&[1.0, 2.0], &[3.0], &[4.0, 5.0]],
        &[&[0, 1], &[2], &[0, 3]],
    );

    let sliced = csr.slice_rows(1, 3).unwrap();

    // 验证指针相同（零拷贝）
    // 注意：由于 Span clone 只增加引用计数，指针应该相同
    assert_eq!(
        csr.row_values_span(1).as_ptr(),
        sliced.row_values_span(0).as_ptr()
    );
    assert_eq!(
        csr.row_values_span(2).as_ptr(),
        sliced.row_values_span(1).as_ptr()
    );
}

#[test]
fn test_csc_col_slice_is_zero_copy() {
    let csc = make_test_csc::<32>(
        3,
        4,
        &[&[1.0], &[2.0], &[3.0], &[4.0]],
        &[&[0], &[1], &[2], &[0]],
    );

    let sliced = csc.slice_cols(1, 3).unwrap();

    assert_eq!(
        csc.col_values_span(1).as_ptr(),
        sliced.col_values_span(0).as_ptr()
    );
    assert_eq!(
        csc.col_values_span(2).as_ptr(),
        sliced.col_values_span(1).as_ptr()
    );
}
