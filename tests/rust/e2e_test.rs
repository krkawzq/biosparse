//! 端到端集成测试
//!
//! 测试内容：
//! - 完整工作流程测试
//! - 格式转换往返测试
//! - 大规模数据测试
//! - 边界情况综合测试

use std::ptr::NonNull;

use scl_core::convert::{
    csc_from_csr, csr_from_csc, csr_from_dense, csr_from_scipy_csr_copy, csr_to_dense,
    AllocStrategy, DenseLayout,
};
use scl_core::span::{Span, SpanFlags};
use scl_core::sparse::CSR;

// =============================================================================
// 辅助函数
// =============================================================================

fn empty_span<T>() -> Span<T> {
    let ptr = NonNull::dangling();
    unsafe { Span::from_raw_parts_unchecked(ptr, 0, SpanFlags::VIEW) }
}

fn make_csr(
    rows: usize,
    cols: usize,
    row_data: &[&[f64]],
    row_indices: &[&[i64]],
) -> CSR<f64, i64> {
    let mut values = Vec::with_capacity(rows);
    let mut indices = Vec::with_capacity(rows);

    for i in 0..rows {
        if row_data[i].is_empty() {
            values.push(empty_span());
            indices.push(empty_span());
        } else {
            values.push(Span::copy_from::<32>(row_data[i]).unwrap());
            indices.push(Span::copy_from::<32>(row_indices[i]).unwrap());
        }
    }

    unsafe { CSR::from_raw_parts(values, indices, rows as i64, cols as i64) }
}

// =============================================================================
// 完整工作流程测试
// =============================================================================

#[test]
fn test_e2e_dense_to_csr_to_csc_to_dense() {
    // Dense -> CSR -> CSC -> Dense 完整流程
    let original = vec![
        1.0f64, 0.0, 2.0, 0.0, //
        0.0, 3.0, 0.0, 4.0, //
        5.0, 0.0, 6.0, 0.0, //
    ];

    // Dense -> CSR
    let csr =
        csr_from_dense::<f64, i64, 32>(&original, 3, 4, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nrows(), 3);
    assert_eq!(csr.ncols(), 4);
    assert_eq!(csr.nnz(), 6);

    // CSR -> CSC
    let csc = csc_from_csr::<f64, i64, 32>(&csr, AllocStrategy::Auto).unwrap();

    assert_eq!(csc.nrows(), 3);
    assert_eq!(csc.ncols(), 4);
    assert_eq!(csc.nnz(), 6);

    // CSC -> Dense (通过 CSR)
    let csr_back = csr_from_csc::<f64, i64, 32>(&csc, AllocStrategy::Auto).unwrap();

    let mut recovered = vec![0.0f64; 12];
    csr_to_dense(&csr_back, &mut recovered, DenseLayout::RowMajor).unwrap();

    assert_eq!(original, recovered);
}

#[test]
fn test_e2e_scipy_csr_with_slicing() {
    // 从 scipy 格式创建 -> 切片 -> 验证
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let indices = vec![0i64, 1, 2, 0, 2, 0, 1, 2, 1];
    let indptr = vec![0i64, 3, 5, 9];

    let csr = csr_from_scipy_csr_copy::<f64, i64, 32>(
        3,
        3,
        &data,
        &indices,
        &indptr,
        AllocStrategy::Auto,
    )
    .unwrap();

    // 行切片
    let row_sliced = csr.slice_rows(0, 2).unwrap();
    assert_eq!(row_sliced.nrows(), 2);
    assert_eq!(row_sliced.nnz(), 5);

    // 列切片
    let col_sliced = csr.slice_cols::<32>(1, 3).unwrap();
    assert_eq!(col_sliced.ncols(), 2);
}

#[test]
fn test_e2e_stack_operations() {
    // 创建两个矩阵
    let a = make_csr(2, 3, &[&[1.0, 2.0], &[3.0]], &[&[0, 1], &[2]]);
    let b = make_csr(2, 3, &[&[4.0], &[5.0, 6.0]], &[&[0], &[1, 2]]);

    // vstack
    let vstacked = CSR::vstack(&[&a, &b]).unwrap();
    assert_eq!(vstacked.nrows(), 4);
    assert_eq!(vstacked.ncols(), 3);
    assert_eq!(vstacked.nnz(), 6);

    // hstack
    let c = make_csr(2, 2, &[&[7.0], &[8.0]], &[&[0], &[1]]);
    let hstacked = CSR::hstack::<32>(&[&a, &c], AllocStrategy::Auto).unwrap();
    assert_eq!(hstacked.nrows(), 2);
    assert_eq!(hstacked.ncols(), 5);
}

#[test]
fn test_e2e_validation_and_sorting() {
    fn make_csr_unsorted(
        rows: usize,
        cols: usize,
        row_data: &[&[f64]],
        row_indices: &[&[i64]],
    ) -> CSR<f64, i64> {
        let mut values: Vec<Span<f64>> = Vec::with_capacity(rows);
        let mut indices: Vec<Span<i64>> = Vec::with_capacity(rows);

        for i in 0..rows {
            if row_data[i].is_empty() {
                values.push(empty_span());
                indices.push(empty_span());
            } else {
                values.push(Span::copy_from::<32>(row_data[i]).unwrap());
                indices.push(Span::copy_from::<32>(row_indices[i]).unwrap());
            }
        }

        unsafe { CSR::from_raw_parts(values, indices, rows as i64, cols as i64) }
    }

    // 创建未排序的矩阵
    let mut csr = make_csr_unsorted(
        2,
        5,
        &[&[3.0, 1.0, 2.0], &[5.0, 4.0]],
        &[&[4, 0, 2], &[3, 1]],
    );

    // 验证未排序
    assert!(!csr.is_sorted());
    assert!(csr.is_valid()); // 结构有效

    // 排序
    csr.ensure_sorted();

    // 验证已排序
    assert!(csr.is_sorted());
    assert!(csr.validate());

    // 验证数据正确性
    assert_eq!(csr.row_indices(0), &[0, 2, 4]);
    assert_eq!(csr.row_values(0), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_e2e_large_sparse_matrix() {
    // 创建一个较大的稀疏矩阵
    let n = 500;

    // 创建随机稀疏矩阵（对角 + 一些额外元素）
    let mut dense = vec![0.0f64; n * n];

    // 对角线
    for i in 0..n {
        dense[i * n + i] = (i + 1) as f64;
    }

    // 一些额外的非零元素（避免覆盖对角线）
    for i in 0..n / 10 {
        let row = (i * 37) % n;
        let col = (i * 53) % n;
        if row != col {
            dense[row * n + col] = (i + 1) as f64 * 0.1;
        }
    }

    // Dense -> CSR
    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, n, n, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nrows() as usize, n);
    assert_eq!(csr.ncols() as usize, n);
    assert!(csr.nnz() >= n as i64); // 至少有对角元素

    // 验证对角元素
    for i in 0..n {
        let row_vals = csr.row_values(i as i64);
        let row_idxs = csr.row_indices(i as i64);

        // 对角元素应该存在
        let diag_pos = row_idxs.iter().position(|&idx| idx == i as i64);
        if let Some(pos) = diag_pos {
            assert_eq!(row_vals[pos], (i + 1) as f64);
        }
    }

    // 切片操作
    let sliced = csr.slice_rows(n as i64 / 4, n as i64 * 3 / 4).unwrap();
    assert_eq!(sliced.nrows() as usize, n / 2);

    // CSR -> CSC -> CSR 往返
    let csc = csc_from_csr::<f64, i64, 32>(&csr, AllocStrategy::Auto).unwrap();
    let csr_back = csr_from_csc::<f64, i64, 32>(&csc, AllocStrategy::Auto).unwrap();

    assert_eq!(csr.nnz(), csr_back.nnz());
}

// =============================================================================
// 零拷贝效率测试
// =============================================================================

#[test]
fn test_zero_copy_efficiency() {
    // 预先创建数据数组
    let row_values: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64 + 1.0]).collect();
    let row_indices: Vec<Vec<i64>> = (0..100).map(|i| vec![i as i64]).collect();

    let row_values_refs: Vec<&[f64]> = row_values.iter().map(|v| v.as_slice()).collect();
    let row_indices_refs: Vec<&[i64]> = row_indices.iter().map(|v| v.as_slice()).collect();

    let csr = make_csr(100, 100, &row_values_refs, &row_indices_refs);

    // 行切片应该是零拷贝
    let sliced = csr.slice_rows(10, 90).unwrap();

    // 验证指针共享（零拷贝证明）
    for i in 0..80 {
        let orig_ptr = csr.row_values_span((i + 10) as i64).as_ptr();
        let sliced_ptr = sliced.row_values_span(i as i64).as_ptr();
        assert_eq!(orig_ptr, sliced_ptr, "Row {} should share data", i);
    }
}

// =============================================================================
// 边界情况测试 - 稀疏矩阵
// =============================================================================

#[test]
fn test_edge_single_element_matrix() {
    let dense = vec![42.0f64];
    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 1, 1, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nrows(), 1);
    assert_eq!(csr.ncols(), 1);
    assert_eq!(csr.nnz(), 1);
    assert_eq!(csr.row_values(0), &[42.0]);
    assert_eq!(csr.row_indices(0), &[0]);
    assert!(csr.validate());
}

#[test]
fn test_edge_single_row_matrix() {
    let dense = vec![1.0f64, 0.0, 2.0, 0.0, 3.0];
    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 1, 5, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nrows(), 1);
    assert_eq!(csr.ncols(), 5);
    assert_eq!(csr.nnz(), 3);
    assert_eq!(csr.row_values(0), &[1.0, 2.0, 3.0]);
    assert_eq!(csr.row_indices(0), &[0, 2, 4]);
}

#[test]
fn test_edge_single_column_matrix() {
    let dense = vec![1.0f64, 0.0, 2.0, 0.0, 3.0];
    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 5, 1, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nrows(), 5);
    assert_eq!(csr.ncols(), 1);
    assert_eq!(csr.nnz(), 3);

    assert_eq!(csr.row_values(0), &[1.0]);
    assert_eq!(csr.row_values(2), &[2.0]);
    assert_eq!(csr.row_values(4), &[3.0]);
}

#[test]
fn test_edge_empty_matrix() {
    let csr: CSR<f64, i64> = CSR::default();

    assert_eq!(csr.nrows(), 0);
    assert_eq!(csr.ncols(), 0);
    assert_eq!(csr.nnz(), 0);
    assert!(csr.is_empty());
    assert!(csr.is_zero());
    assert!(csr.validate());
}

#[test]
fn test_edge_all_zeros_matrix() {
    let dense = vec![0.0f64; 16];
    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 4, 4, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nrows(), 4);
    assert_eq!(csr.ncols(), 4);
    assert_eq!(csr.nnz(), 0);
    assert!(!csr.is_empty());
    assert!(csr.is_zero());
    assert!((csr.sparsity() - 1.0).abs() < 1e-10);
}

#[test]
fn test_edge_full_dense_matrix() {
    let dense: Vec<f64> = (1..=9).map(|x| x as f64).collect();
    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 3, 3, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nrows(), 3);
    assert_eq!(csr.ncols(), 3);
    assert_eq!(csr.nnz(), 9);
    assert!((csr.density() - 1.0).abs() < 1e-10);
    assert!((csr.sparsity() - 0.0).abs() < 1e-10);
}

#[test]
fn test_edge_very_wide_matrix() {
    let mut dense = vec![0.0f64; 2000];
    dense[0] = 1.0;
    dense[999] = 2.0;
    dense[1000] = 3.0;
    dense[1999] = 4.0;

    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 2, 1000, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nrows(), 2);
    assert_eq!(csr.ncols(), 1000);
    assert_eq!(csr.nnz(), 4);

    let middle = csr.slice_cols::<32>(100, 900).unwrap();
    assert_eq!(middle.nnz(), 0);
}

#[test]
fn test_edge_very_tall_matrix() {
    let mut dense = vec![0.0f64; 2000];
    dense[0] = 1.0;
    dense[1] = 2.0;
    dense[1998] = 3.0;
    dense[1999] = 4.0;

    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 1000, 2, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nrows(), 1000);
    assert_eq!(csr.ncols(), 2);
    assert_eq!(csr.nnz(), 4);

    let middle = csr.slice_rows(100, 900).unwrap();
    assert_eq!(middle.nnz(), 0);
}

#[test]
fn test_edge_diagonal_matrix() {
    let n = 10;
    let mut dense = vec![0.0f64; n * n];
    for i in 0..n {
        dense[i * n + i] = (i + 1) as f64;
    }

    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, n, n, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nnz(), n as i64);

    for i in 0..n {
        assert_eq!(csr.row_nnz(i as i64), 1);
        assert_eq!(csr.row_indices(i as i64), &[i as i64]);
        assert_eq!(csr.row_values(i as i64), &[(i + 1) as f64]);
    }
}

#[test]
fn test_edge_lower_triangular() {
    let dense = vec![
        1.0, 0.0, 0.0, //
        2.0, 3.0, 0.0, //
        4.0, 5.0, 6.0, //
    ];

    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 3, 3, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nnz(), 6);
    assert_eq!(csr.row_nnz(0), 1);
    assert_eq!(csr.row_nnz(1), 2);
    assert_eq!(csr.row_nnz(2), 3);
}

#[test]
fn test_edge_upper_triangular() {
    let dense = vec![
        1.0, 2.0, 3.0, //
        0.0, 4.0, 5.0, //
        0.0, 0.0, 6.0, //
    ];

    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 3, 3, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nnz(), 6);
    assert_eq!(csr.row_nnz(0), 3);
    assert_eq!(csr.row_nnz(1), 2);
    assert_eq!(csr.row_nnz(2), 1);
}

// =============================================================================
// 边界情况测试 - 切片操作
// =============================================================================

#[test]
fn test_edge_slice_at_boundaries() {
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let indices = vec![0i64, 1, 2, 3, 4];
    let indptr = vec![0i64, 1, 2, 3, 4, 5];

    let csr =
        csr_from_scipy_csr_copy::<f64, i64, 32>(5, 5, &data, &indices, &indptr, AllocStrategy::Auto)
            .unwrap();

    let first = csr.slice_rows(0, 1).unwrap();
    assert_eq!(first.nrows(), 1);
    assert_eq!(first.row_values(0), &[1.0]);

    let last = csr.slice_rows(4, 5).unwrap();
    assert_eq!(last.nrows(), 1);
    assert_eq!(last.row_values(0), &[5.0]);

    let first_col = csr.slice_cols::<32>(0, 1).unwrap();
    assert_eq!(first_col.ncols(), 1);

    let last_col = csr.slice_cols::<32>(4, 5).unwrap();
    assert_eq!(last_col.ncols(), 1);
}

#[test]
fn test_edge_slice_entire_matrix() {
    let data = vec![1.0f64, 2.0, 3.0];
    let indices = vec![0i64, 1, 2];
    let indptr = vec![0i64, 1, 2, 3];

    let csr =
        csr_from_scipy_csr_copy::<f64, i64, 32>(3, 3, &data, &indices, &indptr, AllocStrategy::Auto)
            .unwrap();

    let full_rows = csr.slice_rows(0, 3).unwrap();
    assert_eq!(full_rows.nrows(), 3);
    assert_eq!(full_rows.nnz(), 3);

    let full_cols = csr.slice_cols::<32>(0, 3).unwrap();
    assert_eq!(full_cols.ncols(), 3);
    assert_eq!(full_cols.nnz(), 3);
}

#[test]
fn test_edge_slice_empty_range() {
    let data = vec![1.0f64, 2.0, 3.0];
    let indices = vec![0i64, 1, 2];
    let indptr = vec![0i64, 1, 2, 3];

    let csr =
        csr_from_scipy_csr_copy::<f64, i64, 32>(3, 3, &data, &indices, &indptr, AllocStrategy::Auto)
            .unwrap();

    let empty_rows = csr.slice_rows(1, 1).unwrap();
    assert_eq!(empty_rows.nrows(), 0);
    assert_eq!(empty_rows.nnz(), 0);

    let empty_cols = csr.slice_cols::<32>(2, 2).unwrap();
    assert_eq!(empty_cols.ncols(), 0);
}

#[test]
fn test_edge_chained_slices() {
    let dense: Vec<f64> = (1..=25).map(|x| x as f64).collect();
    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 5, 5, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    let row_sliced = csr.slice_rows(1, 4).unwrap();
    assert_eq!(row_sliced.nrows(), 3);

    let col_sliced = row_sliced.slice_cols::<32>(1, 4).unwrap();
    assert_eq!(col_sliced.nrows(), 3);
    assert_eq!(col_sliced.ncols(), 3);

    let final_slice = col_sliced.slice_rows(1, 2).unwrap();
    assert_eq!(final_slice.nrows(), 1);
}

#[test]
fn test_edge_slice_with_mask_all_true() {
    let data = vec![1.0f64, 2.0, 3.0];
    let indices = vec![0i64, 1, 2];
    let indptr = vec![0i64, 1, 2, 3];

    let csr =
        csr_from_scipy_csr_copy::<f64, i64, 32>(3, 3, &data, &indices, &indptr, AllocStrategy::Auto)
            .unwrap();

    let mask = vec![true, true, true];
    let sliced = csr.slice_rows_mask(&mask).unwrap();
    assert_eq!(sliced.nrows(), 3);
    assert_eq!(sliced.nnz(), 3);
}

#[test]
fn test_edge_slice_with_mask_all_false() {
    let data = vec![1.0f64, 2.0, 3.0];
    let indices = vec![0i64, 1, 2];
    let indptr = vec![0i64, 1, 2, 3];

    let csr =
        csr_from_scipy_csr_copy::<f64, i64, 32>(3, 3, &data, &indices, &indptr, AllocStrategy::Auto)
            .unwrap();

    let mask = vec![false, false, false];
    let sliced = csr.slice_rows_mask(&mask).unwrap();
    assert_eq!(sliced.nrows(), 0);
    assert_eq!(sliced.nnz(), 0);
}

#[test]
fn test_edge_slice_with_alternating_mask() {
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let indices = vec![0i64, 1, 2, 3, 4];
    let indptr = vec![0i64, 1, 2, 3, 4, 5];

    let csr =
        csr_from_scipy_csr_copy::<f64, i64, 32>(5, 5, &data, &indices, &indptr, AllocStrategy::Auto)
            .unwrap();

    let mask = vec![true, false, true, false, true];
    let sliced = csr.slice_rows_mask(&mask).unwrap();
    assert_eq!(sliced.nrows(), 3);
    assert_eq!(sliced.row_values(0), &[1.0]);
    assert_eq!(sliced.row_values(1), &[3.0]);
    assert_eq!(sliced.row_values(2), &[5.0]);
}

// =============================================================================
// 边界情况测试 - 转换操作
// =============================================================================

#[test]
fn test_edge_conversion_roundtrip_preserves_data() {
    let dense = vec![
        1.0, 0.0, 2.0, //
        0.0, 3.0, 0.0, //
        4.0, 0.0, 5.0, //
    ];

    let csr1 =
        csr_from_dense::<f64, i64, 32>(&dense, 3, 3, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    let csc = csc_from_csr::<f64, i64, 32>(&csr1, AllocStrategy::Auto).unwrap();
    let csr2 = csr_from_csc::<f64, i64, 32>(&csc, AllocStrategy::Auto).unwrap();

    assert_eq!(csr1.nrows(), csr2.nrows());
    assert_eq!(csr1.ncols(), csr2.ncols());
    assert_eq!(csr1.nnz(), csr2.nnz());

    let mut recovered = vec![0.0f64; 9];
    csr_to_dense(&csr2, &mut recovered, DenseLayout::RowMajor).unwrap();
    assert_eq!(dense, recovered);
}

#[test]
fn test_edge_conversion_col_major_layout() {
    let dense_row_major = vec![
        1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0, //
    ];

    let dense_col_major = vec![
        1.0, 4.0, // 第 0 列
        2.0, 5.0, // 第 1 列
        3.0, 6.0, // 第 2 列
    ];

    let csr_from_row = csr_from_dense::<f64, i64, 32>(
        &dense_row_major,
        2,
        3,
        DenseLayout::RowMajor,
        AllocStrategy::Auto,
    )
    .unwrap();

    let csr_from_col = csr_from_dense::<f64, i64, 32>(
        &dense_col_major,
        2,
        3,
        DenseLayout::ColMajor,
        AllocStrategy::Auto,
    )
    .unwrap();

    assert_eq!(csr_from_row.nnz(), csr_from_col.nnz());

    let mut recovered1 = vec![0.0f64; 6];
    let mut recovered2 = vec![0.0f64; 6];
    csr_to_dense(&csr_from_row, &mut recovered1, DenseLayout::RowMajor).unwrap();
    csr_to_dense(&csr_from_col, &mut recovered2, DenseLayout::RowMajor).unwrap();
    assert_eq!(recovered1, recovered2);
}

// =============================================================================
// 边界情况测试 - 数值精度
// =============================================================================

#[test]
fn test_edge_very_small_values() {
    let dense = vec![1e-300f64, 0.0, 1e-300, 0.0];
    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 2, 2, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nnz(), 2);

    let mut recovered = vec![0.0f64; 4];
    csr_to_dense(&csr, &mut recovered, DenseLayout::RowMajor).unwrap();
    assert_eq!(dense, recovered);
}

#[test]
fn test_edge_very_large_values() {
    let dense = vec![1e300f64, 0.0, 1e300, 0.0];
    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 2, 2, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nnz(), 2);

    let mut recovered = vec![0.0f64; 4];
    csr_to_dense(&csr, &mut recovered, DenseLayout::RowMajor).unwrap();
    assert_eq!(dense, recovered);
}

#[test]
fn test_edge_negative_values() {
    let dense = vec![-1.0f64, 0.0, -2.0, 0.0, 3.0, -4.0];
    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 2, 3, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nnz(), 4);

    let mut recovered = vec![0.0f64; 6];
    csr_to_dense(&csr, &mut recovered, DenseLayout::RowMajor).unwrap();
    assert_eq!(dense, recovered);
}

#[test]
fn test_edge_mixed_sign_values() {
    let dense = vec![
        -1.0, 2.0, -3.0, //
        4.0, -5.0, 6.0, //
    ];

    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 2, 3, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    let csc = csc_from_csr::<f64, i64, 32>(&csr, AllocStrategy::Auto).unwrap();
    let csr_back = csr_from_csc::<f64, i64, 32>(&csc, AllocStrategy::Auto).unwrap();

    let mut recovered = vec![0.0f64; 6];
    csr_to_dense(&csr_back, &mut recovered, DenseLayout::RowMajor).unwrap();
    assert_eq!(dense, recovered);
}

// =============================================================================
// 边界情况测试 - 特殊模式矩阵
// =============================================================================

#[test]
fn test_edge_checkerboard_pattern() {
    let dense = vec![
        1.0, 0.0, 1.0, 0.0, //
        0.0, 1.0, 0.0, 1.0, //
        1.0, 0.0, 1.0, 0.0, //
        0.0, 1.0, 0.0, 1.0, //
    ];

    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 4, 4, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nnz(), 8);
    assert!((csr.density() - 0.5).abs() < 1e-10);

    for i in 0..4 {
        assert_eq!(csr.row_nnz(i), 2);
    }
}

#[test]
fn test_edge_band_matrix() {
    let n = 5;
    let mut dense = vec![0.0f64; n * n];

    for i in 0..n {
        dense[i * n + i] = 2.0;
    }
    for i in 0..n - 1 {
        dense[i * n + (i + 1)] = -1.0;
    }
    for i in 1..n {
        dense[i * n + (i - 1)] = -1.0;
    }

    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, n, n, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.row_nnz(0), 2);
    assert_eq!(csr.row_nnz(1), 3);
    assert_eq!(csr.row_nnz((n - 2) as i64), 3);
    assert_eq!(csr.row_nnz((n - 1) as i64), 2);
}

#[test]
fn test_edge_block_diagonal() {
    let dense = vec![
        1.0, 2.0, 0.0, 0.0, //
        3.0, 4.0, 0.0, 0.0, //
        0.0, 0.0, 5.0, 6.0, //
        0.0, 0.0, 7.0, 8.0, //
    ];

    let csr =
        csr_from_dense::<f64, i64, 32>(&dense, 4, 4, DenseLayout::RowMajor, AllocStrategy::Auto)
            .unwrap();

    assert_eq!(csr.nnz(), 8);

    assert_eq!(csr.row_indices(0), &[0, 1]);
    assert_eq!(csr.row_indices(1), &[0, 1]);
    assert_eq!(csr.row_indices(2), &[2, 3]);
    assert_eq!(csr.row_indices(3), &[2, 3]);
}

// =============================================================================
// CSR 内部结构验证
// =============================================================================

#[test]
fn test_csr_internal_spans() {
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let indices = vec![0i64, 1, 2, 0, 3];
    let indptr = vec![0i64, 2, 3, 5];

    let csr =
        csr_from_scipy_csr_copy::<f64, i64, 32>(3, 4, &data, &indices, &indptr, AllocStrategy::Auto)
            .unwrap();

    for i in 0..3 {
        let row_vals = csr.row_values_span(i);
        let row_idxs = csr.row_indices_span(i);

        assert!(row_vals.has_storage());
        assert!(row_idxs.has_storage());

        assert!(row_vals.is_aligned() || row_vals.len() == 0);
    }
}

#[test]
fn test_csr_slice_preserves_span_properties() {
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let indices = vec![0i64, 1, 2, 0, 2, 0, 1, 2, 1];
    let indptr = vec![0i64, 3, 5, 9];

    let csr =
        csr_from_scipy_csr_copy::<f64, i64, 32>(3, 3, &data, &indices, &indptr, AllocStrategy::Auto)
            .unwrap();

    let sliced = csr.slice_rows(0, 2).unwrap();

    let orig_ptr = csr.row_values_span(0).as_ptr();
    let sliced_ptr = sliced.row_values_span(0).as_ptr();
    assert_eq!(orig_ptr, sliced_ptr, "Row slice should share data");
}
