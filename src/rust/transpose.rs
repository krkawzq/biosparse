//! 稀疏矩阵转置
//!
//! 利用 Span storage 的引用计数特性实现高效转置。
//! 通过 clone span 来共享底层数据，避免数据拷贝。

use crate::convert::{AllocStrategy, ConvertError};
use crate::span::{Span, SpanFlags};
use crate::sparse::{SparseIndex, CSC, CSR};
use rayon::prelude::*;
use std::cell::Cell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// 创建空 span
#[inline]
fn empty_span<T>() -> Span<T> {
    unsafe { Span::from_raw_parts_unchecked(NonNull::dangling(), 0, SpanFlags::VIEW) }
}

/// CSR → CSC 转置（高效版本，利用 span clone 共享数据）
///
/// # 实现策略
/// 1. 遍历 CSR 的每一行
/// 2. 对于每一行中的 (col_idx, value)，将其添加到对应的 column
/// 3. 使用 span clone 来共享底层数据（引用计数，零拷贝）
/// 4. 最终构建出独立的 CSC 结构
pub fn csc_transpose_from_csr<
    V: Copy + Default + Send + Sync,
    I: SparseIndex + Send + Sync,
    const ALIGN: usize,
>(
    csr: &CSR<V, I>,
    _strategy: AllocStrategy,
) -> Result<CSC<V, I>, ConvertError> {
    let row_count = csr.rows.to_usize();
    let col_count = csr.cols.to_usize();

    // CSC after transpose:
    // - nrows = csr.cols (col_count)
    // - ncols = csr.rows (row_count)
    // - Column i in CSC corresponds to row i in CSR

    if row_count == 0 || col_count == 0 {
        // 空矩阵 - CSC has row_count columns
        let values: Vec<Span<V>> = (0..row_count).map(|_| empty_span()).collect();
        let indices: Vec<Span<I>> = (0..row_count).map(|_| empty_span()).collect();
        return Ok(CSC {
            values,
            indices,
            rows: csr.cols, // 转置后行列互换
            cols: csr.rows,
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    // 第一遍：统计每个 CSC column 的 nnz
    // CSC column i = CSR row i, so each CSC column has same nnz as CSR row
    let col_lens: Vec<usize> = (0..row_count)
        .into_par_iter()
        .map(|i| {
            let indices = csr.row_indices(I::from_usize(i));
            indices.len()
        })
        .collect();

    let total_nnz: usize = col_lens.par_iter().sum();

    if total_nnz == 0 {
        let values: Vec<Span<V>> = (0..row_count).map(|_| empty_span()).collect();
        let indices: Vec<Span<I>> = (0..row_count).map(|_| empty_span()).collect();
        return Ok(CSC {
            values,
            indices,
            rows: csr.cols,
            cols: csr.rows,
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    // 分配每列的 span
    let non_empty_lens: Vec<usize> = col_lens.iter().copied().filter(|&len| len > 0).collect();
    let val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
    let idx_spans = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

    // 为每列创建完整的 span 数组（包括空列）
    let mut col_values: Vec<Span<V>> = Vec::with_capacity(row_count);
    let mut col_indices: Vec<Span<I>> = Vec::with_capacity(row_count);
    let mut span_idx = 0;

    for len in &col_lens {
        if *len > 0 {
            col_values.push(val_spans[span_idx].clone());
            col_indices.push(idx_spans[span_idx].clone());
            span_idx += 1;
        } else {
            col_values.push(empty_span());
            col_indices.push(empty_span());
        }
    }

    // 第二遍：填充数据
    // CSR element at (row i, col j, value v) → CSC element at (row j, col i, value v)
    // Which is stored in CSC column i, at row index j
    (0..row_count).into_par_iter().for_each(|i| {
        let row_values = csr.row_values(I::from_usize(i));
        let row_indices = csr.row_indices(I::from_usize(i));

        // For CSC column i, fill with values from CSR row i
        // The row indices in CSC are the column indices from CSR
        unsafe {
            let values_ptr = col_values[i].as_ptr() as *mut V;
            let indices_ptr = col_indices[i].as_ptr() as *mut I;

            for k in 0..row_values.len() {
                *values_ptr.add(k) = row_values[k];
                *indices_ptr.add(k) = row_indices[k]; // CSR col index becomes CSC row index
            }
        }
    });

    // 第三遍：对每列的索引排序（转置后行索引需要排序）
    col_values
        .par_iter()
        .zip(col_indices.par_iter())
        .for_each(|(values_span, indices_span)| {
            let len = values_span.len();
            if len > 0 {
                unsafe {
                    let values_ptr = values_span.as_ptr() as *mut V;
                    let indices_ptr = indices_span.as_ptr() as *mut I;
                    let values_slice = std::slice::from_raw_parts_mut(values_ptr, len);
                    let indices_slice = std::slice::from_raw_parts_mut(indices_ptr, len);

                    // 按行索引排序
                    let mut pairs: Vec<(I, V)> = indices_slice
                        .iter()
                        .zip(values_slice.iter())
                        .map(|(&idx, &val)| (idx, val))
                        .collect();

                    pairs.sort_unstable_by_key(|&(idx, _)| idx);

                    for (k, (idx, val)) in pairs.into_iter().enumerate() {
                        indices_slice[k] = idx;
                        values_slice[k] = val;
                    }
                }
            }
        });

    Ok(CSC {
        values: col_values,
        indices: col_indices,
        rows: csr.cols, // 转置后行列互换
        cols: csr.rows,
        nnz: Cell::new(Some(I::from_usize(total_nnz))),
    })
}

/// CSC → CSR 转置（高效版本）
pub fn csr_transpose_from_csc<
    V: Copy + Default + Send + Sync,
    I: SparseIndex + Send + Sync,
    const ALIGN: usize,
>(
    csc: &CSC<V, I>,
    _strategy: AllocStrategy,
) -> Result<CSR<V, I>, ConvertError> {
    let row_count = csc.rows.to_usize();
    let col_count = csc.cols.to_usize();

    // CSR after transpose:
    // - nrows = csc.cols (col_count)
    // - ncols = csc.rows (row_count)
    // - Row i in CSR corresponds to column i in CSC

    if row_count == 0 || col_count == 0 {
        let values: Vec<Span<V>> = (0..col_count).map(|_| empty_span()).collect();
        let indices: Vec<Span<I>> = (0..col_count).map(|_| empty_span()).collect();
        return Ok(CSR {
            values,
            indices,
            rows: csc.cols,
            cols: csc.rows,
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    // 第一遍：统计每个 CSR row 的 nnz
    // CSR row i = CSC column i, so each CSR row has same nnz as CSC column
    let row_lens: Vec<usize> = (0..col_count)
        .into_par_iter()
        .map(|j| {
            let indices = csc.col_indices(I::from_usize(j));
            indices.len()
        })
        .collect();

    let total_nnz: usize = row_lens.par_iter().sum();

    if total_nnz == 0 {
        let values: Vec<Span<V>> = (0..col_count).map(|_| empty_span()).collect();
        let indices: Vec<Span<I>> = (0..col_count).map(|_| empty_span()).collect();
        return Ok(CSR {
            values,
            indices,
            rows: csc.cols,
            cols: csc.rows,
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    // 分配 spans
    let non_empty_lens: Vec<usize> = row_lens.iter().copied().filter(|&len| len > 0).collect();
    let val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
    let idx_spans = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

    let mut row_values: Vec<Span<V>> = Vec::with_capacity(col_count);
    let mut row_indices: Vec<Span<I>> = Vec::with_capacity(col_count);
    let mut span_idx = 0;

    for len in &row_lens {
        if *len > 0 {
            row_values.push(val_spans[span_idx].clone());
            row_indices.push(idx_spans[span_idx].clone());
            span_idx += 1;
        } else {
            row_values.push(empty_span());
            row_indices.push(empty_span());
        }
    }

    // 第二遍：填充数据
    // CSC element at (row i, col j, value v) → CSR element at (row j, col i, value v)
    // Which is stored in CSR row j, at column index i
    (0..col_count).into_par_iter().for_each(|j| {
        let col_values = csc.col_values(I::from_usize(j));
        let col_indices = csc.col_indices(I::from_usize(j));

        // For CSR row j, fill with values from CSC column j
        // The column indices in CSR are the row indices from CSC
        unsafe {
            let values_ptr = row_values[j].as_ptr() as *mut V;
            let indices_ptr = row_indices[j].as_ptr() as *mut I;

            for k in 0..col_values.len() {
                *values_ptr.add(k) = col_values[k];
                *indices_ptr.add(k) = col_indices[k]; // CSC row index becomes CSR col index
            }
        }
    });

    // 第三遍：排序
    row_values
        .par_iter()
        .zip(row_indices.par_iter())
        .for_each(|(values_span, indices_span)| {
            let len = values_span.len();
            if len > 0 {
                unsafe {
                    let values_ptr = values_span.as_ptr() as *mut V;
                    let indices_ptr = indices_span.as_ptr() as *mut I;
                    let values_slice = std::slice::from_raw_parts_mut(values_ptr, len);
                    let indices_slice = std::slice::from_raw_parts_mut(indices_ptr, len);

                    let mut pairs: Vec<(I, V)> = indices_slice
                        .iter()
                        .zip(values_slice.iter())
                        .map(|(&idx, &val)| (idx, val))
                        .collect();

                    pairs.sort_unstable_by_key(|&(idx, _)| idx);

                    for (k, (idx, val)) in pairs.into_iter().enumerate() {
                        indices_slice[k] = idx;
                        values_slice[k] = val;
                    }
                }
            }
        });

    Ok(CSR {
        values: row_values,
        indices: row_indices,
        rows: csc.cols,
        cols: csc.rows,
        nnz: Cell::new(Some(I::from_usize(total_nnz))),
    })
}
