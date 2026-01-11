//! 稀疏矩阵切片操作
//!
//! # 设计理念
//!
//! - **主轴切片（行切片/列切片）**：利用 Span 的引用计数实现零拷贝
//! - **副轴切片**：需要遍历过滤，创建新的 Span
//! - **支持两种索引方式**：
//!   - `Slice`: 范围切片 `[start, end)`
//!   - `Mask`: 布尔掩码切片
//! - **不支持 Scatter（花式索引）**
//!
//! # 性能优化
//!
//! - 使用 `unlikely` 标记错误路径
//! - 对于有序索引使用二分搜索优化范围查找
//! - 并行处理每行/列
//! - 16 路循环展开 + prefetch 诱导 SIMD 向量化
//!
//! # 算法复杂度
//!
//! | 操作 | CSR 行切片 | CSR 列切片 | CSC 列切片 | CSC 行切片 |
//! |------|-----------|-----------|-----------|-----------|
//! | Slice | O(1) 零拷贝 | O(nnz) | O(1) 零拷贝 | O(nnz) |
//! | Mask | O(selected) | O(nnz) | O(selected) | O(nnz) |

#![allow(clippy::missing_safety_doc)]

use std::cell::Cell;
use std::mem::size_of;
use std::ptr::NonNull;

use rayon::prelude::*;

use crate::span::{Span, SpanFlags};
use crate::sparse::{SparseIndex, CSC, CSR};
use crate::storage::AllocError;
use crate::tools::{prefetch_read, prefetch_write, unlikely};

// =============================================================================
// Error Type
// =============================================================================

/// 切片操作错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceError {
    /// 范围越界
    OutOfBounds {
        start: usize,
        end: usize,
        length: usize,
    },
    /// 掩码长度不匹配
    MaskLengthMismatch { expected: usize, got: usize },
    /// 内存分配失败
    AllocError(AllocError),
}

impl From<AllocError> for SliceError {
    #[inline]
    fn from(e: AllocError) -> Self {
        SliceError::AllocError(e)
    }
}

impl std::fmt::Display for SliceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SliceError::OutOfBounds { start, end, length } => {
                write!(
                    f,
                    "slice out of bounds: [{}, {}) for length {}",
                    start, end, length
                )
            }
            SliceError::MaskLengthMismatch { expected, got } => {
                write!(
                    f,
                    "mask length mismatch: expected {}, got {}",
                    expected, got
                )
            }
            SliceError::AllocError(e) => write!(f, "allocation error: {:?}", e),
        }
    }
}

impl std::error::Error for SliceError {}

/// 切片操作结果
pub type SliceResult<T> = Result<T, SliceError>;

// =============================================================================
// 常量和辅助
// =============================================================================

/// 并行处理的最小元素阈值
const PARALLEL_THRESHOLD: usize = 256;

/// prefetch 的目标字节偏移（约 512 字节，适应缓存行预取）
const PREFETCH_BYTES: usize = 512;

/// 创建空 Span（用于空行/列）
///
/// 使用 `NonNull::dangling()` 创建一个悬空指针，这是 Rust 标准库处理
/// 空容器的惯用方式，比使用固定静态地址更安全。
#[inline(always)]
fn empty_span<T>() -> Span<T> {
    let ptr = NonNull::dangling();
    // SAFETY: 长度为 0，永远不会解引用该指针
    unsafe { Span::from_raw_parts_unchecked(ptr, 0, SpanFlags::VIEW) }
}

/// 根据数量决定是否并行
#[inline(always)]
fn should_parallelize(count: usize) -> bool {
    count >= PARALLEL_THRESHOLD
}

// =============================================================================
// 二分搜索辅助
// =============================================================================

/// 二分搜索找到第一个 >= target 的位置
///
/// 等价于标准库的 `partition_point`，但使用 `get_unchecked` 避免边界检查
#[inline(always)]
fn lower_bound<I: SparseIndex>(slice: &[I], target: I) -> usize {
    slice.partition_point(|&x| x < target)
}

/// 使用二分搜索统计范围内的元素数量（用于有序索引）
#[inline(always)]
fn count_in_range_sorted<I: SparseIndex>(indices: &[I], start: I, end: I) -> usize {
    if indices.is_empty() {
        return 0;
    }
    let left = lower_bound(indices, start);
    let right = lower_bound(indices, end);
    right.saturating_sub(left)
}

// =============================================================================
// SIMD 辅助
// =============================================================================

/// 快速统计掩码中 true 的数量（8 路展开诱导 SIMD）
#[inline(always)]
fn count_true_fast(mask: &[bool]) -> usize {
    let len = mask.len();
    if len < 64 {
        return mask.iter().filter(|&&b| b).count();
    }

    let ptr = mask.as_ptr();
    let main_len = len & !7;
    let mut count = 0usize;
    let mut i = 0;

    // 8 路展开主循环
    while i < main_len {
        unsafe {
            count += *ptr.add(i) as usize;
            count += *ptr.add(i + 1) as usize;
            count += *ptr.add(i + 2) as usize;
            count += *ptr.add(i + 3) as usize;
            count += *ptr.add(i + 4) as usize;
            count += *ptr.add(i + 5) as usize;
            count += *ptr.add(i + 6) as usize;
            count += *ptr.add(i + 7) as usize;
        }
        i += 8;
    }

    // 处理余数
    while i < len {
        unsafe {
            count += *ptr.add(i) as usize;
        }
        i += 1;
    }

    count
}

/// 复制并减去偏移（16 路展开诱导 SIMD）
///
/// # Safety
///
/// - src 和 dst 不重叠
/// - src 可读 count 个元素
/// - dst 可写 count 个元素
#[inline(always)]
unsafe fn copy_sub_offset<I: SparseIndex>(src: *const I, dst: *mut I, count: usize, offset: I) {
    if count < 16 {
        for i in 0..count {
            *dst.add(i) = *src.add(i) - offset;
        }
        return;
    }

    // 根据元素大小计算 prefetch 元素偏移量
    let prefetch_elements = PREFETCH_BYTES / size_of::<I>();
    let main_count = count & !15;
    let mut i = 0;

    while i < main_count {
        // prefetch 基于字节偏移量，而非固定元素数
        prefetch_read(src.add(i + prefetch_elements));
        prefetch_write(dst.add(i + prefetch_elements));

        *dst.add(i) = *src.add(i) - offset;
        *dst.add(i + 1) = *src.add(i + 1) - offset;
        *dst.add(i + 2) = *src.add(i + 2) - offset;
        *dst.add(i + 3) = *src.add(i + 3) - offset;
        *dst.add(i + 4) = *src.add(i + 4) - offset;
        *dst.add(i + 5) = *src.add(i + 5) - offset;
        *dst.add(i + 6) = *src.add(i + 6) - offset;
        *dst.add(i + 7) = *src.add(i + 7) - offset;
        *dst.add(i + 8) = *src.add(i + 8) - offset;
        *dst.add(i + 9) = *src.add(i + 9) - offset;
        *dst.add(i + 10) = *src.add(i + 10) - offset;
        *dst.add(i + 11) = *src.add(i + 11) - offset;
        *dst.add(i + 12) = *src.add(i + 12) - offset;
        *dst.add(i + 13) = *src.add(i + 13) - offset;
        *dst.add(i + 14) = *src.add(i + 14) - offset;
        *dst.add(i + 15) = *src.add(i + 15) - offset;
        i += 16;
    }

    while i < count {
        *dst.add(i) = *src.add(i) - offset;
        i += 1;
    }
}

/// 过滤复制带掩码映射（4 路展开）
///
/// # Safety
///
/// 调用者需确保所有指针和索引有效
#[inline(always)]
unsafe fn filter_copy_with_map<V: Copy, I: SparseIndex>(
    src_vals: &[V],
    src_idxs: &[I],
    dst_vals: *mut V,
    dst_idxs: *mut I,
    col_map: &[I],
    sentinel: I,
) -> usize {
    let len = src_vals.len();
    let main_len = len & !3;
    let mut write_pos = 0;
    let mut i = 0;

    // 4 路展开主循环
    while i < main_len {
        let col0 = *src_idxs.get_unchecked(i);
        let col1 = *src_idxs.get_unchecked(i + 1);
        let col2 = *src_idxs.get_unchecked(i + 2);
        let col3 = *src_idxs.get_unchecked(i + 3);

        let map0 = *col_map.get_unchecked(col0.to_usize());
        let map1 = *col_map.get_unchecked(col1.to_usize());
        let map2 = *col_map.get_unchecked(col2.to_usize());
        let map3 = *col_map.get_unchecked(col3.to_usize());

        if map0 != sentinel {
            *dst_vals.add(write_pos) = *src_vals.get_unchecked(i);
            *dst_idxs.add(write_pos) = map0;
            write_pos += 1;
        }
        if map1 != sentinel {
            *dst_vals.add(write_pos) = *src_vals.get_unchecked(i + 1);
            *dst_idxs.add(write_pos) = map1;
            write_pos += 1;
        }
        if map2 != sentinel {
            *dst_vals.add(write_pos) = *src_vals.get_unchecked(i + 2);
            *dst_idxs.add(write_pos) = map2;
            write_pos += 1;
        }
        if map3 != sentinel {
            *dst_vals.add(write_pos) = *src_vals.get_unchecked(i + 3);
            *dst_idxs.add(write_pos) = map3;
            write_pos += 1;
        }
        i += 4;
    }

    // 余数
    while i < len {
        let col = *src_idxs.get_unchecked(i);
        let mapped = *col_map.get_unchecked(col.to_usize());
        if mapped != sentinel {
            *dst_vals.add(write_pos) = *src_vals.get_unchecked(i);
            *dst_idxs.add(write_pos) = mapped;
            write_pos += 1;
        }
        i += 1;
    }

    write_pos
}

// =============================================================================
// 内部辅助函数 - 减少代码重复
// =============================================================================

/// 过滤非空行/列并返回 (索引, nnz) 对
#[inline]
fn collect_non_empty(nnz_per_item: &[usize]) -> Vec<(usize, usize)> {
    nnz_per_item
        .iter()
        .enumerate()
        .filter(|(_, &nnz)| nnz > 0)
        .map(|(i, &nnz)| (i, nnz))
        .collect()
}

/// 组装最终结果的通用函数
///
/// 使用双指针法将分配的 Span 按正确顺序放入结果 Vec
#[inline]
fn assemble_result<V, I>(
    total_count: usize,
    non_empty: &[(usize, usize)],
    val_spans: &mut [Span<V>],
    idx_spans: &mut [Span<I>],
) -> (Vec<Span<V>>, Vec<Span<I>>)
where
    V: Copy,
    I: Copy,
{
    let mut values = Vec::with_capacity(total_count);
    let mut indices = Vec::with_capacity(total_count);
    let mut non_empty_idx = 0;

    for i in 0..total_count {
        if non_empty_idx < non_empty.len() && non_empty[non_empty_idx].0 == i {
            values.push(unsafe { Span::take_at(val_spans, non_empty_idx) });
            indices.push(unsafe { Span::take_at(idx_spans, non_empty_idx) });
            non_empty_idx += 1;
        } else {
            values.push(empty_span());
            indices.push(empty_span());
        }
    }

    (values, indices)
}

// =============================================================================
// CSR Slicing - 行切片（主轴，零拷贝）
// =============================================================================

impl<V: Clone + Sync, I: SparseIndex> CSR<V, I> {
    /// 行范围切片 - O(1) 零拷贝
    ///
    /// 返回 `[row_start, row_end)` 范围内的行组成的新 CSR 矩阵。
    /// 由于 Span 的引用计数机制，不需要复制数据。
    #[inline]
    pub fn slice_rows(&self, row_start: I, row_end: I) -> SliceResult<Self> {
        let start = row_start.to_usize();
        let end = row_end.to_usize();
        let nrows = self.rows.to_usize();

        if unlikely(start > end || end > nrows) {
            return Err(SliceError::OutOfBounds {
                start,
                end,
                length: nrows,
            });
        }

        let new_rows = end - start;

        if unlikely(new_rows == 0) {
            return Ok(Self {
                values: Vec::new(),
                indices: Vec::new(),
                rows: I::ZERO,
                cols: self.cols,
                nnz: Cell::new(Some(I::ZERO)),
            });
        }

        // 直接切片，编译器可从上面的边界检查推断安全性
        let values: Vec<Span<V>> = self.values[start..end].to_vec();
        let indices: Vec<Span<I>> = self.indices[start..end].to_vec();

        Ok(Self {
            values,
            indices,
            rows: I::from_usize(new_rows),
            cols: self.cols,
            nnz: Cell::new(None),
        })
    }

    /// 行掩码切片 - O(selected_rows) 零拷贝
    #[inline]
    pub fn slice_rows_mask(&self, mask: &[bool]) -> SliceResult<Self> {
        let nrows = self.rows.to_usize();

        if unlikely(mask.len() != nrows) {
            return Err(SliceError::MaskLengthMismatch {
                expected: nrows,
                got: mask.len(),
            });
        }

        let selected_count = count_true_fast(mask);

        if unlikely(selected_count == 0) {
            return Ok(Self {
                values: Vec::new(),
                indices: Vec::new(),
                rows: I::ZERO,
                cols: self.cols,
                nnz: Cell::new(Some(I::ZERO)),
            });
        }

        let mut values = Vec::with_capacity(selected_count);
        let mut indices = Vec::with_capacity(selected_count);

        for (i, &selected) in mask.iter().enumerate() {
            if selected {
                // mask.len() == nrows == self.values.len()，所以 i 必然有效
                values.push(self.values[i].clone());
                indices.push(self.indices[i].clone());
            }
        }

        Ok(Self {
            values,
            indices,
            rows: I::from_usize(selected_count),
            cols: self.cols,
            nnz: Cell::new(None),
        })
    }
}

// =============================================================================
// CSR Slicing - 列切片（副轴，需要过滤）
// =============================================================================

impl<V: Copy + Sync + Send, I: SparseIndex + Send + Sync> CSR<V, I> {
    /// 列范围切片 - O(nnz)
    ///
    /// # Arguments
    ///
    /// * `col_start` - 起始列（包含）
    /// * `col_end` - 结束列（不包含）
    pub fn slice_cols<const ALIGN: usize>(&self, col_start: I, col_end: I) -> SliceResult<Self> {
        let c_start = col_start.to_usize();
        let c_end = col_end.to_usize();
        let ncols = self.cols.to_usize();

        if unlikely(c_start > c_end || c_end > ncols) {
            return Err(SliceError::OutOfBounds {
                start: c_start,
                end: c_end,
                length: ncols,
            });
        }

        let nrows = self.rows.to_usize();
        let new_cols = c_end - c_start;

        if unlikely(new_cols == 0 || nrows == 0) {
            let values = (0..nrows).map(|_| empty_span()).collect();
            let indices = (0..nrows).map(|_| empty_span()).collect();
            return Ok(Self {
                values,
                indices,
                rows: self.rows,
                cols: I::ZERO,
                nnz: Cell::new(Some(I::ZERO)),
            });
        }

        // 计算每行的新 nnz
        let compute_nnz =
            |idx_span: &Span<I>| count_in_range_sorted(idx_span.as_slice(), col_start, col_end);

        let row_nnz: Vec<usize> = if should_parallelize(nrows) {
            self.indices.par_iter().map(compute_nnz).collect()
        } else {
            self.indices.iter().map(compute_nnz).collect()
        };

        let total_nnz: usize = row_nnz.iter().sum();

        // 快速路径：所有行都为空
        if unlikely(total_nnz == 0) {
            let values = (0..nrows).map(|_| empty_span()).collect();
            let indices = (0..nrows).map(|_| empty_span()).collect();
            return Ok(Self {
                values,
                indices,
                rows: self.rows,
                cols: I::from_usize(new_cols),
                nnz: Cell::new(Some(I::ZERO)),
            });
        }

        // 过滤非空行
        let non_empty = collect_non_empty(&row_nnz);
        let non_empty_lens: Vec<usize> = non_empty.iter().map(|(_, nnz)| *nnz).collect();

        // 分配 Storage
        let mut val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
        let mut idx_spans = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

        // 填充数据的闭包
        let fill_row = |(alloc_idx, &(row_idx, _)): (usize, &(usize, usize))| {
            let val_span = unsafe { val_spans.get_unchecked(alloc_idx) };
            let idx_span = unsafe { idx_spans.get_unchecked(alloc_idx) };

            let old_vals = unsafe { self.values.get_unchecked(row_idx).as_slice() };
            let old_idxs = unsafe { self.indices.get_unchecked(row_idx).as_slice() };

            let left = lower_bound(old_idxs, col_start);
            let right = lower_bound(old_idxs, col_end);
            let copy_len = right - left;

            let val_ptr = val_span.as_ptr() as *mut V;
            let idx_ptr = idx_span.as_ptr() as *mut I;

            unsafe {
                std::ptr::copy_nonoverlapping(old_vals.as_ptr().add(left), val_ptr, copy_len);
                copy_sub_offset(old_idxs.as_ptr().add(left), idx_ptr, copy_len, col_start);
            }
        };

        if should_parallelize(non_empty.len()) {
            non_empty.par_iter().enumerate().for_each(fill_row);
        } else {
            non_empty.iter().enumerate().for_each(fill_row);
        }

        // 组装最终结果
        let (values, indices) = assemble_result(nrows, &non_empty, &mut val_spans, &mut idx_spans);

        Ok(Self {
            values,
            indices,
            rows: self.rows,
            cols: I::from_usize(new_cols),
            nnz: Cell::new(Some(I::from_usize(total_nnz))),
        })
    }

    /// 列掩码切片 - O(nnz)
    pub fn slice_cols_mask<const ALIGN: usize>(&self, mask: &[bool]) -> SliceResult<Self> {
        let ncols = self.cols.to_usize();

        if unlikely(mask.len() != ncols) {
            return Err(SliceError::MaskLengthMismatch {
                expected: ncols,
                got: mask.len(),
            });
        }

        let nrows = self.rows.to_usize();
        let selected_cols = count_true_fast(mask);

        if unlikely(selected_cols == 0 || nrows == 0) {
            let values = (0..nrows).map(|_| empty_span()).collect();
            let indices = (0..nrows).map(|_| empty_span()).collect();
            return Ok(Self {
                values,
                indices,
                rows: self.rows,
                cols: I::from_usize(selected_cols),
                nnz: Cell::new(Some(I::ZERO)),
            });
        }

        // 构建列索引映射表：old_col -> new_col，使用 I::MAX 表示未选中
        let mut col_map: Vec<I> = vec![I::MAX; ncols];
        let mut new_col_idx = I::ZERO;
        for (old_col, &selected) in mask.iter().enumerate() {
            if selected {
                col_map[old_col] = new_col_idx;
                new_col_idx = new_col_idx + I::ONE;
            }
        }
        let new_cols = new_col_idx;

        // 计算每行的新 nnz
        let col_map_ref = &col_map;
        let compute_nnz = |idx_span: &Span<I>| {
            idx_span
                .as_slice()
                .iter()
                .filter(|&&col| col_map_ref[col.to_usize()] != I::MAX)
                .count()
        };

        let row_nnz: Vec<usize> = if should_parallelize(nrows) {
            self.indices.par_iter().map(compute_nnz).collect()
        } else {
            self.indices.iter().map(compute_nnz).collect()
        };

        let total_nnz: usize = row_nnz.iter().sum();

        if unlikely(total_nnz == 0) {
            let values = (0..nrows).map(|_| empty_span()).collect();
            let indices = (0..nrows).map(|_| empty_span()).collect();
            return Ok(Self {
                values,
                indices,
                rows: self.rows,
                cols: new_cols,
                nnz: Cell::new(Some(I::ZERO)),
            });
        }

        // 过滤非空行
        let non_empty = collect_non_empty(&row_nnz);
        let non_empty_lens: Vec<usize> = non_empty.iter().map(|(_, nnz)| *nnz).collect();

        // 分配 Storage
        let mut val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
        let mut idx_spans = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

        // 填充数据的闭包
        let fill_row = |(alloc_idx, &(row_idx, _)): (usize, &(usize, usize))| {
            let val_span = unsafe { val_spans.get_unchecked(alloc_idx) };
            let idx_span = unsafe { idx_spans.get_unchecked(alloc_idx) };

            let old_vals = unsafe { self.values.get_unchecked(row_idx).as_slice() };
            let old_idxs = unsafe { self.indices.get_unchecked(row_idx).as_slice() };

            let val_ptr = val_span.as_ptr() as *mut V;
            let idx_ptr = idx_span.as_ptr() as *mut I;

            unsafe {
                filter_copy_with_map(old_vals, old_idxs, val_ptr, idx_ptr, col_map_ref, I::MAX);
            }
        };

        if should_parallelize(non_empty.len()) {
            non_empty.par_iter().enumerate().for_each(fill_row);
        } else {
            non_empty.iter().enumerate().for_each(fill_row);
        }

        // 组装最终结果
        let (values, indices) = assemble_result(nrows, &non_empty, &mut val_spans, &mut idx_spans);

        Ok(Self {
            values,
            indices,
            rows: self.rows,
            cols: new_cols,
            nnz: Cell::new(Some(I::from_usize(total_nnz))),
        })
    }
}

// =============================================================================
// CSC Slicing - 列切片（主轴，零拷贝）
// =============================================================================

impl<V: Clone + Sync, I: SparseIndex> CSC<V, I> {
    /// 列范围切片 - O(1) 零拷贝
    #[inline]
    pub fn slice_cols(&self, col_start: I, col_end: I) -> SliceResult<Self> {
        let start = col_start.to_usize();
        let end = col_end.to_usize();
        let ncols = self.cols.to_usize();

        if unlikely(start > end || end > ncols) {
            return Err(SliceError::OutOfBounds {
                start,
                end,
                length: ncols,
            });
        }

        let new_cols = end - start;

        if unlikely(new_cols == 0) {
            return Ok(Self {
                values: Vec::new(),
                indices: Vec::new(),
                rows: self.rows,
                cols: I::ZERO,
                nnz: Cell::new(Some(I::ZERO)),
            });
        }

        let values: Vec<Span<V>> = self.values[start..end].to_vec();
        let indices: Vec<Span<I>> = self.indices[start..end].to_vec();

        Ok(Self {
            values,
            indices,
            rows: self.rows,
            cols: I::from_usize(new_cols),
            nnz: Cell::new(None),
        })
    }

    /// 列掩码切片 - O(selected_cols) 零拷贝
    #[inline]
    pub fn slice_cols_mask(&self, mask: &[bool]) -> SliceResult<Self> {
        let ncols = self.cols.to_usize();

        if unlikely(mask.len() != ncols) {
            return Err(SliceError::MaskLengthMismatch {
                expected: ncols,
                got: mask.len(),
            });
        }

        let selected_count = count_true_fast(mask);

        if unlikely(selected_count == 0) {
            return Ok(Self {
                values: Vec::new(),
                indices: Vec::new(),
                rows: self.rows,
                cols: I::ZERO,
                nnz: Cell::new(Some(I::ZERO)),
            });
        }

        let mut values = Vec::with_capacity(selected_count);
        let mut indices = Vec::with_capacity(selected_count);

        for (j, &selected) in mask.iter().enumerate() {
            if selected {
                values.push(self.values[j].clone());
                indices.push(self.indices[j].clone());
            }
        }

        Ok(Self {
            values,
            indices,
            rows: self.rows,
            cols: I::from_usize(selected_count),
            nnz: Cell::new(None),
        })
    }
}

// =============================================================================
// CSC Slicing - 行切片（副轴，需要过滤）
// =============================================================================

impl<V: Copy + Sync + Send, I: SparseIndex + Send + Sync> CSC<V, I> {
    /// 行范围切片 - O(nnz)
    pub fn slice_rows<const ALIGN: usize>(&self, row_start: I, row_end: I) -> SliceResult<Self> {
        let r_start = row_start.to_usize();
        let r_end = row_end.to_usize();
        let nrows = self.rows.to_usize();

        if unlikely(r_start > r_end || r_end > nrows) {
            return Err(SliceError::OutOfBounds {
                start: r_start,
                end: r_end,
                length: nrows,
            });
        }

        let ncols = self.cols.to_usize();
        let new_rows = r_end - r_start;

        if unlikely(new_rows == 0 || ncols == 0) {
            let values = (0..ncols).map(|_| empty_span()).collect();
            let indices = (0..ncols).map(|_| empty_span()).collect();
            return Ok(Self {
                values,
                indices,
                rows: I::ZERO,
                cols: self.cols,
                nnz: Cell::new(Some(I::ZERO)),
            });
        }

        // 计算每列的新 nnz
        let compute_nnz =
            |idx_span: &Span<I>| count_in_range_sorted(idx_span.as_slice(), row_start, row_end);

        let col_nnz: Vec<usize> = if should_parallelize(ncols) {
            self.indices.par_iter().map(compute_nnz).collect()
        } else {
            self.indices.iter().map(compute_nnz).collect()
        };

        let total_nnz: usize = col_nnz.iter().sum();

        if unlikely(total_nnz == 0) {
            let values = (0..ncols).map(|_| empty_span()).collect();
            let indices = (0..ncols).map(|_| empty_span()).collect();
            return Ok(Self {
                values,
                indices,
                rows: I::from_usize(new_rows),
                cols: self.cols,
                nnz: Cell::new(Some(I::ZERO)),
            });
        }

        // 过滤非空列
        let non_empty = collect_non_empty(&col_nnz);
        let non_empty_lens: Vec<usize> = non_empty.iter().map(|(_, nnz)| *nnz).collect();

        // 分配 Storage
        let mut val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
        let mut idx_spans = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

        // 填充数据的闭包
        let fill_col = |(alloc_idx, &(col_idx, _)): (usize, &(usize, usize))| {
            let val_span = unsafe { val_spans.get_unchecked(alloc_idx) };
            let idx_span = unsafe { idx_spans.get_unchecked(alloc_idx) };

            let old_vals = unsafe { self.values.get_unchecked(col_idx).as_slice() };
            let old_idxs = unsafe { self.indices.get_unchecked(col_idx).as_slice() };

            let left = lower_bound(old_idxs, row_start);
            let right = lower_bound(old_idxs, row_end);
            let copy_len = right - left;

            let val_ptr = val_span.as_ptr() as *mut V;
            let idx_ptr = idx_span.as_ptr() as *mut I;

            unsafe {
                std::ptr::copy_nonoverlapping(old_vals.as_ptr().add(left), val_ptr, copy_len);
                copy_sub_offset(old_idxs.as_ptr().add(left), idx_ptr, copy_len, row_start);
            }
        };

        if should_parallelize(non_empty.len()) {
            non_empty.par_iter().enumerate().for_each(fill_col);
        } else {
            non_empty.iter().enumerate().for_each(fill_col);
        }

        // 组装最终结果
        let (values, indices) = assemble_result(ncols, &non_empty, &mut val_spans, &mut idx_spans);

        Ok(Self {
            values,
            indices,
            rows: I::from_usize(new_rows),
            cols: self.cols,
            nnz: Cell::new(Some(I::from_usize(total_nnz))),
        })
    }

    /// 行掩码切片 - O(nnz)
    pub fn slice_rows_mask<const ALIGN: usize>(&self, mask: &[bool]) -> SliceResult<Self> {
        let nrows = self.rows.to_usize();

        if unlikely(mask.len() != nrows) {
            return Err(SliceError::MaskLengthMismatch {
                expected: nrows,
                got: mask.len(),
            });
        }

        let ncols = self.cols.to_usize();
        let selected_rows = count_true_fast(mask);

        if unlikely(selected_rows == 0 || ncols == 0) {
            let values = (0..ncols).map(|_| empty_span()).collect();
            let indices = (0..ncols).map(|_| empty_span()).collect();
            return Ok(Self {
                values,
                indices,
                rows: I::from_usize(selected_rows),
                cols: self.cols,
                nnz: Cell::new(Some(I::ZERO)),
            });
        }

        // 构建行索引映射表
        let mut row_map: Vec<I> = vec![I::MAX; nrows];
        let mut new_row_idx = I::ZERO;
        for (old_row, &selected) in mask.iter().enumerate() {
            if selected {
                row_map[old_row] = new_row_idx;
                new_row_idx = new_row_idx + I::ONE;
            }
        }
        let new_rows = new_row_idx;

        // 计算每列的新 nnz
        let row_map_ref = &row_map;
        let compute_nnz = |idx_span: &Span<I>| {
            idx_span
                .as_slice()
                .iter()
                .filter(|&&row| row_map_ref[row.to_usize()] != I::MAX)
                .count()
        };

        let col_nnz: Vec<usize> = if should_parallelize(ncols) {
            self.indices.par_iter().map(compute_nnz).collect()
        } else {
            self.indices.iter().map(compute_nnz).collect()
        };

        let total_nnz: usize = col_nnz.iter().sum();

        if unlikely(total_nnz == 0) {
            let values = (0..ncols).map(|_| empty_span()).collect();
            let indices = (0..ncols).map(|_| empty_span()).collect();
            return Ok(Self {
                values,
                indices,
                rows: new_rows,
                cols: self.cols,
                nnz: Cell::new(Some(I::ZERO)),
            });
        }

        // 过滤非空列
        let non_empty = collect_non_empty(&col_nnz);
        let non_empty_lens: Vec<usize> = non_empty.iter().map(|(_, nnz)| *nnz).collect();

        // 分配 Storage
        let mut val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
        let mut idx_spans = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

        // 填充数据的闭包
        let fill_col = |(alloc_idx, &(col_idx, _)): (usize, &(usize, usize))| {
            let val_span = unsafe { val_spans.get_unchecked(alloc_idx) };
            let idx_span = unsafe { idx_spans.get_unchecked(alloc_idx) };

            let old_vals = unsafe { self.values.get_unchecked(col_idx).as_slice() };
            let old_idxs = unsafe { self.indices.get_unchecked(col_idx).as_slice() };

            let val_ptr = val_span.as_ptr() as *mut V;
            let idx_ptr = idx_span.as_ptr() as *mut I;

            unsafe {
                filter_copy_with_map(old_vals, old_idxs, val_ptr, idx_ptr, row_map_ref, I::MAX);
            }
        };

        if should_parallelize(non_empty.len()) {
            non_empty.par_iter().enumerate().for_each(fill_col);
        } else {
            non_empty.iter().enumerate().for_each(fill_col);
        }

        // 组装最终结果
        let (values, indices) = assemble_result(ncols, &non_empty, &mut val_spans, &mut idx_spans);

        Ok(Self {
            values,
            indices,
            rows: new_rows,
            cols: self.cols,
            nnz: Cell::new(Some(I::from_usize(total_nnz))),
        })
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_csr() -> CSR<f64, i64> {
        // 3x4 CSR:
        // [1 0 2 0]
        // [0 3 0 4]
        // [5 0 6 0]
        let values = vec![
            Span::copy_from::<32>(&[1.0, 2.0]).unwrap(),
            Span::copy_from::<32>(&[3.0, 4.0]).unwrap(),
            Span::copy_from::<32>(&[5.0, 6.0]).unwrap(),
        ];
        let indices = vec![
            Span::copy_from::<32>(&[0i64, 2]).unwrap(),
            Span::copy_from::<32>(&[1i64, 3]).unwrap(),
            Span::copy_from::<32>(&[0i64, 2]).unwrap(),
        ];
        unsafe { CSR::from_raw_parts_with_nnz(values, indices, 3, 4, 6) }
    }

    fn create_test_csc() -> CSC<f64, i64> {
        // Same 3x4 matrix in CSC
        let values = vec![
            Span::copy_from::<32>(&[1.0, 5.0]).unwrap(),
            Span::copy_from::<32>(&[3.0]).unwrap(),
            Span::copy_from::<32>(&[2.0, 6.0]).unwrap(),
            Span::copy_from::<32>(&[4.0]).unwrap(),
        ];
        let indices = vec![
            Span::copy_from::<32>(&[0i64, 2]).unwrap(),
            Span::copy_from::<32>(&[1i64]).unwrap(),
            Span::copy_from::<32>(&[0i64, 2]).unwrap(),
            Span::copy_from::<32>(&[1i64]).unwrap(),
        ];
        unsafe { CSC::from_raw_parts_with_nnz(values, indices, 3, 4, 6) }
    }

    #[test]
    fn test_csr_slice_rows() {
        let csr = create_test_csr();
        let sub = csr.slice_rows(0, 2).unwrap();
        assert_eq!(sub.nrows(), 2);
        assert_eq!(sub.ncols(), 4);
        assert_eq!(sub.nnz(), 4);
        assert_eq!(sub.row_values(0), &[1.0, 2.0]);
        assert_eq!(sub.row_values(1), &[3.0, 4.0]);
    }

    #[test]
    fn test_csr_slice_rows_mask() {
        let csr = create_test_csr();
        let mask = [true, false, true];
        let sub = csr.slice_rows_mask(&mask).unwrap();
        assert_eq!(sub.nrows(), 2);
        assert_eq!(sub.nnz(), 4);
        assert_eq!(sub.row_values(0), &[1.0, 2.0]);
        assert_eq!(sub.row_values(1), &[5.0, 6.0]);
    }

    #[test]
    fn test_csr_slice_cols() {
        let csr = create_test_csr();
        // [0, 2) -> col 0 and col 1
        let sub = csr.slice_cols::<32>(0, 2).unwrap();
        assert_eq!(sub.nrows(), 3);
        assert_eq!(sub.ncols(), 2);
        assert_eq!(sub.nnz(), 3); // col0: 1,5; col1: 3
        assert_eq!(sub.row_values(0), &[1.0]);
        assert_eq!(sub.row_indices(0), &[0]);
        assert_eq!(sub.row_values(1), &[3.0]);
        assert_eq!(sub.row_indices(1), &[1]);
        assert_eq!(sub.row_values(2), &[5.0]);
        assert_eq!(sub.row_indices(2), &[0]);
    }

    #[test]
    fn test_csr_slice_cols_mask() {
        let csr = create_test_csr();
        // Select col 0 and col 2
        let mask = [true, false, true, false];
        let sub = csr.slice_cols_mask::<32>(&mask).unwrap();
        assert_eq!(sub.nrows(), 3);
        assert_eq!(sub.ncols(), 2);
        assert_eq!(sub.nnz(), 4);
        assert_eq!(sub.row_values(0), &[1.0, 2.0]);
        assert_eq!(sub.row_indices(0), &[0, 1]); // remapped
    }

    #[test]
    fn test_csc_slice_cols() {
        let csc = create_test_csc();
        let sub = csc.slice_cols(1, 3).unwrap();
        assert_eq!(sub.nrows(), 3);
        assert_eq!(sub.ncols(), 2);
        assert_eq!(sub.nnz(), 3);
    }

    #[test]
    fn test_csc_slice_rows() {
        let csc = create_test_csc();
        let sub = csc.slice_rows::<32>(0, 2).unwrap();
        assert_eq!(sub.nrows(), 2);
        assert_eq!(sub.ncols(), 4);
        assert_eq!(sub.nnz(), 4);
        assert_eq!(sub.col_indices(0), &[0]); // only row 0 in range
    }

    #[test]
    fn test_empty_slice() {
        let csr = create_test_csr();
        let sub = csr.slice_rows(1, 1).unwrap();
        assert_eq!(sub.nrows(), 0);
        assert_eq!(sub.nnz(), 0);

        let mask = [false, false, false];
        let sub = csr.slice_rows_mask(&mask).unwrap();
        assert_eq!(sub.nrows(), 0);
    }

    #[test]
    fn test_out_of_bounds() {
        let csr = create_test_csr();
        assert!(matches!(
            csr.slice_rows(0, 5),
            Err(SliceError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn test_mask_length_mismatch() {
        let csr = create_test_csr();
        let bad_mask = [true, false];
        assert!(matches!(
            csr.slice_rows_mask(&bad_mask),
            Err(SliceError::MaskLengthMismatch { .. })
        ));
    }

    #[test]
    fn test_lower_bound() {
        let arr: Vec<i64> = vec![1, 3, 5, 7, 9];
        assert_eq!(lower_bound(&arr, 0), 0);
        assert_eq!(lower_bound(&arr, 1), 0);
        assert_eq!(lower_bound(&arr, 2), 1);
        assert_eq!(lower_bound(&arr, 5), 2);
        assert_eq!(lower_bound(&arr, 10), 5);
    }

    #[test]
    fn test_count_true_fast() {
        let mask = [true, false, true, false, true, true, false, true];
        assert_eq!(count_true_fast(&mask), 5);
        assert_eq!(count_true_fast(&vec![true; 100]), 100);
        assert_eq!(count_true_fast(&vec![false; 100]), 0);
    }

    #[test]
    fn test_empty_span_is_valid() {
        let span: Span<f64> = empty_span();
        assert_eq!(span.len(), 0);
        assert!(span.is_empty());
        // 确保 as_slice 不会 panic
        assert_eq!(span.as_slice(), &[]);
    }
}
