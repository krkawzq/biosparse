//! 稀疏矩阵堆叠操作
//!
//! # 设计理念
//!
//! - **CSR::vstack / CSC::hstack** - 廉价操作（Span clone，仅增加引用计数）
//! - **CSR::hstack / CSC::vstack** - 需要重新分配（indices 加偏移）
//!
//! # 并行策略
//!
//! - 使用 rayon 并行化：
//!   - Span clone（大规模时）
//!   - 每行/列 nnz 计算
//!   - 数据复制 + indices 加偏移
//!
//! # SIMD 优化
//!
//! - indices 加偏移使用 16 路循环展开 + prefetch 诱导向量化
//! - 小数据（< 16 元素）使用快速路径避免分支开销
//! - 编译器在 release 模式下会自动向量化
//!
//! # 边界情况处理
//!
//! - 空输入 → `Err(StackError::EmptyInput)`
//! - 单个矩阵 → `clone()` 返回
//! - 维度不匹配 → `Err(StackError::DimensionMismatch)`
//! - 某些行/列全空 → 返回 empty Span
//! - 全部为空 → 返回有效的零矩阵

#![allow(clippy::missing_safety_doc)]

use std::cell::Cell;
use std::ptr::NonNull;

use rayon::prelude::*;

use crate::convert::AllocStrategy;
use crate::span::{Span, SpanFlags};
use crate::sparse::{SparseIndex, CSC, CSR};
use crate::storage::AllocError;
use crate::tools::{prefetch_read, prefetch_write, unlikely};

// =============================================================================
// 错误类型
// =============================================================================

/// Stack 操作错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StackError {
    /// 空输入（矩阵列表为空）
    EmptyInput,

    /// 维度不匹配
    DimensionMismatch {
        /// 期望的维度
        expected: usize,
        /// 实际得到的维度
        got: usize,
        /// 出错的矩阵索引
        index: usize,
    },

    /// 内存分配失败
    Alloc(AllocError),
}

impl From<AllocError> for StackError {
    #[inline]
    fn from(e: AllocError) -> Self {
        StackError::Alloc(e)
    }
}

impl std::fmt::Display for StackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StackError::EmptyInput => write!(f, "empty input"),
            StackError::DimensionMismatch {
                expected,
                got,
                index,
            } => {
                write!(
                    f,
                    "dimension mismatch at index {}: expected {}, got {}",
                    index, expected, got
                )
            }
            StackError::Alloc(e) => write!(f, "allocation error: {:?}", e),
        }
    }
}

impl std::error::Error for StackError {}

// =============================================================================
// 常量和辅助
// =============================================================================

/// 并行处理的最小元素阈值（行/列数）
///
/// 对于昂贵操作（如 hstack/vstack 中的数据复制），256 是一个合理的阈值。
/// 对于廉价操作（如 Span clone），由于只涉及原子引用计数操作，
/// 并行化开销可能超过收益，因此使用更高的阈值。
const PARALLEL_THRESHOLD_COUNT: usize = 256;

/// Span clone 并行化的阈值
///
/// Span::clone() 只增加 Arc 的引用计数（原子操作），非常快。
/// 并行化的线程调度开销可能超过收益，因此使用较高阈值。
const PARALLEL_CLONE_THRESHOLD: usize = 4096;

/// 创建空 Span（用于空行/列）
///
/// 使用 `NonNull::dangling()` 创建一个有效但不可解引用的指针。
/// 由于 len=0，Span 的所有方法都不会实际访问该指针。
#[inline(always)]
fn empty_span<T>() -> Span<T> {
    // SAFETY: dangling() 返回一个对齐的非空指针，适用于零长度切片
    unsafe { Span::from_raw_parts_unchecked(NonNull::dangling(), 0, SpanFlags::VIEW) }
}

/// 根据行/列数决定是否并行（用于昂贵操作）
#[inline(always)]
fn should_parallelize_count(count: usize) -> bool {
    count >= PARALLEL_THRESHOLD_COUNT
}

/// 根据行/列数决定是否并行 clone Span（用于廉价操作）
#[inline(always)]
fn should_parallelize_clone(count: usize) -> bool {
    count >= PARALLEL_CLONE_THRESHOLD
}

// =============================================================================
// SIMD 辅助：indices 加偏移
// =============================================================================

/// 预取距离（元素数）
///
/// 设置为 32 个元素，对于 i32 是 128 字节（2 个缓存行），
/// 对于 i64 是 256 字节（4 个缓存行）。
/// 这个距离在大多数现代 CPU 上能提供良好的预取效果。
const PREFETCH_DISTANCE: usize = 32;

/// 复制并添加偏移（诱导 SIMD 向量化）
///
/// 将 src 中的每个元素加上 offset，写入 dst
///
/// # Safety
///
/// - src 和 dst 不重叠
/// - src 可读 count 个元素
/// - dst 可写 count 个元素
///
/// # 优化
///
/// - 小数据（< 16）：直接循环，避免分支开销
/// - 大数据：16 路展开 + prefetch，诱导 SIMD 向量化
#[inline(always)]
unsafe fn copy_add_offset<I: SparseIndex>(src: *const I, dst: *mut I, count: usize, offset: I) {
    // 小数据快速路径
    if count < 16 {
        for i in 0..count {
            *dst.add(i) = *src.add(i) + offset;
        }
        return;
    }

    // 16 路展开诱导 SIMD
    let main_count = count & !15;
    let mut i = 0;

    // 主循环：16 路展开 + prefetch
    while i < main_count {
        // 预取后续数据（PREFETCH_DISTANCE 个元素）
        prefetch_read(src.add(i + PREFETCH_DISTANCE));
        prefetch_write(dst.add(i + PREFETCH_DISTANCE));

        // 编译器会将这些操作向量化
        *dst.add(i) = *src.add(i) + offset;
        *dst.add(i + 1) = *src.add(i + 1) + offset;
        *dst.add(i + 2) = *src.add(i + 2) + offset;
        *dst.add(i + 3) = *src.add(i + 3) + offset;
        *dst.add(i + 4) = *src.add(i + 4) + offset;
        *dst.add(i + 5) = *src.add(i + 5) + offset;
        *dst.add(i + 6) = *src.add(i + 6) + offset;
        *dst.add(i + 7) = *src.add(i + 7) + offset;
        *dst.add(i + 8) = *src.add(i + 8) + offset;
        *dst.add(i + 9) = *src.add(i + 9) + offset;
        *dst.add(i + 10) = *src.add(i + 10) + offset;
        *dst.add(i + 11) = *src.add(i + 11) + offset;
        *dst.add(i + 12) = *src.add(i + 12) + offset;
        *dst.add(i + 13) = *src.add(i + 13) + offset;
        *dst.add(i + 14) = *src.add(i + 14) + offset;
        *dst.add(i + 15) = *src.add(i + 15) + offset;
        i += 16;
    }

    // 余数处理
    while i < count {
        *dst.add(i) = *src.add(i) + offset;
        i += 1;
    }
}

// =============================================================================
// CSR Stack Operations
// =============================================================================

impl<V: Clone + Send + Sync, I: SparseIndex + Send + Sync> CSR<V, I> {
    /// 垂直堆叠（增加行）- 廉价操作
    ///
    /// 时间复杂度: O(n) 其中 n 是总行数
    /// 空间复杂度: O(1) 额外空间（Span clone 只增加引用计数）
    ///
    /// # 边界情况
    ///
    /// - 空输入: 返回错误
    /// - 单个矩阵: clone 返回
    /// - 某些矩阵 nnz=0: 正常处理
    ///
    /// # 注意
    ///
    /// 如果 `total_rows` 超过索引类型 `I` 的最大值，`I::from_usize()` 可能会
    /// panic 或产生不正确的结果。
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = CSR::vstack(&[&a, &b, &c])?;
    /// assert_eq!(result.rows, a.rows + b.rows + c.rows);
    /// assert_eq!(result.cols, a.cols); // cols 保持不变
    /// ```
    pub fn vstack(matrices: &[&Self]) -> Result<Self, StackError> {
        // 边界检查
        if unlikely(matrices.is_empty()) {
            return Err(StackError::EmptyInput);
        }
        if matrices.len() == 1 {
            return Ok((*matrices[0]).clone());
        }

        // 检查 cols 一致性
        let cols = matrices[0].cols;
        for (i, m) in matrices.iter().enumerate().skip(1) {
            if unlikely(m.cols != cols) {
                return Err(StackError::DimensionMismatch {
                    expected: cols.to_usize(),
                    got: m.cols.to_usize(),
                    index: i,
                });
            }
        }

        // 并行计算 total_rows 和 total_nnz
        let (total_rows, total_nnz): (usize, usize) = if should_parallelize_count(matrices.len()) {
            matrices
                .par_iter()
                .map(|m| (m.rows.to_usize(), m.nnz().to_usize()))
                .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
        } else {
            matrices.iter().fold((0, 0), |acc, m| {
                (acc.0 + m.rows.to_usize(), acc.1 + m.nnz().to_usize())
            })
        };

        // 预分配
        let mut values = Vec::with_capacity(total_rows);
        let mut indices = Vec::with_capacity(total_rows);

        // 收集 Span（根据规模决定是否并行）
        // 注意：Span::clone() 只增加引用计数，非常快，使用较高的并行阈值
        for m in matrices {
            let row_count = m.values.len();

            if should_parallelize_clone(row_count) {
                // 并行 clone Span（仅对超大规模数据有收益）
                let (v, i): (Vec<_>, Vec<_>) = m
                    .values
                    .par_iter()
                    .zip(m.indices.par_iter())
                    .map(|(v, i)| (v.clone(), i.clone()))
                    .unzip();
                values.extend(v);
                indices.extend(i);
            } else {
                // 顺序 clone（对于大多数情况更快）
                values.extend(m.values.iter().cloned());
                indices.extend(m.indices.iter().cloned());
            }
        }

        Ok(CSR {
            values,
            indices,
            rows: I::from_usize(total_rows),
            cols,
            nnz: Cell::new(Some(I::from_usize(total_nnz))),
        })
    }
}

impl<V: Copy + Send + Sync, I: SparseIndex + Send + Sync> CSR<V, I> {
    /// 水平堆叠（增加列）- 需要分配
    ///
    /// 时间复杂度: O(nnz_total)
    /// 空间复杂度: O(nnz_total)
    ///
    /// # 边界情况
    ///
    /// - 空输入: 返回错误
    /// - 单个矩阵: clone 返回
    /// - 某些行全空: 返回空 Span
    ///
    /// # 并行策略
    ///
    /// 1. 并行计算每行的总 nnz
    /// 2. 顺序分配 Storage
    /// 3. 并行复制数据 + SIMD 加偏移
    ///
    /// # 注意
    ///
    /// - `strategy` 参数当前未使用，预留用于未来的分配策略优化
    /// - 如果 `total_rows` 或 `total_cols` 超过索引类型 `I` 的最大值，
    ///   `I::from_usize()` 可能会 panic 或产生不正确的结果
    pub fn hstack<const ALIGN: usize>(
        matrices: &[&Self],
        #[allow(unused_variables)] strategy: AllocStrategy,
    ) -> Result<Self, StackError> {
        // TODO: 根据 strategy 参数实现不同的分配策略
        //       - Fragmented: 每行独立 Storage
        //       - SingleBuffer: 所有行共享一个 Storage
        //       - MinBufferSize/BufferCount: 分组策略
        let _ = strategy;
        // 边界检查
        if unlikely(matrices.is_empty()) {
            return Err(StackError::EmptyInput);
        }
        if matrices.len() == 1 {
            return Ok((*matrices[0]).clone());
        }

        // 检查 rows 一致性
        let rows = matrices[0].rows;
        let row_count = rows.to_usize();
        for (i, m) in matrices.iter().enumerate().skip(1) {
            if unlikely(m.rows != rows) {
                return Err(StackError::DimensionMismatch {
                    expected: row_count,
                    got: m.rows.to_usize(),
                    index: i,
                });
            }
        }

        // 计算 total_cols 和列偏移，同时检查是否全空
        let mut col_offsets = Vec::with_capacity(matrices.len());
        let mut total_cols = I::ZERO;
        let mut all_empty = true;
        for m in matrices {
            col_offsets.push(total_cols);
            total_cols = total_cols + m.cols;
            if all_empty && m.nnz() != I::ZERO {
                all_empty = false;
            }
        }

        // 快速路径：所有输入矩阵都为空
        if all_empty {
            let values = (0..row_count).map(|_| empty_span()).collect();
            let indices = (0..row_count).map(|_| empty_span()).collect();
            return Ok(unsafe {
                CSR::from_raw_parts_with_nnz(values, indices, rows, total_cols, I::ZERO)
            });
        }

        // 并行计算每行的 nnz
        let row_nnzs: Vec<usize> = (0..row_count)
            .into_par_iter()
            .map(|i| {
                let ii = I::from_usize(i);
                matrices
                    .iter()
                    .map(|m| unsafe { m.row_nnz_unchecked(ii).to_usize() })
                    .sum()
            })
            .collect();

        let total_nnz: usize = row_nnzs.par_iter().sum();

        // 过滤非空行并记录索引
        let non_empty: Vec<(usize, usize)> = row_nnzs
            .iter()
            .enumerate()
            .filter(|(_, &nnz)| nnz > 0)
            .map(|(i, &nnz)| (i, nnz))
            .collect();

        let non_empty_lens: Vec<usize> = non_empty.iter().map(|(_, nnz)| *nnz).collect();

        // 分配 Storage
        let mut val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
        let mut idx_spans = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

        // 并行复制数据 + 加偏移
        non_empty
            .par_iter()
            .enumerate()
            .for_each(|(alloc_idx, &(row_idx, expected_len))| {
                // SAFETY: alloc_idx 是唯一的，每个 Span 只被一个线程访问
                let val_span = unsafe { val_spans.get_unchecked(alloc_idx) };
                let idx_span = unsafe { idx_spans.get_unchecked(alloc_idx) };

                let val_ptr = val_span.as_ptr() as *mut V;
                let idx_ptr = idx_span.as_ptr() as *mut I;

                let mut pos = 0usize;
                let ii = I::from_usize(row_idx);

                for (mat_idx, m) in matrices.iter().enumerate() {
                    let row_vals = unsafe { m.row_values_unchecked(ii) };
                    let row_idxs = unsafe { m.row_indices_unchecked(ii) };
                    let len = row_vals.len();

                    if len == 0 {
                        continue;
                    }

                    let col_offset = col_offsets[mat_idx];

                    unsafe {
                        // 复制 values
                        std::ptr::copy_nonoverlapping(row_vals.as_ptr(), val_ptr.add(pos), len);

                        // 复制 indices + 加偏移（SIMD 优化）
                        copy_add_offset(row_idxs.as_ptr(), idx_ptr.add(pos), len, col_offset);
                    }

                    pos += len;
                }

                // 断言：确保填充了正确的长度
                debug_assert_eq!(pos, expected_len);
            });

        // 构建最终矩阵（双指针法，避免中间映射表）
        let mut values = Vec::with_capacity(row_count);
        let mut indices = Vec::with_capacity(row_count);
        let mut non_empty_idx = 0;

        for i in 0..row_count {
            if non_empty_idx < non_empty.len() && non_empty[non_empty_idx].0 == i {
                // 使用移动语义
                values.push(unsafe { Span::take_at(&mut val_spans, non_empty_idx) });
                indices.push(unsafe { Span::take_at(&mut idx_spans, non_empty_idx) });
                non_empty_idx += 1;
            } else {
                values.push(empty_span());
                indices.push(empty_span());
            }
        }

        Ok(unsafe {
            CSR::from_raw_parts_with_nnz(
                values,
                indices,
                rows,
                total_cols,
                I::from_usize(total_nnz),
            )
        })
    }
}

// =============================================================================
// CSC Stack Operations
// =============================================================================

impl<V: Clone + Send + Sync, I: SparseIndex + Send + Sync> CSC<V, I> {
    /// 水平堆叠（增加列）- 廉价操作
    ///
    /// 时间复杂度: O(n) 其中 n 是总列数
    /// 空间复杂度: O(1) 额外空间（Span clone 只增加引用计数）
    ///
    /// # 边界情况
    ///
    /// - 空输入: 返回错误
    /// - 单个矩阵: clone 返回
    /// - 某些矩阵 nnz=0: 正常处理
    ///
    /// # 注意
    ///
    /// 如果 `total_cols` 超过索引类型 `I` 的最大值，`I::from_usize()` 可能会
    /// panic 或产生不正确的结果。
    pub fn hstack(matrices: &[&Self]) -> Result<Self, StackError> {
        // 边界检查
        if unlikely(matrices.is_empty()) {
            return Err(StackError::EmptyInput);
        }
        if matrices.len() == 1 {
            return Ok((*matrices[0]).clone());
        }

        // 检查 rows 一致性
        let rows = matrices[0].rows;
        for (i, m) in matrices.iter().enumerate().skip(1) {
            if unlikely(m.rows != rows) {
                return Err(StackError::DimensionMismatch {
                    expected: rows.to_usize(),
                    got: m.rows.to_usize(),
                    index: i,
                });
            }
        }

        // 并行计算 total_cols 和 total_nnz
        let (total_cols, total_nnz): (usize, usize) = if should_parallelize_count(matrices.len()) {
            matrices
                .par_iter()
                .map(|m| (m.cols.to_usize(), m.nnz().to_usize()))
                .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
        } else {
            matrices.iter().fold((0, 0), |acc, m| {
                (acc.0 + m.cols.to_usize(), acc.1 + m.nnz().to_usize())
            })
        };

        // 预分配
        let mut values = Vec::with_capacity(total_cols);
        let mut indices = Vec::with_capacity(total_cols);

        // 收集 Span（根据规模决定是否并行）
        // 注意：Span::clone() 只增加引用计数，非常快，使用较高的并行阈值
        for m in matrices {
            let col_count = m.values.len();

            if should_parallelize_clone(col_count) {
                // 并行 clone Span（仅对超大规模数据有收益）
                let (v, i): (Vec<_>, Vec<_>) = m
                    .values
                    .par_iter()
                    .zip(m.indices.par_iter())
                    .map(|(v, i)| (v.clone(), i.clone()))
                    .unzip();
                values.extend(v);
                indices.extend(i);
            } else {
                // 顺序 clone（对于大多数情况更快）
                values.extend(m.values.iter().cloned());
                indices.extend(m.indices.iter().cloned());
            }
        }

        Ok(CSC {
            values,
            indices,
            rows,
            cols: I::from_usize(total_cols),
            nnz: Cell::new(Some(I::from_usize(total_nnz))),
        })
    }
}

impl<V: Copy + Send + Sync, I: SparseIndex + Send + Sync> CSC<V, I> {
    /// 垂直堆叠（增加行）- 需要分配
    ///
    /// 时间复杂度: O(nnz_total)
    /// 空间复杂度: O(nnz_total)
    ///
    /// # 边界情况
    ///
    /// - 空输入: 返回错误
    /// - 单个矩阵: clone 返回
    /// - 某些列全空: 返回空 Span
    ///
    /// # 注意
    ///
    /// - `strategy` 参数当前未使用，预留用于未来的分配策略优化
    /// - 如果 `total_rows` 或 `total_cols` 超过索引类型 `I` 的最大值，
    ///   `I::from_usize()` 可能会 panic 或产生不正确的结果
    pub fn vstack<const ALIGN: usize>(
        matrices: &[&Self],
        #[allow(unused_variables)] strategy: AllocStrategy,
    ) -> Result<Self, StackError> {
        // TODO: 根据 strategy 参数实现不同的分配策略
        let _ = strategy;
        // 边界检查
        if unlikely(matrices.is_empty()) {
            return Err(StackError::EmptyInput);
        }
        if matrices.len() == 1 {
            return Ok((*matrices[0]).clone());
        }

        // 检查 cols 一致性
        let cols = matrices[0].cols;
        let col_count = cols.to_usize();
        for (i, m) in matrices.iter().enumerate().skip(1) {
            if unlikely(m.cols != cols) {
                return Err(StackError::DimensionMismatch {
                    expected: col_count,
                    got: m.cols.to_usize(),
                    index: i,
                });
            }
        }

        // 计算 total_rows 和行偏移，同时检查是否全空
        let mut row_offsets = Vec::with_capacity(matrices.len());
        let mut total_rows = I::ZERO;
        let mut all_empty = true;
        for m in matrices {
            row_offsets.push(total_rows);
            total_rows = total_rows + m.rows;
            if all_empty && m.nnz() != I::ZERO {
                all_empty = false;
            }
        }

        // 快速路径：所有输入矩阵都为空
        if all_empty {
            let values = (0..col_count).map(|_| empty_span()).collect();
            let indices = (0..col_count).map(|_| empty_span()).collect();
            return Ok(unsafe {
                CSC::from_raw_parts_with_nnz(values, indices, total_rows, cols, I::ZERO)
            });
        }

        // 并行计算每列的 nnz
        let col_nnzs: Vec<usize> = (0..col_count)
            .into_par_iter()
            .map(|j| {
                let jj = I::from_usize(j);
                matrices
                    .iter()
                    .map(|m| unsafe { m.col_nnz_unchecked(jj).to_usize() })
                    .sum()
            })
            .collect();

        let total_nnz: usize = col_nnzs.par_iter().sum();

        // 过滤非空列
        let non_empty: Vec<(usize, usize)> = col_nnzs
            .iter()
            .enumerate()
            .filter(|(_, &nnz)| nnz > 0)
            .map(|(j, &nnz)| (j, nnz))
            .collect();

        let non_empty_lens: Vec<usize> = non_empty.iter().map(|(_, nnz)| *nnz).collect();

        // 分配 Storage
        let mut val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
        let mut idx_spans = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

        // 并行复制数据 + 加偏移
        non_empty
            .par_iter()
            .enumerate()
            .for_each(|(alloc_idx, &(col_idx, expected_len))| {
                let val_span = unsafe { val_spans.get_unchecked(alloc_idx) };
                let idx_span = unsafe { idx_spans.get_unchecked(alloc_idx) };

                let val_ptr = val_span.as_ptr() as *mut V;
                let idx_ptr = idx_span.as_ptr() as *mut I;

                let mut pos = 0usize;
                let jj = I::from_usize(col_idx);

                for (mat_idx, m) in matrices.iter().enumerate() {
                    let col_vals = unsafe { m.col_values_unchecked(jj) };
                    let col_idxs = unsafe { m.col_indices_unchecked(jj) };
                    let len = col_vals.len();

                    if len == 0 {
                        continue;
                    }

                    let row_offset = row_offsets[mat_idx];

                    unsafe {
                        // 复制 values
                        std::ptr::copy_nonoverlapping(col_vals.as_ptr(), val_ptr.add(pos), len);

                        // 复制 indices + 加偏移
                        copy_add_offset(col_idxs.as_ptr(), idx_ptr.add(pos), len, row_offset);
                    }

                    pos += len;
                }

                debug_assert_eq!(pos, expected_len);
            });

        // 构建最终矩阵（双指针法，避免中间映射表）
        let mut values = Vec::with_capacity(col_count);
        let mut indices = Vec::with_capacity(col_count);
        let mut non_empty_idx = 0;

        for j in 0..col_count {
            if non_empty_idx < non_empty.len() && non_empty[non_empty_idx].0 == j {
                values.push(unsafe { Span::take_at(&mut val_spans, non_empty_idx) });
                indices.push(unsafe { Span::take_at(&mut idx_spans, non_empty_idx) });
                non_empty_idx += 1;
            } else {
                values.push(empty_span());
                indices.push(empty_span());
            }
        }

        Ok(unsafe {
            CSC::from_raw_parts_with_nnz(
                values,
                indices,
                total_rows,
                cols,
                I::from_usize(total_nnz),
            )
        })
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: 创建测试用 CSR
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

    // Helper: 创建测试用 CSC
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

    // =========================================================================
    // CSR::vstack tests
    // =========================================================================

    #[test]
    fn test_csr_vstack_empty_input() {
        let result = CSR::<f64, i64>::vstack(&[]);
        assert!(matches!(result, Err(StackError::EmptyInput)));
    }

    #[test]
    fn test_csr_vstack_single() {
        let a = make_test_csr::<32>(2, 3, &[&[1.0, 2.0], &[3.0]], &[&[0, 1], &[2]]);

        let result = CSR::vstack(&[&a]).unwrap();
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 3);
        assert_eq!(result.nnz(), 3);
    }

    #[test]
    fn test_csr_vstack_two() {
        // A: 2x3
        //   [1, 2, 0]
        //   [0, 0, 3]
        let a = make_test_csr::<32>(2, 3, &[&[1.0, 2.0], &[3.0]], &[&[0, 1], &[2]]);

        // B: 2x3
        //   [4, 0, 0]
        //   [0, 5, 6]
        let b = make_test_csr::<32>(2, 3, &[&[4.0], &[5.0, 6.0]], &[&[0], &[1, 2]]);

        let result = CSR::vstack(&[&a, &b]).unwrap();

        assert_eq!(result.rows, 4);
        assert_eq!(result.cols, 3);
        assert_eq!(result.nnz(), 6);

        // 检查行数据
        assert_eq!(result.row_values(0), &[1.0, 2.0]);
        assert_eq!(result.row_indices(0), &[0, 1]);
        assert_eq!(result.row_values(1), &[3.0]);
        assert_eq!(result.row_indices(1), &[2]);
        assert_eq!(result.row_values(2), &[4.0]);
        assert_eq!(result.row_indices(2), &[0]);
        assert_eq!(result.row_values(3), &[5.0, 6.0]);
        assert_eq!(result.row_indices(3), &[1, 2]);
    }

    #[test]
    fn test_csr_vstack_dimension_mismatch() {
        let a = make_test_csr::<32>(2, 3, &[&[1.0], &[]], &[&[0], &[]]);
        let b = make_test_csr::<32>(2, 4, &[&[1.0], &[]], &[&[0], &[]]); // cols 不一致

        let result = CSR::vstack(&[&a, &b]);
        assert!(matches!(
            result,
            Err(StackError::DimensionMismatch {
                expected: 3,
                got: 4,
                index: 1
            })
        ));
    }

    #[test]
    fn test_csr_vstack_with_empty_rows() {
        let a = make_test_csr::<32>(2, 3, &[&[1.0], &[]], &[&[0], &[]]);
        let b = make_test_csr::<32>(2, 3, &[&[], &[2.0]], &[&[], &[1]]);

        let result = CSR::vstack(&[&a, &b]).unwrap();
        assert_eq!(result.rows, 4);
        assert_eq!(result.nnz(), 2);
    }

    // =========================================================================
    // CSR::hstack tests
    // =========================================================================

    #[test]
    fn test_csr_hstack_empty_input() {
        let result = CSR::<f64, i64>::hstack::<32>(&[], AllocStrategy::Auto);
        assert!(matches!(result, Err(StackError::EmptyInput)));
    }

    #[test]
    fn test_csr_hstack_single() {
        let a = make_test_csr::<32>(2, 3, &[&[1.0, 2.0], &[3.0]], &[&[0, 1], &[2]]);

        let result = CSR::hstack::<32>(&[&a], AllocStrategy::Auto).unwrap();
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 3);
        assert_eq!(result.nnz(), 3);
    }

    #[test]
    fn test_csr_hstack_two() {
        // A: 2x3
        //   [1, 2, 0]
        //   [0, 0, 3]
        let a = make_test_csr::<32>(2, 3, &[&[1.0, 2.0], &[3.0]], &[&[0, 1], &[2]]);

        // B: 2x2
        //   [4, 0]
        //   [5, 6]
        let b = make_test_csr::<32>(2, 2, &[&[4.0], &[5.0, 6.0]], &[&[0], &[0, 1]]);

        let result = CSR::hstack::<32>(&[&a, &b], AllocStrategy::Auto).unwrap();

        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 5); // 3 + 2

        // 行 0: [1, 2, 0, 4, 0]
        assert_eq!(result.row_values(0), &[1.0, 2.0, 4.0]);
        assert_eq!(result.row_indices(0), &[0, 1, 3]); // 4.0 在 col 3 (0+3)

        // 行 1: [0, 0, 3, 5, 6]
        assert_eq!(result.row_values(1), &[3.0, 5.0, 6.0]);
        assert_eq!(result.row_indices(1), &[2, 3, 4]); // 5.0 在 col 3 (0+3), 6.0 在 col 4 (1+3)
    }

    #[test]
    fn test_csr_hstack_dimension_mismatch() {
        let a = make_test_csr::<32>(2, 3, &[&[1.0], &[]], &[&[0], &[]]);
        let b = make_test_csr::<32>(3, 2, &[&[1.0], &[], &[]], &[&[0], &[], &[]]); // rows 不一致

        let result = CSR::hstack::<32>(&[&a, &b], AllocStrategy::Auto);
        assert!(matches!(
            result,
            Err(StackError::DimensionMismatch {
                expected: 2,
                got: 3,
                index: 1
            })
        ));
    }

    #[test]
    fn test_csr_hstack_with_empty_rows() {
        // A: 3x2, 行1为空
        let a = make_test_csr::<32>(3, 2, &[&[1.0], &[], &[2.0]], &[&[0], &[], &[1]]);

        // B: 3x2, 行0为空
        let b = make_test_csr::<32>(3, 2, &[&[], &[3.0], &[4.0]], &[&[], &[0], &[1]]);

        let result = CSR::hstack::<32>(&[&a, &b], AllocStrategy::Auto).unwrap();

        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 4);

        // 行 0: [1, 0, 0, 0]
        assert_eq!(result.row_values(0), &[1.0]);
        assert_eq!(result.row_indices(0), &[0]);

        // 行 1: [0, 0, 3, 0]
        assert_eq!(result.row_values(1), &[3.0]);
        assert_eq!(result.row_indices(1), &[2]); // 0 + 2 (col offset)

        // 行 2: [0, 2, 0, 4]
        assert_eq!(result.row_values(2), &[2.0, 4.0]);
        assert_eq!(result.row_indices(2), &[1, 3]);
    }

    #[test]
    fn test_csr_hstack_all_empty() {
        let a = make_test_csr::<32>(2, 3, &[&[], &[]], &[&[], &[]]);
        let b = make_test_csr::<32>(2, 2, &[&[], &[]], &[&[], &[]]);

        let result = CSR::hstack::<32>(&[&a, &b], AllocStrategy::Auto).unwrap();

        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 5);
        assert_eq!(result.nnz(), 0);
    }

    // =========================================================================
    // CSC::hstack tests
    // =========================================================================

    #[test]
    fn test_csc_hstack_empty_input() {
        let result = CSC::<f64, i64>::hstack(&[]);
        assert!(matches!(result, Err(StackError::EmptyInput)));
    }

    #[test]
    fn test_csc_hstack_two() {
        // A: 3x2
        let a = make_test_csc::<32>(3, 2, &[&[1.0, 2.0], &[3.0]], &[&[0, 1], &[2]]);

        // B: 3x2
        let b = make_test_csc::<32>(3, 2, &[&[4.0], &[5.0, 6.0]], &[&[0], &[1, 2]]);

        let result = CSC::hstack(&[&a, &b]).unwrap();

        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 4);
        assert_eq!(result.nnz(), 6);
    }

    // =========================================================================
    // CSC::vstack tests
    // =========================================================================

    #[test]
    fn test_csc_vstack_empty_input() {
        let result = CSC::<f64, i64>::vstack::<32>(&[], AllocStrategy::Auto);
        assert!(matches!(result, Err(StackError::EmptyInput)));
    }

    #[test]
    fn test_csc_vstack_two() {
        // A: 2x3
        let a = make_test_csc::<32>(2, 3, &[&[1.0], &[2.0], &[3.0]], &[&[0], &[1], &[0]]);

        // B: 2x3
        let b = make_test_csc::<32>(2, 3, &[&[4.0], &[5.0], &[6.0]], &[&[1], &[0], &[1]]);

        let result = CSC::vstack::<32>(&[&a, &b], AllocStrategy::Auto).unwrap();

        assert_eq!(result.rows, 4); // 2 + 2
        assert_eq!(result.cols, 3);

        // 列 0: 值 [1.0, 4.0], 行索引 [0, 3] (1 + 2 = 3)
        assert_eq!(result.col_values(0), &[1.0, 4.0]);
        assert_eq!(result.col_indices(0), &[0, 3]);

        // 列 1: 值 [2.0, 5.0], 行索引 [1, 2] (0 + 2 = 2)
        assert_eq!(result.col_values(1), &[2.0, 5.0]);
        assert_eq!(result.col_indices(1), &[1, 2]);
    }

    #[test]
    fn test_csc_vstack_with_empty_cols() {
        // A: 2x3, 列1为空
        let a = make_test_csc::<32>(2, 3, &[&[1.0], &[], &[2.0]], &[&[0], &[], &[1]]);

        // B: 2x3, 列2为空
        let b = make_test_csc::<32>(2, 3, &[&[3.0], &[4.0], &[]], &[&[0], &[1], &[]]);

        let result = CSC::vstack::<32>(&[&a, &b], AllocStrategy::Auto).unwrap();

        assert_eq!(result.rows, 4);
        assert_eq!(result.cols, 3);

        // 列 0: [1, 3] → 行 [0, 2]
        assert_eq!(result.col_values(0), &[1.0, 3.0]);
        assert_eq!(result.col_indices(0), &[0, 2]);

        // 列 1: [4] → 行 [3] (1+2)
        assert_eq!(result.col_values(1), &[4.0]);
        assert_eq!(result.col_indices(1), &[3]);

        // 列 2: [2] → 行 [1]
        assert_eq!(result.col_values(2), &[2.0]);
        assert_eq!(result.col_indices(2), &[1]);
    }
}
