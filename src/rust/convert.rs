//! 稀疏矩阵格式转换
//!
//! # 设计理念
//!
//! - 所有输入使用原生 Rust 类型（无额外封装）
//! - 充分利用 Span/Storage 的分配接口
//! - 对于未知 size 的情况，使用探测法：先计算 len 数组再分配
//!
//! # 转换矩阵
//!
//! | 源 \ 目标 | CSR | CSC |
//! |-----------|-----|-----|
//! | scipy CSR | view/copy | copy |
//! | scipy CSC | copy | view/copy |
//! | scipy COO | copy | copy |
//! | LIL | copy | copy |
//! | Dense | copy | copy |
//! | CSR | - | copy |
//! | CSC | copy | - |

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use std::cell::Cell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::span::{Span, SpanFlags};
use crate::sparse::{SparseIndex, CSC, CSR};
use crate::storage::AllocError;
use crate::tools::SendPtr;

/// 内存分配策略
///
/// TODO: 目前所有转换函数都使用类似 SingleBuffer 的策略（批量分配），
/// 尚未根据此参数实现不同的分配策略。未来可以根据矩阵特性（如 nnz 分布）
/// 选择最优策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocStrategy {
    /// 每行/列独立 Storage（最灵活，但可能碎片化）
    Fragmented,

    /// 所有行/列共享一个 Storage（最紧凑）
    SingleBuffer,

    /// 限制每个 Storage 的最小字节数（平衡方案）
    MinBufferSize(usize),

    /// 限制 Storage 总数（平衡方案）
    BufferCount(usize),

    /// 自动选择（基于 nnz 启发式）
    #[default]
    Auto,
}

/// Dense 矩阵布局
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenseLayout {
    /// 行主序 (C-order): data[i * cols + j]
    #[default]
    RowMajor,

    /// 列主序 (Fortran-order): data[j * rows + i]
    ColMajor,
}

/// 转换错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvertError {
    /// 维度不匹配
    DimensionMismatch,
    /// values 和 indices 长度不匹配
    LengthMismatch,
    /// indptr 格式错误
    InvalidIndptr,
    /// 索引越界
    IndexOutOfBounds,
    /// Buffer 太小
    BufferTooSmall,
    /// 内存分配失败
    Alloc(AllocError),
}

impl std::fmt::Display for ConvertError {
    #[cold]
    #[inline(never)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch => write!(f, "dimension mismatch"),
            Self::LengthMismatch => write!(f, "values and indices length mismatch"),
            Self::InvalidIndptr => write!(f, "invalid indptr format"),
            Self::IndexOutOfBounds => write!(f, "index out of bounds"),
            Self::BufferTooSmall => write!(f, "output buffer too small"),
            Self::Alloc(e) => write!(f, "allocation error: {}", e),
        }
    }
}

impl std::error::Error for ConvertError {}

impl From<AllocError> for ConvertError {
    #[inline]
    fn from(e: AllocError) -> Self {
        ConvertError::Alloc(e)
    }
}

// =============================================================================
// Helper: 空 Span
// =============================================================================

/// 空 Span 的共享地址（静态变量，所有空 Span 指向这里）
///
/// 使用 u8 数组并保证对齐，避免 UB
#[repr(C, align(64))]
struct EmptySpanAddr {
    _data: [u8; 64],
}

static EMPTY_SPAN_ADDR: EmptySpanAddr = EmptySpanAddr { _data: [0; 64] };

/// 创建空 Span（用于空行/列）
///
/// 所有空 Span 共享同一个静态地址
/// 注意：空 Span 也需要标记为 MUTABLE，因为 ensure_sorted 等函数会调用 as_slice_mut
#[inline(always)]
fn empty_span<T>() -> Span<T> {
    // SAFETY: EMPTY_SPAN_ADDR 是静态变量，地址永远有效
    // len=0 的 Span 不会实际读写这个地址
    let ptr = unsafe { NonNull::new_unchecked(EMPTY_SPAN_ADDR._data.as_ptr() as *mut T) };
    // 空 Span 标记为 VIEW | MUTABLE，因为 len=0 不会有实际写操作
    unsafe { Span::from_raw_parts_unchecked(ptr, 0, SpanFlags::VIEW | SpanFlags::MUTABLE) }
}

// =============================================================================
// Scipy CSR → CSR
// =============================================================================

/// scipy CSR → 我们的 CSR（View 模式，零拷贝）
///
/// # Safety
///
/// 调用者需确保：
/// - `data`, `indices`, `indptr` 指向有效内存
/// - 生命周期覆盖返回的 CSR
/// - `indptr` 长度为 rows + 1
#[inline]
pub unsafe fn csr_from_scipy_csr_view<V, I: SparseIndex>(
    rows: I,
    cols: I,
    data: *const V,
    indices: *const I,
    indptr: *const I,
) -> CSR<V, I> {
    let row_count = rows.to_usize();

    let mut values = Vec::with_capacity(row_count);
    let mut idx_spans = Vec::with_capacity(row_count);

    for i in 0..row_count {
        let start = (*indptr.add(i)).to_usize();
        let end = (*indptr.add(i + 1)).to_usize();
        let len = end - start;

        if len == 0 {
            values.push(empty_span());
            idx_spans.push(empty_span());
        } else {
            let val_ptr = NonNull::new_unchecked(data.add(start) as *mut V);
            let idx_ptr = NonNull::new_unchecked(indices.add(start) as *mut I);

            values.push(Span::from_raw_parts_unchecked(
                val_ptr,
                len,
                SpanFlags::VIEW,
            ));
            idx_spans.push(Span::from_raw_parts_unchecked(
                idx_ptr,
                len,
                SpanFlags::VIEW,
            ));
        }
    }

    CSR {
        values,
        indices: idx_spans,
        rows,
        cols,
        nnz: Cell::new(None),
    }
}

/// scipy CSR → 我们的 CSR（Copy 模式，并行）
pub fn csr_from_scipy_csr_copy<
    V: Copy + Send + Sync,
    I: SparseIndex + Send + Sync,
    const ALIGN: usize,
>(
    rows: I,
    cols: I,
    data: &[V],
    indices: &[I],
    indptr: &[I],
    _strategy: AllocStrategy,
) -> Result<CSR<V, I>, ConvertError> {
    let row_count = rows.to_usize();

    // 验证 indptr
    if indptr.len() != row_count + 1 {
        return Err(ConvertError::InvalidIndptr);
    }
    if row_count > 0 && indptr[0] != I::ZERO {
        return Err(ConvertError::InvalidIndptr);
    }

    // 验证 data 和 indices 长度
    let nnz = indptr[row_count].to_usize();
    if data.len() != nnz || indices.len() != nnz {
        return Err(ConvertError::LengthMismatch);
    }

    // 并行计算每行的长度
    let lens: Vec<usize> = (0..row_count)
        .into_par_iter()
        .map(|i| {
            let start = indptr[i].to_usize();
            let end = indptr[i + 1].to_usize();
            end.saturating_sub(start)
        })
        .collect();

    let total_nnz: usize = lens.par_iter().sum();

    // 处理全空的情况
    if total_nnz == 0 {
        let values: Vec<Span<V>> = (0..row_count).map(|_| empty_span()).collect();
        let idx_spans: Vec<Span<I>> = (0..row_count).map(|_| empty_span()).collect();

        return Ok(CSR {
            values,
            indices: idx_spans,
            rows,
            cols,
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    // 构建行索引到分配索引的映射
    let mut row_to_alloc = vec![usize::MAX; row_count];
    let non_empty_lens: Vec<usize> = lens
        .iter()
        .enumerate()
        .filter_map(|(_, &len)| if len > 0 { Some(len) } else { None })
        .collect();

    let mut alloc_idx = 0usize;
    for i in 0..row_count {
        if lens[i] > 0 {
            row_to_alloc[i] = alloc_idx;
            alloc_idx += 1;
        }
    }

    // 分配 Spans（SingleBuffer 策略）
    let val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
    let idx_spans_alloc = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

    // 并行复制数据
    (0..row_count).into_par_iter().for_each(|i| {
        let len = lens[i];
        if len > 0 {
            let start = indptr[i].to_usize();
            let alloc_idx = row_to_alloc[i];

            // SAFETY: 每个分配索引只被一个行使用，无并发冲突
            unsafe {
                let val_ptr = val_spans[alloc_idx].as_ptr() as *mut V;
                let idx_ptr = idx_spans_alloc[alloc_idx].as_ptr() as *mut I;

                std::ptr::copy_nonoverlapping(data.as_ptr().add(start), val_ptr, len);
                std::ptr::copy_nonoverlapping(indices.as_ptr().add(start), idx_ptr, len);
            }
        }
    });

    // 构建完整的 values 和 indices 数组
    let values: Vec<Span<V>> = (0..row_count)
        .map(|i| {
            if lens[i] == 0 {
                empty_span()
            } else {
                val_spans[row_to_alloc[i]].clone()
            }
        })
        .collect();

    let idx_spans: Vec<Span<I>> = (0..row_count)
        .map(|i| {
            if lens[i] == 0 {
                empty_span()
            } else {
                idx_spans_alloc[row_to_alloc[i]].clone()
            }
        })
        .collect();

    Ok(CSR {
        values,
        indices: idx_spans,
        rows,
        cols,
        nnz: Cell::new(Some(I::from_usize(total_nnz))),
    })
}

// =============================================================================
// Scipy CSC → CSC
// =============================================================================

/// scipy CSC → 我们的 CSC（View 模式）
#[inline]
pub unsafe fn csc_from_scipy_csc_view<V, I: SparseIndex>(
    rows: I,
    cols: I,
    data: *const V,
    indices: *const I,
    indptr: *const I,
) -> CSC<V, I> {
    let col_count = cols.to_usize();

    let mut values = Vec::with_capacity(col_count);
    let mut idx_spans = Vec::with_capacity(col_count);

    for j in 0..col_count {
        let start = (*indptr.add(j)).to_usize();
        let end = (*indptr.add(j + 1)).to_usize();
        let len = end - start;

        if len == 0 {
            values.push(empty_span());
            idx_spans.push(empty_span());
        } else {
            let val_ptr = NonNull::new_unchecked(data.add(start) as *mut V);
            let idx_ptr = NonNull::new_unchecked(indices.add(start) as *mut I);

            values.push(Span::from_raw_parts_unchecked(
                val_ptr,
                len,
                SpanFlags::VIEW,
            ));
            idx_spans.push(Span::from_raw_parts_unchecked(
                idx_ptr,
                len,
                SpanFlags::VIEW,
            ));
        }
    }

    CSC {
        values,
        indices: idx_spans,
        rows,
        cols,
        nnz: Cell::new(None),
    }
}

/// scipy CSC → 我们的 CSC（Copy 模式，并行）
pub fn csc_from_scipy_csc_copy<
    V: Copy + Send + Sync,
    I: SparseIndex + Send + Sync,
    const ALIGN: usize,
>(
    rows: I,
    cols: I,
    data: &[V],
    indices: &[I],
    indptr: &[I],
    _strategy: AllocStrategy,
) -> Result<CSC<V, I>, ConvertError> {
    let col_count = cols.to_usize();

    // 验证 indptr
    if indptr.len() != col_count + 1 {
        return Err(ConvertError::InvalidIndptr);
    }
    if col_count > 0 && indptr[0] != I::ZERO {
        return Err(ConvertError::InvalidIndptr);
    }

    let nnz = indptr[col_count].to_usize();
    if data.len() != nnz || indices.len() != nnz {
        return Err(ConvertError::LengthMismatch);
    }

    // 并行计算每列的长度
    let lens: Vec<usize> = (0..col_count)
        .into_par_iter()
        .map(|j| {
            let start = indptr[j].to_usize();
            let end = indptr[j + 1].to_usize();
            end.saturating_sub(start)
        })
        .collect();

    let total_nnz: usize = lens.par_iter().sum();

    // 处理全空
    if total_nnz == 0 {
        let values: Vec<Span<V>> = (0..col_count).map(|_| empty_span()).collect();
        let idx_spans: Vec<Span<I>> = (0..col_count).map(|_| empty_span()).collect();

        return Ok(CSC {
            values,
            indices: idx_spans,
            rows,
            cols,
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    // 构建列索引到分配索引的映射
    let mut col_to_alloc = vec![usize::MAX; col_count];
    let non_empty_lens: Vec<usize> = lens.iter().copied().filter(|&len| len > 0).collect();

    let mut alloc_idx = 0usize;
    for j in 0..col_count {
        if lens[j] > 0 {
            col_to_alloc[j] = alloc_idx;
            alloc_idx += 1;
        }
    }

    // 分配 Spans
    let val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
    let idx_spans_alloc = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

    // 并行复制数据
    (0..col_count).into_par_iter().for_each(|j| {
        let len = lens[j];
        if len > 0 {
            let start = indptr[j].to_usize();
            let alloc_idx = col_to_alloc[j];

            // SAFETY: 每个分配索引只被一个列使用，无并发冲突
            unsafe {
                let val_ptr = val_spans[alloc_idx].as_ptr() as *mut V;
                let idx_ptr = idx_spans_alloc[alloc_idx].as_ptr() as *mut I;

                std::ptr::copy_nonoverlapping(data.as_ptr().add(start), val_ptr, len);
                std::ptr::copy_nonoverlapping(indices.as_ptr().add(start), idx_ptr, len);
            }
        }
    });

    // 构建完整的 values 和 indices 数组
    let values: Vec<Span<V>> = (0..col_count)
        .map(|j| {
            if lens[j] == 0 {
                empty_span()
            } else {
                val_spans[col_to_alloc[j]].clone()
            }
        })
        .collect();

    let idx_spans: Vec<Span<I>> = (0..col_count)
        .map(|j| {
            if lens[j] == 0 {
                empty_span()
            } else {
                idx_spans_alloc[col_to_alloc[j]].clone()
            }
        })
        .collect();

    Ok(CSC {
        values,
        indices: idx_spans,
        rows,
        cols,
        nnz: Cell::new(Some(I::from_usize(total_nnz))),
    })
}

// =============================================================================
// Scipy 交叉转换：CSR → CSC, CSC → CSR
// =============================================================================

/// scipy CSR → 我们的 CSC（直接转换，并行）
///
/// 算法：探测法 + 原子位置分配
/// 1. 并行统计每列的 nnz（原子操作）
/// 2. 分配 CSC 存储
/// 3. 并行填充数据（原子位置分配）
pub fn csc_from_scipy_csr_copy<
    V: Copy + Default + Send + Sync,
    I: SparseIndex + Send + Sync,
    const ALIGN: usize,
>(
    rows: I,
    cols: I,
    data: &[V],
    col_indices: &[I],
    row_indptr: &[I],
    _strategy: AllocStrategy,
) -> Result<CSC<V, I>, ConvertError> {
    let row_count = rows.to_usize();
    let col_count = cols.to_usize();

    // 验证
    if row_indptr.len() != row_count + 1 {
        return Err(ConvertError::InvalidIndptr);
    }

    let nnz = if row_count > 0 {
        row_indptr[row_count].to_usize()
    } else {
        0
    };

    if data.len() != nnz || col_indices.len() != nnz {
        return Err(ConvertError::LengthMismatch);
    }

    // 第一遍：并行统计每列的 nnz（使用原子操作）
    let col_lens_atomic: Vec<AtomicUsize> = (0..col_count).map(|_| AtomicUsize::new(0)).collect();

    col_indices.par_iter().for_each(|&col_idx| {
        let j = col_idx.to_usize();
        debug_assert!(
            j < col_count,
            "column index {} out of bounds {}",
            j,
            col_count
        );
        col_lens_atomic[j].fetch_add(1, Ordering::Relaxed);
    });

    let col_lens: Vec<usize> = col_lens_atomic
        .iter()
        .map(|a| a.load(Ordering::Relaxed))
        .collect();

    let total_nnz: usize = col_lens.par_iter().sum();

    // 处理全空
    if total_nnz == 0 {
        let values: Vec<Span<V>> = (0..col_count).map(|_| empty_span()).collect();
        let idx_spans: Vec<Span<I>> = (0..col_count).map(|_| empty_span()).collect();

        return Ok(CSC {
            values,
            indices: idx_spans,
            rows,
            cols,
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    // 分配存储
    let non_empty_lens: Vec<usize> = col_lens.iter().copied().filter(|&len| len > 0).collect();
    let val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
    let idx_spans_alloc = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

    // 构建映射：列索引 → 分配索引
    let mut col_to_alloc = vec![usize::MAX; col_count];
    let mut alloc_idx = 0usize;
    for j in 0..col_count {
        if col_lens[j] > 0 {
            col_to_alloc[j] = alloc_idx;
            alloc_idx += 1;
        }
    }

    // 每列当前写入位置（原子）
    let col_pos: Vec<AtomicUsize> = (0..col_count).map(|_| AtomicUsize::new(0)).collect();

    // 第二遍：并行填充数据
    (0..row_count).into_par_iter().for_each(|i| {
        let start = row_indptr[i].to_usize();
        let end = row_indptr[i + 1].to_usize();

        for k in start..end {
            let j = col_indices[k].to_usize();
            let val = data[k];

            let alloc_idx = col_to_alloc[j];
            let pos = col_pos[j].fetch_add(1, Ordering::Relaxed);

            // SAFETY: 每个位置只写入一次，无并发冲突
            unsafe {
                let val_ptr = val_spans[alloc_idx].as_ptr() as *mut V;
                let idx_ptr = idx_spans_alloc[alloc_idx].as_ptr() as *mut I;
                *val_ptr.add(pos) = val;
                *idx_ptr.add(pos) = I::from_usize(i);
            }
        }
    });

    // 构建完整的 values 和 indices
    let values: Vec<Span<V>> = (0..col_count)
        .map(|j| {
            if col_lens[j] == 0 {
                empty_span()
            } else {
                val_spans[col_to_alloc[j]].clone()
            }
        })
        .collect();

    let idx_spans: Vec<Span<I>> = (0..col_count)
        .map(|j| {
            if col_lens[j] == 0 {
                empty_span()
            } else {
                idx_spans_alloc[col_to_alloc[j]].clone()
            }
        })
        .collect();

    let mut result = CSC {
        values,
        indices: idx_spans,
        rows,
        cols,
        nnz: Cell::new(Some(I::from_usize(total_nnz))),
    };

    // 并行填充不保证顺序，需要排序
    result.ensure_sorted();

    Ok(result)
}

/// scipy CSC → 我们的 CSR（直接转换，并行）
pub fn csr_from_scipy_csc_copy<
    V: Copy + Default + Send + Sync,
    I: SparseIndex + Send + Sync,
    const ALIGN: usize,
>(
    rows: I,
    cols: I,
    data: &[V],
    row_indices: &[I],
    col_indptr: &[I],
    _strategy: AllocStrategy,
) -> Result<CSR<V, I>, ConvertError> {
    let row_count = rows.to_usize();
    let col_count = cols.to_usize();

    if col_indptr.len() != col_count + 1 {
        return Err(ConvertError::InvalidIndptr);
    }

    let nnz = if col_count > 0 {
        col_indptr[col_count].to_usize()
    } else {
        0
    };

    if data.len() != nnz || row_indices.len() != nnz {
        return Err(ConvertError::LengthMismatch);
    }

    // 第一遍：并行统计每行的 nnz（使用原子操作）
    let row_lens_atomic: Vec<AtomicUsize> = (0..row_count).map(|_| AtomicUsize::new(0)).collect();

    row_indices.par_iter().for_each(|&row_idx| {
        let i = row_idx.to_usize();
        debug_assert!(i < row_count, "row index {} out of bounds {}", i, row_count);
        row_lens_atomic[i].fetch_add(1, Ordering::Relaxed);
    });

    let row_lens: Vec<usize> = row_lens_atomic
        .iter()
        .map(|a| a.load(Ordering::Relaxed))
        .collect();

    let total_nnz: usize = row_lens.par_iter().sum();

    if total_nnz == 0 {
        let values: Vec<Span<V>> = (0..row_count).map(|_| empty_span()).collect();
        let idx_spans: Vec<Span<I>> = (0..row_count).map(|_| empty_span()).collect();

        return Ok(CSR {
            values,
            indices: idx_spans,
            rows,
            cols,
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    let non_empty_lens: Vec<usize> = row_lens.iter().copied().filter(|&len| len > 0).collect();
    let val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
    let idx_spans_alloc = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

    let mut row_to_alloc = vec![usize::MAX; row_count];
    let mut alloc_idx = 0usize;
    for i in 0..row_count {
        if row_lens[i] > 0 {
            row_to_alloc[i] = alloc_idx;
            alloc_idx += 1;
        }
    }

    // 每行当前写入位置（原子）
    let row_pos: Vec<AtomicUsize> = (0..row_count).map(|_| AtomicUsize::new(0)).collect();

    // 第二遍：并行填充数据
    (0..col_count).into_par_iter().for_each(|j| {
        let start = col_indptr[j].to_usize();
        let end = col_indptr[j + 1].to_usize();

        for k in start..end {
            let i = row_indices[k].to_usize();
            let val = data[k];

            let alloc_idx = row_to_alloc[i];
            let pos = row_pos[i].fetch_add(1, Ordering::Relaxed);

            // SAFETY: 每个位置只写入一次，无并发冲突
            unsafe {
                let val_ptr = val_spans[alloc_idx].as_ptr() as *mut V;
                let idx_ptr = idx_spans_alloc[alloc_idx].as_ptr() as *mut I;
                *val_ptr.add(pos) = val;
                *idx_ptr.add(pos) = I::from_usize(j);
            }
        }
    });

    // 构建完整的 values 和 indices
    let values: Vec<Span<V>> = (0..row_count)
        .map(|i| {
            if row_lens[i] == 0 {
                empty_span()
            } else {
                val_spans[row_to_alloc[i]].clone()
            }
        })
        .collect();

    let idx_spans: Vec<Span<I>> = (0..row_count)
        .map(|i| {
            if row_lens[i] == 0 {
                empty_span()
            } else {
                idx_spans_alloc[row_to_alloc[i]].clone()
            }
        })
        .collect();

    let mut result = CSR {
        values,
        indices: idx_spans,
        rows,
        cols,
        nnz: Cell::new(Some(I::from_usize(total_nnz))),
    };

    // 并行填充不保证顺序，需要排序
    result.ensure_sorted();

    Ok(result)
}

// =============================================================================
// Scipy COO → CSR/CSC
// =============================================================================

/// scipy COO → 我们的 CSR（并行）
pub fn csr_from_scipy_coo_copy<
    V: Copy + Default + Send + Sync,
    I: SparseIndex + Send + Sync,
    const ALIGN: usize,
>(
    rows: I,
    cols: I,
    row_indices: &[I],
    col_indices: &[I],
    data: &[V],
    _strategy: AllocStrategy,
) -> Result<CSR<V, I>, ConvertError> {
    let row_count = rows.to_usize();
    let nnz = data.len();

    if row_indices.len() != nnz || col_indices.len() != nnz {
        return Err(ConvertError::LengthMismatch);
    }

    // 第一遍：并行统计每行的 nnz
    let row_lens_atomic: Vec<AtomicUsize> = (0..row_count).map(|_| AtomicUsize::new(0)).collect();

    row_indices.par_iter().for_each(|&row_idx| {
        let i = row_idx.to_usize();
        debug_assert!(i < row_count, "row index {} out of bounds {}", i, row_count);
        row_lens_atomic[i].fetch_add(1, Ordering::Relaxed);
    });

    let row_lens: Vec<usize> = row_lens_atomic
        .iter()
        .map(|a| a.load(Ordering::Relaxed))
        .collect();

    let total_nnz: usize = row_lens.par_iter().sum();

    if total_nnz == 0 {
        let values: Vec<Span<V>> = (0..row_count).map(|_| empty_span()).collect();
        let idx_spans: Vec<Span<I>> = (0..row_count).map(|_| empty_span()).collect();

        return Ok(CSR {
            values,
            indices: idx_spans,
            rows,
            cols,
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    let non_empty_lens: Vec<usize> = row_lens.iter().copied().filter(|&len| len > 0).collect();
    let val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
    let idx_spans_alloc = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

    let mut row_to_alloc = vec![usize::MAX; row_count];
    let mut alloc_idx = 0usize;
    for i in 0..row_count {
        if row_lens[i] > 0 {
            row_to_alloc[i] = alloc_idx;
            alloc_idx += 1;
        }
    }

    // 每行当前写入位置（原子）
    let row_pos: Vec<AtomicUsize> = (0..row_count).map(|_| AtomicUsize::new(0)).collect();

    // 第二遍：并行填充数据
    (0..nnz).into_par_iter().for_each(|k| {
        let i = row_indices[k].to_usize();
        let j = col_indices[k];
        let val = data[k];

        let alloc_idx = row_to_alloc[i];
        let pos = row_pos[i].fetch_add(1, Ordering::Relaxed);

        // SAFETY: COO 已去重，每个位置只写入一次
        unsafe {
            let val_ptr = val_spans[alloc_idx].as_ptr() as *mut V;
            let idx_ptr = idx_spans_alloc[alloc_idx].as_ptr() as *mut I;
            *val_ptr.add(pos) = val;
            *idx_ptr.add(pos) = j;
        }
    });

    // 构建完整的 values 和 indices
    let values: Vec<Span<V>> = (0..row_count)
        .map(|i| {
            if row_lens[i] == 0 {
                empty_span()
            } else {
                val_spans[row_to_alloc[i]].clone()
            }
        })
        .collect();

    let idx_spans: Vec<Span<I>> = (0..row_count)
        .map(|i| {
            if row_lens[i] == 0 {
                empty_span()
            } else {
                idx_spans_alloc[row_to_alloc[i]].clone()
            }
        })
        .collect();

    let mut result = CSR {
        values,
        indices: idx_spans,
        rows,
        cols,
        nnz: Cell::new(Some(I::from_usize(total_nnz))),
    };

    // COO 并行填充不保证顺序，需要排序
    result.ensure_sorted();

    Ok(result)
}

/// scipy COO → 我们的 CSC（并行）
pub fn csc_from_scipy_coo_copy<
    V: Copy + Default + Send + Sync,
    I: SparseIndex + Send + Sync,
    const ALIGN: usize,
>(
    rows: I,
    cols: I,
    row_indices: &[I],
    col_indices: &[I],
    data: &[V],
    _strategy: AllocStrategy,
) -> Result<CSC<V, I>, ConvertError> {
    let col_count = cols.to_usize();
    let nnz = data.len();

    if row_indices.len() != nnz || col_indices.len() != nnz {
        return Err(ConvertError::LengthMismatch);
    }

    // 第一遍：并行统计每列的 nnz
    let col_lens_atomic: Vec<AtomicUsize> = (0..col_count).map(|_| AtomicUsize::new(0)).collect();

    col_indices.par_iter().for_each(|&col_idx| {
        let j = col_idx.to_usize();
        debug_assert!(
            j < col_count,
            "column index {} out of bounds {}",
            j,
            col_count
        );
        col_lens_atomic[j].fetch_add(1, Ordering::Relaxed);
    });

    let col_lens: Vec<usize> = col_lens_atomic
        .iter()
        .map(|a| a.load(Ordering::Relaxed))
        .collect();

    let total_nnz: usize = col_lens.par_iter().sum();

    if total_nnz == 0 {
        let values: Vec<Span<V>> = (0..col_count).map(|_| empty_span()).collect();
        let idx_spans: Vec<Span<I>> = (0..col_count).map(|_| empty_span()).collect();

        return Ok(CSC {
            values,
            indices: idx_spans,
            rows,
            cols,
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    let non_empty_lens: Vec<usize> = col_lens.iter().copied().filter(|&len| len > 0).collect();
    let val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
    let idx_spans_alloc = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

    let mut col_to_alloc = vec![usize::MAX; col_count];
    let mut alloc_idx = 0usize;
    for j in 0..col_count {
        if col_lens[j] > 0 {
            col_to_alloc[j] = alloc_idx;
            alloc_idx += 1;
        }
    }

    // 每列当前写入位置（原子）
    let col_pos: Vec<AtomicUsize> = (0..col_count).map(|_| AtomicUsize::new(0)).collect();

    // 第二遍：并行填充数据
    (0..nnz).into_par_iter().for_each(|k| {
        let i = row_indices[k];
        let j = col_indices[k].to_usize();
        let val = data[k];

        let alloc_idx = col_to_alloc[j];
        let pos = col_pos[j].fetch_add(1, Ordering::Relaxed);

        // SAFETY: COO 已去重，每个位置只写入一次
        unsafe {
            let val_ptr = val_spans[alloc_idx].as_ptr() as *mut V;
            let idx_ptr = idx_spans_alloc[alloc_idx].as_ptr() as *mut I;
            *val_ptr.add(pos) = val;
            *idx_ptr.add(pos) = i;
        }
    });

    // 构建完整的 values 和 indices
    let values: Vec<Span<V>> = (0..col_count)
        .map(|j| {
            if col_lens[j] == 0 {
                empty_span()
            } else {
                val_spans[col_to_alloc[j]].clone()
            }
        })
        .collect();

    let idx_spans: Vec<Span<I>> = (0..col_count)
        .map(|j| {
            if col_lens[j] == 0 {
                empty_span()
            } else {
                idx_spans_alloc[col_to_alloc[j]].clone()
            }
        })
        .collect();

    let mut result = CSC {
        values,
        indices: idx_spans,
        rows,
        cols,
        nnz: Cell::new(Some(I::from_usize(total_nnz))),
    };

    // COO 并行填充不保证顺序，需要排序
    result.ensure_sorted();

    Ok(result)
}

// =============================================================================
// LIL → CSR/CSC
// =============================================================================

/// LIL → CSR（并行）
pub fn csr_from_lil<V: Copy + Send + Sync, I: SparseIndex + Send + Sync, const ALIGN: usize>(
    rows: I,
    cols: I,
    row_indices: &[&[I]],
    row_values: &[&[V]],
    _strategy: AllocStrategy,
) -> Result<CSR<V, I>, ConvertError> {
    let row_count = rows.to_usize();

    if row_indices.len() != row_count || row_values.len() != row_count {
        return Err(ConvertError::DimensionMismatch);
    }

    // 并行验证每行的 indices 和 values 长度匹配
    let valid = (0..row_count)
        .into_par_iter()
        .all(|i| row_indices[i].len() == row_values[i].len());

    if !valid {
        return Err(ConvertError::LengthMismatch);
    }

    // 计算长度
    let lens: Vec<usize> = row_values.par_iter().map(|v| v.len()).collect();
    let total_nnz: usize = lens.par_iter().sum();

    if total_nnz == 0 {
        let values: Vec<Span<V>> = (0..row_count).map(|_| empty_span()).collect();
        let idx_spans: Vec<Span<I>> = (0..row_count).map(|_| empty_span()).collect();

        return Ok(CSR {
            values,
            indices: idx_spans,
            rows,
            cols,
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    // 构建行索引到分配索引的映射
    let mut row_to_alloc = vec![usize::MAX; row_count];
    let non_empty_lens: Vec<usize> = lens.iter().copied().filter(|&len| len > 0).collect();

    let mut alloc_idx = 0usize;
    for i in 0..row_count {
        if lens[i] > 0 {
            row_to_alloc[i] = alloc_idx;
            alloc_idx += 1;
        }
    }

    let val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
    let idx_spans_alloc = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

    // 并行复制数据
    (0..row_count).into_par_iter().for_each(|i| {
        let len = lens[i];
        if len > 0 {
            let alloc_idx = row_to_alloc[i];

            // SAFETY: 每个分配索引只被一个行使用，无并发冲突
            unsafe {
                let val_ptr = val_spans[alloc_idx].as_ptr() as *mut V;
                let idx_ptr = idx_spans_alloc[alloc_idx].as_ptr() as *mut I;

                std::ptr::copy_nonoverlapping(row_values[i].as_ptr(), val_ptr, len);
                std::ptr::copy_nonoverlapping(row_indices[i].as_ptr(), idx_ptr, len);
            }
        }
    });

    // 构建完整的 values 和 indices
    let values: Vec<Span<V>> = (0..row_count)
        .map(|i| {
            if lens[i] == 0 {
                empty_span()
            } else {
                val_spans[row_to_alloc[i]].clone()
            }
        })
        .collect();

    let idx_spans: Vec<Span<I>> = (0..row_count)
        .map(|i| {
            if lens[i] == 0 {
                empty_span()
            } else {
                idx_spans_alloc[row_to_alloc[i]].clone()
            }
        })
        .collect();

    Ok(CSR {
        values,
        indices: idx_spans,
        rows,
        cols,
        nnz: Cell::new(Some(I::from_usize(total_nnz))),
    })
}

/// LIL → CSC（并行）
pub fn csc_from_lil<V: Copy + Send + Sync, I: SparseIndex + Send + Sync, const ALIGN: usize>(
    rows: I,
    cols: I,
    col_indices: &[&[I]],
    col_values: &[&[V]],
    _strategy: AllocStrategy,
) -> Result<CSC<V, I>, ConvertError> {
    let col_count = cols.to_usize();

    if col_indices.len() != col_count || col_values.len() != col_count {
        return Err(ConvertError::DimensionMismatch);
    }

    // 并行验证每列的 indices 和 values 长度匹配
    let valid = (0..col_count)
        .into_par_iter()
        .all(|j| col_indices[j].len() == col_values[j].len());

    if !valid {
        return Err(ConvertError::LengthMismatch);
    }

    // 计算长度
    let lens: Vec<usize> = col_values.par_iter().map(|v| v.len()).collect();
    let total_nnz: usize = lens.par_iter().sum();

    if total_nnz == 0 {
        let values: Vec<Span<V>> = (0..col_count).map(|_| empty_span()).collect();
        let idx_spans: Vec<Span<I>> = (0..col_count).map(|_| empty_span()).collect();

        return Ok(CSC {
            values,
            indices: idx_spans,
            rows,
            cols,
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    // 构建列索引到分配索引的映射
    let mut col_to_alloc = vec![usize::MAX; col_count];
    let non_empty_lens: Vec<usize> = lens.iter().copied().filter(|&len| len > 0).collect();

    let mut alloc_idx = 0usize;
    for j in 0..col_count {
        if lens[j] > 0 {
            col_to_alloc[j] = alloc_idx;
            alloc_idx += 1;
        }
    }

    let val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
    let idx_spans_alloc = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

    // 并行复制数据
    (0..col_count).into_par_iter().for_each(|j| {
        let len = lens[j];
        if len > 0 {
            let alloc_idx = col_to_alloc[j];

            // SAFETY: 每个分配索引只被一个列使用，无并发冲突
            unsafe {
                let val_ptr = val_spans[alloc_idx].as_ptr() as *mut V;
                let idx_ptr = idx_spans_alloc[alloc_idx].as_ptr() as *mut I;

                std::ptr::copy_nonoverlapping(col_values[j].as_ptr(), val_ptr, len);
                std::ptr::copy_nonoverlapping(col_indices[j].as_ptr(), idx_ptr, len);
            }
        }
    });

    // 构建完整的 values 和 indices
    let values: Vec<Span<V>> = (0..col_count)
        .map(|j| {
            if lens[j] == 0 {
                empty_span()
            } else {
                val_spans[col_to_alloc[j]].clone()
            }
        })
        .collect();

    let idx_spans: Vec<Span<I>> = (0..col_count)
        .map(|j| {
            if lens[j] == 0 {
                empty_span()
            } else {
                idx_spans_alloc[col_to_alloc[j]].clone()
            }
        })
        .collect();

    Ok(CSC {
        values,
        indices: idx_spans,
        rows,
        cols,
        nnz: Cell::new(Some(I::from_usize(total_nnz))),
    })
}

// =============================================================================
// CSR ↔ CSC 互转
// =============================================================================

/// 我们的 CSR → 我们的 CSC（并行）
pub fn csc_from_csr<
    V: Copy + Default + Send + Sync,
    I: SparseIndex + Send + Sync,
    const ALIGN: usize,
>(
    csr: &CSR<V, I>,
    _strategy: AllocStrategy,
) -> Result<CSC<V, I>, ConvertError> {
    let row_count = csr.rows.to_usize();
    let col_count = csr.cols.to_usize();

    // 第一遍：并行统计每列的 nnz（使用原子操作）
    let col_lens_atomic: Vec<AtomicUsize> = (0..col_count).map(|_| AtomicUsize::new(0)).collect();

    (0..row_count).into_par_iter().for_each(|i| {
        for &col_idx in csr.row_indices(I::from_usize(i)) {
            let j = col_idx.to_usize();
            // SAFETY: indices 已验证在有效范围内
            col_lens_atomic[j].fetch_add(1, Ordering::Relaxed);
        }
    });

    let col_lens: Vec<usize> = col_lens_atomic
        .iter()
        .map(|a| a.load(Ordering::Relaxed))
        .collect();

    let total_nnz: usize = col_lens.par_iter().sum();

    if total_nnz == 0 {
        let values: Vec<Span<V>> = (0..col_count).map(|_| empty_span()).collect();
        let idx_spans: Vec<Span<I>> = (0..col_count).map(|_| empty_span()).collect();

        return Ok(CSC {
            values,
            indices: idx_spans,
            rows: csr.rows,
            cols: csr.cols,
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    let non_empty_lens: Vec<usize> = col_lens.iter().copied().filter(|&len| len > 0).collect();
    let val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
    let idx_spans_alloc = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

    let mut col_to_alloc = vec![usize::MAX; col_count];
    let mut alloc_idx = 0usize;
    for j in 0..col_count {
        if col_lens[j] > 0 {
            col_to_alloc[j] = alloc_idx;
            alloc_idx += 1;
        }
    }

    // 每列当前写入位置（原子）
    let col_pos: Vec<AtomicUsize> = (0..col_count).map(|_| AtomicUsize::new(0)).collect();

    // 第二遍：并行填充数据
    (0..row_count).into_par_iter().for_each(|i| {
        let row_vals = csr.row_values(I::from_usize(i));
        let row_idxs = csr.row_indices(I::from_usize(i));

        for k in 0..row_vals.len() {
            let j = row_idxs[k].to_usize();
            let val = row_vals[k];

            let alloc_idx = col_to_alloc[j];
            let pos = col_pos[j].fetch_add(1, Ordering::Relaxed);

            // SAFETY: 每个位置只写入一次
            unsafe {
                let val_ptr = val_spans[alloc_idx].as_ptr() as *mut V;
                let idx_ptr = idx_spans_alloc[alloc_idx].as_ptr() as *mut I;
                *val_ptr.add(pos) = val;
                *idx_ptr.add(pos) = I::from_usize(i);
            }
        }
    });

    // 构建完整的 values 和 indices
    let values: Vec<Span<V>> = (0..col_count)
        .map(|j| {
            if col_lens[j] == 0 {
                empty_span()
            } else {
                val_spans[col_to_alloc[j]].clone()
            }
        })
        .collect();

    let idx_spans: Vec<Span<I>> = (0..col_count)
        .map(|j| {
            if col_lens[j] == 0 {
                empty_span()
            } else {
                idx_spans_alloc[col_to_alloc[j]].clone()
            }
        })
        .collect();

    let mut result = CSC {
        values,
        indices: idx_spans,
        rows: csr.rows,
        cols: csr.cols,
        nnz: Cell::new(Some(I::from_usize(total_nnz))),
    };

    // 并行填充不保证顺序，需要排序
    result.ensure_sorted();

    Ok(result)
}

/// 我们的 CSC → 我们的 CSR（并行）
pub fn csr_from_csc<
    V: Copy + Default + Send + Sync,
    I: SparseIndex + Send + Sync,
    const ALIGN: usize,
>(
    csc: &CSC<V, I>,
    _strategy: AllocStrategy,
) -> Result<CSR<V, I>, ConvertError> {
    let row_count = csc.rows.to_usize();
    let col_count = csc.cols.to_usize();

    // 第一遍：并行统计每行的 nnz（使用原子操作）
    let row_lens_atomic: Vec<AtomicUsize> = (0..row_count).map(|_| AtomicUsize::new(0)).collect();

    (0..col_count).into_par_iter().for_each(|j| {
        for &row_idx in csc.col_indices(I::from_usize(j)) {
            let i = row_idx.to_usize();
            // SAFETY: indices 已验证在有效范围内
            row_lens_atomic[i].fetch_add(1, Ordering::Relaxed);
        }
    });

    let row_lens: Vec<usize> = row_lens_atomic
        .iter()
        .map(|a| a.load(Ordering::Relaxed))
        .collect();

    let total_nnz: usize = row_lens.par_iter().sum();

    if total_nnz == 0 {
        let values: Vec<Span<V>> = (0..row_count).map(|_| empty_span()).collect();
        let idx_spans: Vec<Span<I>> = (0..row_count).map(|_| empty_span()).collect();

        return Ok(CSR {
            values,
            indices: idx_spans,
            rows: csc.rows,
            cols: csc.cols,
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    let non_empty_lens: Vec<usize> = row_lens.iter().copied().filter(|&len| len > 0).collect();
    let val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
    let idx_spans_alloc = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

    let mut row_to_alloc = vec![usize::MAX; row_count];
    let mut alloc_idx = 0usize;
    for i in 0..row_count {
        if row_lens[i] > 0 {
            row_to_alloc[i] = alloc_idx;
            alloc_idx += 1;
        }
    }

    // 每行当前写入位置（原子）
    let row_pos: Vec<AtomicUsize> = (0..row_count).map(|_| AtomicUsize::new(0)).collect();

    // 第二遍：并行填充数据
    (0..col_count).into_par_iter().for_each(|j| {
        let col_vals = csc.col_values(I::from_usize(j));
        let col_idxs = csc.col_indices(I::from_usize(j));

        for k in 0..col_vals.len() {
            let i = col_idxs[k].to_usize();
            let val = col_vals[k];

            let alloc_idx = row_to_alloc[i];
            let pos = row_pos[i].fetch_add(1, Ordering::Relaxed);

            // SAFETY: 每个位置只写入一次
            unsafe {
                let val_ptr = val_spans[alloc_idx].as_ptr() as *mut V;
                let idx_ptr = idx_spans_alloc[alloc_idx].as_ptr() as *mut I;
                *val_ptr.add(pos) = val;
                *idx_ptr.add(pos) = I::from_usize(j);
            }
        }
    });

    // 构建完整的 values 和 indices
    let values: Vec<Span<V>> = (0..row_count)
        .map(|i| {
            if row_lens[i] == 0 {
                empty_span()
            } else {
                val_spans[row_to_alloc[i]].clone()
            }
        })
        .collect();

    let idx_spans: Vec<Span<I>> = (0..row_count)
        .map(|i| {
            if row_lens[i] == 0 {
                empty_span()
            } else {
                idx_spans_alloc[row_to_alloc[i]].clone()
            }
        })
        .collect();

    let mut result = CSR {
        values,
        indices: idx_spans,
        rows: csc.rows,
        cols: csc.cols,
        nnz: Cell::new(Some(I::from_usize(total_nnz))),
    };

    // 并行填充不保证顺序，需要排序
    result.ensure_sorted();

    Ok(result)
}

// =============================================================================
// Dense → CSR/CSC
// =============================================================================

/// Dense → CSR（并行）
pub fn csr_from_dense<
    V: Copy + PartialEq + Default + Send + Sync,
    I: SparseIndex + Send + Sync,
    const ALIGN: usize,
>(
    dense: &[V],
    rows: usize,
    cols: usize,
    layout: DenseLayout,
    _strategy: AllocStrategy,
) -> Result<CSR<V, I>, ConvertError> {
    if dense.len() != rows * cols {
        return Err(ConvertError::DimensionMismatch);
    }

    let zero = V::default();

    // 第一遍：并行统计每行的 nnz
    let row_lens: Vec<usize> = match layout {
        DenseLayout::RowMajor => (0..rows)
            .into_par_iter()
            .map(|i| {
                let row_start = i * cols;
                (0..cols).filter(|&j| dense[row_start + j] != zero).count()
            })
            .collect(),
        DenseLayout::ColMajor => {
            // ColMajor 布局需要原子操作统计
            let row_lens_atomic: Vec<AtomicUsize> =
                (0..rows).map(|_| AtomicUsize::new(0)).collect();
            (0..cols).into_par_iter().for_each(|j| {
                let col_start = j * rows;
                for i in 0..rows {
                    if dense[col_start + i] != zero {
                        row_lens_atomic[i].fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
            row_lens_atomic
                .iter()
                .map(|a| a.load(Ordering::Relaxed))
                .collect()
        }
    };

    let total_nnz: usize = row_lens.par_iter().sum();

    if total_nnz == 0 {
        let values: Vec<Span<V>> = (0..rows).map(|_| empty_span()).collect();
        let idx_spans: Vec<Span<I>> = (0..rows).map(|_| empty_span()).collect();

        return Ok(CSR {
            values,
            indices: idx_spans,
            rows: I::from_usize(rows),
            cols: I::from_usize(cols),
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    let non_empty_lens: Vec<usize> = row_lens.iter().copied().filter(|&len| len > 0).collect();
    let val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
    let idx_spans_alloc = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

    let mut row_to_alloc = vec![usize::MAX; rows];
    let mut alloc_idx = 0usize;
    for i in 0..rows {
        if row_lens[i] > 0 {
            row_to_alloc[i] = alloc_idx;
            alloc_idx += 1;
        }
    }

    // 第二遍：并行填充数据
    // 跟踪是否需要排序（ColMajor 布局使用原子位置分配，索引可能乱序）
    let needs_sort = matches!(layout, DenseLayout::ColMajor);

    match layout {
        DenseLayout::RowMajor => {
            // RowMajor: 每行独立，完全并行，行内按列顺序填充
            (0..rows).into_par_iter().for_each(|i| {
                if row_lens[i] > 0 {
                    let alloc_idx = row_to_alloc[i];
                    let row_start = i * cols;
                    let mut pos = 0usize;

                    for j in 0..cols {
                        let val = dense[row_start + j];
                        if val != zero {
                            unsafe {
                                let val_ptr = val_spans[alloc_idx].as_ptr() as *mut V;
                                let idx_ptr = idx_spans_alloc[alloc_idx].as_ptr() as *mut I;
                                *val_ptr.add(pos) = val;
                                *idx_ptr.add(pos) = I::from_usize(j);
                            }
                            pos += 1;
                        }
                    }
                }
            });
        }
        DenseLayout::ColMajor => {
            // ColMajor: 需要原子位置分配，索引可能乱序
            let row_pos: Vec<AtomicUsize> = (0..rows).map(|_| AtomicUsize::new(0)).collect();

            (0..cols).into_par_iter().for_each(|j| {
                let col_start = j * rows;
                for i in 0..rows {
                    let val = dense[col_start + i];
                    if val != zero {
                        let alloc_idx = row_to_alloc[i];
                        let pos = row_pos[i].fetch_add(1, Ordering::Relaxed);

                        unsafe {
                            let val_ptr = val_spans[alloc_idx].as_ptr() as *mut V;
                            let idx_ptr = idx_spans_alloc[alloc_idx].as_ptr() as *mut I;
                            *val_ptr.add(pos) = val;
                            *idx_ptr.add(pos) = I::from_usize(j);
                        }
                    }
                }
            });
        }
    }

    // 构建完整的 values 和 indices
    let values: Vec<Span<V>> = (0..rows)
        .map(|i| {
            if row_lens[i] == 0 {
                empty_span()
            } else {
                val_spans[row_to_alloc[i]].clone()
            }
        })
        .collect();

    let idx_spans: Vec<Span<I>> = (0..rows)
        .map(|i| {
            if row_lens[i] == 0 {
                empty_span()
            } else {
                idx_spans_alloc[row_to_alloc[i]].clone()
            }
        })
        .collect();

    let mut result = CSR {
        values,
        indices: idx_spans,
        rows: I::from_usize(rows),
        cols: I::from_usize(cols),
        nnz: Cell::new(Some(I::from_usize(total_nnz))),
    };

    // ColMajor 布局使用原子位置分配，需要排序
    if needs_sort {
        result.ensure_sorted();
    }

    Ok(result)
}

/// Dense → CSC（并行）
pub fn csc_from_dense<
    V: Copy + PartialEq + Default + Send + Sync,
    I: SparseIndex + Send + Sync,
    const ALIGN: usize,
>(
    dense: &[V],
    rows: usize,
    cols: usize,
    layout: DenseLayout,
    _strategy: AllocStrategy,
) -> Result<CSC<V, I>, ConvertError> {
    if dense.len() != rows * cols {
        return Err(ConvertError::DimensionMismatch);
    }

    let zero = V::default();

    // 第一遍：并行统计每列的 nnz
    let col_lens: Vec<usize> = match layout {
        DenseLayout::ColMajor => (0..cols)
            .into_par_iter()
            .map(|j| {
                let col_start = j * rows;
                (0..rows).filter(|&i| dense[col_start + i] != zero).count()
            })
            .collect(),
        DenseLayout::RowMajor => {
            // RowMajor 布局需要原子操作统计
            let col_lens_atomic: Vec<AtomicUsize> =
                (0..cols).map(|_| AtomicUsize::new(0)).collect();
            (0..rows).into_par_iter().for_each(|i| {
                let row_start = i * cols;
                for j in 0..cols {
                    if dense[row_start + j] != zero {
                        col_lens_atomic[j].fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
            col_lens_atomic
                .iter()
                .map(|a| a.load(Ordering::Relaxed))
                .collect()
        }
    };

    let total_nnz: usize = col_lens.par_iter().sum();

    if total_nnz == 0 {
        let values: Vec<Span<V>> = (0..cols).map(|_| empty_span()).collect();
        let idx_spans: Vec<Span<I>> = (0..cols).map(|_| empty_span()).collect();

        return Ok(CSC {
            values,
            indices: idx_spans,
            rows: I::from_usize(rows),
            cols: I::from_usize(cols),
            nnz: Cell::new(Some(I::ZERO)),
        });
    }

    let non_empty_lens: Vec<usize> = col_lens.iter().copied().filter(|&len| len > 0).collect();
    let val_spans = Span::<V>::alloc_slices::<ALIGN>(&non_empty_lens)?;
    let idx_spans_alloc = Span::<I>::alloc_slices::<ALIGN>(&non_empty_lens)?;

    let mut col_to_alloc = vec![usize::MAX; cols];
    let mut alloc_idx = 0usize;
    for j in 0..cols {
        if col_lens[j] > 0 {
            col_to_alloc[j] = alloc_idx;
            alloc_idx += 1;
        }
    }

    // 第二遍：并行填充数据
    // 跟踪是否需要排序（RowMajor 布局使用原子位置分配，索引可能乱序）
    let needs_sort = matches!(layout, DenseLayout::RowMajor);

    match layout {
        DenseLayout::ColMajor => {
            // ColMajor: 每列独立，完全并行，列内按行顺序填充
            (0..cols).into_par_iter().for_each(|j| {
                if col_lens[j] > 0 {
                    let alloc_idx = col_to_alloc[j];
                    let col_start = j * rows;
                    let mut pos = 0usize;

                    for i in 0..rows {
                        let val = dense[col_start + i];
                        if val != zero {
                            unsafe {
                                let val_ptr = val_spans[alloc_idx].as_ptr() as *mut V;
                                let idx_ptr = idx_spans_alloc[alloc_idx].as_ptr() as *mut I;
                                *val_ptr.add(pos) = val;
                                *idx_ptr.add(pos) = I::from_usize(i);
                            }
                            pos += 1;
                        }
                    }
                }
            });
        }
        DenseLayout::RowMajor => {
            // RowMajor: 需要原子位置分配，索引可能乱序
            let col_pos: Vec<AtomicUsize> = (0..cols).map(|_| AtomicUsize::new(0)).collect();

            (0..rows).into_par_iter().for_each(|i| {
                let row_start = i * cols;
                for j in 0..cols {
                    let val = dense[row_start + j];
                    if val != zero {
                        let alloc_idx = col_to_alloc[j];
                        let pos = col_pos[j].fetch_add(1, Ordering::Relaxed);

                        unsafe {
                            let val_ptr = val_spans[alloc_idx].as_ptr() as *mut V;
                            let idx_ptr = idx_spans_alloc[alloc_idx].as_ptr() as *mut I;
                            *val_ptr.add(pos) = val;
                            *idx_ptr.add(pos) = I::from_usize(i);
                        }
                    }
                }
            });
        }
    }

    // 构建完整的 values 和 indices
    let values: Vec<Span<V>> = (0..cols)
        .map(|j| {
            if col_lens[j] == 0 {
                empty_span()
            } else {
                val_spans[col_to_alloc[j]].clone()
            }
        })
        .collect();

    let idx_spans: Vec<Span<I>> = (0..cols)
        .map(|j| {
            if col_lens[j] == 0 {
                empty_span()
            } else {
                idx_spans_alloc[col_to_alloc[j]].clone()
            }
        })
        .collect();

    let mut result = CSC {
        values,
        indices: idx_spans,
        rows: I::from_usize(rows),
        cols: I::from_usize(cols),
        nnz: Cell::new(Some(I::from_usize(total_nnz))),
    };

    // RowMajor 布局使用原子位置分配，需要排序
    if needs_sort {
        result.ensure_sorted();
    }

    Ok(result)
}

// =============================================================================
// CSR/CSC → Dense
// =============================================================================

/// CSR → Dense（写入外部 buffer，并行）
pub fn csr_to_dense<V: Copy + Default + Send + Sync, I: SparseIndex + Send + Sync>(
    csr: &CSR<V, I>,
    out: &mut [V],
    layout: DenseLayout,
) -> Result<(), ConvertError> {
    let rows = csr.rows.to_usize();
    let cols = csr.cols.to_usize();

    if out.len() < rows * cols {
        return Err(ConvertError::BufferTooSmall);
    }

    // 并行填充零
    out[..rows * cols]
        .par_iter_mut()
        .for_each(|v| *v = V::default());

    // 获取 out 的原始指针用于并行写入
    let out_ptr = SendPtr::new(out.as_mut_ptr());

    // 并行填充非零元素
    match layout {
        DenseLayout::RowMajor => {
            (0..rows).into_par_iter().for_each(|i| {
                let row_vals = csr.row_values(I::from_usize(i));
                let row_idxs = csr.row_indices(I::from_usize(i));

                for k in 0..row_vals.len() {
                    let j = row_idxs[k].to_usize();
                    // SAFETY: 每行写入不同的区域，无并发冲突
                    unsafe {
                        *out_ptr.ptr().add(i * cols + j) = row_vals[k];
                    }
                }
            });
        }
        DenseLayout::ColMajor => {
            (0..rows).into_par_iter().for_each(|i| {
                let row_vals = csr.row_values(I::from_usize(i));
                let row_idxs = csr.row_indices(I::from_usize(i));

                for k in 0..row_vals.len() {
                    let j = row_idxs[k].to_usize();
                    // SAFETY: indices 已排序且去重，无并发冲突
                    unsafe {
                        *out_ptr.ptr().add(j * rows + i) = row_vals[k];
                    }
                }
            });
        }
    }

    Ok(())
}

/// CSC → Dense（写入外部 buffer，并行）
pub fn csc_to_dense<V: Copy + Default + Send + Sync, I: SparseIndex + Send + Sync>(
    csc: &CSC<V, I>,
    out: &mut [V],
    layout: DenseLayout,
) -> Result<(), ConvertError> {
    let rows = csc.rows.to_usize();
    let cols = csc.cols.to_usize();

    if out.len() < rows * cols {
        return Err(ConvertError::BufferTooSmall);
    }

    // 并行填充零
    out[..rows * cols]
        .par_iter_mut()
        .for_each(|v| *v = V::default());

    // 获取 out 的原始指针用于并行写入
    let out_ptr = SendPtr::new(out.as_mut_ptr());

    // 并行填充非零元素
    match layout {
        DenseLayout::RowMajor => {
            (0..cols).into_par_iter().for_each(|j| {
                let col_vals = csc.col_values(I::from_usize(j));
                let col_idxs = csc.col_indices(I::from_usize(j));

                for k in 0..col_vals.len() {
                    let i = col_idxs[k].to_usize();
                    // SAFETY: indices 已排序且去重，无并发冲突
                    unsafe {
                        *out_ptr.ptr().add(i * cols + j) = col_vals[k];
                    }
                }
            });
        }
        DenseLayout::ColMajor => {
            (0..cols).into_par_iter().for_each(|j| {
                let col_vals = csc.col_values(I::from_usize(j));
                let col_idxs = csc.col_indices(I::from_usize(j));

                for k in 0..col_vals.len() {
                    let i = col_idxs[k].to_usize();
                    // SAFETY: 每列写入不同的区域，无并发冲突
                    unsafe {
                        *out_ptr.ptr().add(j * rows + i) = col_vals[k];
                    }
                }
            });
        }
    }

    Ok(())
}

// =============================================================================
// CSR/CSC → COO
// =============================================================================

/// CSR → COO（写入外部 buffer）
pub fn csr_to_coo<V: Copy + Send + Sync, I: SparseIndex + Send + Sync>(
    csr: &CSR<V, I>,
    out_row_indices: &mut [I],
    out_col_indices: &mut [I],
    out_data: &mut [V],
) -> Result<(), ConvertError> {
    let nnz = csr.nnz().to_usize();

    if out_row_indices.len() < nnz || out_col_indices.len() < nnz || out_data.len() < nnz {
        return Err(ConvertError::BufferTooSmall);
    }

    let row_count = csr.rows.to_usize();
    let mut pos = 0usize;

    for i in 0..row_count {
        let row_vals = csr.row_values(I::from_usize(i));
        let row_idxs = csr.row_indices(I::from_usize(i));

        for k in 0..row_vals.len() {
            out_row_indices[pos] = I::from_usize(i);
            out_col_indices[pos] = row_idxs[k];
            out_data[pos] = row_vals[k];
            pos += 1;
        }
    }

    Ok(())
}

/// CSC → COO（写入外部 buffer）
pub fn csc_to_coo<V: Copy + Send + Sync, I: SparseIndex + Send + Sync>(
    csc: &CSC<V, I>,
    out_row_indices: &mut [I],
    out_col_indices: &mut [I],
    out_data: &mut [V],
) -> Result<(), ConvertError> {
    let nnz = csc.nnz().to_usize();

    if out_row_indices.len() < nnz || out_col_indices.len() < nnz || out_data.len() < nnz {
        return Err(ConvertError::BufferTooSmall);
    }

    let col_count = csc.cols.to_usize();
    let mut pos = 0usize;

    for j in 0..col_count {
        let col_vals = csc.col_values(I::from_usize(j));
        let col_idxs = csc.col_indices(I::from_usize(j));

        for k in 0..col_vals.len() {
            out_row_indices[pos] = col_idxs[k];
            out_col_indices[pos] = I::from_usize(j);
            out_data[pos] = col_vals[k];
            pos += 1;
        }
    }

    Ok(())
}

// =============================================================================
// 从 Vec<Span> 直接构造
// =============================================================================

/// 从 Vec<Span> 构造 CSR（移动语义）
#[inline]
pub fn csr_from_spans<V, I: SparseIndex>(
    rows: I,
    cols: I,
    values: Vec<Span<V>>,
    indices: Vec<Span<I>>,
    nnz: Option<I>,
) -> Result<CSR<V, I>, ConvertError> {
    let row_count = rows.to_usize();

    if values.len() != row_count || indices.len() != row_count {
        return Err(ConvertError::DimensionMismatch);
    }

    // 验证每对长度匹配
    for i in 0..row_count {
        if values[i].len() != indices[i].len() {
            return Err(ConvertError::LengthMismatch);
        }
    }

    Ok(CSR {
        values,
        indices,
        rows,
        cols,
        nnz: Cell::new(nnz),
    })
}

/// 从 Vec<Span> 构造 CSC（移动语义）
#[inline]
pub fn csc_from_spans<V, I: SparseIndex>(
    rows: I,
    cols: I,
    values: Vec<Span<V>>,
    indices: Vec<Span<I>>,
    nnz: Option<I>,
) -> Result<CSC<V, I>, ConvertError> {
    let col_count = cols.to_usize();

    if values.len() != col_count || indices.len() != col_count {
        return Err(ConvertError::DimensionMismatch);
    }

    for j in 0..col_count {
        if values[j].len() != indices[j].len() {
            return Err(ConvertError::LengthMismatch);
        }
    }

    Ok(CSC {
        values,
        indices,
        rows,
        cols,
        nnz: Cell::new(nnz),
    })
}
