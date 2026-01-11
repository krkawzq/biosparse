//! Sparse - CSR/CSC 稀疏矩阵
//!
//! # 设计理念
//!
//! - **Rust 负责内存布局管理**
//! - **Python 负责高性能算子（使用 JIT）**
//! - **面向句柄**：字段直接 pub，外部可直接访问
//!
//! # 存储结构
//!
//! 每行(CSR)/列(CSC) 存储为独立的 Span 对，支持：
//! - 零拷贝行/列切片（利用 Span 的引用计数）
//! - 灵活的内存管理（owned/shared/view 模式）
//! - 自然的行/列独立性
//!
//! # Performance Notes
//!
//! - `*_unchecked` 方法跳过边界检查，用于热路径
//! - 使用 `assume_unchecked!` 帮助编译器优化
//! - 所有热路径使用 `#[inline(always)]`
//! - 验证循环使用 early exit 优化

#![allow(clippy::missing_safety_doc)]

use std::cell::Cell;

use rayon::prelude::*;

use crate::assume_unchecked;
use crate::span::Span;
use crate::tools::{unlikely, SendPtr};

// =============================================================================
// 辅助函数
// =============================================================================

/// 应用排列到 values 和 indices（in-place）
///
/// 使用循环跟踪法，O(n) 时间，O(1) 额外空间（除了 perm 本身）
#[inline]
fn apply_permutation<V: Copy, I: Copy>(values: &mut [V], indices: &mut [I], perm: &[usize]) {
    let len = perm.len();
    if len <= 1 {
        return;
    }

    // 使用临时数组（对于小数据更简单高效）
    if len <= 64 {
        let old_values: Vec<V> = values.to_vec();
        let old_indices: Vec<I> = indices.to_vec();
        for (new_pos, &old_pos) in perm.iter().enumerate() {
            values[new_pos] = old_values[old_pos];
            indices[new_pos] = old_indices[old_pos];
        }
        return;
    }

    // 对于大数据，使用循环跟踪法
    let mut visited = vec![false; len];
    for start in 0..len {
        if visited[start] || perm[start] == start {
            visited[start] = true;
            continue;
        }

        let mut current = start;
        let temp_val = values[start];
        let temp_idx = indices[start];

        loop {
            let next = perm[current];
            visited[current] = true;

            if next == start {
                values[current] = temp_val;
                indices[current] = temp_idx;
                break;
            }

            values[current] = values[next];
            indices[current] = indices[next];
            current = next;
        }
    }
}

// =============================================================================
// Index Trait
// =============================================================================

/// 稀疏矩阵索引类型约束
pub trait SparseIndex:
    Copy
    + Default
    + Ord
    + Send
    + Sync
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + TryFrom<usize>
    + TryInto<usize>
{
    const ZERO: Self;
    const ONE: Self;
    const MAX: Self;

    fn to_usize(self) -> usize;
    fn from_usize(v: usize) -> Self;

    /// 安全的 usize 转换（可能截断）
    #[inline(always)]
    fn saturating_from_usize(v: usize) -> Self {
        Self::from_usize(v)
    }
}

impl SparseIndex for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = i32::MAX;

    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v as Self
    }
}

impl SparseIndex for i64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = i64::MAX;

    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v as Self
    }
}

impl SparseIndex for u32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = u32::MAX;

    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v as Self
    }
}

impl SparseIndex for u64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = u64::MAX;

    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v as Self
    }
}

impl SparseIndex for usize {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MAX: Self = usize::MAX;

    #[inline(always)]
    fn to_usize(self) -> usize {
        self
    }

    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v
    }
}

// =============================================================================
// CSR - Compressed Sparse Row
// =============================================================================

/// CSR（压缩稀疏行）矩阵
///
/// # Type Parameters
///
/// * `V` - 值类型
/// * `I` - 索引类型（默认 i64）
///
/// # 字段
///
/// 所有字段 pub，面向句柄设计：
/// - `values`: 每行的非零值
/// - `indices`: 每行的列索引
/// - `rows`, `cols`: 维度
/// - `nnz`: 懒计算缓存
#[derive(Debug)]
pub struct CSR<V, I: SparseIndex = i64> {
    /// 每行的非零值（Vec 长度 = rows）
    pub values: Vec<Span<V>>,

    /// 每行的列索引（Vec 长度 = rows）
    pub indices: Vec<Span<I>>,

    /// 行数
    pub rows: I,

    /// 列数
    pub cols: I,

    /// NNZ 缓存（懒惰计算）
    pub nnz: Cell<Option<I>>,
}

// Safety: CSR 可以安全地跨线程共享（如果 V 和 I 满足 Send/Sync）
unsafe impl<V: Send, I: SparseIndex + Send> Send for CSR<V, I> {}
unsafe impl<V: Sync, I: SparseIndex + Sync> Sync for CSR<V, I> {}

impl<V: Send + Sync, I: SparseIndex> CSR<V, I> {
    // =========================================================================
    // 构造函数
    // =========================================================================

    /// 创建空的 CSR 矩阵
    #[inline]
    #[must_use]
    pub fn new(rows: I, cols: I) -> Self {
        let row_count = rows.to_usize();
        Self {
            values: Vec::with_capacity(row_count),
            indices: Vec::with_capacity(row_count),
            rows,
            cols,
            nnz: Cell::new(None),
        }
    }

    /// 从现有数据创建 CSR（不检查有效性）
    ///
    /// # Safety
    ///
    /// 调用者需确保：
    /// - values.len() == rows.to_usize()
    /// - indices.len() == rows.to_usize()
    /// - values[i].len() == indices[i].len() 对所有 i
    #[inline]
    pub unsafe fn from_raw_parts(
        values: Vec<Span<V>>,
        indices: Vec<Span<I>>,
        rows: I,
        cols: I,
    ) -> Self {
        debug_assert_eq!(values.len(), rows.to_usize());
        debug_assert_eq!(indices.len(), rows.to_usize());
        Self {
            values,
            indices,
            rows,
            cols,
            nnz: Cell::new(None),
        }
    }

    /// 从现有数据创建 CSR（带 nnz 缓存）
    ///
    /// # Safety
    ///
    /// 同 `from_raw_parts`，另外 nnz 必须正确
    #[inline]
    pub unsafe fn from_raw_parts_with_nnz(
        values: Vec<Span<V>>,
        indices: Vec<Span<I>>,
        rows: I,
        cols: I,
        nnz: I,
    ) -> Self {
        debug_assert_eq!(values.len(), rows.to_usize());
        debug_assert_eq!(indices.len(), rows.to_usize());
        Self {
            values,
            indices,
            rows,
            cols,
            nnz: Cell::new(Some(nnz)),
        }
    }

    // =========================================================================
    // 维度查询
    // =========================================================================

    /// 获取行数
    #[inline(always)]
    #[must_use]
    pub fn nrows(&self) -> I {
        self.rows
    }

    /// 获取列数
    #[inline(always)]
    #[must_use]
    pub fn ncols(&self) -> I {
        self.cols
    }

    /// 获取形状 (rows, cols)
    #[inline(always)]
    #[must_use]
    pub fn shape(&self) -> (I, I) {
        (self.rows, self.cols)
    }

    /// 获取非零元素个数（懒惰计算，缓存结果）
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> I {
        if let Some(cached) = self.nnz.get() {
            return cached;
        }

        let result = self.compute_nnz();
        self.nnz.set(Some(result));
        result
    }

    /// 计算 NNZ（不使用缓存，并行计算）
    #[inline]
    fn compute_nnz(&self) -> I {
        let total: usize = self.values.par_iter().map(|span| span.len()).sum();
        I::from_usize(total)
    }

    /// 检查是否为空矩阵（rows == 0 或 cols == 0）
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows == I::ZERO || self.cols == I::ZERO
    }

    /// 检查是否无非零元素
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.nnz() == I::ZERO
    }

    /// 计算稀疏度（零元素比例）
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        if unlikely(self.is_empty()) {
            return 1.0;
        }
        let total = self.rows.to_usize() as f64 * self.cols.to_usize() as f64;
        1.0 - (self.nnz().to_usize() as f64 / total)
    }

    /// 计算密度（非零元素比例）
    #[inline]
    #[must_use]
    pub fn density(&self) -> f64 {
        1.0 - self.sparsity()
    }

    // =========================================================================
    // 行访问 - Checked
    // =========================================================================

    /// 获取第 i 行的值 Span
    #[inline]
    #[must_use]
    pub fn row_values_span(&self, i: I) -> &Span<V> {
        &self.values[i.to_usize()]
    }

    /// 获取第 i 行的值 Span（可变）
    #[inline]
    #[must_use]
    pub fn row_values_span_mut(&mut self, i: I) -> &mut Span<V> {
        &mut self.values[i.to_usize()]
    }

    /// 获取第 i 行的索引 Span
    #[inline]
    #[must_use]
    pub fn row_indices_span(&self, i: I) -> &Span<I> {
        &self.indices[i.to_usize()]
    }

    /// 获取第 i 行的索引 Span（可变）
    #[inline]
    #[must_use]
    pub fn row_indices_span_mut(&mut self, i: I) -> &mut Span<I> {
        &mut self.indices[i.to_usize()]
    }

    /// 获取第 i 行的值切片
    #[inline]
    #[must_use]
    pub fn row_values(&self, i: I) -> &[V] {
        self.values[i.to_usize()].as_slice()
    }

    /// 获取第 i 行的值切片（可变）
    #[inline]
    #[must_use]
    pub fn row_values_mut(&mut self, i: I) -> &mut [V] {
        self.values[i.to_usize()].as_slice_mut()
    }

    /// 获取第 i 行的索引切片
    #[inline]
    #[must_use]
    pub fn row_indices(&self, i: I) -> &[I] {
        self.indices[i.to_usize()].as_slice()
    }

    /// 获取第 i 行的索引切片（可变）
    #[inline]
    #[must_use]
    pub fn row_indices_mut(&mut self, i: I) -> &mut [I] {
        self.indices[i.to_usize()].as_slice_mut()
    }

    /// 获取第 i 行的长度（非零元素个数）
    #[inline]
    #[must_use]
    pub fn row_nnz(&self, i: I) -> I {
        I::from_usize(self.values[i.to_usize()].len())
    }

    /// 获取第 i 行的值和索引切片
    #[inline]
    #[must_use]
    pub fn row(&self, i: I) -> (&[V], &[I]) {
        let idx = i.to_usize();
        (self.values[idx].as_slice(), self.indices[idx].as_slice())
    }

    /// 获取第 i 行的值和索引切片（可变）
    #[inline]
    #[must_use]
    pub fn row_mut(&mut self, i: I) -> (&mut [V], &mut [I]) {
        let idx = i.to_usize();
        // 分别借用 values 和 indices
        let vals = self.values[idx].as_slice_mut();
        let idxs = unsafe {
            // SAFETY: values 和 indices 是不同的 Vec，不会重叠
            let ptr = self.indices.as_mut_ptr().add(idx);
            (*ptr).as_slice_mut()
        };
        (vals, idxs)
    }

    // =========================================================================
    // 行访问 - Unchecked
    // =========================================================================

    /// 获取第 i 行的值 Span（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `i < rows`
    #[inline(always)]
    #[must_use]
    pub unsafe fn row_values_span_unchecked(&self, i: I) -> &Span<V> {
        let idx = i.to_usize();
        debug_assert!(idx < self.values.len());
        assume_unchecked!(idx < self.values.len());
        self.values.get_unchecked(idx)
    }

    /// 获取第 i 行的值 Span（不检查边界，可变）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `i < rows`
    #[inline(always)]
    #[must_use]
    pub unsafe fn row_values_span_unchecked_mut(&mut self, i: I) -> &mut Span<V> {
        let idx = i.to_usize();
        debug_assert!(idx < self.values.len());
        assume_unchecked!(idx < self.values.len());
        self.values.get_unchecked_mut(idx)
    }

    /// 获取第 i 行的索引 Span（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `i < rows`
    #[inline(always)]
    #[must_use]
    pub unsafe fn row_indices_span_unchecked(&self, i: I) -> &Span<I> {
        let idx = i.to_usize();
        debug_assert!(idx < self.indices.len());
        assume_unchecked!(idx < self.indices.len());
        self.indices.get_unchecked(idx)
    }

    /// 获取第 i 行的索引 Span（不检查边界，可变）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `i < rows`
    #[inline(always)]
    #[must_use]
    pub unsafe fn row_indices_span_unchecked_mut(&mut self, i: I) -> &mut Span<I> {
        let idx = i.to_usize();
        debug_assert!(idx < self.indices.len());
        assume_unchecked!(idx < self.indices.len());
        self.indices.get_unchecked_mut(idx)
    }

    /// 获取第 i 行的值切片（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `i < rows`
    #[inline(always)]
    #[must_use]
    pub unsafe fn row_values_unchecked(&self, i: I) -> &[V] {
        self.row_values_span_unchecked(i).as_slice()
    }

    /// 获取第 i 行的值切片（不检查边界，可变）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `i < rows`
    #[inline(always)]
    #[must_use]
    pub unsafe fn row_values_unchecked_mut(&mut self, i: I) -> &mut [V] {
        self.row_values_span_unchecked_mut(i).as_slice_mut()
    }

    /// 获取第 i 行的索引切片（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `i < rows`
    #[inline(always)]
    #[must_use]
    pub unsafe fn row_indices_unchecked(&self, i: I) -> &[I] {
        self.row_indices_span_unchecked(i).as_slice()
    }

    /// 获取第 i 行的索引切片（不检查边界，可变）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `i < rows`
    #[inline(always)]
    #[must_use]
    pub unsafe fn row_indices_unchecked_mut(&mut self, i: I) -> &mut [I] {
        self.row_indices_span_unchecked_mut(i).as_slice_mut()
    }

    /// 获取第 i 行的 nnz（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `i < rows`
    #[inline(always)]
    #[must_use]
    pub unsafe fn row_nnz_unchecked(&self, i: I) -> I {
        I::from_usize(self.row_values_span_unchecked(i).len())
    }

    /// 获取第 i 行的值和索引切片（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `i < rows`
    #[inline(always)]
    #[must_use]
    pub unsafe fn row_unchecked(&self, i: I) -> (&[V], &[I]) {
        (self.row_values_unchecked(i), self.row_indices_unchecked(i))
    }

    /// 获取第 i 行的值和索引切片（不检查边界，可变）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `i < rows`
    #[inline(always)]
    #[must_use]
    pub unsafe fn row_unchecked_mut(&mut self, i: I) -> (&mut [V], &mut [I]) {
        let idx = i.to_usize();
        debug_assert!(idx < self.values.len());
        assume_unchecked!(idx < self.values.len());

        let vals = self.values.get_unchecked_mut(idx).as_slice_mut();
        let idxs = {
            let ptr = self.indices.as_mut_ptr().add(idx);
            (*ptr).as_slice_mut()
        };
        (vals, idxs)
    }

    // =========================================================================
    // 原始指针访问（用于 FFI）
    // =========================================================================

    /// 获取第 i 行值的原始指针
    #[inline]
    #[must_use]
    pub fn row_values_ptr(&self, i: I) -> *const V {
        self.values[i.to_usize()].as_ptr()
    }

    /// 获取第 i 行值的原始可变指针
    #[inline]
    #[must_use]
    pub fn row_values_ptr_mut(&mut self, i: I) -> *mut V {
        self.values[i.to_usize()].as_mut_ptr()
    }

    /// 获取第 i 行索引的原始指针
    #[inline]
    #[must_use]
    pub fn row_indices_ptr(&self, i: I) -> *const I {
        self.indices[i.to_usize()].as_ptr()
    }

    /// 获取第 i 行索引的原始可变指针
    #[inline]
    #[must_use]
    pub fn row_indices_ptr_mut(&mut self, i: I) -> *mut I {
        self.indices[i.to_usize()].as_mut_ptr()
    }

    /// 获取第 i 行值的原始指针（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `i < rows`
    #[inline(always)]
    #[must_use]
    pub unsafe fn row_values_ptr_unchecked(&self, i: I) -> *const V {
        self.row_values_span_unchecked(i).as_ptr()
    }

    /// 获取第 i 行索引的原始指针（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `i < rows`
    #[inline(always)]
    #[must_use]
    pub unsafe fn row_indices_ptr_unchecked(&self, i: I) -> *const I {
        self.row_indices_span_unchecked(i).as_ptr()
    }

    // =========================================================================
    // 迭代器
    // =========================================================================

    /// 迭代所有行（值切片）
    #[inline]
    pub fn iter_row_values(&self) -> impl Iterator<Item = &[V]> + '_ {
        self.values.iter().map(|span| span.as_slice())
    }

    /// 迭代所有行（索引切片）
    #[inline]
    pub fn iter_row_indices(&self) -> impl Iterator<Item = &[I]> + '_ {
        self.indices.iter().map(|span| span.as_slice())
    }

    /// 迭代所有行（值和索引）
    #[inline]
    pub fn iter_rows(&self) -> impl Iterator<Item = (&[V], &[I])> + '_ {
        self.values
            .iter()
            .zip(self.indices.iter())
            .map(|(v, i)| (v.as_slice(), i.as_slice()))
    }

    /// 带行号迭代
    #[inline]
    pub fn iter_rows_enumerated(&self) -> impl Iterator<Item = (I, &[V], &[I])> + '_ {
        self.values
            .iter()
            .zip(self.indices.iter())
            .enumerate()
            .map(|(i, (v, idx))| (I::from_usize(i), v.as_slice(), idx.as_slice()))
    }

    // =========================================================================
    // 缓存管理
    // =========================================================================

    /// 使 NNZ 缓存失效
    #[inline(always)]
    pub fn invalidate_nnz(&self) {
        self.nnz.set(None);
    }

    /// 设置 NNZ 缓存（外部已知 nnz 时使用）
    #[inline(always)]
    pub fn set_nnz(&self, nnz: I) {
        self.nnz.set(Some(nnz));
    }

    /// 检查 NNZ 缓存是否有效
    #[inline(always)]
    #[must_use]
    pub fn has_nnz_cache(&self) -> bool {
        self.nnz.get().is_some()
    }

    // =========================================================================
    // 验证
    // =========================================================================

    /// 检查矩阵结构是否有效（并行）
    ///
    /// 验证：
    /// - values.len() == rows
    /// - indices.len() == rows
    /// - values[i].len() == indices[i].len() 对所有 i
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let row_count = self.rows.to_usize();

        if unlikely(self.values.len() != row_count || self.indices.len() != row_count) {
            return false;
        }

        self.values
            .par_iter()
            .zip(self.indices.par_iter())
            .all(|(v, i)| v.len() == i.len())
    }

    /// 检查每行的索引是否已排序（严格递增，并行）
    #[must_use]
    pub fn is_sorted(&self) -> bool {
        self.indices.par_iter().all(|row_idx| {
            let slice = row_idx.as_slice();
            slice.windows(2).all(|w| w[0] < w[1])
        })
    }

    /// 检查所有索引是否在有效范围内 [0, cols)（并行）
    #[must_use]
    pub fn indices_in_bounds(&self) -> bool {
        let cols = self.cols;
        self.indices.par_iter().all(|row_idx| {
            row_idx
                .as_slice()
                .iter()
                .all(|&idx| idx >= I::ZERO && idx < cols)
        })
    }

    /// 完整验证（结构 + 排序 + 边界，并行）
    #[must_use]
    pub fn validate(&self) -> bool {
        let row_count = self.rows.to_usize();
        let cols = self.cols;

        if unlikely(self.values.len() != row_count || self.indices.len() != row_count) {
            return false;
        }

        // 合并所有验证为一次并行遍历
        self.values
            .par_iter()
            .zip(self.indices.par_iter())
            .all(|(v, idx_span)| {
                // 长度匹配
                if v.len() != idx_span.len() {
                    return false;
                }
                let slice = idx_span.as_slice();
                // 排序 + 边界检查
                slice.iter().all(|&idx| idx >= I::ZERO && idx < cols)
                    && slice.windows(2).all(|w| w[0] < w[1])
            })
    }

    // =========================================================================
    // 排序
    // =========================================================================

    /// 确保每行的 indices 是有序的（并行，in-place）
    ///
    /// 对于每行，如果 indices 无序，则对 (indices, values) 进行排序。
    /// 如果已经有序，则不做任何操作。
    ///
    /// # 时间复杂度
    ///
    /// - 最好情况（已排序）：O(nnz) - 仅检查
    /// - 最坏情况：O(nnz log k)，其中 k 是最大行长度
    ///
    /// # 注意
    ///
    /// 此方法需要 Span 是可变的（owned 或 mutable view）。
    /// 如果 Span 是共享的（引用计数 > 1），行为未定义。
    pub fn ensure_sorted(&mut self)
    where
        V: Copy + Send,
    {
        let row_count = self.rows.to_usize();
        let values_ptr = SendPtr::new(self.values.as_mut_ptr());
        let indices_ptr = SendPtr::new(self.indices.as_mut_ptr());

        (0..row_count).into_par_iter().for_each(|i| {
            // SAFETY: 每行独立访问，无数据竞争
            let val_span = unsafe { &mut *values_ptr.add(i) };
            let idx_span = unsafe { &mut *indices_ptr.add(i) };

            let indices = idx_span.as_slice_mut();
            let values = val_span.as_slice_mut();

            // 快速检查：是否已严格递增排序
            // 使用 < 与 is_sorted() 保持一致，不允许重复索引
            if indices.windows(2).all(|w| w[0] < w[1]) {
                return;
            }

            // 需要排序：创建索引数组
            let len = indices.len();
            let mut perm: Vec<usize> = (0..len).collect();
            perm.sort_unstable_by_key(|&k| indices[k]);

            // 应用排列（in-place）
            apply_permutation(values, indices, &perm);
        });
    }

    /// 检查是否需要排序，如果需要则排序，返回是否进行了排序
    ///
    /// 这是 `ensure_sorted` 的变体，返回是否实际进行了排序操作。
    #[must_use]
    pub fn ensure_sorted_checked(&mut self) -> bool
    where
        V: Copy + Send,
    {
        use std::sync::atomic::{AtomicBool, Ordering};

        let did_sort = AtomicBool::new(false);
        let row_count = self.rows.to_usize();
        let values_ptr = SendPtr::new(self.values.as_mut_ptr());
        let indices_ptr = SendPtr::new(self.indices.as_mut_ptr());

        (0..row_count).into_par_iter().for_each(|i| {
            // SAFETY: 每行独立访问，无数据竞争
            let val_span = unsafe { &mut *values_ptr.add(i) };
            let idx_span = unsafe { &mut *indices_ptr.add(i) };

            let indices = idx_span.as_slice_mut();
            let values = val_span.as_slice_mut();

            // 使用 < 与 is_sorted() 保持一致
            if !indices.windows(2).all(|w| w[0] < w[1]) {
                let len = indices.len();
                let mut perm: Vec<usize> = (0..len).collect();
                perm.sort_unstable_by_key(|&k| indices[k]);
                apply_permutation(values, indices, &perm);
                did_sort.store(true, Ordering::Relaxed);
            }
        });

        did_sort.load(Ordering::Relaxed)
    }
}

// =============================================================================
// Clone
// =============================================================================

impl<V: Clone, I: SparseIndex> Clone for CSR<V, I> {
    /// 浅拷贝（Span 引用计数增加）
    #[inline]
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            indices: self.indices.clone(),
            rows: self.rows,
            cols: self.cols,
            nnz: self.nnz.clone(),
        }
    }
}

// =============================================================================
// Default
// =============================================================================

impl<V, I: SparseIndex> Default for CSR<V, I> {
    /// 创建空的 0x0 矩阵
    #[inline]
    fn default() -> Self {
        Self {
            values: Vec::new(),
            indices: Vec::new(),
            rows: I::ZERO,
            cols: I::ZERO,
            nnz: Cell::new(Some(I::ZERO)),
        }
    }
}

// =============================================================================
// CSC - Compressed Sparse Column
// =============================================================================

/// CSC（压缩稀疏列）矩阵
///
/// # Type Parameters
///
/// * `V` - 值类型
/// * `I` - 索引类型（默认 i64）
///
/// # 字段
///
/// 所有字段 pub，面向句柄设计：
/// - `values`: 每列的非零值
/// - `indices`: 每列的行索引
/// - `rows`, `cols`: 维度
/// - `nnz`: 懒计算缓存
#[derive(Debug)]
pub struct CSC<V, I: SparseIndex = i64> {
    /// 每列的非零值（Vec 长度 = cols）
    pub values: Vec<Span<V>>,

    /// 每列的行索引（Vec 长度 = cols）
    pub indices: Vec<Span<I>>,

    /// 行数
    pub rows: I,

    /// 列数
    pub cols: I,

    /// NNZ 缓存（懒惰计算）
    pub nnz: Cell<Option<I>>,
}

// Safety: CSC 可以安全地跨线程共享
unsafe impl<V: Send, I: SparseIndex + Send> Send for CSC<V, I> {}
unsafe impl<V: Sync, I: SparseIndex + Sync> Sync for CSC<V, I> {}

impl<V: Send + Sync, I: SparseIndex> CSC<V, I> {
    // =========================================================================
    // 构造函数
    // =========================================================================

    /// 创建空的 CSC 矩阵
    #[inline]
    #[must_use]
    pub fn new(rows: I, cols: I) -> Self {
        let col_count = cols.to_usize();
        Self {
            values: Vec::with_capacity(col_count),
            indices: Vec::with_capacity(col_count),
            rows,
            cols,
            nnz: Cell::new(None),
        }
    }

    /// 从现有数据创建 CSC（不检查有效性）
    ///
    /// # Safety
    ///
    /// 调用者需确保数据有效
    #[inline]
    pub unsafe fn from_raw_parts(
        values: Vec<Span<V>>,
        indices: Vec<Span<I>>,
        rows: I,
        cols: I,
    ) -> Self {
        debug_assert_eq!(values.len(), cols.to_usize());
        debug_assert_eq!(indices.len(), cols.to_usize());
        Self {
            values,
            indices,
            rows,
            cols,
            nnz: Cell::new(None),
        }
    }

    /// 从现有数据创建 CSC（带 nnz 缓存）
    ///
    /// # Safety
    ///
    /// 同 `from_raw_parts`
    #[inline]
    pub unsafe fn from_raw_parts_with_nnz(
        values: Vec<Span<V>>,
        indices: Vec<Span<I>>,
        rows: I,
        cols: I,
        nnz: I,
    ) -> Self {
        debug_assert_eq!(values.len(), cols.to_usize());
        debug_assert_eq!(indices.len(), cols.to_usize());
        Self {
            values,
            indices,
            rows,
            cols,
            nnz: Cell::new(Some(nnz)),
        }
    }

    // =========================================================================
    // 维度查询
    // =========================================================================

    /// 获取行数
    #[inline(always)]
    #[must_use]
    pub fn nrows(&self) -> I {
        self.rows
    }

    /// 获取列数
    #[inline(always)]
    #[must_use]
    pub fn ncols(&self) -> I {
        self.cols
    }

    /// 获取形状 (rows, cols)
    #[inline(always)]
    #[must_use]
    pub fn shape(&self) -> (I, I) {
        (self.rows, self.cols)
    }

    /// 获取非零元素个数（懒惰计算，缓存结果）
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> I {
        if let Some(cached) = self.nnz.get() {
            return cached;
        }

        let result = self.compute_nnz();
        self.nnz.set(Some(result));
        result
    }

    /// 计算 NNZ（不使用缓存，并行计算）
    #[inline]
    fn compute_nnz(&self) -> I {
        let total: usize = self.values.par_iter().map(|span| span.len()).sum();
        I::from_usize(total)
    }

    /// 检查是否为空矩阵（rows == 0 或 cols == 0）
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows == I::ZERO || self.cols == I::ZERO
    }

    /// 检查是否无非零元素
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.nnz() == I::ZERO
    }

    /// 计算稀疏度（零元素比例）
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        if unlikely(self.is_empty()) {
            return 1.0;
        }
        let total = self.rows.to_usize() as f64 * self.cols.to_usize() as f64;
        1.0 - (self.nnz().to_usize() as f64 / total)
    }

    /// 计算密度（非零元素比例）
    #[inline]
    #[must_use]
    pub fn density(&self) -> f64 {
        1.0 - self.sparsity()
    }

    // =========================================================================
    // 列访问 - Checked
    // =========================================================================

    /// 获取第 j 列的值 Span
    #[inline]
    #[must_use]
    pub fn col_values_span(&self, j: I) -> &Span<V> {
        &self.values[j.to_usize()]
    }

    /// 获取第 j 列的值 Span（可变）
    #[inline]
    #[must_use]
    pub fn col_values_span_mut(&mut self, j: I) -> &mut Span<V> {
        &mut self.values[j.to_usize()]
    }

    /// 获取第 j 列的索引 Span
    #[inline]
    #[must_use]
    pub fn col_indices_span(&self, j: I) -> &Span<I> {
        &self.indices[j.to_usize()]
    }

    /// 获取第 j 列的索引 Span（可变）
    #[inline]
    #[must_use]
    pub fn col_indices_span_mut(&mut self, j: I) -> &mut Span<I> {
        &mut self.indices[j.to_usize()]
    }

    /// 获取第 j 列的值切片
    #[inline]
    #[must_use]
    pub fn col_values(&self, j: I) -> &[V] {
        self.values[j.to_usize()].as_slice()
    }

    /// 获取第 j 列的值切片（可变）
    #[inline]
    #[must_use]
    pub fn col_values_mut(&mut self, j: I) -> &mut [V] {
        self.values[j.to_usize()].as_slice_mut()
    }

    /// 获取第 j 列的索引切片
    #[inline]
    #[must_use]
    pub fn col_indices(&self, j: I) -> &[I] {
        self.indices[j.to_usize()].as_slice()
    }

    /// 获取第 j 列的索引切片（可变）
    #[inline]
    #[must_use]
    pub fn col_indices_mut(&mut self, j: I) -> &mut [I] {
        self.indices[j.to_usize()].as_slice_mut()
    }

    /// 获取第 j 列的长度（非零元素个数）
    #[inline]
    #[must_use]
    pub fn col_nnz(&self, j: I) -> I {
        I::from_usize(self.values[j.to_usize()].len())
    }

    /// 获取第 j 列的值和索引切片
    #[inline]
    #[must_use]
    pub fn col(&self, j: I) -> (&[V], &[I]) {
        let idx = j.to_usize();
        (self.values[idx].as_slice(), self.indices[idx].as_slice())
    }

    /// 获取第 j 列的值和索引切片（可变）
    #[inline]
    #[must_use]
    pub fn col_mut(&mut self, j: I) -> (&mut [V], &mut [I]) {
        let idx = j.to_usize();
        let vals = self.values[idx].as_slice_mut();
        let idxs = unsafe {
            // SAFETY: values 和 indices 是不同的 Vec
            let ptr = self.indices.as_mut_ptr().add(idx);
            (*ptr).as_slice_mut()
        };
        (vals, idxs)
    }

    // =========================================================================
    // 列访问 - Unchecked
    // =========================================================================

    /// 获取第 j 列的值 Span（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `j < cols`
    #[inline(always)]
    #[must_use]
    pub unsafe fn col_values_span_unchecked(&self, j: I) -> &Span<V> {
        let idx = j.to_usize();
        debug_assert!(idx < self.values.len());
        assume_unchecked!(idx < self.values.len());
        self.values.get_unchecked(idx)
    }

    /// 获取第 j 列的值 Span（不检查边界，可变）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `j < cols`
    #[inline(always)]
    #[must_use]
    pub unsafe fn col_values_span_unchecked_mut(&mut self, j: I) -> &mut Span<V> {
        let idx = j.to_usize();
        debug_assert!(idx < self.values.len());
        assume_unchecked!(idx < self.values.len());
        self.values.get_unchecked_mut(idx)
    }

    /// 获取第 j 列的索引 Span（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `j < cols`
    #[inline(always)]
    #[must_use]
    pub unsafe fn col_indices_span_unchecked(&self, j: I) -> &Span<I> {
        let idx = j.to_usize();
        debug_assert!(idx < self.indices.len());
        assume_unchecked!(idx < self.indices.len());
        self.indices.get_unchecked(idx)
    }

    /// 获取第 j 列的索引 Span（不检查边界，可变）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `j < cols`
    #[inline(always)]
    #[must_use]
    pub unsafe fn col_indices_span_unchecked_mut(&mut self, j: I) -> &mut Span<I> {
        let idx = j.to_usize();
        debug_assert!(idx < self.indices.len());
        assume_unchecked!(idx < self.indices.len());
        self.indices.get_unchecked_mut(idx)
    }

    /// 获取第 j 列的值切片（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `j < cols`
    #[inline(always)]
    #[must_use]
    pub unsafe fn col_values_unchecked(&self, j: I) -> &[V] {
        self.col_values_span_unchecked(j).as_slice()
    }

    /// 获取第 j 列的值切片（不检查边界，可变）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `j < cols`
    #[inline(always)]
    #[must_use]
    pub unsafe fn col_values_unchecked_mut(&mut self, j: I) -> &mut [V] {
        self.col_values_span_unchecked_mut(j).as_slice_mut()
    }

    /// 获取第 j 列的索引切片（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `j < cols`
    #[inline(always)]
    #[must_use]
    pub unsafe fn col_indices_unchecked(&self, j: I) -> &[I] {
        self.col_indices_span_unchecked(j).as_slice()
    }

    /// 获取第 j 列的索引切片（不检查边界，可变）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `j < cols`
    #[inline(always)]
    #[must_use]
    pub unsafe fn col_indices_unchecked_mut(&mut self, j: I) -> &mut [I] {
        self.col_indices_span_unchecked_mut(j).as_slice_mut()
    }

    /// 获取第 j 列的 nnz（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `j < cols`
    #[inline(always)]
    #[must_use]
    pub unsafe fn col_nnz_unchecked(&self, j: I) -> I {
        I::from_usize(self.col_values_span_unchecked(j).len())
    }

    /// 获取第 j 列的值和索引切片（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `j < cols`
    #[inline(always)]
    #[must_use]
    pub unsafe fn col_unchecked(&self, j: I) -> (&[V], &[I]) {
        (self.col_values_unchecked(j), self.col_indices_unchecked(j))
    }

    /// 获取第 j 列的值和索引切片（不检查边界，可变）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `j < cols`
    #[inline(always)]
    #[must_use]
    pub unsafe fn col_unchecked_mut(&mut self, j: I) -> (&mut [V], &mut [I]) {
        let idx = j.to_usize();
        debug_assert!(idx < self.values.len());
        assume_unchecked!(idx < self.values.len());

        let vals = self.values.get_unchecked_mut(idx).as_slice_mut();
        let idxs = {
            let ptr = self.indices.as_mut_ptr().add(idx);
            (*ptr).as_slice_mut()
        };
        (vals, idxs)
    }

    // =========================================================================
    // 原始指针访问（用于 FFI）
    // =========================================================================

    /// 获取第 j 列值的原始指针
    #[inline]
    #[must_use]
    pub fn col_values_ptr(&self, j: I) -> *const V {
        self.values[j.to_usize()].as_ptr()
    }

    /// 获取第 j 列值的原始可变指针
    #[inline]
    #[must_use]
    pub fn col_values_ptr_mut(&mut self, j: I) -> *mut V {
        self.values[j.to_usize()].as_mut_ptr()
    }

    /// 获取第 j 列索引的原始指针
    #[inline]
    #[must_use]
    pub fn col_indices_ptr(&self, j: I) -> *const I {
        self.indices[j.to_usize()].as_ptr()
    }

    /// 获取第 j 列索引的原始可变指针
    #[inline]
    #[must_use]
    pub fn col_indices_ptr_mut(&mut self, j: I) -> *mut I {
        self.indices[j.to_usize()].as_mut_ptr()
    }

    /// 获取第 j 列值的原始指针（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `j < cols`
    #[inline(always)]
    #[must_use]
    pub unsafe fn col_values_ptr_unchecked(&self, j: I) -> *const V {
        self.col_values_span_unchecked(j).as_ptr()
    }

    /// 获取第 j 列索引的原始指针（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `j < cols`
    #[inline(always)]
    #[must_use]
    pub unsafe fn col_indices_ptr_unchecked(&self, j: I) -> *const I {
        self.col_indices_span_unchecked(j).as_ptr()
    }

    // =========================================================================
    // 迭代器
    // =========================================================================

    /// 迭代所有列（值切片）
    #[inline]
    pub fn iter_col_values(&self) -> impl Iterator<Item = &[V]> + '_ {
        self.values.iter().map(|span| span.as_slice())
    }

    /// 迭代所有列（索引切片）
    #[inline]
    pub fn iter_col_indices(&self) -> impl Iterator<Item = &[I]> + '_ {
        self.indices.iter().map(|span| span.as_slice())
    }

    /// 迭代所有列（值和索引）
    #[inline]
    pub fn iter_cols(&self) -> impl Iterator<Item = (&[V], &[I])> + '_ {
        self.values
            .iter()
            .zip(self.indices.iter())
            .map(|(v, i)| (v.as_slice(), i.as_slice()))
    }

    /// 带列号迭代
    #[inline]
    pub fn iter_cols_enumerated(&self) -> impl Iterator<Item = (I, &[V], &[I])> + '_ {
        self.values
            .iter()
            .zip(self.indices.iter())
            .enumerate()
            .map(|(j, (v, idx))| (I::from_usize(j), v.as_slice(), idx.as_slice()))
    }

    // =========================================================================
    // 缓存管理
    // =========================================================================

    /// 使 NNZ 缓存失效
    #[inline(always)]
    pub fn invalidate_nnz(&self) {
        self.nnz.set(None);
    }

    /// 设置 NNZ 缓存（外部已知 nnz 时使用）
    #[inline(always)]
    pub fn set_nnz(&self, nnz: I) {
        self.nnz.set(Some(nnz));
    }

    /// 检查 NNZ 缓存是否有效
    #[inline(always)]
    #[must_use]
    pub fn has_nnz_cache(&self) -> bool {
        self.nnz.get().is_some()
    }

    // =========================================================================
    // 验证
    // =========================================================================

    /// 检查矩阵结构是否有效（并行）
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let col_count = self.cols.to_usize();

        if unlikely(self.values.len() != col_count || self.indices.len() != col_count) {
            return false;
        }

        self.values
            .par_iter()
            .zip(self.indices.par_iter())
            .all(|(v, i)| v.len() == i.len())
    }

    /// 检查每列的索引是否已排序（严格递增，并行）
    #[must_use]
    pub fn is_sorted(&self) -> bool {
        self.indices.par_iter().all(|col_idx| {
            let slice = col_idx.as_slice();
            slice.windows(2).all(|w| w[0] < w[1])
        })
    }

    /// 检查所有索引是否在有效范围内 [0, rows)（并行）
    #[must_use]
    pub fn indices_in_bounds(&self) -> bool {
        let rows = self.rows;
        self.indices.par_iter().all(|col_idx| {
            col_idx
                .as_slice()
                .iter()
                .all(|&idx| idx >= I::ZERO && idx < rows)
        })
    }

    /// 完整验证（结构 + 排序 + 边界，并行）
    #[must_use]
    pub fn validate(&self) -> bool {
        let col_count = self.cols.to_usize();
        let rows = self.rows;

        if unlikely(self.values.len() != col_count || self.indices.len() != col_count) {
            return false;
        }

        // 合并所有验证为一次并行遍历
        self.values
            .par_iter()
            .zip(self.indices.par_iter())
            .all(|(v, idx_span)| {
                // 长度匹配
                if v.len() != idx_span.len() {
                    return false;
                }
                let slice = idx_span.as_slice();
                // 排序 + 边界检查
                slice.iter().all(|&idx| idx >= I::ZERO && idx < rows)
                    && slice.windows(2).all(|w| w[0] < w[1])
            })
    }

    // =========================================================================
    // 排序
    // =========================================================================

    /// 确保每列的 indices 是有序的（并行，in-place）
    ///
    /// 对于每列，如果 indices 无序，则对 (indices, values) 进行排序。
    /// 如果已经有序，则不做任何操作。
    ///
    /// # 时间复杂度
    ///
    /// - 最好情况（已排序）：O(nnz) - 仅检查
    /// - 最坏情况：O(nnz log k)，其中 k 是最大列长度
    ///
    /// # 注意
    ///
    /// 此方法需要 Span 是可变的（owned 或 mutable view）。
    /// 如果 Span 是共享的（引用计数 > 1），行为未定义。
    pub fn ensure_sorted(&mut self)
    where
        V: Copy + Send,
    {
        let col_count = self.cols.to_usize();
        let values_ptr = SendPtr::new(self.values.as_mut_ptr());
        let indices_ptr = SendPtr::new(self.indices.as_mut_ptr());

        (0..col_count).into_par_iter().for_each(|j| {
            // SAFETY: 每列独立访问，无数据竞争
            let val_span = unsafe { &mut *values_ptr.add(j) };
            let idx_span = unsafe { &mut *indices_ptr.add(j) };

            let indices = idx_span.as_slice_mut();
            let values = val_span.as_slice_mut();

            // 快速检查：是否已严格递增排序
            // 使用 < 与 is_sorted() 保持一致，不允许重复索引
            if indices.windows(2).all(|w| w[0] < w[1]) {
                return;
            }

            // 需要排序：创建索引数组
            let len = indices.len();
            let mut perm: Vec<usize> = (0..len).collect();
            perm.sort_unstable_by_key(|&k| indices[k]);

            // 应用排列（in-place）
            apply_permutation(values, indices, &perm);
        });
    }

    /// 检查是否需要排序，如果需要则排序，返回是否进行了排序
    ///
    /// 这是 `ensure_sorted` 的变体，返回是否实际进行了排序操作。
    #[must_use]
    pub fn ensure_sorted_checked(&mut self) -> bool
    where
        V: Copy + Send,
    {
        use std::sync::atomic::{AtomicBool, Ordering};

        let did_sort = AtomicBool::new(false);
        let col_count = self.cols.to_usize();
        let values_ptr = SendPtr::new(self.values.as_mut_ptr());
        let indices_ptr = SendPtr::new(self.indices.as_mut_ptr());

        (0..col_count).into_par_iter().for_each(|j| {
            // SAFETY: 每列独立访问，无数据竞争
            let val_span = unsafe { &mut *values_ptr.add(j) };
            let idx_span = unsafe { &mut *indices_ptr.add(j) };

            let indices = idx_span.as_slice_mut();
            let values = val_span.as_slice_mut();

            // 使用 < 与 is_sorted() 保持一致
            if !indices.windows(2).all(|w| w[0] < w[1]) {
                let len = indices.len();
                let mut perm: Vec<usize> = (0..len).collect();
                perm.sort_unstable_by_key(|&k| indices[k]);
                apply_permutation(values, indices, &perm);
                did_sort.store(true, Ordering::Relaxed);
            }
        });

        did_sort.load(Ordering::Relaxed)
    }
}

// =============================================================================
// Clone
// =============================================================================

impl<V: Clone, I: SparseIndex> Clone for CSC<V, I> {
    /// 浅拷贝（Span 引用计数增加）
    #[inline]
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            indices: self.indices.clone(),
            rows: self.rows,
            cols: self.cols,
            nnz: self.nnz.clone(),
        }
    }
}

// =============================================================================
// Default
// =============================================================================

impl<V, I: SparseIndex> Default for CSC<V, I> {
    /// 创建空的 0x0 矩阵
    #[inline]
    fn default() -> Self {
        Self {
            values: Vec::new(),
            indices: Vec::new(),
            rows: I::ZERO,
            cols: I::ZERO,
            nnz: Cell::new(Some(I::ZERO)),
        }
    }
}

// =============================================================================
// Type Aliases
// =============================================================================

/// CSR with f32 values and i64 indices
pub type CSRf32 = CSR<f32, i64>;

/// CSR with f64 values and i64 indices
pub type CSRf64 = CSR<f64, i64>;

/// CSC with f32 values and i64 indices
pub type CSCf32 = CSC<f32, i64>;

/// CSC with f64 values and i64 indices
pub type CSCf64 = CSC<f64, i64>;

/// CSR with i32 indices (for compatibility)
pub type CSRf32i32 = CSR<f32, i32>;
pub type CSRf64i32 = CSR<f64, i32>;

/// CSC with i32 indices
pub type CSCf32i32 = CSC<f32, i32>;
pub type CSCf64i32 = CSC<f64, i32>;

/// CSR with usize indices (native)
pub type CSRf32usize = CSR<f32, usize>;
pub type CSRf64usize = CSR<f64, usize>;

/// CSC with usize indices (native)
pub type CSCf32usize = CSC<f32, usize>;
pub type CSCf64usize = CSC<f64, usize>;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::DEFAULT_ALIGN;

    // =========================================================================
    // Helper: 创建测试用的 CSR/CSC
    // =========================================================================

    /// 创建一个简单的 3x4 CSR 矩阵用于测试
    ///
    /// 矩阵内容:
    /// ```text
    /// [1.0, 0.0, 2.0, 0.0]
    /// [0.0, 3.0, 0.0, 4.0]
    /// [5.0, 0.0, 0.0, 6.0]
    /// ```
    fn create_test_csr() -> CSRf64 {
        let row0_vals: &[f64] = &[1.0, 2.0];
        let row0_idxs: &[i64] = &[0, 2];
        let row1_vals: &[f64] = &[3.0, 4.0];
        let row1_idxs: &[i64] = &[1, 3];
        let row2_vals: &[f64] = &[5.0, 6.0];
        let row2_idxs: &[i64] = &[0, 3];

        let values = Span::copy_from_slices::<DEFAULT_ALIGN>(&[row0_vals, row1_vals, row2_vals])
            .expect("alloc failed");
        let indices = Span::copy_from_slices::<DEFAULT_ALIGN>(&[row0_idxs, row1_idxs, row2_idxs])
            .expect("alloc failed");

        unsafe { CSR::from_raw_parts_with_nnz(values, indices, 3, 4, 6) }
    }

    /// 创建一个简单的 4x3 CSC 矩阵用于测试
    fn create_test_csc() -> CSCf64 {
        let col0_vals: &[f64] = &[1.0, 5.0];
        let col0_idxs: &[i64] = &[0, 2];
        let col1_vals: &[f64] = &[3.0];
        let col1_idxs: &[i64] = &[1];
        let col2_vals: &[f64] = &[2.0];
        let col2_idxs: &[i64] = &[0];

        let values = Span::copy_from_slices::<DEFAULT_ALIGN>(&[col0_vals, col1_vals, col2_vals])
            .expect("alloc failed");
        let indices = Span::copy_from_slices::<DEFAULT_ALIGN>(&[col0_idxs, col1_idxs, col2_idxs])
            .expect("alloc failed");

        unsafe { CSC::from_raw_parts_with_nnz(values, indices, 4, 3, 5) }
    }

    // =========================================================================
    // apply_permutation 测试
    // =========================================================================

    #[test]
    fn test_apply_permutation_small() {
        let mut values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut indices: Vec<i64> = vec![10, 20, 30, 40];
        let perm = vec![2, 0, 3, 1]; // 重排为 [3, 1, 4, 2], [30, 10, 40, 20]

        apply_permutation(&mut values, &mut indices, &perm);

        assert_eq!(values, vec![3.0, 1.0, 4.0, 2.0]);
        assert_eq!(indices, vec![30, 10, 40, 20]);
    }

    #[test]
    fn test_apply_permutation_empty() {
        let mut values: Vec<f64> = vec![];
        let mut indices: Vec<i64> = vec![];
        let perm: Vec<usize> = vec![];

        apply_permutation(&mut values, &mut indices, &perm);

        assert!(values.is_empty());
        assert!(indices.is_empty());
    }

    #[test]
    fn test_apply_permutation_single() {
        let mut values: Vec<f64> = vec![42.0];
        let mut indices: Vec<i64> = vec![99];
        let perm = vec![0];

        apply_permutation(&mut values, &mut indices, &perm);

        assert_eq!(values, vec![42.0]);
        assert_eq!(indices, vec![99]);
    }

    #[test]
    fn test_apply_permutation_identity() {
        let mut values: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut indices: Vec<i64> = vec![10, 20, 30];
        let perm = vec![0, 1, 2]; // 恒等排列

        apply_permutation(&mut values, &mut indices, &perm);

        assert_eq!(values, vec![1.0, 2.0, 3.0]);
        assert_eq!(indices, vec![10, 20, 30]);
    }

    #[test]
    fn test_apply_permutation_large() {
        // 超过 64 元素，测试循环跟踪法
        let n = 100;
        let mut values: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut indices: Vec<i64> = (0..n as i64).collect();
        // 反转排列
        let perm: Vec<usize> = (0..n).rev().collect();

        apply_permutation(&mut values, &mut indices, &perm);

        let expected_values: Vec<f64> = (0..n).rev().map(|i| i as f64).collect();
        let expected_indices: Vec<i64> = (0..n as i64).rev().collect();
        assert_eq!(values, expected_values);
        assert_eq!(indices, expected_indices);
    }

    // =========================================================================
    // 基本测试
    // =========================================================================

    #[test]
    fn test_csr_default() {
        let csr: CSRf64 = CSR::default();
        assert_eq!(csr.nrows(), 0);
        assert_eq!(csr.ncols(), 0);
        assert_eq!(csr.nnz(), 0);
        assert!(csr.is_empty());
        assert!(csr.is_valid());
    }

    #[test]
    fn test_csc_default() {
        let csc: CSCf64 = CSC::default();
        assert_eq!(csc.nrows(), 0);
        assert_eq!(csc.ncols(), 0);
        assert_eq!(csc.nnz(), 0);
        assert!(csc.is_empty());
        assert!(csc.is_valid());
    }

    #[test]
    fn test_sparsity_empty() {
        let csr: CSRf64 = CSR::default();
        assert!((csr.sparsity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_new() {
        let csr: CSR<f64, i64> = CSR::new(10, 20);
        assert_eq!(csr.nrows(), 10);
        assert_eq!(csr.ncols(), 20);
        assert!(!csr.has_nnz_cache());
    }

    #[test]
    fn test_csc_new() {
        let csc: CSC<f64, i64> = CSC::new(10, 20);
        assert_eq!(csc.nrows(), 10);
        assert_eq!(csc.ncols(), 20);
        assert!(!csc.has_nnz_cache());
    }

    #[test]
    fn test_sparse_index_types() {
        // Test i32
        assert_eq!(i32::ZERO, 0i32);
        assert_eq!(i32::ONE, 1i32);
        assert_eq!(i32::from_usize(42), 42i32);
        assert_eq!((42i32).to_usize(), 42usize);

        // Test i64
        assert_eq!(i64::ZERO, 0i64);
        assert_eq!(i64::ONE, 1i64);

        // Test u32
        assert_eq!(u32::ZERO, 0u32);
        assert_eq!(u32::ONE, 1u32);

        // Test u64
        assert_eq!(u64::ZERO, 0u64);
        assert_eq!(u64::ONE, 1u64);

        // Test usize
        assert_eq!(usize::ZERO, 0usize);
        assert_eq!(usize::ONE, 1usize);
    }

    #[test]
    fn test_validate_empty() {
        let csr: CSRf64 = CSR::default();
        assert!(csr.validate());

        let csc: CSCf64 = CSC::default();
        assert!(csc.validate());
    }

    // =========================================================================
    // 有数据的矩阵测试
    // =========================================================================

    #[test]
    fn test_csr_with_data() {
        let csr = create_test_csr();

        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 4);
        assert_eq!(csr.nnz(), 6);
        assert!(!csr.is_empty());
        assert!(csr.is_valid());
        assert!(csr.is_sorted());
        assert!(csr.indices_in_bounds());
        assert!(csr.validate());

        // 检查稀疏度 (6 非零 / 12 总 = 0.5 密度)
        assert!((csr.density() - 0.5).abs() < 1e-10);
        assert!((csr.sparsity() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_csr_row_access() {
        let csr = create_test_csr();

        // 第 0 行: [1.0, 2.0] at indices [0, 2]
        assert_eq!(csr.row_values(0), &[1.0, 2.0]);
        assert_eq!(csr.row_indices(0), &[0, 2]);
        assert_eq!(csr.row_nnz(0), 2);

        // 第 1 行: [3.0, 4.0] at indices [1, 3]
        let (vals, idxs) = csr.row(1);
        assert_eq!(vals, &[3.0, 4.0]);
        assert_eq!(idxs, &[1, 3]);

        // 第 2 行: [5.0, 6.0] at indices [0, 3]
        assert_eq!(csr.row_values(2), &[5.0, 6.0]);
        assert_eq!(csr.row_indices(2), &[0, 3]);
    }

    #[test]
    fn test_csr_row_access_unchecked() {
        let csr = create_test_csr();

        unsafe {
            assert_eq!(csr.row_values_unchecked(0), &[1.0, 2.0]);
            assert_eq!(csr.row_indices_unchecked(1), &[1, 3]);
            assert_eq!(csr.row_nnz_unchecked(2), 2);

            let (vals, idxs) = csr.row_unchecked(2);
            assert_eq!(vals, &[5.0, 6.0]);
            assert_eq!(idxs, &[0, 3]);
        }
    }

    #[test]
    fn test_csc_with_data() {
        let csc = create_test_csc();

        assert_eq!(csc.nrows(), 4);
        assert_eq!(csc.ncols(), 3);
        assert_eq!(csc.nnz(), 5);
        assert!(!csc.is_empty());
        assert!(csc.is_valid());
        assert!(csc.is_sorted());
    }

    #[test]
    fn test_csc_col_access() {
        let csc = create_test_csc();

        // 第 0 列: [1.0, 5.0] at indices [0, 2]
        assert_eq!(csc.col_values(0), &[1.0, 5.0]);
        assert_eq!(csc.col_indices(0), &[0, 2]);
        assert_eq!(csc.col_nnz(0), 2);

        // 第 1 列: [3.0] at indices [1]
        let (vals, idxs) = csc.col(1);
        assert_eq!(vals, &[3.0]);
        assert_eq!(idxs, &[1]);
    }

    // =========================================================================
    // 迭代器测试
    // =========================================================================

    #[test]
    fn test_csr_iterators() {
        let csr = create_test_csr();

        let all_values: Vec<&[f64]> = csr.iter_row_values().collect();
        assert_eq!(all_values.len(), 3);
        assert_eq!(all_values[0], &[1.0, 2.0]);
        assert_eq!(all_values[1], &[3.0, 4.0]);
        assert_eq!(all_values[2], &[5.0, 6.0]);

        let enumerated: Vec<(i64, &[f64], &[i64])> = csr.iter_rows_enumerated().collect();
        assert_eq!(enumerated.len(), 3);
        assert_eq!(enumerated[0].0, 0);
        assert_eq!(enumerated[1].0, 1);
        assert_eq!(enumerated[2].0, 2);
    }

    // =========================================================================
    // 排序测试
    // =========================================================================

    #[test]
    fn test_csr_is_sorted_already_sorted() {
        let csr = create_test_csr();
        assert!(csr.is_sorted());
    }

    #[test]
    fn test_csr_is_sorted_unsorted() {
        // 创建一个索引未排序的 CSR
        let row0_vals: &[f64] = &[2.0, 1.0]; // 值顺序: idx 2, idx 0
        let row0_idxs: &[i64] = &[2, 0]; // 未排序

        let values = Span::copy_from_slices::<DEFAULT_ALIGN>(&[row0_vals]).expect("alloc failed");
        let indices = Span::copy_from_slices::<DEFAULT_ALIGN>(&[row0_idxs]).expect("alloc failed");

        let csr: CSRf64 = unsafe { CSR::from_raw_parts(values, indices, 1, 3) };

        assert!(!csr.is_sorted());
    }

    #[test]
    fn test_csr_ensure_sorted() {
        // 创建一个索引未排序的 CSR
        let row0_vals: &[f64] = &[2.0, 1.0, 3.0]; // 对应 idx 2, 0, 1
        let row0_idxs: &[i64] = &[2, 0, 1]; // 未排序

        let values = Span::copy_from_slices::<DEFAULT_ALIGN>(&[row0_vals]).expect("alloc failed");
        let indices = Span::copy_from_slices::<DEFAULT_ALIGN>(&[row0_idxs]).expect("alloc failed");

        let mut csr: CSRf64 = unsafe { CSR::from_raw_parts(values, indices, 1, 3) };

        assert!(!csr.is_sorted());

        csr.ensure_sorted();

        assert!(csr.is_sorted());
        // 排序后: indices [0, 1, 2], values [1.0, 3.0, 2.0]
        assert_eq!(csr.row_indices(0), &[0, 1, 2]);
        assert_eq!(csr.row_values(0), &[1.0, 3.0, 2.0]);
    }

    #[test]
    fn test_csr_ensure_sorted_checked() {
        // 已排序的情况
        let csr = create_test_csr();
        let mut csr_clone = csr.clone();
        let did_sort = csr_clone.ensure_sorted_checked();
        assert!(!did_sort); // 不需要排序

        // 未排序的情况
        let row0_vals: &[f64] = &[2.0, 1.0];
        let row0_idxs: &[i64] = &[2, 0];

        let values = Span::copy_from_slices::<DEFAULT_ALIGN>(&[row0_vals]).expect("alloc failed");
        let indices = Span::copy_from_slices::<DEFAULT_ALIGN>(&[row0_idxs]).expect("alloc failed");

        let mut csr: CSRf64 = unsafe { CSR::from_raw_parts(values, indices, 1, 3) };
        let did_sort = csr.ensure_sorted_checked();
        assert!(did_sort); // 需要排序
        assert!(csr.is_sorted());
    }

    #[test]
    fn test_csr_is_sorted_with_duplicates() {
        // 包含重复索引的行（不应该被认为是排序的）
        let row0_vals: &[f64] = &[1.0, 2.0];
        let row0_idxs: &[i64] = &[1, 1]; // 重复索引

        let values = Span::copy_from_slices::<DEFAULT_ALIGN>(&[row0_vals]).expect("alloc failed");
        let indices = Span::copy_from_slices::<DEFAULT_ALIGN>(&[row0_idxs]).expect("alloc failed");

        let csr: CSRf64 = unsafe { CSR::from_raw_parts(values, indices, 1, 3) };

        // is_sorted 使用严格 < 判断，所以重复索引不被认为是排序的
        assert!(!csr.is_sorted());
    }

    // =========================================================================
    // NNZ 缓存测试
    // =========================================================================

    #[test]
    fn test_nnz_cache() {
        let csr = create_test_csr();

        assert!(csr.has_nnz_cache());
        assert_eq!(csr.nnz(), 6);

        csr.invalidate_nnz();
        assert!(!csr.has_nnz_cache());

        // 再次调用 nnz() 会重新计算并缓存
        assert_eq!(csr.nnz(), 6);
        assert!(csr.has_nnz_cache());

        // 手动设置缓存
        csr.set_nnz(100);
        assert_eq!(csr.nnz(), 100); // 使用缓存值
    }

    // =========================================================================
    // Clone 测试
    // =========================================================================

    #[test]
    fn test_csr_clone() {
        let csr = create_test_csr();
        let cloned = csr.clone();

        assert_eq!(cloned.nrows(), csr.nrows());
        assert_eq!(cloned.ncols(), csr.ncols());
        assert_eq!(cloned.nnz(), csr.nnz());

        // 浅拷贝：数据指针应该相同（共享内存）
        assert_eq!(csr.row_values_ptr(0), cloned.row_values_ptr(0));
    }

    // =========================================================================
    // 边界情况测试
    // =========================================================================

    #[test]
    fn test_is_zero() {
        let csr: CSRf64 = CSR::default();
        assert!(csr.is_zero());

        let csr_with_data = create_test_csr();
        assert!(!csr_with_data.is_zero());
    }
}
