//! Storage - 内部引用计数内存管理器
//!
//! Storage 是 scl 的内部类型，不对外暴露。用户应通过 Span API 访问。
//!
//! # 设计要点
//!
//! - 使用系统分配器的对齐分配，保证 ALIGN 字节对齐（默认 32，适配 AVX2）
//! - 尾部额外分配 ALIGN-1 字节，保证 SIMD mask load 不越界
//! - 通过 `Arc<Storage>` 实现多 Span 共享，引用计数归零时自动释放
//!
//! # API 命名规范
//!
//! - `alloc_*` 系列：纯分配，不复制数据
//! - `copy_from_*` 系列：从外部数据复制构造
//!
//! # Safety
//!
//! Storage 内部持有裸指针，但数据本身是普通字节数组，因此实现了 Send + Sync。
//! 调用者需确保：
//! - 不在 Storage 释放后使用其返回的指针
//! - 多线程写入时自行同步
//!
//! # Performance Notes
//!
//! 本模块使用多种编译器优化技巧：
//! - `#[inline(always)]` 强制内联热路径
//! - `assume_unchecked!` 消除冗余边界检查
//! - const 泛型参数启用编译期计算
//! - 所有对齐运算使用位操作

#![allow(clippy::missing_safety_doc)]

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::assume_unchecked;
use crate::tools::{const_align_up, copy_nonoverlapping_aligned, likely, prefetch_write, unlikely};

// =============================================================================
// Constants
// =============================================================================

/// 默认对齐大小（32 字节，适用于 AVX2）
pub const DEFAULT_ALIGN: usize = 32;

/// AVX-512 对齐
pub const AVX512_ALIGN: usize = 64;

/// 缓存行对齐
pub const CACHE_LINE_ALIGN: usize = 64;

// =============================================================================
// Error Types
// =============================================================================

/// 内存分配错误
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AllocError {
    /// 请求的大小为零
    ZeroSize,
    /// 大小数组为空
    EmptySizes,
    /// Layout 构造失败（通常是 size 过大）
    LayoutError,
    /// 系统内存不足
    OutOfMemory,
}

impl std::fmt::Display for AllocError {
    #[cold]
    #[inline(never)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSize => write!(f, "allocation size cannot be zero"),
            Self::EmptySizes => write!(f, "sizes array cannot be empty"),
            Self::LayoutError => write!(f, "invalid memory layout (size too large?)"),
            Self::OutOfMemory => write!(f, "out of memory"),
        }
    }
}

impl std::error::Error for AllocError {}

/// 分配结果类型别名
pub type AllocResult<T> = Result<T, AllocError>;

// =============================================================================
// Debug Statistics
// =============================================================================

#[cfg(debug_assertions)]
static TOTAL_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

#[cfg(debug_assertions)]
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

/// 调试统计信息
#[cfg(debug_assertions)]
#[derive(Debug, Clone, Copy)]
pub struct AllocStats {
    pub total_bytes: usize,
    pub allocation_count: usize,
}

#[cfg(debug_assertions)]
impl AllocStats {
    /// 获取当前分配统计
    #[must_use]
    #[inline]
    pub fn current() -> Self {
        Self {
            total_bytes: TOTAL_ALLOCATED.load(Ordering::Relaxed),
            allocation_count: ALLOCATION_COUNT.load(Ordering::Relaxed),
        }
    }

    /// 重置统计（仅用于测试）
    #[inline]
    pub fn reset() {
        TOTAL_ALLOCATED.store(0, Ordering::Relaxed);
        ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    }
}

// =============================================================================
// Storage
// =============================================================================

/// Storage - 引用计数的对齐内存块（内部类型）
///
/// 通过 `Arc<Storage>` 被多个 Span 共享。当最后一个持有者释放时，内存自动回收。
///
/// # Memory Layout
///
/// 使用 ALIGN 对齐分配，buffer 即为 start（无头部 padding）：
///
/// ```text
/// [usable region: size bytes] [tail padding: ALIGN-1]
///          ↑
///    buffer = start (ALIGN 对齐)
/// ```
///
/// 尾部 padding 保证 SIMD mask load 不会越界访问。
#[derive(Debug)]
pub(crate) struct Storage {
    /// 实际从堆分配的地址（用于 dealloc）
    buffer: NonNull<u8>,

    /// 对齐后的起始地址
    start: NonNull<u8>,

    /// 对齐后的可用大小（字节）
    size: usize,

    /// 实际分配的大小（用于 dealloc）
    alloc_size: usize,

    /// 对齐要求
    align: usize,
}

// Safety: Storage 内部只有原始指针指向普通字节数组，线程安全
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl Storage {
    // =========================================================================
    // Alloc 系列 - 纯分配，不复制数据
    // =========================================================================

    /// 分配指定大小的对齐内存（单块）
    ///
    /// # Type Parameters
    ///
    /// * `ALIGN` - 对齐要求（必须是 2 的幂，编译期检查）
    ///
    /// # Arguments
    ///
    /// * `size` - 需要的可用字节数（会向上取整到 ALIGN）
    ///
    /// # Performance
    ///
    /// ALIGN 为编译期常量，所有对齐计算会被优化为常量或移位操作
    #[inline]
    pub fn alloc<const ALIGN: usize>(size: usize) -> AllocResult<Arc<Self>> {
        // 编译期断言
        const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };
        const { assert!(ALIGN >= 1, "ALIGN must be at least 1") };

        if unlikely(size == 0) {
            return Err(AllocError::ZeroSize);
        }

        // 向上取整到 ALIGN（编译期常量掩码）
        let aligned_size = const_align_up::<ALIGN>(size);

        // 尾部 padding: ALIGN - 1 字节，保证 SIMD mask load 不越界
        // 使用 ALIGN 对齐分配，不需要头部 padding
        let tail_padding = ALIGN - 1;

        let alloc_size = match aligned_size.checked_add(tail_padding) {
            Some(v) => v,
            None => return Err(AllocError::LayoutError),
        };

        // SAFETY: alloc_size 已验证不为 0，ALIGN 是 2 的幂
        let layout = match Layout::from_size_align(alloc_size, ALIGN) {
            Ok(l) => l,
            Err(_) => return Err(AllocError::LayoutError),
        };

        // SAFETY: layout 已验证有效
        let buffer = unsafe { alloc_zeroed(layout) };
        let buffer = match NonNull::new(buffer) {
            Some(b) => b,
            None => return Err(AllocError::OutOfMemory),
        };

        // 分配器保证对齐，buffer 即为 start
        let start = buffer;

        #[cfg(debug_assertions)]
        {
            TOTAL_ALLOCATED.fetch_add(alloc_size, Ordering::Relaxed);
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        }

        Ok(Arc::new(Self {
            buffer,
            start,
            size: aligned_size,
            alloc_size,
            align: ALIGN,
        }))
    }

    /// 分配多个连续区域（静态数量）
    ///
    /// # Type Parameters
    ///
    /// * `ALIGN` - 对齐要求
    /// * `N` - 区域数量（编译期常量）
    ///
    /// # Performance
    ///
    /// N 为编译期常量，循环会被完全展开
    #[inline]
    pub fn alloc_n<const ALIGN: usize, const N: usize>(
        sizes: [usize; N],
    ) -> AllocResult<(Arc<Self>, [Range<usize>; N])> {
        const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };
        const { assert!(N > 0, "N must be greater than 0") };

        let mut ranges: [Range<usize>; N] = std::array::from_fn(|_| 0..0);
        let mut current_offset = 0usize;

        // 循环会被展开（N 是编译期常量）
        for i in 0..N {
            // SAFETY: i < N 由循环保证
            let size_i = unsafe { *sizes.get_unchecked(i) };

            if unlikely(size_i == 0) {
                return Err(AllocError::ZeroSize);
            }

            let aligned_size = const_align_up::<ALIGN>(size_i);

            // SAFETY: i < N 由循环保证
            unsafe {
                *ranges.get_unchecked_mut(i) = current_offset..current_offset + aligned_size;
            }

            current_offset = match current_offset.checked_add(aligned_size) {
                Some(v) => v,
                None => return Err(AllocError::LayoutError),
            };
        }

        let storage = Self::alloc::<ALIGN>(current_offset)?;
        Ok((storage, ranges))
    }

    /// 分配多个连续区域（动态数量）
    ///
    /// # Type Parameters
    ///
    /// * `ALIGN` - 对齐要求
    #[inline]
    pub fn alloc_slices<const ALIGN: usize>(
        sizes: &[usize],
    ) -> AllocResult<(Arc<Self>, Vec<Range<usize>>)> {
        const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };

        if unlikely(sizes.is_empty()) {
            return Err(AllocError::EmptySizes);
        }

        let len = sizes.len();
        let mut ranges = Vec::with_capacity(len);
        let mut current_offset = 0usize;

        for i in 0..len {
            // SAFETY: i < len 由循环保证
            let size = unsafe { *sizes.get_unchecked(i) };

            if unlikely(size == 0) {
                return Err(AllocError::ZeroSize);
            }

            let aligned_size = const_align_up::<ALIGN>(size);
            ranges.push(current_offset..current_offset + aligned_size);

            current_offset = match current_offset.checked_add(aligned_size) {
                Some(v) => v,
                None => return Err(AllocError::LayoutError),
            };
        }

        let storage = Self::alloc::<ALIGN>(current_offset)?;
        Ok((storage, ranges))
    }

    // =========================================================================
    // Copy From 系列 - 从外部数据复制构造
    // =========================================================================

    /// 从单个 slice 复制构造
    ///
    /// # Performance
    ///
    /// 对于对齐的数据，LLVM 会自动向量化复制操作
    #[inline]
    pub fn copy_from<T: Copy, const ALIGN: usize>(data: &[T]) -> AllocResult<Arc<Self>> {
        const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };

        if unlikely(data.is_empty()) {
            return Err(AllocError::ZeroSize);
        }

        let len = data.len();
        let elem_size = std::mem::size_of::<T>();

        // 溢出检查
        let byte_size = match len.checked_mul(elem_size) {
            Some(v) => v,
            None => return Err(AllocError::LayoutError),
        };

        let storage = Self::alloc::<ALIGN>(byte_size)?;

        // SAFETY:
        // - data 指向有效的 len 个 T 元素
        // - storage 是新分配的内存，保证与 data 不重叠
        // - 目标地址保证 ALIGN 对齐
        // - T: Copy 保证可以按位复制
        unsafe {
            let dst = storage.data().as_ptr() as *mut T;
            // 使用优化的对齐复制
            copy_nonoverlapping_aligned::<T, ALIGN>(data.as_ptr(), dst, len);
        }

        Ok(storage)
    }

    /// 从多个 slices 复制构造（静态数量）
    ///
    /// # Performance
    ///
    /// N 为编译期常量，循环会被完全展开
    #[inline]
    pub fn copy_from_n<T: Copy, const ALIGN: usize, const N: usize>(
        slices: [&[T]; N],
    ) -> AllocResult<(Arc<Self>, [Range<usize>; N])> {
        const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };
        const { assert!(N > 0, "N must be greater than 0") };

        let elem_size = std::mem::size_of::<T>();

        let mut ranges: [Range<usize>; N] = std::array::from_fn(|_| 0..0);
        let mut current_offset = 0usize;

        // 第一遍：计算布局（会被展开）
        for i in 0..N {
            // SAFETY: i < N
            let slice = unsafe { *slices.get_unchecked(i) };

            if unlikely(slice.is_empty()) {
                return Err(AllocError::ZeroSize);
            }

            let byte_size = match slice.len().checked_mul(elem_size) {
                Some(v) => v,
                None => return Err(AllocError::LayoutError),
            };

            let aligned_size = const_align_up::<ALIGN>(byte_size);

            unsafe {
                *ranges.get_unchecked_mut(i) = current_offset..current_offset + aligned_size;
            }

            current_offset = match current_offset.checked_add(aligned_size) {
                Some(v) => v,
                None => return Err(AllocError::LayoutError),
            };
        }

        let storage = Self::alloc::<ALIGN>(current_offset)?;
        let base_ptr = storage.data().as_ptr();

        // SAFETY: 所有范围和指针已验证
        unsafe {
            assume_unchecked!((base_ptr as usize) % ALIGN == 0);

            // 第二遍：复制数据（会被展开）
            for i in 0..N {
                let slice = *slices.get_unchecked(i);
                let range = ranges.get_unchecked(i);
                let dst = base_ptr.add(range.start) as *mut T;

                // 预取下一个目标区域
                if i + 1 < N {
                    let next_range = ranges.get_unchecked(i + 1);
                    prefetch_write(base_ptr.add(next_range.start) as *mut T);
                }

                copy_nonoverlapping_aligned::<T, ALIGN>(slice.as_ptr(), dst, slice.len());
            }
        }

        Ok((storage, ranges))
    }

    /// 从多个 slices 复制构造（动态数量）
    #[inline]
    pub fn copy_from_slices<T: Copy, const ALIGN: usize>(
        slices: &[&[T]],
    ) -> AllocResult<(Arc<Self>, Vec<Range<usize>>)> {
        const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };

        if unlikely(slices.is_empty()) {
            return Err(AllocError::EmptySizes);
        }

        let elem_size = std::mem::size_of::<T>();
        let num_slices = slices.len();

        let mut ranges = Vec::with_capacity(num_slices);
        let mut current_offset = 0usize;

        // 第一遍：计算布局
        for i in 0..num_slices {
            // SAFETY: i < num_slices
            let slice = unsafe { *slices.get_unchecked(i) };

            if unlikely(slice.is_empty()) {
                return Err(AllocError::ZeroSize);
            }

            let byte_size = match slice.len().checked_mul(elem_size) {
                Some(v) => v,
                None => return Err(AllocError::LayoutError),
            };

            let aligned_size = const_align_up::<ALIGN>(byte_size);
            ranges.push(current_offset..current_offset + aligned_size);

            current_offset = match current_offset.checked_add(aligned_size) {
                Some(v) => v,
                None => return Err(AllocError::LayoutError),
            };
        }

        let storage = Self::alloc::<ALIGN>(current_offset)?;
        let base_ptr = storage.data().as_ptr();

        // SAFETY: 所有范围和指针已验证
        unsafe {
            assume_unchecked!((base_ptr as usize) % ALIGN == 0);
            assume_unchecked!(ranges.len() == num_slices);

            // 第二遍：复制数据
            for i in 0..num_slices {
                let slice = *slices.get_unchecked(i);
                let range = ranges.get_unchecked(i);
                let dst = base_ptr.add(range.start) as *mut T;

                // 预取下一个目标区域
                if i + 1 < num_slices {
                    let next_range = ranges.get_unchecked(i + 1);
                    prefetch_write(base_ptr.add(next_range.start) as *mut T);
                }

                copy_nonoverlapping_aligned::<T, ALIGN>(slice.as_ptr(), dst, slice.len());
            }
        }

        Ok((storage, ranges))
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// 获取对齐后的起始指针（NonNull 保证非空）
    #[inline(always)]
    #[must_use]
    pub fn data(&self) -> NonNull<u8> {
        self.start
    }

    /// 获取裸指针
    #[inline(always)]
    #[must_use]
    pub fn as_ptr(&self) -> *mut u8 {
        self.start.as_ptr()
    }

    /// 获取指定偏移处的指针（带边界检查）
    #[inline]
    #[must_use]
    pub fn ptr_at(&self, offset: usize) -> Option<NonNull<u8>> {
        if likely(offset < self.size) {
            // SAFETY: offset 已检查在范围内
            Some(unsafe { NonNull::new_unchecked(self.start.as_ptr().add(offset)) })
        } else {
            None
        }
    }

    /// 获取指定偏移处的指针（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `offset < self.size()`
    #[inline(always)]
    #[must_use]
    pub unsafe fn ptr_at_unchecked(&self, offset: usize) -> NonNull<u8> {
        debug_assert!(
            offset < self.size,
            "offset {} out of bounds {}",
            offset,
            self.size
        );
        assume_unchecked!(offset < self.size);
        NonNull::new_unchecked(self.start.as_ptr().add(offset))
    }

    /// 获取可用大小（字节）
    #[inline(always)]
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// 获取可用大小（字节）- 带 assume 的版本
    ///
    /// 返回值保证 > 0（Storage 不允许 0 大小分配）
    #[inline(always)]
    #[must_use]
    pub fn size_nonzero(&self) -> usize {
        let s = self.size;
        // SAFETY: 构造时已验证 size > 0
        unsafe { assume_unchecked!(s > 0) };
        s
    }

    /// 获取对齐要求
    #[inline(always)]
    #[must_use]
    pub fn align(&self) -> usize {
        self.align
    }

    /// 获取实际分配的大小（包含 padding）
    #[inline(always)]
    #[must_use]
    pub fn allocated_size(&self) -> usize {
        self.alloc_size
    }

    // =========================================================================
    // Slice Accessors
    // =========================================================================

    /// 检查偏移和长度是否在有效范围内
    #[inline(always)]
    #[must_use]
    pub fn contains(&self, offset: usize, len: usize) -> bool {
        // 使用 checked_add 处理溢出，语义更清晰
        offset
            .checked_add(len)
            .map_or(false, |end| end <= self.size)
    }

    /// 检查偏移和长度是否在有效范围内（不检查溢出）
    ///
    /// # Safety
    ///
    /// 调用者需确保 offset + len 不会溢出
    #[inline(always)]
    #[must_use]
    pub unsafe fn contains_unchecked(&self, offset: usize, len: usize) -> bool {
        offset + len <= self.size
    }

    /// 获取指定范围的切片（通过 offset + len）
    #[inline]
    #[must_use]
    pub fn slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        if likely(self.contains(offset, len)) {
            Some(unsafe { self.slice_unchecked(offset, len) })
        } else {
            None
        }
    }

    /// 获取指定范围的切片（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `offset + len <= self.size()`
    #[inline(always)]
    #[must_use]
    pub unsafe fn slice_unchecked(&self, offset: usize, len: usize) -> &[u8] {
        debug_assert!(offset + len <= self.size);
        assume_unchecked!(offset + len <= self.size);
        std::slice::from_raw_parts(self.start.as_ptr().add(offset), len)
    }

    /// 获取指定范围的切片（通过 Range）
    #[inline]
    #[must_use]
    pub fn slice_range(&self, range: &Range<usize>) -> Option<&[u8]> {
        // 提前检查无效 range（end < start）
        if unlikely(range.end < range.start) {
            return None;
        }
        let len = range.end - range.start;
        self.slice(range.start, len)
    }

    /// 获取指定范围的切片（通过 Range，不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 range 有效且在边界内
    #[inline(always)]
    #[must_use]
    pub unsafe fn slice_range_unchecked(&self, range: &Range<usize>) -> &[u8] {
        debug_assert!(range.start <= range.end);
        assume_unchecked!(range.start <= range.end);
        self.slice_unchecked(range.start, range.end - range.start)
    }

    /// 获取指定范围的可变切片（通过 offset + len）
    #[inline]
    #[must_use]
    pub fn slice_mut(&mut self, offset: usize, len: usize) -> Option<&mut [u8]> {
        if likely(self.contains(offset, len)) {
            Some(unsafe { self.slice_mut_unchecked(offset, len) })
        } else {
            None
        }
    }

    /// 获取指定范围的可变切片（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `offset + len <= self.size()` 且没有其他引用
    #[inline(always)]
    #[must_use]
    pub unsafe fn slice_mut_unchecked(&mut self, offset: usize, len: usize) -> &mut [u8] {
        debug_assert!(offset + len <= self.size);
        assume_unchecked!(offset + len <= self.size);
        std::slice::from_raw_parts_mut(self.start.as_ptr().add(offset), len)
    }

    /// 获取指定范围的可变切片（通过 Range）
    #[inline]
    #[must_use]
    pub fn slice_range_mut(&mut self, range: &Range<usize>) -> Option<&mut [u8]> {
        // 提前检查无效 range（end < start）
        if unlikely(range.end < range.start) {
            return None;
        }
        let len = range.end - range.start;
        self.slice_mut(range.start, len)
    }

    /// 获取指定范围的可变切片（通过 Range，不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 range 有效且在边界内，且没有其他引用
    #[inline(always)]
    #[must_use]
    pub unsafe fn slice_range_mut_unchecked(&mut self, range: &Range<usize>) -> &mut [u8] {
        debug_assert!(range.start <= range.end);
        assume_unchecked!(range.start <= range.end);
        self.slice_mut_unchecked(range.start, range.end - range.start)
    }
}

impl Drop for Storage {
    #[inline]
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        {
            TOTAL_ALLOCATED.fetch_sub(self.alloc_size, Ordering::Relaxed);
            ALLOCATION_COUNT.fetch_sub(1, Ordering::Relaxed);
        }

        // SAFETY: buffer 和 layout 在构造时已验证，使用相同的 align
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.alloc_size, self.align);
            dealloc(self.buffer.as_ptr(), layout);
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// 向上取整到 align 的倍数（编译期常量版本）
#[inline(always)]
#[must_use]
pub(crate) const fn align_up_const<const ALIGN: usize>(value: usize) -> usize {
    const_align_up::<ALIGN>(value)
}

/// 向上取整到 align 的倍数（运行时版本）
#[inline(always)]
#[must_use]
pub(crate) const fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// 检查地址是否对齐（编译期常量版本）
#[inline(always)]
#[must_use]
pub(crate) const fn is_aligned_const<const ALIGN: usize>(addr: usize) -> bool {
    const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };
    addr & (ALIGN - 1) == 0
}

/// 检查地址是否对齐（运行时版本）
#[inline(always)]
#[must_use]
pub(crate) const fn is_aligned(addr: usize, align: usize) -> bool {
    debug_assert!(align.is_power_of_two());
    addr & (align - 1) == 0
}

/// 编译期断言：检查类型大小和对齐关系
#[inline(always)]
pub(crate) const fn assert_align_compatible<T, const ALIGN: usize>() {
    const {
        assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two");
        assert!(
            ALIGN >= std::mem::align_of::<T>(),
            "ALIGN must be >= type alignment"
        );
    };
}

// =============================================================================
// Macros
// =============================================================================

/// 便捷宏：分配单个 Storage（默认对齐）
#[macro_export]
macro_rules! alloc {
    ($size:expr) => {{
        $crate::storage::Storage::alloc::<{ $crate::storage::DEFAULT_ALIGN }>($size)
    }};
}

/// 便捷宏：分配多个区域（静态数量，默认对齐）
#[macro_export]
macro_rules! alloc_n {
    ($($size:expr),+ $(,)?) => {{
        $crate::storage::Storage::alloc_n::<{ $crate::storage::DEFAULT_ALIGN }, { [$($size),+].len() }>([$($size),+])
    }};
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    // =========================================================================
    // Basic Allocation Tests
    // =========================================================================

    #[test]
    fn test_alloc_basic() {
        let storage = Storage::alloc::<32>(100).expect("alloc failed");
        assert!(storage.size() >= 100);
        assert_eq!(storage.align(), 32);
        assert!(is_aligned(storage.as_ptr() as usize, 32));
    }

    #[test]
    fn test_alloc_align64() {
        let storage = Storage::alloc::<64>(100).expect("alloc failed");
        assert!(storage.size() >= 100);
        assert_eq!(storage.align(), 64);
        assert!(is_aligned(storage.as_ptr() as usize, 64));
    }

    #[test]
    fn test_alloc_zero_size_fails() {
        let result = Storage::alloc::<32>(0);
        assert_eq!(result.unwrap_err(), AllocError::ZeroSize);
    }

    // =========================================================================
    // Alloc N Tests
    // =========================================================================

    #[test]
    fn test_alloc_n() {
        let (storage, [r0, r1, r2]) = Storage::alloc_n::<32, 3>([100, 200, 300]).unwrap();

        assert_eq!(r0.start, 0);
        assert_eq!(r1.start, 128); // 100 → 128 (aligned)
        assert_eq!(r2.start, 352); // 128 + 224

        // 所有偏移都对齐
        assert!(is_aligned(storage.as_ptr() as usize + r0.start, 32));
        assert!(is_aligned(storage.as_ptr() as usize + r1.start, 32));
        assert!(is_aligned(storage.as_ptr() as usize + r2.start, 32));
    }

    // =========================================================================
    // Alloc Slices Tests (Dynamic)
    // =========================================================================

    #[test]
    fn test_alloc_slices() {
        let (storage, ranges) = Storage::alloc_slices::<32>(&[100, 200, 300]).unwrap();

        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[1].start, 128);
        assert_eq!(ranges[2].start, 352);

        for range in &ranges {
            let ptr = storage.as_ptr() as usize + range.start;
            assert!(is_aligned(ptr, 32));
        }
    }

    #[test]
    fn test_alloc_slices_empty() {
        let result = Storage::alloc_slices::<32>(&[]);
        assert_eq!(result.unwrap_err(), AllocError::EmptySizes);
    }

    #[test]
    fn test_alloc_slices_zero_size() {
        let result = Storage::alloc_slices::<32>(&[100, 0, 200]);
        assert_eq!(result.unwrap_err(), AllocError::ZeroSize);
    }

    // =========================================================================
    // Copy From Tests
    // =========================================================================

    #[test]
    fn test_copy_from() {
        let data = [1u8, 2, 3, 4, 5];
        let storage = Storage::copy_from::<u8, 32>(&data).unwrap();

        let slice = storage.slice(0, 5).unwrap();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_copy_from_f32() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let storage = Storage::copy_from::<f32, 32>(&data).unwrap();

        let ptr = storage.as_ptr() as *const f32;
        unsafe {
            assert_eq!(*ptr, 1.0);
            assert_eq!(*ptr.add(1), 2.0);
            assert_eq!(*ptr.add(2), 3.0);
            assert_eq!(*ptr.add(3), 4.0);
        }
    }

    // =========================================================================
    // Copy From N Tests
    // =========================================================================

    #[test]
    fn test_copy_from_n() {
        let a = [1u8, 2];
        let b = [3u8, 4, 5];
        let (storage, [r0, r1]) = Storage::copy_from_n::<u8, 32, 2>([&a, &b]).unwrap();

        let slice0 = storage.slice_range(&r0).unwrap();
        let slice1 = storage.slice_range(&r1).unwrap();

        assert_eq!(&slice0[..2], &[1, 2]);
        assert_eq!(&slice1[..3], &[3, 4, 5]);
    }

    // =========================================================================
    // Copy From Slices Tests (Dynamic)
    // =========================================================================

    #[test]
    fn test_copy_from_slices() {
        let a = [1u8, 2];
        let b = [3u8, 4, 5];
        let (storage, ranges) = Storage::copy_from_slices::<u8, 32>(&[&a, &b]).unwrap();

        assert_eq!(ranges.len(), 2);

        let slice0 = storage.slice_range(&ranges[0]).unwrap();
        let slice1 = storage.slice_range(&ranges[1]).unwrap();

        assert_eq!(&slice0[..2], &[1, 2]);
        assert_eq!(&slice1[..3], &[3, 4, 5]);
    }

    // =========================================================================
    // Pointer Access Tests
    // =========================================================================

    #[test]
    fn test_ptr_at() {
        let storage = Storage::alloc::<32>(256).unwrap();

        assert!(storage.ptr_at(0).is_some());
        assert!(storage.ptr_at(255).is_some());
        assert!(storage.ptr_at(256).is_none()); // 边界外
        assert!(storage.ptr_at(1000).is_none());
    }

    #[test]
    fn test_ptr_at_unchecked() {
        let storage = Storage::alloc::<32>(256).unwrap();
        unsafe {
            let ptr0 = storage.ptr_at_unchecked(0);
            let ptr64 = storage.ptr_at_unchecked(64);
            assert_eq!(ptr64.as_ptr() as usize - ptr0.as_ptr() as usize, 64);
        }
    }

    // =========================================================================
    // Slice Tests
    // =========================================================================

    #[test]
    fn test_slice() {
        let storage = Storage::alloc::<32>(256).unwrap();

        assert!(storage.slice(0, 256).is_some());
        assert!(storage.slice(0, 257).is_none());
        assert!(storage.slice(200, 100).is_none());
        assert!(storage.slice(usize::MAX, 1).is_none());
    }

    #[test]
    fn test_slice_range() {
        let storage = Storage::alloc::<32>(256).unwrap();

        assert!(storage.slice_range(&(0..256)).is_some());
        assert!(storage.slice_range(&(0..257)).is_none());
        assert!(storage.slice_range(&(100..300)).is_none());
        // 测试无效 range（end < start）
        assert!(storage.slice_range(&(100..50)).is_none());
    }

    #[test]
    fn test_contains() {
        let storage = Storage::alloc::<32>(256).unwrap();

        assert!(storage.contains(0, 256));
        assert!(storage.contains(0, 0));
        assert!(storage.contains(255, 1));
        assert!(!storage.contains(256, 1));
        assert!(!storage.contains(0, 257));
        assert!(!storage.contains(usize::MAX, 1));
    }

    // =========================================================================
    // Arc Sharing Tests
    // =========================================================================

    #[test]
    fn test_arc_sharing() {
        let storage = Storage::alloc::<32>(100).unwrap();
        let storage2 = Arc::clone(&storage);

        assert_eq!(Arc::strong_count(&storage), 2);
        assert_eq!(storage.as_ptr(), storage2.as_ptr());

        drop(storage2);
        assert_eq!(Arc::strong_count(&storage), 1);
    }

    // =========================================================================
    // Concurrent Access Tests
    // =========================================================================

    #[test]
    fn test_concurrent_access() {
        let storage = Storage::alloc::<32>(1024).unwrap();

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let s = Arc::clone(&storage);
                thread::spawn(move || unsafe {
                    let ptr = s.ptr_at_unchecked(i * 256);
                    std::ptr::write_bytes(ptr.as_ptr(), i as u8, 256);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // 验证写入
        for i in 0..4 {
            let slice = storage.slice(i * 256, 256).unwrap();
            assert!(slice.iter().all(|&b| b == i as u8));
        }
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(100, 32), 128);
    }

    #[test]
    fn test_is_aligned() {
        assert!(is_aligned(0, 32));
        assert!(is_aligned(32, 32));
        assert!(is_aligned(64, 32));
        assert!(!is_aligned(1, 32));
        assert!(!is_aligned(33, 32));
    }

    // =========================================================================
    // Memory Safety Tests (run with miri)
    // =========================================================================

    #[test]
    fn test_no_ub() {
        let storage = Storage::alloc::<32>(100).unwrap();
        unsafe {
            // 写入边界
            std::ptr::write(storage.ptr_at_unchecked(0).as_ptr(), 42);
            let last_idx = storage.size() - 1;
            std::ptr::write(storage.ptr_at_unchecked(last_idx).as_ptr(), 42);
        }
    }

    #[test]
    fn test_alloc_n_into_ranges() {
        let (storage, ranges) = Storage::alloc_n::<32, 2>([100, 200]).unwrap();

        assert_eq!(ranges.len(), 2);
        assert!(storage.size() >= 100 + 200);
    }

    // =========================================================================
    // Macro Tests
    // =========================================================================

    #[test]
    fn test_alloc_macro() {
        let storage = alloc!(1024).unwrap();
        assert!(storage.size() >= 1024);
        assert!(is_aligned(storage.as_ptr() as usize, DEFAULT_ALIGN));
    }

    #[test]
    fn test_alloc_n_macro() {
        let (storage, [r0, r1, r2]) = alloc_n!(100, 200, 300).unwrap();

        assert_eq!(r0.start, 0);
        assert!(storage.size() >= 100 + 200 + 300);
    }

    // =========================================================================
    // Debug Statistics Tests
    // =========================================================================

    #[cfg(debug_assertions)]
    #[test]
    fn test_alloc_stats() {
        AllocStats::reset();

        let storage1 = Storage::alloc::<32>(100).unwrap();
        let stats1 = AllocStats::current();
        assert_eq!(stats1.allocation_count, 1);
        assert!(stats1.total_bytes > 0);

        let storage2 = Storage::alloc::<32>(200).unwrap();
        let stats2 = AllocStats::current();
        assert_eq!(stats2.allocation_count, 2);
        assert!(stats2.total_bytes > stats1.total_bytes);

        drop(storage1);
        let stats3 = AllocStats::current();
        assert_eq!(stats3.allocation_count, 1);

        drop(storage2);
        let stats4 = AllocStats::current();
        assert_eq!(stats4.allocation_count, 0);
        assert_eq!(stats4.total_bytes, 0);
    }
}
