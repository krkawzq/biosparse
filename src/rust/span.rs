//! Span - 类型化的内存视图
//!
//! Span 是 scl 的核心数据结构，提供对连续内存区域的类型化访问。
//!
//! # 设计要点
//!
//! - **两种模式**：
//!   - `Shared`: 持有 `Arc<Storage>`，与其他 Span 共享内存
//!   - `View`: 不持有内存，仅引用外部数据（如 NumPy 数组）
//!
//! - **类型安全**：通过泛型 `T` 提供类型化访问
//!
//! - **对齐感知**：跟踪数据指针是否满足 SIMD 对齐要求
//!
//! # API 命名规范
//!
//! - `alloc_*` 系列：纯分配，零初始化
//! - `copy_from_*` 系列：从外部数据复制构造
//! - `from_*` 系列：创建 View（不持有内存）
//!
//! # Performance Notes
//!
//! 本模块使用多种编译器优化技巧：
//! - `#[inline(always)]` 强制内联热路径
//! - `assume_unchecked!` 消除冗余边界检查
//! - const 泛型参数启用编译期计算

#![allow(clippy::missing_safety_doc)]

use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::assume_unchecked;
use crate::storage::{is_aligned, AllocError, AllocResult, Storage, DEFAULT_ALIGN};
use crate::tools::{copy_nonoverlapping_aligned, likely, unlikely};

// =============================================================================
// Constants
// =============================================================================

/// Span flags 位掩码
pub mod flags {
    /// 是否为 View 模式（不持有 Storage）
    pub const VIEW: usize = 1 << 0;

    /// 数据指针是否对齐
    pub const ALIGNED: usize = 1 << 1;

    /// 是否可变（可写）
    pub const MUTABLE: usize = 1 << 2;
}

// =============================================================================
// SpanFlags
// =============================================================================

/// Span 的状态标志位
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct SpanFlags(usize);

impl SpanFlags {
    /// 空标志
    pub const EMPTY: Self = Self(0);

    /// View 模式
    pub const VIEW: Self = Self(flags::VIEW);

    /// 对齐标志
    pub const ALIGNED: Self = Self(flags::ALIGNED);

    /// 可变标志
    pub const MUTABLE: Self = Self(flags::MUTABLE);

    /// 创建新的标志集
    #[inline(always)]
    #[must_use]
    pub const fn new(bits: usize) -> Self {
        Self(bits)
    }

    /// 获取原始位
    #[inline(always)]
    #[must_use]
    pub const fn bits(self) -> usize {
        self.0
    }

    /// 检查是否设置了指定标志
    #[inline(always)]
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// 检查是否为 View 模式
    #[inline(always)]
    #[must_use]
    pub const fn is_view(self) -> bool {
        self.contains(Self::VIEW)
    }

    /// 检查是否对齐
    #[inline(always)]
    #[must_use]
    pub const fn is_aligned(self) -> bool {
        self.contains(Self::ALIGNED)
    }

    /// 检查是否可变
    #[inline(always)]
    #[must_use]
    pub const fn is_mutable(self) -> bool {
        self.contains(Self::MUTABLE)
    }

    /// 设置标志
    #[inline(always)]
    #[must_use]
    pub const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// 清除标志
    #[inline(always)]
    #[must_use]
    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// 根据条件设置或清除标志
    #[inline(always)]
    #[must_use]
    pub const fn set(self, flag: Self, value: bool) -> Self {
        if value {
            self.with(flag)
        } else {
            self.without(flag)
        }
    }
}

impl std::ops::BitOr for SpanFlags {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitAnd for SpanFlags {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

// =============================================================================
// Span
// =============================================================================

/// Span - 类型化的连续内存视图
///
/// # Memory Layout
///
/// ```text
/// Span<T> {
///     storage: Option<Arc<Storage>>,  // 8 bytes (None for View)
///     data: NonNull<T>,               // 8 bytes
///     len: usize,                     // 8 bytes (element count)
///     flags: SpanFlags,               // 8 bytes
/// }
/// ```
///
/// Total: 32 bytes on 64-bit platforms
#[derive(Debug)]
pub struct Span<T> {
    /// 底层存储（View 模式为 None）
    storage: Option<Arc<Storage>>,

    /// 数据指针（始终非空，指向第一个元素）
    data: NonNull<T>,

    /// 元素个数
    len: usize,

    /// 状态标志
    flags: SpanFlags,

    /// 类型标记
    _marker: PhantomData<T>,
}

// Safety: Span 可以跨线程发送（如果 T: Send）
// Span 可以被多线程共享读取（如果 T: Send + Sync）
// 注意：Sync 要求 T: Send + Sync，因为多线程可能同时读取同一个 Span
unsafe impl<T: Send> Send for Span<T> {}
unsafe impl<T: Send + Sync> Sync for Span<T> {}

impl<T> Span<T> {
    // =========================================================================
    // Alloc 系列 - 纯分配，零初始化
    // =========================================================================

    /// 分配新的对齐内存（单个 Span）
    ///
    /// # Type Parameters
    ///
    /// * `ALIGN` - 对齐要求（必须是 2 的幂）
    ///
    /// # Arguments
    ///
    /// * `len` - 元素个数
    ///
    /// # Returns
    ///
    /// * `Ok(Span<T>)` - 零初始化的可变 Span
    /// * `Err(AllocError)` - 分配失败
    #[inline]
    pub fn alloc<const ALIGN: usize>(len: usize) -> AllocResult<Self> {
        const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };

        if unlikely(len == 0) {
            return Err(AllocError::ZeroSize);
        }

        let byte_size = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(AllocError::LayoutError)?;

        let storage = Storage::alloc::<ALIGN>(byte_size)?;
        let data = storage.data().cast::<T>();

        Ok(Self {
            storage: Some(storage),
            data,
            len,
            flags: SpanFlags::ALIGNED | SpanFlags::MUTABLE,
            _marker: PhantomData,
        })
    }

    /// 分配多个 Span（静态数量，共享一个 Storage）
    ///
    /// # Type Parameters
    ///
    /// * `ALIGN` - 对齐要求
    /// * `N` - Span 数量（编译期常量）
    ///
    /// # Performance
    ///
    /// N 为编译期常量，循环会被完全展开
    #[inline]
    pub fn alloc_n<const ALIGN: usize, const N: usize>(lens: [usize; N]) -> AllocResult<[Self; N]> {
        const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };
        const { assert!(N > 0, "N must be greater than 0") };

        let elem_size = std::mem::size_of::<T>();

        // 计算每个 Span 的字节大小（会被展开）
        let mut byte_sizes: [usize; N] = [0; N];
        for i in 0..N {
            // SAFETY: i < N
            let len_i = unsafe { *lens.get_unchecked(i) };
            let byte_size = len_i
                .checked_mul(elem_size)
                .ok_or(AllocError::LayoutError)?;
            unsafe { *byte_sizes.get_unchecked_mut(i) = byte_size };
        }

        let (storage, ranges) = Storage::alloc_n::<ALIGN, N>(byte_sizes)?;
        let base_ptr = storage.data().as_ptr();

        // 构造 Span 数组（会被展开）
        let spans: [Self; N] = std::array::from_fn(|i| {
            // SAFETY: i < N 由 from_fn 保证
            let len_i = unsafe { *lens.get_unchecked(i) };
            let range = unsafe { ranges.get_unchecked(i) };
            let ptr = unsafe { base_ptr.add(range.start) as *mut T };

            Self {
                storage: Some(Arc::clone(&storage)),
                data: unsafe { NonNull::new_unchecked(ptr) },
                len: len_i,
                flags: SpanFlags::ALIGNED | SpanFlags::MUTABLE,
                _marker: PhantomData,
            }
        });

        Ok(spans)
    }

    /// 分配多个 Span（动态数量，共享一个 Storage）
    #[inline]
    pub fn alloc_slices<const ALIGN: usize>(lens: &[usize]) -> AllocResult<Vec<Self>> {
        const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };

        if unlikely(lens.is_empty()) {
            return Err(AllocError::EmptySizes);
        }

        let elem_size = std::mem::size_of::<T>();
        let num_spans = lens.len();

        // 计算每个 Span 的字节大小
        let mut byte_sizes = Vec::with_capacity(num_spans);
        for i in 0..num_spans {
            // SAFETY: i < num_spans
            let len_i = unsafe { *lens.get_unchecked(i) };
            let byte_size = len_i
                .checked_mul(elem_size)
                .ok_or(AllocError::LayoutError)?;
            byte_sizes.push(byte_size);
        }

        let (storage, ranges) = Storage::alloc_slices::<ALIGN>(&byte_sizes)?;
        let base_ptr = storage.data().as_ptr();

        // 构造 Span 列表
        let mut spans = Vec::with_capacity(num_spans);
        for i in 0..num_spans {
            // SAFETY: i < num_spans
            let len_i = unsafe { *lens.get_unchecked(i) };
            let range = unsafe { ranges.get_unchecked(i) };
            let ptr = unsafe { base_ptr.add(range.start) as *mut T };

            spans.push(Self {
                storage: Some(Arc::clone(&storage)),
                data: unsafe { NonNull::new_unchecked(ptr) },
                len: len_i,
                flags: SpanFlags::ALIGNED | SpanFlags::MUTABLE,
                _marker: PhantomData,
            });
        }

        Ok(spans)
    }

    // =========================================================================
    // Copy From 系列 - 从外部数据复制构造
    // =========================================================================

    /// 从单个 slice 复制构造
    ///
    /// # Performance
    ///
    /// 对于对齐的数据，使用优化的内存复制
    #[inline]
    pub fn copy_from<const ALIGN: usize>(data: &[T]) -> AllocResult<Self>
    where
        T: Copy,
    {
        const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };

        if unlikely(data.is_empty()) {
            return Err(AllocError::ZeroSize);
        }

        let storage = Storage::copy_from::<T, ALIGN>(data)?;
        let ptr = storage.data().cast::<T>();

        Ok(Self {
            storage: Some(storage),
            data: ptr,
            len: data.len(),
            flags: SpanFlags::ALIGNED | SpanFlags::MUTABLE,
            _marker: PhantomData,
        })
    }

    /// 从多个 slices 复制构造（静态数量）
    ///
    /// # Performance
    ///
    /// N 为编译期常量，循环会被完全展开
    #[inline]
    pub fn copy_from_n<const ALIGN: usize, const N: usize>(
        slices: [&[T]; N],
    ) -> AllocResult<[Self; N]>
    where
        T: Copy,
    {
        const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };
        const { assert!(N > 0, "N must be greater than 0") };

        let (storage, ranges) = Storage::copy_from_n::<T, ALIGN, N>(slices)?;
        let base_ptr = storage.data().as_ptr();

        let spans: [Self; N] = std::array::from_fn(|i| {
            // SAFETY: i < N
            let slice = unsafe { *slices.get_unchecked(i) };
            let range = unsafe { ranges.get_unchecked(i) };
            let ptr = unsafe { base_ptr.add(range.start) as *mut T };

            Self {
                storage: Some(Arc::clone(&storage)),
                data: unsafe { NonNull::new_unchecked(ptr) },
                len: slice.len(),
                flags: SpanFlags::ALIGNED | SpanFlags::MUTABLE,
                _marker: PhantomData,
            }
        });

        Ok(spans)
    }

    /// 从多个 slices 复制构造（动态数量）
    #[inline]
    pub fn copy_from_slices<const ALIGN: usize>(slices: &[&[T]]) -> AllocResult<Vec<Self>>
    where
        T: Copy,
    {
        const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };

        if unlikely(slices.is_empty()) {
            return Err(AllocError::EmptySizes);
        }

        let (storage, ranges) = Storage::copy_from_slices::<T, ALIGN>(slices)?;
        let base_ptr = storage.data().as_ptr();
        let num_slices = slices.len();

        let mut spans = Vec::with_capacity(num_slices);

        for i in 0..num_slices {
            // SAFETY: i < num_slices
            let slice = unsafe { *slices.get_unchecked(i) };
            let range = unsafe { ranges.get_unchecked(i) };
            let ptr = unsafe { base_ptr.add(range.start) as *mut T };

            spans.push(Self {
                storage: Some(Arc::clone(&storage)),
                data: unsafe { NonNull::new_unchecked(ptr) },
                len: slice.len(),
                flags: SpanFlags::ALIGNED | SpanFlags::MUTABLE,
                _marker: PhantomData,
            });
        }

        Ok(spans)
    }

    // =========================================================================
    // From 系列 - 创建 View（不持有内存）
    // =========================================================================

    /// 从原始指针创建 View
    ///
    /// # Safety
    ///
    /// 调用者必须保证：
    /// - `ptr` 指向至少 `len` 个连续的 `T` 类型元素
    /// - 指针在 Span 生命周期内有效
    /// - 如果设置 MUTABLE，调用者需确保独占访问
    #[inline(always)]
    pub unsafe fn from_raw_parts(ptr: NonNull<T>, len: usize, flags: SpanFlags) -> Self {
        let is_aligned = is_aligned(ptr.as_ptr() as usize, DEFAULT_ALIGN);
        let flags = flags
            .with(SpanFlags::VIEW)
            .set(SpanFlags::ALIGNED, is_aligned);

        Self {
            storage: None,
            data: ptr,
            len,
            flags,
            _marker: PhantomData,
        }
    }

    /// 从原始指针创建 View（指定对齐检测）
    ///
    /// # Safety
    ///
    /// 同 `from_raw_parts`
    #[inline(always)]
    pub unsafe fn from_raw_parts_aligned<const ALIGN: usize>(
        ptr: NonNull<T>,
        len: usize,
        flags: SpanFlags,
    ) -> Self {
        const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };

        let is_aligned = is_aligned(ptr.as_ptr() as usize, ALIGN);
        let flags = flags
            .with(SpanFlags::VIEW)
            .set(SpanFlags::ALIGNED, is_aligned);

        Self {
            storage: None,
            data: ptr,
            len,
            flags,
            _marker: PhantomData,
        }
    }

    /// 从原始指针创建 View（不做对齐检测，直接使用传入的 flags）
    ///
    /// # Safety
    ///
    /// 同 `from_raw_parts`，调用者需自行确保 flags 正确
    #[inline(always)]
    pub unsafe fn from_raw_parts_unchecked(ptr: NonNull<T>, len: usize, flags: SpanFlags) -> Self {
        Self {
            storage: None,
            data: ptr,
            len,
            flags: flags.with(SpanFlags::VIEW),
            _marker: PhantomData,
        }
    }

    /// 从切片创建只读 View
    ///
    /// 返回的 Span 是 View 模式，不持有内存
    #[inline(always)]
    #[must_use]
    pub fn from_slice(slice: &[T]) -> Self {
        // SAFETY: slice 保证有效
        let ptr = unsafe { NonNull::new_unchecked(slice.as_ptr() as *mut T) };
        let is_aligned = is_aligned(ptr.as_ptr() as usize, DEFAULT_ALIGN);

        Self {
            storage: None,
            data: ptr,
            len: slice.len(),
            flags: SpanFlags::VIEW.set(SpanFlags::ALIGNED, is_aligned),
            _marker: PhantomData,
        }
    }

    /// 从可变切片创建可变 View
    #[inline(always)]
    #[must_use]
    pub fn from_slice_mut(slice: &mut [T]) -> Self {
        // SAFETY: slice 保证有效
        let ptr = unsafe { NonNull::new_unchecked(slice.as_mut_ptr()) };
        let is_aligned = is_aligned(ptr.as_ptr() as usize, DEFAULT_ALIGN);

        Self {
            storage: None,
            data: ptr,
            len: slice.len(),
            flags: SpanFlags::VIEW
                .with(SpanFlags::MUTABLE)
                .set(SpanFlags::ALIGNED, is_aligned),
            _marker: PhantomData,
        }
    }

    // =========================================================================
    // Deep Copy
    // =========================================================================

    /// 深拷贝 Span（创建独立的内存副本）
    ///
    /// # Type Parameters
    ///
    /// * `ALIGN` - 新 Storage 的对齐要求
    #[cold]
    #[inline(never)]
    pub fn deep_copy<const ALIGN: usize>(&self) -> AllocResult<Self>
    where
        T: Copy,
    {
        Self::copy_from::<ALIGN>(self.as_slice())
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// 获取数据指针
    #[inline(always)]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// 获取可变数据指针
    ///
    /// # Panics
    ///
    /// 如果 Span 不可变则 panic
    #[inline(always)]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        assert!(self.flags.is_mutable(), "Span is not mutable");
        self.data.as_ptr()
    }

    /// 获取可变数据指针（不检查）
    ///
    /// # Safety
    ///
    /// 调用者需确保 Span 可变
    #[inline(always)]
    #[must_use]
    pub unsafe fn as_mut_ptr_unchecked(&self) -> *mut T {
        debug_assert!(self.flags.is_mutable(), "Span is not mutable");
        self.data.as_ptr()
    }

    /// 获取 NonNull 指针
    #[inline(always)]
    #[must_use]
    pub fn data(&self) -> NonNull<T> {
        self.data
    }

    /// 元素个数
    #[inline(always)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 元素个数（带 assume > 0）
    ///
    /// # Safety
    ///
    /// 调用者需确保 Span 非空
    #[inline(always)]
    #[must_use]
    pub unsafe fn len_nonzero(&self) -> usize {
        debug_assert!(self.len > 0);
        assume_unchecked!(self.len > 0);
        self.len
    }

    /// 是否为空
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 字节大小
    #[inline(always)]
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// 获取标志
    #[inline(always)]
    #[must_use]
    pub fn flags(&self) -> SpanFlags {
        self.flags
    }

    /// 是否为 View 模式
    #[inline(always)]
    #[must_use]
    pub fn is_view(&self) -> bool {
        self.flags.is_view()
    }

    /// 是否对齐
    #[inline(always)]
    #[must_use]
    pub fn is_aligned(&self) -> bool {
        self.flags.is_aligned()
    }

    /// 是否可变
    #[inline(always)]
    #[must_use]
    pub fn is_mutable(&self) -> bool {
        self.flags.is_mutable()
    }

    /// 检查是否持有 Storage（Shared 模式）
    #[inline(always)]
    #[must_use]
    pub fn has_storage(&self) -> bool {
        self.storage.is_some()
    }

    /// 获取 Storage 的引用计数（仅 Shared 模式有效）
    #[inline]
    #[must_use]
    pub fn storage_ref_count(&self) -> Option<usize> {
        self.storage.as_ref().map(Arc::strong_count)
    }

    // =========================================================================
    // Slice Access
    // =========================================================================

    /// 获取只读切片
    #[inline(always)]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: data 指向有效的 len 个 T 元素
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    /// 获取可变切片
    ///
    /// # Panics
    ///
    /// 如果 Span 不可变则 panic
    #[inline]
    #[must_use]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        assert!(self.flags.is_mutable(), "Span is not mutable");
        // SAFETY: data 指向有效的 len 个 T 元素，且已检查可变
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }

    /// 获取可变切片（不检查）
    ///
    /// # Safety
    ///
    /// 调用者需确保 Span 可变且没有其他引用
    #[inline(always)]
    #[must_use]
    pub unsafe fn as_slice_mut_unchecked(&self) -> &mut [T] {
        debug_assert!(self.flags.is_mutable(), "Span is not mutable");
        std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len)
    }

    // =========================================================================
    // Element Access
    // =========================================================================

    /// 获取元素引用（带边界检查）
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        if likely(index < self.len) {
            Some(unsafe { &*self.data.as_ptr().add(index) })
        } else {
            None
        }
    }

    /// 获取可变元素引用（带边界检查）
    #[inline]
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if likely(index < self.len && self.flags.is_mutable()) {
            Some(unsafe { &mut *self.data.as_ptr().add(index) })
        } else {
            None
        }
    }

    /// 获取元素引用（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `index < self.len()`
    #[inline(always)]
    #[must_use]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        debug_assert!(index < self.len, "index out of bounds");
        assume_unchecked!(index < self.len);
        &*self.data.as_ptr().add(index)
    }

    /// 获取可变元素引用（不检查边界和可变性）
    ///
    /// # Safety
    ///
    /// 调用者需确保 `index < self.len()` 且 Span 可变
    #[inline(always)]
    #[must_use]
    pub unsafe fn get_unchecked_mut(&self, index: usize) -> &mut T {
        debug_assert!(index < self.len, "index out of bounds");
        debug_assert!(self.flags.is_mutable(), "Span is not mutable");
        assume_unchecked!(index < self.len);
        &mut *self.data.as_ptr().add(index)
    }

    // =========================================================================
    // Subspan / Slice Operations
    // =========================================================================

    /// 获取子范围（带边界检查）
    #[inline]
    #[must_use]
    pub fn subspan(&self, offset: usize, len: usize) -> Option<Self> {
        let end = offset.checked_add(len)?;
        if end > self.len {
            return None;
        }

        let ptr = unsafe { NonNull::new_unchecked(self.data.as_ptr().add(offset)) };

        Some(Self {
            storage: self.storage.clone(),
            data: ptr,
            len,
            flags: self.flags,
            _marker: PhantomData,
        })
    }

    /// 获取子范围（不检查边界）
    ///
    /// # Safety
    ///
    /// 调用者需确保 offset + len <= self.len 且不会溢出
    #[inline(always)]
    #[must_use]
    pub unsafe fn subspan_unchecked(&self, offset: usize, len: usize) -> Self {
        debug_assert!(
            offset.checked_add(len).map_or(false, |end| end <= self.len),
            "subspan_unchecked: offset + len overflow or out of bounds"
        );
        assume_unchecked!(offset + len <= self.len);

        let ptr = NonNull::new_unchecked(self.data.as_ptr().add(offset));

        Self {
            storage: self.storage.clone(),
            data: ptr,
            len,
            flags: self.flags,
            _marker: PhantomData,
        }
    }

    // =========================================================================
    // Bulk Operations (诱导 SIMD)
    // =========================================================================

    /// 填充所有元素
    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Copy,
    {
        assert!(self.flags.is_mutable(), "Span is not mutable");
        let slice = unsafe { self.as_slice_mut_unchecked() };
        slice.fill(value);
    }

    /// 填充所有元素（不检查可变性）
    ///
    /// # Safety
    ///
    /// 调用者需确保 Span 可变
    #[inline]
    pub unsafe fn fill_unchecked(&self, value: T)
    where
        T: Copy,
    {
        debug_assert!(self.flags.is_mutable());
        let slice = self.as_slice_mut_unchecked();
        slice.fill(value);
    }

    /// 从另一个 Span 复制数据
    #[inline]
    pub fn copy_from_span(&mut self, src: &Span<T>)
    where
        T: Copy,
    {
        assert!(self.flags.is_mutable(), "Span is not mutable");
        assert!(src.len <= self.len, "source too large");

        unsafe {
            copy_nonoverlapping_aligned::<T, DEFAULT_ALIGN>(
                src.data.as_ptr(),
                self.data.as_ptr(),
                src.len,
            );
        }
    }

    /// 从另一个 Span 复制数据（不检查）
    ///
    /// # Safety
    ///
    /// 调用者需确保：
    /// - self 可变
    /// - src.len <= self.len
    /// - 两个 Span 不重叠
    #[inline]
    pub unsafe fn copy_from_span_unchecked<const ALIGN: usize>(&self, src: &Span<T>)
    where
        T: Copy,
    {
        debug_assert!(self.flags.is_mutable());
        debug_assert!(src.len <= self.len);

        copy_nonoverlapping_aligned::<T, ALIGN>(src.data.as_ptr(), self.data.as_ptr(), src.len);
    }

    // =========================================================================
    // Iteration
    // =========================================================================

    /// 返回元素迭代器
    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// 返回可变元素迭代器
    ///
    /// # Panics
    ///
    /// 如果 Span 不可变则 panic
    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.as_slice_mut().iter_mut()
    }
}

// =============================================================================
// Clone
// =============================================================================

impl<T> Clone for Span<T> {
    /// 克隆 Span
    ///
    /// - Shared 模式：增加 Storage 引用计数
    /// - View 模式：复制指针（调用者需确保外部数据生命周期）
    #[inline]
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            data: self.data,
            len: self.len,
            flags: self.flags,
            _marker: PhantomData,
        }
    }
}

// =============================================================================
// Default (for take operations)
// =============================================================================

impl<T> Default for Span<T> {
    /// 创建空的 View Span
    ///
    /// 返回一个长度为 0 的 View，指向静态空地址
    #[inline]
    fn default() -> Self {
        // 使用 NonNull::dangling() 作为空指针
        Self {
            storage: None,
            data: NonNull::dangling(),
            len: 0,
            flags: SpanFlags::VIEW,
            _marker: PhantomData,
        }
    }
}

// =============================================================================
// Move Semantics (零成本所有权转移)
// =============================================================================

impl<T> Span<T> {
    /// 取走所有权，将自身变成空 View（零成本移动）
    ///
    /// 这个方法实现了真正的移动语义，完全避免 Arc 引用计数操作。
    /// 调用后原 Span 变成空 View（len=0, storage=None）。
    ///
    /// # Safety
    ///
    /// 调用后原 Span 不再持有 Storage 的所有权。
    /// 调用者需确保不再通过原 Span 访问数据。
    ///
    /// # Performance
    ///
    /// - 零原子操作（不触发 Arc clone/drop）
    /// - 仅涉及几个字段的内存复制
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut span1 = Span::alloc::<32>(100)?;
    /// let span2 = unsafe { span1.take_ownership() };
    /// // span1 现在是空 View，span2 持有所有权
    /// assert!(span1.is_empty());
    /// assert!(span2.has_storage());
    /// ```
    #[inline(always)]
    pub unsafe fn take_ownership(&mut self) -> Self {
        // 直接按字节读取 storage（不触发 Arc clone）
        let storage = std::ptr::read(&self.storage);

        // 读取其他字段
        let data = self.data;
        let len = self.len;
        let flags = self.flags;

        // 将原 Span 变成空 View（阻止 Drop 时减少引用计数）
        std::ptr::write(&mut self.storage, None);
        self.data = NonNull::dangling(); // 防止误用原指针
        self.len = 0;
        self.flags = SpanFlags::VIEW;

        Self {
            storage,
            data,
            len,
            flags,
            _marker: PhantomData,
        }
    }

    /// 安全版本的所有权转移
    ///
    /// 如果 Span 非空（len > 0 或持有 Storage），转移所有权并返回 Some；
    /// 如果 Span 已经是空的（len == 0 且无 Storage），返回 None。
    ///
    /// # Note
    ///
    /// 此方法对 Shared 和 View 模式都有效：
    /// - Shared 模式：转移 Storage 所有权
    /// - View 模式（非空）：转移 View 本身
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut span1 = Span::alloc::<32>(100)?;
    /// let span2 = span1.take().unwrap();
    /// assert!(span1.is_empty());
    /// assert!(span1.take().is_none()); // 再次 take 返回 None
    /// ```
    #[inline]
    pub fn take(&mut self) -> Option<Self> {
        // 已经是空 Span 时返回 None
        if self.len == 0 && self.storage.is_none() {
            return None;
        }
        Some(unsafe { self.take_ownership() })
    }

    /// 使用 std::mem::take 的快捷方式
    ///
    /// 与 `std::mem::take(span)` 等效，但更明确语义
    #[inline(always)]
    pub fn take_replace(&mut self) -> Self {
        std::mem::take(self)
    }

    /// 批量移动：将 Vec<Span> 中的所有元素移动到另一个 Vec
    ///
    /// 这是最高效的批量移动方式，直接 append 并清空源 Vec。
    ///
    /// # Arguments
    ///
    /// * `src` - 源 Vec，移动后变为空
    /// * `dst` - 目标 Vec，接收所有元素
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut src = Span::<f32>::alloc_slices::<32>(&[100, 200, 300])?;
    /// let mut dst = Vec::new();
    /// Span::drain_all(&mut src, &mut dst);
    /// assert!(src.is_empty());
    /// assert_eq!(dst.len(), 3);
    /// ```
    #[inline]
    pub fn drain_all(src: &mut Vec<Self>, dst: &mut Vec<Self>) {
        dst.append(src);
    }

    /// 批量移动：从 Vec<Span> 中按索引顺序移动到另一个 Vec
    ///
    /// 适用于需要交错插入（如空行/非空行交替）的场景。
    /// 源 Vec 中被移动的元素变成空 View。
    ///
    /// # Arguments
    ///
    /// * `src` - 源 Vec
    /// * `indices` - 要移动的索引列表（必须有效）
    /// * `dst` - 目标 Vec
    ///
    /// # Safety
    ///
    /// 调用者需确保 indices 中的所有索引都在 src 的有效范围内
    #[inline]
    pub unsafe fn drain_by_indices(src: &mut [Self], indices: &[usize], dst: &mut Vec<Self>) {
        dst.reserve(indices.len());
        for &idx in indices {
            debug_assert!(idx < src.len());
            dst.push(src.get_unchecked_mut(idx).take_ownership());
        }
    }

    /// 创建一个消费迭代器，按顺序移动 Vec 中的所有元素
    ///
    /// 这与 `Vec::into_iter()` 类似，但可以用于需要自定义处理的场景。
    #[inline]
    pub fn drain_iter(src: Vec<Self>) -> impl Iterator<Item = Self> {
        src.into_iter()
    }

    /// 按索引从 Vec 中取出单个元素（移动语义）
    ///
    /// 等效于 `std::mem::take(&mut vec[idx])`
    ///
    /// # Safety
    ///
    /// 调用者需确保 idx < src.len()
    #[inline(always)]
    pub unsafe fn take_at(src: &mut [Self], idx: usize) -> Self {
        debug_assert!(idx < src.len());
        src.get_unchecked_mut(idx).take_ownership()
    }

    /// 按索引从 Vec 中取出单个元素（带边界检查）
    #[inline]
    pub fn take_at_checked(src: &mut [Self], idx: usize) -> Option<Self> {
        if idx >= src.len() {
            return None;
        }
        Some(unsafe { Self::take_at(src, idx) })
    }
}

// =============================================================================
// Index
// =============================================================================

impl<T> std::ops::Index<usize> for Span<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        assert!(
            index < self.len,
            "index {} out of bounds (len: {})",
            index,
            self.len
        );
        unsafe { &*self.data.as_ptr().add(index) }
    }
}

impl<T> std::ops::IndexMut<usize> for Span<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(
            index < self.len,
            "index {} out of bounds (len: {})",
            index,
            self.len
        );
        assert!(self.flags.is_mutable(), "Span is not mutable");
        unsafe { &mut *self.data.as_ptr().add(index) }
    }
}

// =============================================================================
// AsRef / AsMut
// =============================================================================

impl<T> AsRef<[T]> for Span<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for Span<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}

// =============================================================================
// IntoIterator
// =============================================================================

impl<'a, T> IntoIterator for &'a Span<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Span<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// =============================================================================
// Macros
// =============================================================================

/// 便捷宏：分配单个 Span（默认对齐）
#[macro_export]
macro_rules! span {
    ($len:expr) => {{
        $crate::span::Span::alloc::<{ $crate::DEFAULT_ALIGN }>($len)
    }};
}

/// 便捷宏：分配多个 Span（静态数量，默认对齐）
#[macro_export]
macro_rules! span_n {
    ($ty:ty; $($len:expr),+ $(,)?) => {{
        $crate::span::Span::<$ty>::alloc_n::<{ $crate::DEFAULT_ALIGN }, { [$($len),+].len() }>([$($len),+])
    }};
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Basic Allocation Tests
    // =========================================================================

    #[test]
    fn test_span_alloc() {
        let span: Span<f32> = Span::alloc::<32>(100).expect("alloc failed");

        assert_eq!(span.len(), 100);
        assert!(!span.is_view());
        assert!(span.is_aligned());
        assert!(span.is_mutable());
        assert!(span.has_storage());
    }

    #[test]
    fn test_span_alloc_zero_fails() {
        let result = Span::<f32>::alloc::<32>(0);
        assert_eq!(result.unwrap_err(), AllocError::ZeroSize);
    }

    // =========================================================================
    // Alloc N Tests
    // =========================================================================

    #[test]
    fn test_span_alloc_n() {
        let [s0, s1, s2] = Span::<f32>::alloc_n::<32, 3>([100, 200, 300]).unwrap();

        assert_eq!(s0.len(), 100);
        assert_eq!(s1.len(), 200);
        assert_eq!(s2.len(), 300);

        // 所有 Span 共享同一个 Storage（引用计数为 3）
        assert_eq!(s0.storage_ref_count(), Some(3));
        assert_eq!(s1.storage_ref_count(), Some(3));
        assert_eq!(s2.storage_ref_count(), Some(3));

        // 所有 Span 都是可变且对齐的
        assert!(s0.is_mutable() && s0.is_aligned());
        assert!(s1.is_mutable() && s1.is_aligned());
        assert!(s2.is_mutable() && s2.is_aligned());
    }

    // =========================================================================
    // Alloc Slices Tests
    // =========================================================================

    #[test]
    fn test_span_alloc_slices() {
        let spans = Span::<f32>::alloc_slices::<32>(&[100, 200, 300]).unwrap();

        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].len(), 100);
        assert_eq!(spans[1].len(), 200);
        assert_eq!(spans[2].len(), 300);

        // 所有 Span 共享同一个 Storage
        assert_eq!(spans[0].storage_ref_count(), Some(3));
    }

    #[test]
    fn test_span_alloc_slices_empty_fails() {
        let result = Span::<f32>::alloc_slices::<32>(&[]);
        assert_eq!(result.unwrap_err(), AllocError::EmptySizes);
    }

    // =========================================================================
    // Copy From Tests
    // =========================================================================

    #[test]
    fn test_span_copy_from() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let span = Span::copy_from::<32>(&data).unwrap();

        assert_eq!(span.len(), 4);
        assert_eq!(span.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert!(span.is_mutable());
        assert!(span.is_aligned());
    }

    #[test]
    fn test_span_copy_from_empty_fails() {
        let data: [f32; 0] = [];
        let result = Span::copy_from::<32>(&data);
        assert_eq!(result.unwrap_err(), AllocError::ZeroSize);
    }

    // =========================================================================
    // Copy From N Tests
    // =========================================================================

    #[test]
    fn test_span_copy_from_n() {
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0, 5.0];
        let [s0, s1] = Span::copy_from_n::<32, 2>([&a, &b]).unwrap();

        assert_eq!(s0.len(), 2);
        assert_eq!(s1.len(), 3);
        assert_eq!(s0.as_slice(), &[1.0, 2.0]);
        assert_eq!(s1.as_slice(), &[3.0, 4.0, 5.0]);

        // 两个 Span 共享同一个 Storage
        assert_eq!(s0.storage_ref_count(), Some(2));
    }

    // =========================================================================
    // Copy From Slices Tests
    // =========================================================================

    #[test]
    fn test_span_copy_from_slices() {
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0, 5.0];
        let spans = Span::copy_from_slices::<32>(&[&a, &b]).unwrap();

        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].as_slice(), &[1.0, 2.0]);
        assert_eq!(spans[1].as_slice(), &[3.0, 4.0, 5.0]);

        // 两个 Span 共享同一个 Storage
        assert_eq!(spans[0].storage_ref_count(), Some(2));
    }

    // =========================================================================
    // From Slice Tests
    // =========================================================================

    #[test]
    fn test_span_from_slice() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let span = Span::from_slice(&data);

        assert_eq!(span.len(), 4);
        assert!(span.is_view());
        assert!(!span.is_mutable());
        assert_eq!(span[0], 1.0);
        assert_eq!(span[3], 4.0);
    }

    #[test]
    fn test_span_from_slice_mut() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut span = Span::from_slice_mut(&mut data);

        assert!(span.is_view());
        assert!(span.is_mutable());

        span[0] = 10.0;
        assert_eq!(data[0], 10.0);
    }

    // =========================================================================
    // Deep Copy Tests
    // =========================================================================

    #[test]
    fn test_deep_copy() {
        let mut span1: Span<i32> = Span::alloc::<32>(5).unwrap();
        for i in 0..5 {
            span1[i] = (i * 10) as i32;
        }

        let span2 = span1.deep_copy::<32>().unwrap();

        // 验证是独立的内存
        assert_ne!(span1.as_ptr(), span2.as_ptr());

        // 验证数据已复制
        assert_eq!(span2.as_slice(), &[0, 10, 20, 30, 40]);

        // 验证新 Span 是独立的 Storage
        assert_eq!(span1.storage_ref_count(), Some(1));
        assert_eq!(span2.storage_ref_count(), Some(1));

        // 验证新 Span 是可变的
        assert!(span2.is_mutable());
        assert!(span2.is_aligned());
        assert!(!span2.is_view());
    }

    #[test]
    fn test_deep_copy_from_view() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let view = Span::from_slice(&data);

        let copied = view.deep_copy::<32>().unwrap();

        assert_ne!(view.as_ptr(), copied.as_ptr());
        assert_eq!(copied.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert!(copied.has_storage());
        assert!(copied.is_mutable());
    }

    // =========================================================================
    // Subspan Tests
    // =========================================================================

    #[test]
    fn test_subspan() {
        let span: Span<i32> = Span::alloc::<32>(10).unwrap();

        let sub = span.subspan(2, 5).unwrap();
        assert_eq!(sub.len(), 5);

        // 边界检查
        assert!(span.subspan(0, 10).is_some());
        assert!(span.subspan(0, 11).is_none());
        assert!(span.subspan(10, 0).is_some());
        assert!(span.subspan(10, 1).is_none());
    }

    // =========================================================================
    // Fill Tests
    // =========================================================================

    #[test]
    fn test_fill() {
        let mut span: Span<i32> = Span::alloc::<32>(5).unwrap();
        span.fill(42);
        assert_eq!(span.as_slice(), &[42, 42, 42, 42, 42]);
    }

    // =========================================================================
    // Index Tests
    // =========================================================================

    #[test]
    fn test_span_index() {
        let mut span: Span<i32> = Span::alloc::<32>(10).unwrap();

        for i in 0..10 {
            span[i] = i as i32;
        }

        for i in 0..10 {
            assert_eq!(span[i], i as i32);
        }
    }

    #[test]
    fn test_span_get() {
        let span: Span<u32> = Span::alloc::<32>(5).unwrap();

        assert!(span.get(0).is_some());
        assert!(span.get(4).is_some());
        assert!(span.get(5).is_none());
        assert!(span.get(100).is_none());
    }

    #[test]
    #[should_panic(expected = "index 10 out of bounds")]
    fn test_span_index_out_of_bounds() {
        let span: Span<i32> = Span::alloc::<32>(5).unwrap();
        let _ = span[10];
    }

    // =========================================================================
    // Slice Access Tests
    // =========================================================================

    #[test]
    fn test_span_as_slice() {
        let mut span: Span<u8> = Span::alloc::<32>(4).unwrap();

        {
            let slice = span.as_slice_mut();
            slice[0] = 1;
            slice[1] = 2;
            slice[2] = 3;
            slice[3] = 4;
        }

        let slice = span.as_slice();
        assert_eq!(slice, &[1, 2, 3, 4]);
    }

    // =========================================================================
    // Clone Tests
    // =========================================================================

    #[test]
    fn test_span_clone_shared() {
        let span1: Span<f64> = Span::alloc::<32>(100).unwrap();
        let span2 = span1.clone();

        assert!(span1.has_storage());
        assert!(span2.has_storage());

        assert_eq!(span1.storage_ref_count(), Some(2));
        assert_eq!(span2.storage_ref_count(), Some(2));
    }

    // =========================================================================
    // Iteration Tests
    // =========================================================================

    #[test]
    fn test_iteration() {
        let mut span: Span<i32> = Span::alloc::<32>(5).unwrap();
        for (i, x) in span.iter_mut().enumerate() {
            *x = i as i32;
        }

        let sum: i32 = span.iter().sum();
        assert_eq!(sum, 10);
    }

    // =========================================================================
    // Misc Tests
    // =========================================================================

    #[test]
    fn test_span_size_bytes() {
        let span: Span<f32> = Span::alloc::<32>(100).unwrap();
        assert_eq!(span.size_bytes(), 100 * std::mem::size_of::<f32>());

        let span: Span<f64> = Span::alloc::<32>(50).unwrap();
        assert_eq!(span.size_bytes(), 50 * std::mem::size_of::<f64>());
    }

    #[test]
    fn test_flags_operations() {
        let flags = SpanFlags::EMPTY;
        assert!(!flags.is_view());
        assert!(!flags.is_aligned());
        assert!(!flags.is_mutable());

        let flags = flags.with(SpanFlags::VIEW);
        assert!(flags.is_view());

        let flags = flags.with(SpanFlags::ALIGNED).with(SpanFlags::MUTABLE);
        assert!(flags.is_aligned());
        assert!(flags.is_mutable());

        let flags = flags.without(SpanFlags::MUTABLE);
        assert!(!flags.is_mutable());
        assert!(flags.is_aligned());
    }

    #[test]
    #[should_panic(expected = "Span is not mutable")]
    fn test_span_immutable_write_panics() {
        let data = vec![1, 2, 3];
        let mut span = Span::from_slice(&data);
        span[0] = 10;
    }

    // =========================================================================
    // Macro Tests
    // =========================================================================

    #[test]
    fn test_span_macro() {
        let span: Span<f32> = span!(100).unwrap();
        assert_eq!(span.len(), 100);
        assert!(span.is_aligned());
    }

    #[test]
    fn test_span_n_macro() {
        let [s0, s1, s2] = span_n!(f32; 100, 200, 300).unwrap();

        assert_eq!(s0.len(), 100);
        assert_eq!(s1.len(), 200);
        assert_eq!(s2.len(), 300);
        assert_eq!(s0.storage_ref_count(), Some(3));
    }

    // =========================================================================
    // Move Semantics Tests
    // =========================================================================

    #[test]
    fn test_take_ownership() {
        let mut span1: Span<i32> = Span::alloc::<32>(100).unwrap();
        span1[0] = 42;

        // 初始引用计数为 1
        assert_eq!(span1.storage_ref_count(), Some(1));

        // 移动所有权
        let span2 = unsafe { span1.take_ownership() };

        // span1 变成空 View
        assert!(span1.is_empty());
        assert!(span1.is_view());
        assert!(!span1.has_storage());

        // span2 持有所有权
        assert_eq!(span2.len(), 100);
        assert!(span2.has_storage());
        assert_eq!(span2.storage_ref_count(), Some(1)); // 引用计数仍为 1，没有增加！
        assert_eq!(span2[0], 42);
    }

    #[test]
    fn test_take() {
        let mut span1: Span<i32> = Span::alloc::<32>(50).unwrap();

        // 有 Storage 时返回 Some
        let span2 = span1.take().unwrap();
        assert!(span1.is_empty());
        assert_eq!(span2.len(), 50);

        // 再次 take 返回 None
        assert!(span1.take().is_none());
    }

    #[test]
    fn test_take_replace() {
        let mut span1: Span<i32> = Span::alloc::<32>(50).unwrap();

        let span2 = span1.take_replace();

        // span1 被替换为默认值
        assert!(span1.is_empty());
        assert!(span1.is_view());

        // span2 持有原来的值
        assert_eq!(span2.len(), 50);
        assert!(span2.has_storage());
    }

    #[test]
    fn test_take_at() {
        let mut spans = Span::<i32>::alloc_slices::<32>(&[10, 20, 30]).unwrap();
        assert_eq!(spans[0].storage_ref_count(), Some(3));

        // 取出第一个
        let s0 = unsafe { Span::take_at(&mut spans, 0) };
        assert_eq!(s0.len(), 10);
        assert!(spans[0].is_empty()); // 原位置变空

        // 取出第二个
        let s1 = unsafe { Span::take_at(&mut spans, 1) };
        assert_eq!(s1.len(), 20);

        // 引用计数应该仍为 3（因为没有 clone/drop，只是移动）
        assert_eq!(s0.storage_ref_count(), Some(3));
        assert_eq!(s1.storage_ref_count(), Some(3));
    }

    #[test]
    fn test_drain_all() {
        let mut src = Span::<i32>::alloc_slices::<32>(&[10, 20, 30]).unwrap();
        let mut dst = Vec::new();

        Span::drain_all(&mut src, &mut dst);

        assert!(src.is_empty());
        assert_eq!(dst.len(), 3);
        assert_eq!(dst[0].len(), 10);
        assert_eq!(dst[1].len(), 20);
        assert_eq!(dst[2].len(), 30);
    }

    #[test]
    fn test_default() {
        let span: Span<i32> = Span::default();
        assert!(span.is_empty());
        assert!(span.is_view());
        assert!(!span.has_storage());
    }
}
