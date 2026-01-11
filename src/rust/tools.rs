//! 编译器优化工具
//!
//! 提供用于诱导编译器优化的宏和辅助函数。
//!
//! # 主要工具
//!
//! - [`assume!`] / [`assume_unchecked!`] - 断言条件为真，帮助编译器消除检查
//! - [`likely`] / [`unlikely`] - 分支预测提示
//! - [`prefetch_read`] / [`prefetch_write`] - 预取提示
//! - [`copy_nonoverlapping_aligned`] - 对齐内存复制
//!
//! # 性能说明
//!
//! 所有函数都使用 `#[inline(always)]` 确保零开销抽象。
//! 在 release 模式下，大部分函数会被完全内联并消除。

#![allow(unused_unsafe)]

use std::mem::MaybeUninit;

// =============================================================================
// Compiler Intrinsics Re-exports
// =============================================================================

/// 防止编译器优化掉某个值（用于基准测试）
#[inline(always)]
pub fn black_box<T>(dummy: T) -> T {
    std::hint::black_box(dummy)
}

/// 自旋等待提示（告知 CPU 这是自旋循环）
#[inline(always)]
pub fn spin_loop() {
    std::hint::spin_loop();
}

// =============================================================================
// Assume Macros
// =============================================================================

/// 向编译器断言条件为真，用于消除冗余检查
///
/// 在 debug 模式下，如果条件为假会 panic。
/// 在 release 模式下，如果条件为假则是未定义行为。
///
/// # Safety
///
/// 调用者必须保证条件始终为真，否则在 release 模式下是 UB。
///
/// # Example
///
/// ```ignore
/// assume!(index < len);
/// // 编译器现在可以假设 index < len
/// ```
#[macro_export]
macro_rules! assume {
    ($cond:expr) => {{
        if !$cond {
            #[cfg(debug_assertions)]
            {
                panic!("assume! condition failed: {}", stringify!($cond));
            }
            #[cfg(not(debug_assertions))]
            {
                // SAFETY: 调用者保证条件为真
                unsafe { ::std::hint::unreachable_unchecked() }
            }
        }
    }};
}

/// 向编译器断言条件为真（unsafe 版本，无 debug 检查开销）
///
/// 即使在 debug 模式下也不会检查条件，适用于性能关键路径。
/// 使用 debug_assert 在 debug 模式下进行非阻塞检查。
///
/// # Safety
///
/// 调用者必须保证条件始终为真，否则是未定义行为。
///
/// # Example
///
/// ```ignore
/// unsafe {
///     assume_unchecked!(ptr.is_aligned());
/// }
/// ```
#[macro_export]
macro_rules! assume_unchecked {
    ($cond:expr) => {{
        debug_assert!($cond, "assume_unchecked! violated: {}", stringify!($cond));
        if !$cond {
            // SAFETY: 调用者保证条件为真
            ::std::hint::unreachable_unchecked()
        }
    }};
}

/// 断言索引在边界内并返回索引（用于消除边界检查）
///
/// # Safety
///
/// 调用者必须保证 `index < len`
///
/// # Example
///
/// ```ignore
/// let idx = assume_in_bounds!(i, arr.len());
/// arr.get_unchecked(idx)
/// ```
#[macro_export]
macro_rules! assume_in_bounds {
    ($index:expr, $len:expr) => {{
        let idx = $index;
        let len = $len;
        debug_assert!(idx < len, "index {} out of bounds {}", idx, len);
        $crate::assume_unchecked!(idx < len);
        idx
    }};
}

/// 断言两个范围不重叠（用于优化内存操作）
///
/// # Safety
///
/// 调用者必须保证范围确实不重叠
#[macro_export]
macro_rules! assume_nonoverlapping {
    ($src:expr, $dst:expr, $len:expr) => {{
        let src = $src as usize;
        let dst = $dst as usize;
        let len = $len;
        debug_assert!(
            src + len <= dst || dst + len <= src,
            "ranges overlap: src={:#x}, dst={:#x}, len={}",
            src,
            dst,
            len
        );
        $crate::assume_unchecked!(src + len <= dst || dst + len <= src);
    }};
}

// =============================================================================
// Branch Prediction Hints
// =============================================================================

/// 标记永不执行的分支（帮助编译器优化代码布局）
#[cold]
#[inline(never)]
pub fn cold_path() {
    // 永不实际调用，仅用于标记冷路径
}

/// 标记不太可能执行的分支
///
/// 通过调用 `#[cold]` 函数来暗示编译器该分支不太可能执行。
///
/// # Example
///
/// ```ignore
/// if unlikely(error_occurred) {
///     return Err(e);
/// }
/// ```
#[inline(always)]
#[must_use]
pub fn unlikely(cond: bool) -> bool {
    if cond {
        cold_path();
    }
    cond
}

/// 标记很可能执行的分支
///
/// # Example
///
/// ```ignore
/// if likely(index < len) {
///     process(data[index]);
/// }
/// ```
#[inline(always)]
#[must_use]
pub fn likely(cond: bool) -> bool {
    if !cond {
        cold_path();
    }
    cond
}

// =============================================================================
// Loop Optimization Hints
// =============================================================================

/// 断言循环迭代次数的上界（帮助循环展开）
///
/// # Example
///
/// ```ignore
/// for i in 0..n {
///     assume_loop_bound::<1024>(i);
///     // 编译器可以假设 i < 1024
/// }
/// ```
#[inline(always)]
pub const fn assume_loop_bound<const MAX: usize>(index: usize) {
    // NOTE: 不能在 const fn 中使用 debug_assert! 的格式化版本
    if index >= MAX {
        // SAFETY: 调用者保证 index < MAX
        unsafe { std::hint::unreachable_unchecked() }
    }
}

/// 返回适合循环展开的步长
///
/// # Example
///
/// ```ignore
/// for chunk in data.chunks_exact(unroll_hint::<8>()) {
///     // 编译器会展开这个循环
/// }
/// ```
#[inline(always)]
pub const fn unroll_hint<const N: usize>() -> usize {
    const { assert!(N > 0, "unroll factor must be > 0") };
    const { assert!(N.is_power_of_two(), "unroll factor should be power of 2") };
    N
}

/// 断言长度是某个值的倍数（帮助向量化）
///
/// # Safety
///
/// 调用者必须保证 len 确实是 DIVISOR 的倍数
#[inline(always)]
pub unsafe fn assume_divisible<const DIVISOR: usize>(len: usize) {
    const { assert!(DIVISOR > 0, "DIVISOR must be > 0") };
    debug_assert!(
        len % DIVISOR == 0,
        "len {} is not divisible by {}",
        len,
        DIVISOR
    );
    assume_unchecked!(len % DIVISOR == 0);
}

// =============================================================================
// Alignment Hints
// =============================================================================

/// 断言指针对齐
///
/// # Safety
///
/// 调用者必须保证指针满足指定的对齐要求
#[inline(always)]
pub unsafe fn assume_aligned<T, const ALIGN: usize>(ptr: *const T) -> *const T {
    const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };
    debug_assert!(
        (ptr as usize) % ALIGN == 0,
        "pointer {:p} is not {}-aligned",
        ptr,
        ALIGN
    );
    assume_unchecked!((ptr as usize) % ALIGN == 0);
    ptr
}

/// 断言可变指针对齐
///
/// # Safety
///
/// 调用者必须保证指针满足指定的对齐要求
#[inline(always)]
pub unsafe fn assume_aligned_mut<T, const ALIGN: usize>(ptr: *mut T) -> *mut T {
    const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };
    debug_assert!(
        (ptr as usize) % ALIGN == 0,
        "pointer {:p} is not {}-aligned",
        ptr,
        ALIGN
    );
    assume_unchecked!((ptr as usize) % ALIGN == 0);
    ptr
}

/// 检查指针是否对齐
#[inline(always)]
#[must_use]
pub const fn is_aligned_to<const ALIGN: usize>(ptr: usize) -> bool {
    const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };
    ptr & (ALIGN - 1) == 0
}

// =============================================================================
// Memory Operations
// =============================================================================

/// 对齐的非重叠内存复制（目标对齐）
///
/// 比 std::ptr::copy_nonoverlapping 更激进的优化版本。
/// 通过 assume 告知编译器目标对齐信息。
///
/// # Safety
///
/// - src 和 dst 必须有效
/// - dst 必须满足 ALIGN 对齐
/// - 区域不能重叠
/// - count 个元素必须可读/可写
#[inline(always)]
pub unsafe fn copy_nonoverlapping_aligned<T: Copy, const ALIGN: usize>(
    src: *const T,
    dst: *mut T,
    count: usize,
) {
    const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };

    // 只假设目标对齐（源可能不对齐，这是常见情况）
    debug_assert!((dst as usize) % ALIGN == 0, "dst is not aligned");
    assume_unchecked!((dst as usize) % ALIGN == 0);

    std::ptr::copy_nonoverlapping(src, dst, count);
}

/// 对齐的非重叠内存复制（源和目标都对齐）
///
/// 最激进的优化版本，源和目标都假设对齐。
///
/// # Safety
///
/// - src 和 dst 都必须满足 ALIGN 对齐
/// - 区域不能重叠
/// - count 个元素必须可读/可写
#[inline(always)]
pub unsafe fn copy_nonoverlapping_both_aligned<T: Copy, const ALIGN: usize>(
    src: *const T,
    dst: *mut T,
    count: usize,
) {
    const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };

    debug_assert!((src as usize) % ALIGN == 0, "src is not aligned");
    debug_assert!((dst as usize) % ALIGN == 0, "dst is not aligned");
    assume_unchecked!((src as usize) % ALIGN == 0);
    assume_unchecked!((dst as usize) % ALIGN == 0);

    std::ptr::copy_nonoverlapping(src, dst, count);
}

/// 对齐的内存填充
///
/// # Safety
///
/// - dst 必须有效且对齐
/// - count 个元素必须可写
#[inline(always)]
pub unsafe fn write_bytes_aligned<T, const ALIGN: usize>(dst: *mut T, val: u8, count: usize) {
    const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };

    assume_unchecked!((dst as usize) % ALIGN == 0);
    std::ptr::write_bytes(dst, val, count);
}

/// 零初始化内存
///
/// # Safety
///
/// - dst 必须有效
/// - count 个元素必须可写
#[inline(always)]
pub unsafe fn zero_memory<T>(dst: *mut T, count: usize) {
    std::ptr::write_bytes(dst, 0, count);
}

// =============================================================================
// Prefetch Hints
// =============================================================================

/// 预取数据到缓存（用于读取）
///
/// 使用平台特定的预取指令（x86/x86_64）。
/// 在其他平台上是空操作。
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: 预取指令不会导致段错误，即使地址无效
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }
    #[cfg(target_arch = "x86")]
    {
        unsafe {
            std::arch::x86::_mm_prefetch(ptr as *const i8, std::arch::x86::_MM_HINT_T0);
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        let _ = ptr;
    }
}

/// 预取数据到缓存（用于写入）
///
/// 使用 T1 级缓存提示，适合即将写入的数据。
#[inline(always)]
pub fn prefetch_write<T>(ptr: *mut T) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T1);
        }
    }
    #[cfg(target_arch = "x86")]
    {
        unsafe {
            std::arch::x86::_mm_prefetch(ptr as *const i8, std::arch::x86::_MM_HINT_T1);
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        let _ = ptr;
    }
}

/// 预取数据到 L3 缓存（非临时访问）
///
/// 用于只访问一次的流式数据。
#[inline(always)]
pub fn prefetch_nta<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_NTA);
        }
    }
    #[cfg(target_arch = "x86")]
    {
        unsafe {
            std::arch::x86::_mm_prefetch(ptr as *const i8, std::arch::x86::_MM_HINT_NTA);
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        let _ = ptr;
    }
}

// =============================================================================
// Unchecked Operations
// =============================================================================

/// 无边界检查的切片索引
///
/// # Safety
///
/// 调用者必须保证 index < slice.len()
#[inline(always)]
pub unsafe fn slice_get_unchecked<T>(slice: &[T], index: usize) -> &T {
    debug_assert!(index < slice.len(), "index out of bounds");
    assume_unchecked!(index < slice.len());
    slice.get_unchecked(index)
}

/// 无边界检查的可变切片索引
///
/// # Safety
///
/// 调用者必须保证 index < slice.len()
#[inline(always)]
pub unsafe fn slice_get_unchecked_mut<T>(slice: &mut [T], index: usize) -> &mut T {
    debug_assert!(index < slice.len(), "index out of bounds");
    let len = slice.len();
    assume_unchecked!(index < len);
    slice.get_unchecked_mut(index)
}

/// 无边界检查的切片范围
///
/// # Safety
///
/// 调用者必须保证 start <= end <= slice.len()
#[inline(always)]
pub unsafe fn slice_range_unchecked<T>(slice: &[T], start: usize, end: usize) -> &[T] {
    debug_assert!(start <= end && end <= slice.len());
    assume_unchecked!(start <= end);
    assume_unchecked!(end <= slice.len());
    slice.get_unchecked(start..end)
}

/// 无边界检查的可变切片范围
///
/// # Safety
///
/// 调用者必须保证 start <= end <= slice.len()
#[inline(always)]
pub unsafe fn slice_range_unchecked_mut<T>(slice: &mut [T], start: usize, end: usize) -> &mut [T] {
    debug_assert!(start <= end && end <= slice.len());
    let len = slice.len();
    assume_unchecked!(start <= end);
    assume_unchecked!(end <= len);
    slice.get_unchecked_mut(start..end)
}

// =============================================================================
// Const Utilities
// =============================================================================

/// 编译期断言对齐兼容性
#[inline(always)]
pub const fn const_assert_align<T, const ALIGN: usize>() {
    const {
        assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two");
        assert!(
            ALIGN >= std::mem::align_of::<T>(),
            "ALIGN must be >= type alignment"
        );
    };
}

/// 编译期计算向上对齐
#[inline(always)]
#[must_use]
pub const fn const_align_up<const ALIGN: usize>(value: usize) -> usize {
    const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };
    (value + ALIGN - 1) & !(ALIGN - 1)
}

/// 编译期计算向下对齐
#[inline(always)]
#[must_use]
pub const fn const_align_down<const ALIGN: usize>(value: usize) -> usize {
    const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };
    value & !(ALIGN - 1)
}

/// 编译期计算是否对齐
#[inline(always)]
#[must_use]
pub const fn const_is_aligned<const ALIGN: usize>(value: usize) -> bool {
    const { assert!(ALIGN.is_power_of_two(), "ALIGN must be power of two") };
    value & (ALIGN - 1) == 0
}

/// 编译期选择（类似三元运算符）
#[inline(always)]
#[must_use]
pub const fn const_select<T: Copy>(cond: bool, if_true: T, if_false: T) -> T {
    if cond {
        if_true
    } else {
        if_false
    }
}

/// 编译期最大值
#[inline(always)]
#[must_use]
pub const fn const_max(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

/// 编译期最小值
#[inline(always)]
#[must_use]
pub const fn const_min(a: usize, b: usize) -> usize {
    if a < b {
        a
    } else {
        b
    }
}

// =============================================================================
// Array Utilities
// =============================================================================

/// 使用函数初始化数组（编译期友好）
///
/// # Example
///
/// ```ignore
/// let arr: [u32; 8] = array_init(|i| i as u32 * 2);
/// ```
#[inline(always)]
pub fn array_init<T, F: FnMut(usize) -> T, const N: usize>(mut f: F) -> [T; N] {
    // 使用 MaybeUninit 避免默认值要求
    let mut array: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };

    for i in 0..N {
        array[i] = MaybeUninit::new(f(i));
    }

    // SAFETY: 所有元素都已初始化
    unsafe { std::mem::transmute_copy::<[MaybeUninit<T>; N], [T; N]>(&array) }
}

/// 复制数组的一部分
///
/// # Safety
///
/// 调用者必须保证 start + LEN <= src.len()
#[inline(always)]
pub unsafe fn array_copy_from_slice<T: Copy, const LEN: usize>(
    src: &[T],
    start: usize,
) -> [T; LEN] {
    debug_assert!(start + LEN <= src.len());
    assume_unchecked!(start + LEN <= src.len());

    let ptr = src.as_ptr().add(start);
    std::ptr::read(ptr as *const [T; LEN])
}

// =============================================================================
// Iterator Utilities
// =============================================================================

/// 创建一个带 assume 的范围迭代器
///
/// 告知编译器迭代次数的上界。
#[inline(always)]
pub fn bounded_range<const MAX: usize>(start: usize, end: usize) -> impl Iterator<Item = usize> {
    debug_assert!(end <= MAX);
    unsafe {
        assume_unchecked!(end <= MAX);
    }
    (start..end).map(move |i| {
        unsafe {
            assume_unchecked!(i < MAX);
        }
        i
    })
}

// =============================================================================
// SIMD Helpers (诱导向量化)
// =============================================================================

/// 处理对齐的数组块（诱导 SIMD 优化）
///
/// # Safety
///
/// - slice 长度必须是 CHUNK_SIZE 的倍数
/// - CHUNK_SIZE 应该是 SIMD 向量宽度
#[inline(always)]
pub unsafe fn process_aligned_chunks<T, F, const CHUNK_SIZE: usize>(slice: &mut [T], mut f: F)
where
    F: FnMut(&mut [T; CHUNK_SIZE]),
{
    const { assert!(CHUNK_SIZE > 0 && CHUNK_SIZE.is_power_of_two()) };

    debug_assert!(slice.len() % CHUNK_SIZE == 0);
    assume_unchecked!(slice.len() % CHUNK_SIZE == 0);

    let ptr = slice.as_mut_ptr();
    let chunks = slice.len() / CHUNK_SIZE;

    for i in 0..chunks {
        let chunk_ptr = ptr.add(i * CHUNK_SIZE) as *mut [T; CHUNK_SIZE];
        f(&mut *chunk_ptr);
    }
}

/// 使用 4 路展开处理数组（诱导循环展开）
#[inline(always)]
pub fn process_unrolled_4<T, F>(slice: &mut [T], mut f: F)
where
    F: FnMut(&mut T),
{
    let len = slice.len();
    let ptr = slice.as_mut_ptr();

    // 4 路展开主循环
    let main_len = len & !3;
    let mut i = 0;
    while i < main_len {
        unsafe {
            f(&mut *ptr.add(i));
            f(&mut *ptr.add(i + 1));
            f(&mut *ptr.add(i + 2));
            f(&mut *ptr.add(i + 3));
        }
        i += 4;
    }

    // 处理余数
    while i < len {
        unsafe {
            f(&mut *ptr.add(i));
        }
        i += 1;
    }
}

/// 使用 8 路展开处理数组（诱导循环展开）
#[inline(always)]
pub fn process_unrolled_8<T, F>(slice: &mut [T], mut f: F)
where
    F: FnMut(&mut T),
{
    let len = slice.len();
    let ptr = slice.as_mut_ptr();

    // 8 路展开主循环
    let main_len = len & !7;
    let mut i = 0;
    while i < main_len {
        unsafe {
            f(&mut *ptr.add(i));
            f(&mut *ptr.add(i + 1));
            f(&mut *ptr.add(i + 2));
            f(&mut *ptr.add(i + 3));
            f(&mut *ptr.add(i + 4));
            f(&mut *ptr.add(i + 5));
            f(&mut *ptr.add(i + 6));
            f(&mut *ptr.add(i + 7));
        }
        i += 8;
    }

    // 处理余数
    while i < len {
        unsafe {
            f(&mut *ptr.add(i));
        }
        i += 1;
    }
}

// =============================================================================
// Parallel Helpers - SendPtr
// =============================================================================

/// 用于并行写入的原始指针包装器
///
/// 允许在并行迭代中安全地传递可变指针。
///
/// # Safety
///
/// 调用者需保证并行写入不会产生数据竞争。
/// 典型用法是每个并行任务访问不重叠的内存区域。
///
/// # Example
///
/// ```ignore
/// let ptr = SendPtr::new(data.as_mut_ptr());
/// (0..n).into_par_iter().for_each(|i| {
///     // SAFETY: 每个 i 访问不同位置
///     unsafe { *ptr.add(i) = compute(i); }
/// });
/// ```
#[derive(Clone, Copy)]
pub struct SendPtr<T>(*mut T);

unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Send> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// 创建新的 SendPtr
    #[inline(always)]
    pub fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    /// 获取原始指针
    #[inline(always)]
    pub fn ptr(self) -> *mut T {
        self.0
    }

    /// 指针偏移（unsafe）
    ///
    /// # Safety
    ///
    /// 调用者需确保 offset 在有效范围内
    #[inline(always)]
    pub unsafe fn add(&self, offset: usize) -> *mut T {
        self.0.add(offset)
    }

    /// 指针偏移后解引用（unsafe）
    ///
    /// # Safety
    ///
    /// 调用者需确保 offset 在有效范围内，且不存在数据竞争
    #[inline(always)]
    pub unsafe fn get(&self, offset: usize) -> &T {
        &*self.0.add(offset)
    }

    /// 指针偏移后可变解引用（unsafe）
    ///
    /// # Safety
    ///
    /// 调用者需确保 offset 在有效范围内，且不存在数据竞争
    #[inline(always)]
    pub unsafe fn get_mut(&self, offset: usize) -> &mut T {
        &mut *self.0.add(offset)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_likely_unlikely() {
        assert!(likely(true));
        assert!(!likely(false));
        assert!(unlikely(true));
        assert!(!unlikely(false));
    }

    #[test]
    fn test_assume_loop_bound() {
        for i in 0..100 {
            assume_loop_bound::<128>(i);
        }
    }

    #[test]
    fn test_unroll_hint() {
        assert_eq!(unroll_hint::<8>(), 8);
        assert_eq!(unroll_hint::<16>(), 16);
    }

    #[test]
    fn test_assume_aligned() {
        let aligned: [u8; 64] = [0; 64];
        let ptr = aligned.as_ptr();
        unsafe {
            let _ = assume_aligned::<u8, 1>(ptr);
        }
    }

    #[test]
    fn test_assume_macro() {
        assume!(1 + 1 == 2);
        assume!(true);
    }

    #[test]
    #[should_panic(expected = "assume! condition failed")]
    #[cfg(debug_assertions)]
    fn test_assume_fails_in_debug() {
        assume!(false);
    }

    #[test]
    fn test_const_align() {
        assert_eq!(const_align_up::<32>(0), 0);
        assert_eq!(const_align_up::<32>(1), 32);
        assert_eq!(const_align_up::<32>(32), 32);
        assert_eq!(const_align_up::<32>(33), 64);

        assert_eq!(const_align_down::<32>(0), 0);
        assert_eq!(const_align_down::<32>(31), 0);
        assert_eq!(const_align_down::<32>(32), 32);
        assert_eq!(const_align_down::<32>(33), 32);

        assert!(const_is_aligned::<32>(0));
        assert!(const_is_aligned::<32>(32));
        assert!(!const_is_aligned::<32>(1));
    }

    #[test]
    fn test_array_init() {
        let arr: [u32; 8] = array_init(|i| i as u32 * 2);
        assert_eq!(arr, [0, 2, 4, 6, 8, 10, 12, 14]);
    }

    #[test]
    fn test_slice_unchecked() {
        let arr = [1, 2, 3, 4, 5];
        unsafe {
            assert_eq!(*slice_get_unchecked(&arr, 2), 3);
            assert_eq!(slice_range_unchecked(&arr, 1, 4), &[2, 3, 4]);
        }
    }

    #[test]
    fn test_process_unrolled() {
        let mut arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        process_unrolled_4(&mut arr, |x| *x *= 2);
        assert_eq!(arr, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
    }

    #[test]
    fn test_bounded_range() {
        let v: Vec<usize> = bounded_range::<100>(5, 10).collect();
        assert_eq!(v, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_const_min_max() {
        assert_eq!(const_min(3, 5), 3);
        assert_eq!(const_max(3, 5), 5);
    }

    #[test]
    fn test_black_box() {
        let x = black_box(42);
        assert_eq!(x, 42);
    }
}
