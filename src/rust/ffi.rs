//! FFI - 外部函数接口
//!
//! 提供 C ABI 兼容的句柄接口，用于从 Python/C 等语言调用。
//!
//! # 设计理念
//!
//! - **透明句柄**：允许直接获取内部字段指针
//! - **零开销**：短函数体，由调用方决定是否内联
//! - **类型安全**：通过具体化的句柄类型区分不同泛型实例
//!
//! # 句柄类型
//!
//! 所有句柄都是指向 Rust 结构的裸指针：
//! - `SpanF32Handle`, `SpanF64Handle`, `SpanI32Handle`, `SpanI64Handle`
//! - `CSRF32Handle`, `CSRF64Handle`, `CSCF32Handle`, `CSCF64Handle`
//!
//! # 内存管理
//!
//! - 句柄的生命周期由 Rust 管理
//! - 调用者不应手动释放句柄指向的内存
//! - 对于需要外部管理的场景，使用 `*_clone` 和 `*_free` 函数
//!
//! # 注意
//!
//! `#[no_mangle]` 函数上的 `#[inline]` 属性会被忽略，因为这些函数必须在动态库中有固定地址。
//! 如需内联，调用方应使用 LTO 或直接调用 Rust API。

use std::ptr;

use crate::convert::{AllocStrategy, ConvertError, DenseLayout};
use crate::span::{Span, SpanFlags};
use crate::sparse::{CSCf32, CSCf64, CSRf32, CSRf64, SparseIndex};
use crate::stack::StackError;
use crate::storage::AllocError;

// =============================================================================
// ABI 版本
// =============================================================================

/// ABI 版本号，用于检测二进制兼容性
///
/// 当 FFI 接口发生不兼容变更时应递增此值
#[no_mangle]
pub static BIOSPARSE_ABI_VERSION: u32 = 1;

// =============================================================================
// 句柄类型定义
// =============================================================================

/// Span<f32> 的不透明句柄
pub type SpanF32Handle = *const Span<f32>;
/// Span<f32> 的可变句柄
pub type SpanF32HandleMut = *mut Span<f32>;

/// Span<f64> 的不透明句柄
pub type SpanF64Handle = *const Span<f64>;
/// Span<f64> 的可变句柄
pub type SpanF64HandleMut = *mut Span<f64>;

/// Span<i32> 的不透明句柄
pub type SpanI32Handle = *const Span<i32>;
/// Span<i32> 的可变句柄
pub type SpanI32HandleMut = *mut Span<i32>;

/// Span<i64> 的不透明句柄
pub type SpanI64Handle = *const Span<i64>;
/// Span<i64> 的可变句柄
pub type SpanI64HandleMut = *mut Span<i64>;

/// Span<usize> 的不透明句柄
pub type SpanUsizeHandle = *const Span<usize>;
/// Span<usize> 的可变句柄
pub type SpanUsizeHandleMut = *mut Span<usize>;

/// CSR<f32, i64> 的不透明句柄
pub type CSRF32Handle = *const CSRf32;
/// CSR<f32, i64> 的可变句柄
pub type CSRF32HandleMut = *mut CSRf32;

/// CSR<f64, i64> 的不透明句柄
pub type CSRF64Handle = *const CSRf64;
/// CSR<f64, i64> 的可变句柄
pub type CSRF64HandleMut = *mut CSRf64;

/// CSC<f32, i64> 的不透明句柄
pub type CSCF32Handle = *const CSCf32;
/// CSC<f32, i64> 的可变句柄
pub type CSCF32HandleMut = *mut CSCf32;

/// CSC<f64, i64> 的不透明句柄
pub type CSCF64Handle = *const CSCf64;
/// CSC<f64, i64> 的可变句柄
pub type CSCF64HandleMut = *mut CSCf64;

// =============================================================================
// FFI 结果结构体
// =============================================================================

/// FFI 返回的 Span 信息（用于返回多个值）
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SpanInfo {
    /// 数据指针
    pub data: *const u8,
    /// 元素个数
    pub len: usize,
    /// 单个元素的字节大小
    pub element_size: usize,
    /// 标志位
    pub flags: usize,
}

impl SpanInfo {
    /// 空的 SpanInfo
    pub const NULL: Self = Self {
        data: ptr::null(),
        len: 0,
        element_size: 0,
        flags: 0,
    };
}

/// FFI 返回的矩阵形状
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub rows: i64,
    pub cols: i64,
}

// =============================================================================
// Span FFI - 宏生成
// =============================================================================

/// 为指定类型生成 Span FFI 函数
macro_rules! impl_span_ffi {
    ($suffix:ident, $ty:ty, $handle:ty, $handle_mut:ty) => {
        paste::paste! {
            // -----------------------------------------------------------------
            // 基本访问器
            // -----------------------------------------------------------------

            /// 获取数据指针
            #[no_mangle]
            pub unsafe extern "C" fn [<span_ $suffix _data>](handle: $handle) -> *const $ty {
                if handle.is_null() {
                    return ptr::null();
                }
                (*handle).as_ptr()
            }

            /// 获取可变数据指针
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针
            /// - 如果 Span 标记为不可变，返回 null
            #[no_mangle]
            pub unsafe extern "C" fn [<span_ $suffix _data_mut>](handle: $handle_mut) -> *mut $ty {
                if handle.is_null() {
                    return ptr::null_mut();
                }
                let span = &*handle;
                if !span.flags().is_mutable() {
                    return ptr::null_mut();
                }
                (*handle).as_mut_ptr()
            }

            /// 获取元素个数
            #[no_mangle]
            pub unsafe extern "C" fn [<span_ $suffix _len>](handle: $handle) -> usize {
                if handle.is_null() {
                    return 0;
                }
                (*handle).len()
            }

            /// 获取标志位
            #[no_mangle]
            pub unsafe extern "C" fn [<span_ $suffix _flags>](handle: $handle) -> usize {
                if handle.is_null() {
                    return 0;
                }
                (*handle).flags().bits()
            }

            /// 检查是否为 View 模式
            #[no_mangle]
            pub unsafe extern "C" fn [<span_ $suffix _is_view>](handle: $handle) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).flags().is_view()
            }

            /// 检查是否对齐
            #[no_mangle]
            pub unsafe extern "C" fn [<span_ $suffix _is_aligned>](handle: $handle) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).flags().is_aligned()
            }

            /// 检查是否可变
            #[no_mangle]
            pub unsafe extern "C" fn [<span_ $suffix _is_mutable>](handle: $handle) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).flags().is_mutable()
            }

            /// 获取完整信息（一次调用返回所有字段）
            #[no_mangle]
            pub unsafe extern "C" fn [<span_ $suffix _info>](handle: $handle) -> SpanInfo {
                if handle.is_null() {
                    return SpanInfo::NULL;
                }
                let span = &*handle;
                SpanInfo {
                    data: span.as_ptr() as *const u8,
                    len: span.len(),
                    element_size: std::mem::size_of::<$ty>(),
                    flags: span.flags().bits(),
                }
            }

            /// 获取字节大小
            #[no_mangle]
            pub unsafe extern "C" fn [<span_ $suffix _byte_size>](handle: $handle) -> usize {
                if handle.is_null() {
                    return 0;
                }
                (*handle).len() * std::mem::size_of::<$ty>()
            }

            // -----------------------------------------------------------------
            // 克隆和释放（用于外部管理生命周期）
            // -----------------------------------------------------------------

            /// 克隆 Span（返回堆分配的新句柄）
            ///
            /// 调用者负责使用 `span_*_free` 释放返回的句柄。
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针，或者为 null（返回 null）
            #[no_mangle]
            #[must_use]
            pub unsafe extern "C" fn [<span_ $suffix _clone>](handle: $handle) -> $handle_mut {
                if handle.is_null() {
                    return ptr::null_mut();
                }
                let cloned = (*handle).clone();
                Box::into_raw(Box::new(cloned))
            }

            /// 释放克隆的 Span
            ///
            /// # Safety
            ///
            /// - `handle` 必须是由 `span_*_clone` 返回的有效指针，或者为 null
            /// - 每个句柄只能释放一次
            #[no_mangle]
            pub unsafe extern "C" fn [<span_ $suffix _free>](handle: $handle_mut) {
                if !handle.is_null() {
                    drop(Box::from_raw(handle));
                }
            }
        }
    };
}

// 生成各类型的 Span FFI
impl_span_ffi!(f32, f32, SpanF32Handle, SpanF32HandleMut);
impl_span_ffi!(f64, f64, SpanF64Handle, SpanF64HandleMut);
impl_span_ffi!(i32, i32, SpanI32Handle, SpanI32HandleMut);
impl_span_ffi!(i64, i64, SpanI64Handle, SpanI64HandleMut);
impl_span_ffi!(usize, usize, SpanUsizeHandle, SpanUsizeHandleMut);

// =============================================================================
// CSR FFI - 宏生成
// =============================================================================

/// 为指定类型生成 CSR FFI 函数
macro_rules! impl_csr_ffi {
    ($suffix:ident, $val_ty:ty, $idx_ty:ty, $csr_ty:ty, $handle:ty, $handle_mut:ty, $span_val_handle:ty, $span_idx_handle:ty) => {
        paste::paste! {
            // -----------------------------------------------------------------
            // 维度查询
            // -----------------------------------------------------------------

            /// 获取行数
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _rows>](handle: $handle) -> $idx_ty {
                if handle.is_null() {
                    return <$idx_ty>::ZERO;
                }
                (*handle).rows
            }

            /// 获取列数
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _cols>](handle: $handle) -> $idx_ty {
                if handle.is_null() {
                    return <$idx_ty>::ZERO;
                }
                (*handle).cols
            }

            /// 获取形状
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _shape>](handle: $handle) -> Shape {
                if handle.is_null() {
                    return Shape { rows: 0, cols: 0 };
                }
                let csr = &*handle;
                Shape {
                    rows: csr.rows as i64,
                    cols: csr.cols as i64,
                }
            }

            /// 获取非零元素个数
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _nnz>](handle: $handle) -> $idx_ty {
                if handle.is_null() {
                    return <$idx_ty>::ZERO;
                }
                (*handle).nnz()
            }

            // -----------------------------------------------------------------
            // 内部数组访问（Vec<Span> 的指针）
            // -----------------------------------------------------------------

            /// 获取 values Vec 的数据指针（指向 Span<V> 数组）
            ///
            /// 返回指向 `Vec<Span<V>>` 内部数组的指针
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _values_vec_ptr>](handle: $handle) -> *const Span<$val_ty> {
                if handle.is_null() {
                    return ptr::null();
                }
                (*handle).values.as_ptr()
            }

            /// 获取 indices Vec 的数据指针（指向 Span<I> 数组）
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _indices_vec_ptr>](handle: $handle) -> *const Span<$idx_ty> {
                if handle.is_null() {
                    return ptr::null();
                }
                (*handle).indices.as_ptr()
            }

            /// 获取 values Vec 的长度（即行数）
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _values_vec_len>](handle: $handle) -> usize {
                if handle.is_null() {
                    return 0;
                }
                (*handle).values.len()
            }

            // -----------------------------------------------------------------
            // 单行访问
            // -----------------------------------------------------------------

            /// 获取指定行的 values Span 句柄
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _row_values>](handle: $handle, row: usize) -> $span_val_handle {
                if handle.is_null() {
                    return ptr::null();
                }
                let csr = &*handle;
                if row >= csr.values.len() {
                    return ptr::null();
                }
                csr.values.as_ptr().add(row)
            }

            /// 获取指定行的 indices Span 句柄
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _row_indices>](handle: $handle, row: usize) -> $span_idx_handle {
                if handle.is_null() {
                    return ptr::null();
                }
                let csr = &*handle;
                if row >= csr.indices.len() {
                    return ptr::null();
                }
                csr.indices.as_ptr().add(row)
            }

            /// 获取指定行的 values 数据指针（直接跳过 Span 层）
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _row_values_ptr>](handle: $handle, row: usize) -> *const $val_ty {
                if handle.is_null() {
                    return ptr::null();
                }
                let csr = &*handle;
                match csr.values.get(row) {
                    Some(span) => span.as_ptr(),
                    None => ptr::null(),
                }
            }

            /// 获取指定行的 indices 数据指针（直接跳过 Span 层）
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _row_indices_ptr>](handle: $handle, row: usize) -> *const $idx_ty {
                if handle.is_null() {
                    return ptr::null();
                }
                let csr = &*handle;
                match csr.indices.get(row) {
                    Some(span) => span.as_ptr(),
                    None => ptr::null(),
                }
            }

            /// 获取指定行的元素个数
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _row_len>](handle: $handle, row: usize) -> usize {
                if handle.is_null() {
                    return 0;
                }
                let csr = &*handle;
                match csr.values.get(row) {
                    Some(span) => span.len(),
                    None => 0,
                }
            }

            // -----------------------------------------------------------------
            // 无边界检查版本（用于热路径）
            // -----------------------------------------------------------------

            /// 获取指定行的 values 数据指针（无边界检查）
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针
            /// - `row` 必须小于矩阵行数
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _row_values_ptr_unchecked>](handle: $handle, row: usize) -> *const $val_ty {
                let csr = &*handle;
                csr.values.get_unchecked(row).as_ptr()
            }

            /// 获取指定行的 indices 数据指针（无边界检查）
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针
            /// - `row` 必须小于矩阵行数
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _row_indices_ptr_unchecked>](handle: $handle, row: usize) -> *const $idx_ty {
                let csr = &*handle;
                csr.indices.get_unchecked(row).as_ptr()
            }

            /// 获取指定行的元素个数（无边界检查）
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针
            /// - `row` 必须小于矩阵行数
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _row_len_unchecked>](handle: $handle, row: usize) -> usize {
                let csr = &*handle;
                csr.values.get_unchecked(row).len()
            }

            // -----------------------------------------------------------------
            // 验证函数
            // -----------------------------------------------------------------

            /// 检查矩阵结构是否有效
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _is_valid>](handle: $handle) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).is_valid()
            }

            /// 检查每行的索引是否已排序
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _is_sorted>](handle: $handle) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).is_sorted()
            }

            /// 完整验证（结构 + 排序 + 边界）
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _validate>](handle: $handle) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).validate()
            }

            /// 检查所有索引是否在有效范围内
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _indices_in_bounds>](handle: $handle) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).indices_in_bounds()
            }

            // -----------------------------------------------------------------
            // 查询函数
            // -----------------------------------------------------------------

            /// 检查是否为空矩阵
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _is_empty>](handle: $handle) -> bool {
                if handle.is_null() {
                    return true;
                }
                (*handle).is_empty()
            }

            /// 检查是否无非零元素
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _is_zero>](handle: $handle) -> bool {
                if handle.is_null() {
                    return true;
                }
                (*handle).is_zero()
            }

            /// 计算稀疏度（零元素比例）
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _sparsity>](handle: $handle) -> f64 {
                if handle.is_null() {
                    return 1.0;
                }
                (*handle).sparsity()
            }

            /// 计算密度（非零元素比例）
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _density>](handle: $handle) -> f64 {
                if handle.is_null() {
                    return 0.0;
                }
                (*handle).density()
            }

            // -----------------------------------------------------------------
            // 缓存管理
            // -----------------------------------------------------------------

            /// 使 NNZ 缓存失效
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _invalidate_nnz>](handle: $handle) {
                if !handle.is_null() {
                    (*handle).invalidate_nnz();
                }
            }

            /// 设置 NNZ 缓存
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _set_nnz>](handle: $handle, nnz: $idx_ty) {
                if !handle.is_null() {
                    (*handle).set_nnz(nnz);
                }
            }

            /// 检查 NNZ 缓存是否有效
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _has_nnz_cache>](handle: $handle) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).has_nnz_cache()
            }

            // -----------------------------------------------------------------
            // 克隆和释放
            // -----------------------------------------------------------------

            /// 克隆 CSR（返回堆分配的新句柄）
            ///
            /// 调用者负责使用 `csr_*_free` 释放返回的句柄。
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针，或者为 null（返回 null）
            #[no_mangle]
            #[must_use]
            pub unsafe extern "C" fn [<csr_ $suffix _clone>](handle: $handle) -> $handle_mut {
                if handle.is_null() {
                    return ptr::null_mut();
                }
                let cloned = (*handle).clone();
                Box::into_raw(Box::new(cloned))
            }

            /// 释放克隆的 CSR
            ///
            /// # Safety
            ///
            /// - `handle` 必须是由 `csr_*_clone` 返回的有效指针，或者为 null
            /// - 每个句柄只能释放一次
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _free>](handle: $handle_mut) {
                if !handle.is_null() {
                    drop(Box::from_raw(handle));
                }
            }

            // -----------------------------------------------------------------
            // 排序
            // -----------------------------------------------------------------

            /// 确保每行的 indices 是有序的（in-place）
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _ensure_sorted>](handle: $handle_mut) {
                if !handle.is_null() {
                    (*handle).ensure_sorted();
                }
            }

            /// 确保排序并返回是否进行了排序
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _ensure_sorted_checked>](handle: $handle_mut) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).ensure_sorted_checked()
            }
        }
    };
}

// 生成 CSR FFI
impl_csr_ffi!(
    f32,
    f32,
    i64,
    CSRf32,
    CSRF32Handle,
    CSRF32HandleMut,
    SpanF32Handle,
    SpanI64Handle
);
impl_csr_ffi!(
    f64,
    f64,
    i64,
    CSRf64,
    CSRF64Handle,
    CSRF64HandleMut,
    SpanF64Handle,
    SpanI64Handle
);

// =============================================================================
// CSC FFI - 宏生成
// =============================================================================

/// 为指定类型生成 CSC FFI 函数
macro_rules! impl_csc_ffi {
    ($suffix:ident, $val_ty:ty, $idx_ty:ty, $csc_ty:ty, $handle:ty, $handle_mut:ty, $span_val_handle:ty, $span_idx_handle:ty) => {
        paste::paste! {
            // -----------------------------------------------------------------
            // 维度查询
            // -----------------------------------------------------------------

            /// 获取行数
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _rows>](handle: $handle) -> $idx_ty {
                if handle.is_null() {
                    return <$idx_ty>::ZERO;
                }
                (*handle).rows
            }

            /// 获取列数
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _cols>](handle: $handle) -> $idx_ty {
                if handle.is_null() {
                    return <$idx_ty>::ZERO;
                }
                (*handle).cols
            }

            /// 获取形状
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _shape>](handle: $handle) -> Shape {
                if handle.is_null() {
                    return Shape { rows: 0, cols: 0 };
                }
                let csc = &*handle;
                Shape {
                    rows: csc.rows as i64,
                    cols: csc.cols as i64,
                }
            }

            /// 获取非零元素个数
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _nnz>](handle: $handle) -> $idx_ty {
                if handle.is_null() {
                    return <$idx_ty>::ZERO;
                }
                (*handle).nnz()
            }

            // -----------------------------------------------------------------
            // 内部数组访问
            // -----------------------------------------------------------------

            /// 获取 values Vec 的数据指针
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _values_vec_ptr>](handle: $handle) -> *const Span<$val_ty> {
                if handle.is_null() {
                    return ptr::null();
                }
                (*handle).values.as_ptr()
            }

            /// 获取 indices Vec 的数据指针
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _indices_vec_ptr>](handle: $handle) -> *const Span<$idx_ty> {
                if handle.is_null() {
                    return ptr::null();
                }
                (*handle).indices.as_ptr()
            }

            /// 获取 values Vec 的长度（即列数）
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _values_vec_len>](handle: $handle) -> usize {
                if handle.is_null() {
                    return 0;
                }
                (*handle).values.len()
            }

            // -----------------------------------------------------------------
            // 单列访问
            // -----------------------------------------------------------------

            /// 获取指定列的 values Span 句柄
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _col_values>](handle: $handle, col: usize) -> $span_val_handle {
                if handle.is_null() {
                    return ptr::null();
                }
                let csc = &*handle;
                if col >= csc.values.len() {
                    return ptr::null();
                }
                csc.values.as_ptr().add(col)
            }

            /// 获取指定列的 indices Span 句柄
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _col_indices>](handle: $handle, col: usize) -> $span_idx_handle {
                if handle.is_null() {
                    return ptr::null();
                }
                let csc = &*handle;
                if col >= csc.indices.len() {
                    return ptr::null();
                }
                csc.indices.as_ptr().add(col)
            }

            /// 获取指定列的 values 数据指针
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _col_values_ptr>](handle: $handle, col: usize) -> *const $val_ty {
                if handle.is_null() {
                    return ptr::null();
                }
                let csc = &*handle;
                match csc.values.get(col) {
                    Some(span) => span.as_ptr(),
                    None => ptr::null(),
                }
            }

            /// 获取指定列的 indices 数据指针
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _col_indices_ptr>](handle: $handle, col: usize) -> *const $idx_ty {
                if handle.is_null() {
                    return ptr::null();
                }
                let csc = &*handle;
                match csc.indices.get(col) {
                    Some(span) => span.as_ptr(),
                    None => ptr::null(),
                }
            }

            /// 获取指定列的元素个数
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _col_len>](handle: $handle, col: usize) -> usize {
                if handle.is_null() {
                    return 0;
                }
                let csc = &*handle;
                match csc.values.get(col) {
                    Some(span) => span.len(),
                    None => 0,
                }
            }

            // -----------------------------------------------------------------
            // 无边界检查版本
            // -----------------------------------------------------------------

            /// 获取指定列的 values 数据指针（无边界检查）
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针
            /// - `col` 必须小于矩阵列数
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _col_values_ptr_unchecked>](handle: $handle, col: usize) -> *const $val_ty {
                let csc = &*handle;
                csc.values.get_unchecked(col).as_ptr()
            }

            /// 获取指定列的 indices 数据指针（无边界检查）
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针
            /// - `col` 必须小于矩阵列数
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _col_indices_ptr_unchecked>](handle: $handle, col: usize) -> *const $idx_ty {
                let csc = &*handle;
                csc.indices.get_unchecked(col).as_ptr()
            }

            /// 获取指定列的元素个数（无边界检查）
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针
            /// - `col` 必须小于矩阵列数
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _col_len_unchecked>](handle: $handle, col: usize) -> usize {
                let csc = &*handle;
                csc.values.get_unchecked(col).len()
            }

            // -----------------------------------------------------------------
            // 验证函数
            // -----------------------------------------------------------------

            /// 检查矩阵结构是否有效
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _is_valid>](handle: $handle) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).is_valid()
            }

            /// 检查每列的索引是否已排序
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _is_sorted>](handle: $handle) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).is_sorted()
            }

            /// 完整验证（结构 + 排序 + 边界）
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _validate>](handle: $handle) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).validate()
            }

            /// 检查所有索引是否在有效范围内
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _indices_in_bounds>](handle: $handle) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).indices_in_bounds()
            }

            // -----------------------------------------------------------------
            // 查询函数
            // -----------------------------------------------------------------

            /// 检查是否为空矩阵
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _is_empty>](handle: $handle) -> bool {
                if handle.is_null() {
                    return true;
                }
                (*handle).is_empty()
            }

            /// 检查是否无非零元素
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _is_zero>](handle: $handle) -> bool {
                if handle.is_null() {
                    return true;
                }
                (*handle).is_zero()
            }

            /// 计算稀疏度（零元素比例）
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _sparsity>](handle: $handle) -> f64 {
                if handle.is_null() {
                    return 1.0;
                }
                (*handle).sparsity()
            }

            /// 计算密度（非零元素比例）
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _density>](handle: $handle) -> f64 {
                if handle.is_null() {
                    return 0.0;
                }
                (*handle).density()
            }

            // -----------------------------------------------------------------
            // 缓存管理
            // -----------------------------------------------------------------

            /// 使 NNZ 缓存失效
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _invalidate_nnz>](handle: $handle) {
                if !handle.is_null() {
                    (*handle).invalidate_nnz();
                }
            }

            /// 设置 NNZ 缓存
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _set_nnz>](handle: $handle, nnz: $idx_ty) {
                if !handle.is_null() {
                    (*handle).set_nnz(nnz);
                }
            }

            /// 检查 NNZ 缓存是否有效
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _has_nnz_cache>](handle: $handle) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).has_nnz_cache()
            }

            // -----------------------------------------------------------------
            // 克隆和释放
            // -----------------------------------------------------------------

            /// 克隆 CSC（返回堆分配的新句柄）
            ///
            /// 调用者负责使用 `csc_*_free` 释放返回的句柄。
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针，或者为 null（返回 null）
            #[no_mangle]
            #[must_use]
            pub unsafe extern "C" fn [<csc_ $suffix _clone>](handle: $handle) -> $handle_mut {
                if handle.is_null() {
                    return ptr::null_mut();
                }
                let cloned = (*handle).clone();
                Box::into_raw(Box::new(cloned))
            }

            /// 释放克隆的 CSC
            ///
            /// # Safety
            ///
            /// - `handle` 必须是由 `csc_*_clone` 返回的有效指针，或者为 null
            /// - 每个句柄只能释放一次
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _free>](handle: $handle_mut) {
                if !handle.is_null() {
                    drop(Box::from_raw(handle));
                }
            }

            // -----------------------------------------------------------------
            // 排序
            // -----------------------------------------------------------------

            /// 确保每列的 indices 是有序的（in-place）
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _ensure_sorted>](handle: $handle_mut) {
                if !handle.is_null() {
                    (*handle).ensure_sorted();
                }
            }

            /// 确保排序并返回是否进行了排序
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _ensure_sorted_checked>](handle: $handle_mut) -> bool {
                if handle.is_null() {
                    return false;
                }
                (*handle).ensure_sorted_checked()
            }
        }
    };
}

// 生成 CSC FFI
impl_csc_ffi!(
    f32,
    f32,
    i64,
    CSCf32,
    CSCF32Handle,
    CSCF32HandleMut,
    SpanF32Handle,
    SpanI64Handle
);
impl_csc_ffi!(
    f64,
    f64,
    i64,
    CSCf64,
    CSCF64Handle,
    CSCF64HandleMut,
    SpanF64Handle,
    SpanI64Handle
);

// =============================================================================
// 常量导出
// =============================================================================

/// Span 结构体的大小（字节）
#[no_mangle]
pub static SPAN_SIZE: usize = std::mem::size_of::<Span<f32>>();

/// SpanFlags::VIEW 位
#[no_mangle]
pub static SPAN_FLAG_VIEW: usize = SpanFlags::VIEW.bits();

/// SpanFlags::ALIGNED 位
#[no_mangle]
pub static SPAN_FLAG_ALIGNED: usize = SpanFlags::ALIGNED.bits();

/// SpanFlags::MUTABLE 位
#[no_mangle]
pub static SPAN_FLAG_MUTABLE: usize = SpanFlags::MUTABLE.bits();

// =============================================================================
// 类型大小导出（用于外部验证）
// =============================================================================

/// 获取 Span<f32> 的大小
#[no_mangle]
pub extern "C" fn span_f32_size() -> usize {
    std::mem::size_of::<Span<f32>>()
}

/// 获取 Span<f64> 的大小
#[no_mangle]
pub extern "C" fn span_f64_size() -> usize {
    std::mem::size_of::<Span<f64>>()
}

/// 获取 Span<i32> 的大小
#[no_mangle]
pub extern "C" fn span_i32_size() -> usize {
    std::mem::size_of::<Span<i32>>()
}

/// 获取 Span<i64> 的大小
#[no_mangle]
pub extern "C" fn span_i64_size() -> usize {
    std::mem::size_of::<Span<i64>>()
}

/// 获取 Span<usize> 的大小
#[no_mangle]
pub extern "C" fn span_usize_size() -> usize {
    std::mem::size_of::<Span<usize>>()
}

/// 获取 CSR<f32, i64> 的大小
#[no_mangle]
pub extern "C" fn csr_f32_size() -> usize {
    std::mem::size_of::<CSRf32>()
}

/// 获取 CSR<f64, i64> 的大小
#[no_mangle]
pub extern "C" fn csr_f64_size() -> usize {
    std::mem::size_of::<CSRf64>()
}

/// 获取 CSC<f32, i64> 的大小
#[no_mangle]
pub extern "C" fn csc_f32_size() -> usize {
    std::mem::size_of::<CSCf32>()
}

/// 获取 CSC<f64, i64> 的大小
#[no_mangle]
pub extern "C" fn csc_f64_size() -> usize {
    std::mem::size_of::<CSCf64>()
}

// =============================================================================
// FFI 错误码
// =============================================================================

/// FFI 操作结果码
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiResult {
    /// 成功
    Ok = 0,
    /// 空指针
    NullPointer = -1,
    /// 维度不匹配
    DimensionMismatch = -2,
    /// 长度不匹配
    LengthMismatch = -3,
    /// indptr 格式错误
    InvalidIndptr = -4,
    /// 索引越界
    IndexOutOfBounds = -5,
    /// Buffer 太小
    BufferTooSmall = -6,
    /// 内存分配失败
    AllocError = -7,
    /// 空输入
    EmptyInput = -8,
    /// 未知错误
    Unknown = -99,
}

impl From<ConvertError> for FfiResult {
    fn from(e: ConvertError) -> Self {
        match e {
            ConvertError::DimensionMismatch => FfiResult::DimensionMismatch,
            ConvertError::LengthMismatch => FfiResult::LengthMismatch,
            ConvertError::InvalidIndptr => FfiResult::InvalidIndptr,
            ConvertError::IndexOutOfBounds => FfiResult::IndexOutOfBounds,
            ConvertError::BufferTooSmall => FfiResult::BufferTooSmall,
            ConvertError::Alloc(_) => FfiResult::AllocError,
        }
    }
}

impl From<StackError> for FfiResult {
    fn from(e: StackError) -> Self {
        match e {
            StackError::EmptyInput => FfiResult::EmptyInput,
            StackError::DimensionMismatch { .. } => FfiResult::DimensionMismatch,
            StackError::Alloc(_) => FfiResult::AllocError,
        }
    }
}

impl From<AllocError> for FfiResult {
    fn from(_: AllocError) -> Self {
        FfiResult::AllocError
    }
}

// =============================================================================
// 转换函数 - scipy CSR/CSC 格式
// =============================================================================

/// 默认对齐（32 字节，AVX2）
const DEFAULT_ALIGN: usize = 32;

/// 从 scipy CSR 格式创建 CSR（View 模式，零拷贝）
///
/// # Safety
///
/// 调用者需确保指针指向有效内存，且生命周期覆盖返回的 CSR
#[no_mangle]
pub unsafe extern "C" fn csr_f32_from_scipy_view(
    rows: i64,
    cols: i64,
    data: *const f32,
    indices: *const i64,
    indptr: *const i64,
) -> CSRF32HandleMut {
    if data.is_null() || indices.is_null() || indptr.is_null() {
        return ptr::null_mut();
    }
    let csr = crate::convert::csr_from_scipy_csr_view(rows, cols, data, indices, indptr);
    Box::into_raw(Box::new(csr))
}

/// 从 scipy CSR 格式创建 CSR（View 模式，零拷贝）- f64 版本
#[no_mangle]
pub unsafe extern "C" fn csr_f64_from_scipy_view(
    rows: i64,
    cols: i64,
    data: *const f64,
    indices: *const i64,
    indptr: *const i64,
) -> CSRF64HandleMut {
    if data.is_null() || indices.is_null() || indptr.is_null() {
        return ptr::null_mut();
    }
    let csr = crate::convert::csr_from_scipy_csr_view(rows, cols, data, indices, indptr);
    Box::into_raw(Box::new(csr))
}

/// 从 scipy CSR 格式创建 CSR（Copy 模式）
///
/// 返回值通过 out_handle 返回，函数返回错误码
#[no_mangle]
pub unsafe extern "C" fn csr_f32_from_scipy_copy(
    rows: i64,
    cols: i64,
    data: *const f32,
    data_len: usize,
    indices: *const i64,
    indices_len: usize,
    indptr: *const i64,
    indptr_len: usize,
    out_handle: *mut CSRF32HandleMut,
) -> FfiResult {
    if data.is_null() || indices.is_null() || indptr.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let data_slice = std::slice::from_raw_parts(data, data_len);
    let indices_slice = std::slice::from_raw_parts(indices, indices_len);
    let indptr_slice = std::slice::from_raw_parts(indptr, indptr_len);

    match crate::convert::csr_from_scipy_csr_copy::<f32, i64, DEFAULT_ALIGN>(
        rows,
        cols,
        data_slice,
        indices_slice,
        indptr_slice,
        AllocStrategy::Auto,
    ) {
        Ok(csr) => {
            *out_handle = Box::into_raw(Box::new(csr));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// 从 scipy CSR 格式创建 CSR（Copy 模式）- f64 版本
#[no_mangle]
pub unsafe extern "C" fn csr_f64_from_scipy_copy(
    rows: i64,
    cols: i64,
    data: *const f64,
    data_len: usize,
    indices: *const i64,
    indices_len: usize,
    indptr: *const i64,
    indptr_len: usize,
    out_handle: *mut CSRF64HandleMut,
) -> FfiResult {
    if data.is_null() || indices.is_null() || indptr.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let data_slice = std::slice::from_raw_parts(data, data_len);
    let indices_slice = std::slice::from_raw_parts(indices, indices_len);
    let indptr_slice = std::slice::from_raw_parts(indptr, indptr_len);

    match crate::convert::csr_from_scipy_csr_copy::<f64, i64, DEFAULT_ALIGN>(
        rows,
        cols,
        data_slice,
        indices_slice,
        indptr_slice,
        AllocStrategy::Auto,
    ) {
        Ok(csr) => {
            *out_handle = Box::into_raw(Box::new(csr));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// 从 scipy CSC 格式创建 CSC（View 模式，零拷贝）
#[no_mangle]
pub unsafe extern "C" fn csc_f32_from_scipy_view(
    rows: i64,
    cols: i64,
    data: *const f32,
    indices: *const i64,
    indptr: *const i64,
) -> CSCF32HandleMut {
    if data.is_null() || indices.is_null() || indptr.is_null() {
        return ptr::null_mut();
    }
    let csc = crate::convert::csc_from_scipy_csc_view(rows, cols, data, indices, indptr);
    Box::into_raw(Box::new(csc))
}

/// 从 scipy CSC 格式创建 CSC（View 模式，零拷贝）- f64 版本
#[no_mangle]
pub unsafe extern "C" fn csc_f64_from_scipy_view(
    rows: i64,
    cols: i64,
    data: *const f64,
    indices: *const i64,
    indptr: *const i64,
) -> CSCF64HandleMut {
    if data.is_null() || indices.is_null() || indptr.is_null() {
        return ptr::null_mut();
    }
    let csc = crate::convert::csc_from_scipy_csc_view(rows, cols, data, indices, indptr);
    Box::into_raw(Box::new(csc))
}

/// 从 scipy CSC 格式创建 CSC（Copy 模式）
#[no_mangle]
pub unsafe extern "C" fn csc_f32_from_scipy_copy(
    rows: i64,
    cols: i64,
    data: *const f32,
    data_len: usize,
    indices: *const i64,
    indices_len: usize,
    indptr: *const i64,
    indptr_len: usize,
    out_handle: *mut CSCF32HandleMut,
) -> FfiResult {
    if data.is_null() || indices.is_null() || indptr.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let data_slice = std::slice::from_raw_parts(data, data_len);
    let indices_slice = std::slice::from_raw_parts(indices, indices_len);
    let indptr_slice = std::slice::from_raw_parts(indptr, indptr_len);

    match crate::convert::csc_from_scipy_csc_copy::<f32, i64, DEFAULT_ALIGN>(
        rows,
        cols,
        data_slice,
        indices_slice,
        indptr_slice,
        AllocStrategy::Auto,
    ) {
        Ok(csc) => {
            *out_handle = Box::into_raw(Box::new(csc));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// 从 scipy CSC 格式创建 CSC（Copy 模式）- f64 版本
#[no_mangle]
pub unsafe extern "C" fn csc_f64_from_scipy_copy(
    rows: i64,
    cols: i64,
    data: *const f64,
    data_len: usize,
    indices: *const i64,
    indices_len: usize,
    indptr: *const i64,
    indptr_len: usize,
    out_handle: *mut CSCF64HandleMut,
) -> FfiResult {
    if data.is_null() || indices.is_null() || indptr.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let data_slice = std::slice::from_raw_parts(data, data_len);
    let indices_slice = std::slice::from_raw_parts(indices, indices_len);
    let indptr_slice = std::slice::from_raw_parts(indptr, indptr_len);

    match crate::convert::csc_from_scipy_csc_copy::<f64, i64, DEFAULT_ALIGN>(
        rows,
        cols,
        data_slice,
        indices_slice,
        indptr_slice,
        AllocStrategy::Auto,
    ) {
        Ok(csc) => {
            *out_handle = Box::into_raw(Box::new(csc));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

// =============================================================================
// 转换函数 - COO 格式
// =============================================================================

/// 从 scipy COO 格式创建 CSR
#[no_mangle]
pub unsafe extern "C" fn csr_f32_from_coo(
    rows: i64,
    cols: i64,
    row_indices: *const i64,
    col_indices: *const i64,
    data: *const f32,
    nnz: usize,
    out_handle: *mut CSRF32HandleMut,
) -> FfiResult {
    if row_indices.is_null() || col_indices.is_null() || data.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let row_indices_slice = std::slice::from_raw_parts(row_indices, nnz);
    let col_indices_slice = std::slice::from_raw_parts(col_indices, nnz);
    let data_slice = std::slice::from_raw_parts(data, nnz);

    match crate::convert::csr_from_scipy_coo_copy::<f32, i64, DEFAULT_ALIGN>(
        rows,
        cols,
        row_indices_slice,
        col_indices_slice,
        data_slice,
        AllocStrategy::Auto,
    ) {
        Ok(csr) => {
            *out_handle = Box::into_raw(Box::new(csr));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// 从 scipy COO 格式创建 CSR - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csr_f64_from_coo(
    rows: i64,
    cols: i64,
    row_indices: *const i64,
    col_indices: *const i64,
    data: *const f64,
    nnz: usize,
    out_handle: *mut CSRF64HandleMut,
) -> FfiResult {
    if row_indices.is_null() || col_indices.is_null() || data.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let row_indices_slice = std::slice::from_raw_parts(row_indices, nnz);
    let col_indices_slice = std::slice::from_raw_parts(col_indices, nnz);
    let data_slice = std::slice::from_raw_parts(data, nnz);

    match crate::convert::csr_from_scipy_coo_copy::<f64, i64, DEFAULT_ALIGN>(
        rows,
        cols,
        row_indices_slice,
        col_indices_slice,
        data_slice,
        AllocStrategy::Auto,
    ) {
        Ok(csr) => {
            *out_handle = Box::into_raw(Box::new(csr));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// 从 scipy COO 格式创建 CSC
#[no_mangle]
pub unsafe extern "C" fn csc_f32_from_coo(
    rows: i64,
    cols: i64,
    row_indices: *const i64,
    col_indices: *const i64,
    data: *const f32,
    nnz: usize,
    out_handle: *mut CSCF32HandleMut,
) -> FfiResult {
    if row_indices.is_null() || col_indices.is_null() || data.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let row_indices_slice = std::slice::from_raw_parts(row_indices, nnz);
    let col_indices_slice = std::slice::from_raw_parts(col_indices, nnz);
    let data_slice = std::slice::from_raw_parts(data, nnz);

    match crate::convert::csc_from_scipy_coo_copy::<f32, i64, DEFAULT_ALIGN>(
        rows,
        cols,
        row_indices_slice,
        col_indices_slice,
        data_slice,
        AllocStrategy::Auto,
    ) {
        Ok(csc) => {
            *out_handle = Box::into_raw(Box::new(csc));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// 从 scipy COO 格式创建 CSC - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csc_f64_from_coo(
    rows: i64,
    cols: i64,
    row_indices: *const i64,
    col_indices: *const i64,
    data: *const f64,
    nnz: usize,
    out_handle: *mut CSCF64HandleMut,
) -> FfiResult {
    if row_indices.is_null() || col_indices.is_null() || data.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let row_indices_slice = std::slice::from_raw_parts(row_indices, nnz);
    let col_indices_slice = std::slice::from_raw_parts(col_indices, nnz);
    let data_slice = std::slice::from_raw_parts(data, nnz);

    match crate::convert::csc_from_scipy_coo_copy::<f64, i64, DEFAULT_ALIGN>(
        rows,
        cols,
        row_indices_slice,
        col_indices_slice,
        data_slice,
        AllocStrategy::Auto,
    ) {
        Ok(csc) => {
            *out_handle = Box::into_raw(Box::new(csc));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

// =============================================================================
// 转换函数 - CSR ↔ CSC 互转
// =============================================================================

/// CSR 转 CSC
#[no_mangle]
pub unsafe extern "C" fn csc_f32_from_csr(
    csr_handle: CSRF32Handle,
    out_handle: *mut CSCF32HandleMut,
) -> FfiResult {
    if csr_handle.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    match crate::convert::csc_from_csr::<f32, i64, DEFAULT_ALIGN>(&*csr_handle, AllocStrategy::Auto)
    {
        Ok(csc) => {
            *out_handle = Box::into_raw(Box::new(csc));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// CSR 转 CSC - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csc_f64_from_csr(
    csr_handle: CSRF64Handle,
    out_handle: *mut CSCF64HandleMut,
) -> FfiResult {
    if csr_handle.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    match crate::convert::csc_from_csr::<f64, i64, DEFAULT_ALIGN>(&*csr_handle, AllocStrategy::Auto)
    {
        Ok(csc) => {
            *out_handle = Box::into_raw(Box::new(csc));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// CSC 转 CSR
#[no_mangle]
pub unsafe extern "C" fn csr_f32_from_csc(
    csc_handle: CSCF32Handle,
    out_handle: *mut CSRF32HandleMut,
) -> FfiResult {
    if csc_handle.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    match crate::convert::csr_from_csc::<f32, i64, DEFAULT_ALIGN>(&*csc_handle, AllocStrategy::Auto)
    {
        Ok(csr) => {
            *out_handle = Box::into_raw(Box::new(csr));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// CSC 转 CSR - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csr_f64_from_csc(
    csc_handle: CSCF64Handle,
    out_handle: *mut CSRF64HandleMut,
) -> FfiResult {
    if csc_handle.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    match crate::convert::csr_from_csc::<f64, i64, DEFAULT_ALIGN>(&*csc_handle, AllocStrategy::Auto)
    {
        Ok(csr) => {
            *out_handle = Box::into_raw(Box::new(csr));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

// =============================================================================
// 转换函数 - Dense 格式
// =============================================================================

/// CSR 转 Dense（行主序）
#[no_mangle]
pub unsafe extern "C" fn csr_f32_to_dense(
    handle: CSRF32Handle,
    out: *mut f32,
    out_len: usize,
    col_major: bool,
) -> FfiResult {
    if handle.is_null() || out.is_null() {
        return FfiResult::NullPointer;
    }

    let out_slice = std::slice::from_raw_parts_mut(out, out_len);
    let layout = if col_major {
        DenseLayout::ColMajor
    } else {
        DenseLayout::RowMajor
    };

    match crate::convert::csr_to_dense(&*handle, out_slice, layout) {
        Ok(()) => FfiResult::Ok,
        Err(e) => e.into(),
    }
}

/// CSR 转 Dense - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csr_f64_to_dense(
    handle: CSRF64Handle,
    out: *mut f64,
    out_len: usize,
    col_major: bool,
) -> FfiResult {
    if handle.is_null() || out.is_null() {
        return FfiResult::NullPointer;
    }

    let out_slice = std::slice::from_raw_parts_mut(out, out_len);
    let layout = if col_major {
        DenseLayout::ColMajor
    } else {
        DenseLayout::RowMajor
    };

    match crate::convert::csr_to_dense(&*handle, out_slice, layout) {
        Ok(()) => FfiResult::Ok,
        Err(e) => e.into(),
    }
}

/// CSC 转 Dense
#[no_mangle]
pub unsafe extern "C" fn csc_f32_to_dense(
    handle: CSCF32Handle,
    out: *mut f32,
    out_len: usize,
    col_major: bool,
) -> FfiResult {
    if handle.is_null() || out.is_null() {
        return FfiResult::NullPointer;
    }

    let out_slice = std::slice::from_raw_parts_mut(out, out_len);
    let layout = if col_major {
        DenseLayout::ColMajor
    } else {
        DenseLayout::RowMajor
    };

    match crate::convert::csc_to_dense(&*handle, out_slice, layout) {
        Ok(()) => FfiResult::Ok,
        Err(e) => e.into(),
    }
}

/// CSC 转 Dense - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csc_f64_to_dense(
    handle: CSCF64Handle,
    out: *mut f64,
    out_len: usize,
    col_major: bool,
) -> FfiResult {
    if handle.is_null() || out.is_null() {
        return FfiResult::NullPointer;
    }

    let out_slice = std::slice::from_raw_parts_mut(out, out_len);
    let layout = if col_major {
        DenseLayout::ColMajor
    } else {
        DenseLayout::RowMajor
    };

    match crate::convert::csc_to_dense(&*handle, out_slice, layout) {
        Ok(()) => FfiResult::Ok,
        Err(e) => e.into(),
    }
}

/// CSR 转 COO
#[no_mangle]
pub unsafe extern "C" fn csr_f32_to_coo(
    handle: CSRF32Handle,
    out_row_indices: *mut i64,
    out_col_indices: *mut i64,
    out_data: *mut f32,
    out_len: usize,
) -> FfiResult {
    if handle.is_null()
        || out_row_indices.is_null()
        || out_col_indices.is_null()
        || out_data.is_null()
    {
        return FfiResult::NullPointer;
    }

    let row_slice = std::slice::from_raw_parts_mut(out_row_indices, out_len);
    let col_slice = std::slice::from_raw_parts_mut(out_col_indices, out_len);
    let data_slice = std::slice::from_raw_parts_mut(out_data, out_len);

    match crate::convert::csr_to_coo(&*handle, row_slice, col_slice, data_slice) {
        Ok(()) => FfiResult::Ok,
        Err(e) => e.into(),
    }
}

/// CSR 转 COO - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csr_f64_to_coo(
    handle: CSRF64Handle,
    out_row_indices: *mut i64,
    out_col_indices: *mut i64,
    out_data: *mut f64,
    out_len: usize,
) -> FfiResult {
    if handle.is_null()
        || out_row_indices.is_null()
        || out_col_indices.is_null()
        || out_data.is_null()
    {
        return FfiResult::NullPointer;
    }

    let row_slice = std::slice::from_raw_parts_mut(out_row_indices, out_len);
    let col_slice = std::slice::from_raw_parts_mut(out_col_indices, out_len);
    let data_slice = std::slice::from_raw_parts_mut(out_data, out_len);

    match crate::convert::csr_to_coo(&*handle, row_slice, col_slice, data_slice) {
        Ok(()) => FfiResult::Ok,
        Err(e) => e.into(),
    }
}

/// CSC 转 COO
#[no_mangle]
pub unsafe extern "C" fn csc_f32_to_coo(
    handle: CSCF32Handle,
    out_row_indices: *mut i64,
    out_col_indices: *mut i64,
    out_data: *mut f32,
    out_len: usize,
) -> FfiResult {
    if handle.is_null()
        || out_row_indices.is_null()
        || out_col_indices.is_null()
        || out_data.is_null()
    {
        return FfiResult::NullPointer;
    }

    let row_slice = std::slice::from_raw_parts_mut(out_row_indices, out_len);
    let col_slice = std::slice::from_raw_parts_mut(out_col_indices, out_len);
    let data_slice = std::slice::from_raw_parts_mut(out_data, out_len);

    match crate::convert::csc_to_coo(&*handle, row_slice, col_slice, data_slice) {
        Ok(()) => FfiResult::Ok,
        Err(e) => e.into(),
    }
}

/// CSC 转 COO - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csc_f64_to_coo(
    handle: CSCF64Handle,
    out_row_indices: *mut i64,
    out_col_indices: *mut i64,
    out_data: *mut f64,
    out_len: usize,
) -> FfiResult {
    if handle.is_null()
        || out_row_indices.is_null()
        || out_col_indices.is_null()
        || out_data.is_null()
    {
        return FfiResult::NullPointer;
    }

    let row_slice = std::slice::from_raw_parts_mut(out_row_indices, out_len);
    let col_slice = std::slice::from_raw_parts_mut(out_col_indices, out_len);
    let data_slice = std::slice::from_raw_parts_mut(out_data, out_len);

    match crate::convert::csc_to_coo(&*handle, row_slice, col_slice, data_slice) {
        Ok(()) => FfiResult::Ok,
        Err(e) => e.into(),
    }
}

// =============================================================================
// 切片操作
// =============================================================================

/// CSR 行范围切片（零拷贝）
#[no_mangle]
pub unsafe extern "C" fn csr_f32_slice_rows(
    handle: CSRF32Handle,
    row_start: i64,
    row_end: i64,
    out_handle: *mut CSRF32HandleMut,
) -> FfiResult {
    if handle.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    match (*handle).slice_rows(row_start, row_end) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::IndexOutOfBounds,
    }
}

/// CSR 行范围切片 - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csr_f64_slice_rows(
    handle: CSRF64Handle,
    row_start: i64,
    row_end: i64,
    out_handle: *mut CSRF64HandleMut,
) -> FfiResult {
    if handle.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    match (*handle).slice_rows(row_start, row_end) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::IndexOutOfBounds,
    }
}

/// CSR 列范围切片
#[no_mangle]
pub unsafe extern "C" fn csr_f32_slice_cols(
    handle: CSRF32Handle,
    col_start: i64,
    col_end: i64,
    out_handle: *mut CSRF32HandleMut,
) -> FfiResult {
    if handle.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    match (*handle).slice_cols::<DEFAULT_ALIGN>(col_start, col_end) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::IndexOutOfBounds,
    }
}

/// CSR 列范围切片 - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csr_f64_slice_cols(
    handle: CSRF64Handle,
    col_start: i64,
    col_end: i64,
    out_handle: *mut CSRF64HandleMut,
) -> FfiResult {
    if handle.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    match (*handle).slice_cols::<DEFAULT_ALIGN>(col_start, col_end) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::IndexOutOfBounds,
    }
}

/// CSC 列范围切片（零拷贝）
#[no_mangle]
pub unsafe extern "C" fn csc_f32_slice_cols(
    handle: CSCF32Handle,
    col_start: i64,
    col_end: i64,
    out_handle: *mut CSCF32HandleMut,
) -> FfiResult {
    if handle.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    match (*handle).slice_cols(col_start, col_end) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::IndexOutOfBounds,
    }
}

/// CSC 列范围切片 - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csc_f64_slice_cols(
    handle: CSCF64Handle,
    col_start: i64,
    col_end: i64,
    out_handle: *mut CSCF64HandleMut,
) -> FfiResult {
    if handle.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    match (*handle).slice_cols(col_start, col_end) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::IndexOutOfBounds,
    }
}

/// CSC 行范围切片
#[no_mangle]
pub unsafe extern "C" fn csc_f32_slice_rows(
    handle: CSCF32Handle,
    row_start: i64,
    row_end: i64,
    out_handle: *mut CSCF32HandleMut,
) -> FfiResult {
    if handle.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    match (*handle).slice_rows::<DEFAULT_ALIGN>(row_start, row_end) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::IndexOutOfBounds,
    }
}

/// CSC 行范围切片 - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csc_f64_slice_rows(
    handle: CSCF64Handle,
    row_start: i64,
    row_end: i64,
    out_handle: *mut CSCF64HandleMut,
) -> FfiResult {
    if handle.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    match (*handle).slice_rows::<DEFAULT_ALIGN>(row_start, row_end) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::IndexOutOfBounds,
    }
}

/// CSR 行掩码切片
#[no_mangle]
pub unsafe extern "C" fn csr_f32_slice_rows_mask(
    handle: CSRF32Handle,
    mask: *const bool,
    mask_len: usize,
    out_handle: *mut CSRF32HandleMut,
) -> FfiResult {
    if handle.is_null() || mask.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let mask_slice = std::slice::from_raw_parts(mask, mask_len);

    match (*handle).slice_rows_mask(mask_slice) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::LengthMismatch,
    }
}

/// CSR 行掩码切片 - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csr_f64_slice_rows_mask(
    handle: CSRF64Handle,
    mask: *const bool,
    mask_len: usize,
    out_handle: *mut CSRF64HandleMut,
) -> FfiResult {
    if handle.is_null() || mask.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let mask_slice = std::slice::from_raw_parts(mask, mask_len);

    match (*handle).slice_rows_mask(mask_slice) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::LengthMismatch,
    }
}

/// CSR 列掩码切片
#[no_mangle]
pub unsafe extern "C" fn csr_f32_slice_cols_mask(
    handle: CSRF32Handle,
    mask: *const bool,
    mask_len: usize,
    out_handle: *mut CSRF32HandleMut,
) -> FfiResult {
    if handle.is_null() || mask.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let mask_slice = std::slice::from_raw_parts(mask, mask_len);

    match (*handle).slice_cols_mask::<DEFAULT_ALIGN>(mask_slice) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::LengthMismatch,
    }
}

/// CSR 列掩码切片 - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csr_f64_slice_cols_mask(
    handle: CSRF64Handle,
    mask: *const bool,
    mask_len: usize,
    out_handle: *mut CSRF64HandleMut,
) -> FfiResult {
    if handle.is_null() || mask.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let mask_slice = std::slice::from_raw_parts(mask, mask_len);

    match (*handle).slice_cols_mask::<DEFAULT_ALIGN>(mask_slice) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::LengthMismatch,
    }
}

/// CSC 列掩码切片
#[no_mangle]
pub unsafe extern "C" fn csc_f32_slice_cols_mask(
    handle: CSCF32Handle,
    mask: *const bool,
    mask_len: usize,
    out_handle: *mut CSCF32HandleMut,
) -> FfiResult {
    if handle.is_null() || mask.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let mask_slice = std::slice::from_raw_parts(mask, mask_len);

    match (*handle).slice_cols_mask(mask_slice) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::LengthMismatch,
    }
}

/// CSC 列掩码切片 - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csc_f64_slice_cols_mask(
    handle: CSCF64Handle,
    mask: *const bool,
    mask_len: usize,
    out_handle: *mut CSCF64HandleMut,
) -> FfiResult {
    if handle.is_null() || mask.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let mask_slice = std::slice::from_raw_parts(mask, mask_len);

    match (*handle).slice_cols_mask(mask_slice) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::LengthMismatch,
    }
}

/// CSC 行掩码切片
#[no_mangle]
pub unsafe extern "C" fn csc_f32_slice_rows_mask(
    handle: CSCF32Handle,
    mask: *const bool,
    mask_len: usize,
    out_handle: *mut CSCF32HandleMut,
) -> FfiResult {
    if handle.is_null() || mask.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let mask_slice = std::slice::from_raw_parts(mask, mask_len);

    match (*handle).slice_rows_mask::<DEFAULT_ALIGN>(mask_slice) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::LengthMismatch,
    }
}

/// CSC 行掩码切片 - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csc_f64_slice_rows_mask(
    handle: CSCF64Handle,
    mask: *const bool,
    mask_len: usize,
    out_handle: *mut CSCF64HandleMut,
) -> FfiResult {
    if handle.is_null() || mask.is_null() || out_handle.is_null() {
        return FfiResult::NullPointer;
    }

    let mask_slice = std::slice::from_raw_parts(mask, mask_len);

    match (*handle).slice_rows_mask::<DEFAULT_ALIGN>(mask_slice) {
        Ok(sliced) => {
            *out_handle = Box::into_raw(Box::new(sliced));
            FfiResult::Ok
        }
        Err(_) => FfiResult::LengthMismatch,
    }
}

// =============================================================================
// 堆叠操作
// =============================================================================

/// CSR 垂直堆叠（增加行）
#[no_mangle]
pub unsafe extern "C" fn csr_f32_vstack(
    handles: *const CSRF32Handle,
    count: usize,
    out_handle: *mut CSRF32HandleMut,
) -> FfiResult {
    if handles.is_null() || out_handle.is_null() || count == 0 {
        return FfiResult::NullPointer;
    }

    let handle_slice = std::slice::from_raw_parts(handles, count);
    let matrices: Vec<&CSRf32> = handle_slice.iter().map(|&h| &*h).collect();
    let refs: Vec<&CSRf32> = matrices.iter().copied().collect();

    match CSRf32::vstack(&refs) {
        Ok(result) => {
            *out_handle = Box::into_raw(Box::new(result));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// CSR 垂直堆叠 - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csr_f64_vstack(
    handles: *const CSRF64Handle,
    count: usize,
    out_handle: *mut CSRF64HandleMut,
) -> FfiResult {
    if handles.is_null() || out_handle.is_null() || count == 0 {
        return FfiResult::NullPointer;
    }

    let handle_slice = std::slice::from_raw_parts(handles, count);
    let matrices: Vec<&CSRf64> = handle_slice.iter().map(|&h| &*h).collect();
    let refs: Vec<&CSRf64> = matrices.iter().copied().collect();

    match CSRf64::vstack(&refs) {
        Ok(result) => {
            *out_handle = Box::into_raw(Box::new(result));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// CSR 水平堆叠（增加列）
#[no_mangle]
pub unsafe extern "C" fn csr_f32_hstack(
    handles: *const CSRF32Handle,
    count: usize,
    out_handle: *mut CSRF32HandleMut,
) -> FfiResult {
    if handles.is_null() || out_handle.is_null() || count == 0 {
        return FfiResult::NullPointer;
    }

    let handle_slice = std::slice::from_raw_parts(handles, count);
    let matrices: Vec<&CSRf32> = handle_slice.iter().map(|&h| &*h).collect();
    let refs: Vec<&CSRf32> = matrices.iter().copied().collect();

    match CSRf32::hstack::<DEFAULT_ALIGN>(&refs, AllocStrategy::Auto) {
        Ok(result) => {
            *out_handle = Box::into_raw(Box::new(result));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// CSR 水平堆叠 - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csr_f64_hstack(
    handles: *const CSRF64Handle,
    count: usize,
    out_handle: *mut CSRF64HandleMut,
) -> FfiResult {
    if handles.is_null() || out_handle.is_null() || count == 0 {
        return FfiResult::NullPointer;
    }

    let handle_slice = std::slice::from_raw_parts(handles, count);
    let matrices: Vec<&CSRf64> = handle_slice.iter().map(|&h| &*h).collect();
    let refs: Vec<&CSRf64> = matrices.iter().copied().collect();

    match CSRf64::hstack::<DEFAULT_ALIGN>(&refs, AllocStrategy::Auto) {
        Ok(result) => {
            *out_handle = Box::into_raw(Box::new(result));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// CSC 水平堆叠（增加列）
#[no_mangle]
pub unsafe extern "C" fn csc_f32_hstack(
    handles: *const CSCF32Handle,
    count: usize,
    out_handle: *mut CSCF32HandleMut,
) -> FfiResult {
    if handles.is_null() || out_handle.is_null() || count == 0 {
        return FfiResult::NullPointer;
    }

    let handle_slice = std::slice::from_raw_parts(handles, count);
    let matrices: Vec<&CSCf32> = handle_slice.iter().map(|&h| &*h).collect();
    let refs: Vec<&CSCf32> = matrices.iter().copied().collect();

    match CSCf32::hstack(&refs) {
        Ok(result) => {
            *out_handle = Box::into_raw(Box::new(result));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// CSC 水平堆叠 - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csc_f64_hstack(
    handles: *const CSCF64Handle,
    count: usize,
    out_handle: *mut CSCF64HandleMut,
) -> FfiResult {
    if handles.is_null() || out_handle.is_null() || count == 0 {
        return FfiResult::NullPointer;
    }

    let handle_slice = std::slice::from_raw_parts(handles, count);
    let matrices: Vec<&CSCf64> = handle_slice.iter().map(|&h| &*h).collect();
    let refs: Vec<&CSCf64> = matrices.iter().copied().collect();

    match CSCf64::hstack(&refs) {
        Ok(result) => {
            *out_handle = Box::into_raw(Box::new(result));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// CSC 垂直堆叠（增加行）
#[no_mangle]
pub unsafe extern "C" fn csc_f32_vstack(
    handles: *const CSCF32Handle,
    count: usize,
    out_handle: *mut CSCF32HandleMut,
) -> FfiResult {
    if handles.is_null() || out_handle.is_null() || count == 0 {
        return FfiResult::NullPointer;
    }

    let handle_slice = std::slice::from_raw_parts(handles, count);
    let matrices: Vec<&CSCf32> = handle_slice.iter().map(|&h| &*h).collect();
    let refs: Vec<&CSCf32> = matrices.iter().copied().collect();

    match CSCf32::vstack::<DEFAULT_ALIGN>(&refs, AllocStrategy::Auto) {
        Ok(result) => {
            *out_handle = Box::into_raw(Box::new(result));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

/// CSC 垂直堆叠 - f64 版本
#[no_mangle]
pub unsafe extern "C" fn csc_f64_vstack(
    handles: *const CSCF64Handle,
    count: usize,
    out_handle: *mut CSCF64HandleMut,
) -> FfiResult {
    if handles.is_null() || out_handle.is_null() || count == 0 {
        return FfiResult::NullPointer;
    }

    let handle_slice = std::slice::from_raw_parts(handles, count);
    let matrices: Vec<&CSCf64> = handle_slice.iter().map(|&h| &*h).collect();
    let refs: Vec<&CSCf64> = matrices.iter().copied().collect();

    match CSCf64::vstack::<DEFAULT_ALIGN>(&refs, AllocStrategy::Auto) {
        Ok(result) => {
            *out_handle = Box::into_raw(Box::new(result));
            FfiResult::Ok
        }
        Err(e) => e.into(),
    }
}

// =============================================================================
// Numba 支持 - 批量指针获取
// =============================================================================

/// CSR 行指针信息（用于 Numba unbox）
///
/// 一次性返回指定行的所有指针信息，避免多次 FFI 调用
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RowPointerInfo {
    /// 值数据指针
    pub values_ptr: *const u8,
    /// 索引数据指针
    pub indices_ptr: *const u8,
    /// 行长度（非零元素个数）
    pub len: usize,
}

impl RowPointerInfo {
    pub const NULL: Self = Self {
        values_ptr: ptr::null(),
        indices_ptr: ptr::null(),
        len: 0,
    };
}

/// CSC 列指针信息（用于 Numba unbox）
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ColPointerInfo {
    /// 值数据指针
    pub values_ptr: *const u8,
    /// 索引数据指针
    pub indices_ptr: *const u8,
    /// 列长度（非零元素个数）
    pub len: usize,
}

impl ColPointerInfo {
    pub const NULL: Self = Self {
        values_ptr: ptr::null(),
        indices_ptr: ptr::null(),
        len: 0,
    };
}

/// 为 CSR 生成 Numba 批量指针获取函数
macro_rules! impl_csr_numba_ffi {
    ($suffix:ident, $val_ty:ty, $idx_ty:ty, $handle:ty) => {
        paste::paste! {
            /// 批量获取所有行的指针信息（用于 Numba unbox）
            ///
            /// 一次性将所有行的 (values_ptr, indices_ptr, len) 填充到预分配的数组中。
            /// 这样 Numba 在 unbox 时只需一次 FFI 调用即可获取全部信息。
            ///
            /// # 参数
            ///
            /// - `handle`: CSR 句柄
            /// - `out_values_ptrs`: 预分配的 nrows 长度数组，用于存储每行的值指针
            /// - `out_indices_ptrs`: 预分配的 nrows 长度数组，用于存储每行的索引指针
            /// - `out_row_lens`: 预分配的 nrows 长度数组，用于存储每行的长度
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针
            /// - 输出数组必须预分配足够空间（至少 nrows 个元素）
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _get_row_pointers>](
                handle: $handle,
                out_values_ptrs: *mut *const $val_ty,
                out_indices_ptrs: *mut *const $idx_ty,
                out_row_lens: *mut usize,
            ) -> i32 {
                if handle.is_null() || out_values_ptrs.is_null() || out_indices_ptrs.is_null() || out_row_lens.is_null() {
                    return FfiResult::NullPointer as i32;
                }

                let csr = &*handle;
                let nrows = csr.rows.to_usize();

                for i in 0..nrows {
                    let val_span = csr.values.get_unchecked(i);
                    let idx_span = csr.indices.get_unchecked(i);

                    *out_values_ptrs.add(i) = val_span.as_ptr();
                    *out_indices_ptrs.add(i) = idx_span.as_ptr();
                    *out_row_lens.add(i) = val_span.len();
                }

                FfiResult::Ok as i32
            }

            /// 获取单行的指针信息（用于 Numba 单行访问）
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针
            /// - `row` 必须小于矩阵行数
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _get_row_info>](
                handle: $handle,
                row: usize,
            ) -> RowPointerInfo {
                if handle.is_null() {
                    return RowPointerInfo::NULL;
                }

                let csr = &*handle;
                if row >= csr.rows.to_usize() {
                    return RowPointerInfo::NULL;
                }

                let val_span = csr.values.get_unchecked(row);
                let idx_span = csr.indices.get_unchecked(row);

                RowPointerInfo {
                    values_ptr: val_span.as_ptr() as *const u8,
                    indices_ptr: idx_span.as_ptr() as *const u8,
                    len: val_span.len(),
                }
            }

            /// 获取单行的指针信息（无边界检查版本）
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针
            /// - `row` 必须小于矩阵行数
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _get_row_info_unchecked>](
                handle: $handle,
                row: usize,
            ) -> RowPointerInfo {
                let csr = &*handle;
                let val_span = csr.values.get_unchecked(row);
                let idx_span = csr.indices.get_unchecked(row);

                RowPointerInfo {
                    values_ptr: val_span.as_ptr() as *const u8,
                    indices_ptr: idx_span.as_ptr() as *const u8,
                    len: val_span.len(),
                }
            }

            /// 获取 CSR 的元数据（用于 Numba unbox）
            ///
            /// 返回 (nrows, ncols, nnz)，打包在一个结构中
            #[no_mangle]
            pub unsafe extern "C" fn [<csr_ $suffix _get_metadata>](
                handle: $handle,
                out_nrows: *mut i64,
                out_ncols: *mut i64,
                out_nnz: *mut i64,
            ) -> i32 {
                if handle.is_null() || out_nrows.is_null() || out_ncols.is_null() || out_nnz.is_null() {
                    return FfiResult::NullPointer as i32;
                }

                let csr = &*handle;
                *out_nrows = csr.rows;
                *out_ncols = csr.cols;
                *out_nnz = csr.nnz();

                FfiResult::Ok as i32
            }
        }
    };
}

/// 为 CSC 生成 Numba 批量指针获取函数
macro_rules! impl_csc_numba_ffi {
    ($suffix:ident, $val_ty:ty, $idx_ty:ty, $handle:ty) => {
        paste::paste! {
            /// 批量获取所有列的指针信息（用于 Numba unbox）
            ///
            /// 一次性将所有列的 (values_ptr, indices_ptr, len) 填充到预分配的数组中。
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针
            /// - 输出数组必须预分配足够空间（至少 ncols 个元素）
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _get_col_pointers>](
                handle: $handle,
                out_values_ptrs: *mut *const $val_ty,
                out_indices_ptrs: *mut *const $idx_ty,
                out_col_lens: *mut usize,
            ) -> i32 {
                if handle.is_null() || out_values_ptrs.is_null() || out_indices_ptrs.is_null() || out_col_lens.is_null() {
                    return FfiResult::NullPointer as i32;
                }

                let csc = &*handle;
                let ncols = csc.cols.to_usize();

                for j in 0..ncols {
                    let val_span = csc.values.get_unchecked(j);
                    let idx_span = csc.indices.get_unchecked(j);

                    *out_values_ptrs.add(j) = val_span.as_ptr();
                    *out_indices_ptrs.add(j) = idx_span.as_ptr();
                    *out_col_lens.add(j) = val_span.len();
                }

                FfiResult::Ok as i32
            }

            /// 获取单列的指针信息（用于 Numba 单列访问）
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针
            /// - `col` 必须小于矩阵列数
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _get_col_info>](
                handle: $handle,
                col: usize,
            ) -> ColPointerInfo {
                if handle.is_null() {
                    return ColPointerInfo::NULL;
                }

                let csc = &*handle;
                if col >= csc.cols.to_usize() {
                    return ColPointerInfo::NULL;
                }

                let val_span = csc.values.get_unchecked(col);
                let idx_span = csc.indices.get_unchecked(col);

                ColPointerInfo {
                    values_ptr: val_span.as_ptr() as *const u8,
                    indices_ptr: idx_span.as_ptr() as *const u8,
                    len: val_span.len(),
                }
            }

            /// 获取单列的指针信息（无边界检查版本）
            ///
            /// # Safety
            ///
            /// - `handle` 必须是有效的非空指针
            /// - `col` 必须小于矩阵列数
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _get_col_info_unchecked>](
                handle: $handle,
                col: usize,
            ) -> ColPointerInfo {
                let csc = &*handle;
                let val_span = csc.values.get_unchecked(col);
                let idx_span = csc.indices.get_unchecked(col);

                ColPointerInfo {
                    values_ptr: val_span.as_ptr() as *const u8,
                    indices_ptr: idx_span.as_ptr() as *const u8,
                    len: val_span.len(),
                }
            }

            /// 获取 CSC 的元数据（用于 Numba unbox）
            #[no_mangle]
            pub unsafe extern "C" fn [<csc_ $suffix _get_metadata>](
                handle: $handle,
                out_nrows: *mut i64,
                out_ncols: *mut i64,
                out_nnz: *mut i64,
            ) -> i32 {
                if handle.is_null() || out_nrows.is_null() || out_ncols.is_null() || out_nnz.is_null() {
                    return FfiResult::NullPointer as i32;
                }

                let csc = &*handle;
                *out_nrows = csc.rows;
                *out_ncols = csc.cols;
                *out_nnz = csc.nnz();

                FfiResult::Ok as i32
            }
        }
    };
}

// 生成 CSR Numba FFI
impl_csr_numba_ffi!(f32, f32, i64, CSRF32Handle);
impl_csr_numba_ffi!(f64, f64, i64, CSRF64Handle);

// 生成 CSC Numba FFI
impl_csc_numba_ffi!(f32, f32, i64, CSCF32Handle);
impl_csc_numba_ffi!(f64, f64, i64, CSCF64Handle);

// =============================================================================
// 测试
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_info_null() {
        assert!(SpanInfo::NULL.data.is_null());
        assert_eq!(SpanInfo::NULL.len, 0);
        assert_eq!(SpanInfo::NULL.element_size, 0);
        assert_eq!(SpanInfo::NULL.flags, 0);
    }

    #[test]
    fn test_span_size() {
        // 验证 Span 大小为 32 字节（64位平台）
        assert_eq!(span_f32_size(), 32);
        assert_eq!(span_f64_size(), 32);
        assert_eq!(span_i64_size(), 32);
    }

    #[test]
    fn test_span_flags() {
        assert_eq!(SPAN_FLAG_VIEW, 1);
        assert_eq!(SPAN_FLAG_ALIGNED, 2);
        assert_eq!(SPAN_FLAG_MUTABLE, 4);
    }

    #[test]
    fn test_abi_version() {
        assert_eq!(BIOSPARSE_ABI_VERSION, 1);
    }

    #[test]
    fn test_null_handle_safety() {
        unsafe {
            // 所有函数应该安全处理 null 句柄
            assert!(span_f32_data(ptr::null()).is_null());
            assert_eq!(span_f32_len(ptr::null()), 0);
            assert_eq!(span_f32_flags(ptr::null()), 0);

            assert_eq!(csr_f32_rows(ptr::null()), 0);
            assert_eq!(csr_f32_cols(ptr::null()), 0);
            assert!(csr_f32_values_vec_ptr(ptr::null()).is_null());

            assert_eq!(csc_f32_rows(ptr::null()), 0);
            assert_eq!(csc_f32_cols(ptr::null()), 0);
        }
    }
}
