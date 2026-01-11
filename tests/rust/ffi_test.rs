//! FFI 接口测试
//!
//! 测试内容：
//! - Span FFI 函数
//! - CSR/CSC FFI 函数
//! - 内存管理和生命周期
//! - ABI 版本

use scl_core::ffi::*;
use scl_core::span::Span;
use std::ptr;

// =============================================================================
// ABI 版本测试
// =============================================================================

#[test]
fn test_abi_version() {
    // ABI 版本应该是一个有效的正整数
    assert!(SCL_CORE_ABI_VERSION >= 1);
}

// =============================================================================
// SpanInfo 测试
// =============================================================================

#[test]
fn test_span_info_null() {
    let info = SpanInfo::NULL;
    assert!(info.data.is_null());
    assert_eq!(info.len, 0);
    assert_eq!(info.element_size, 0);
    assert_eq!(info.flags, 0);
}

// =============================================================================
// Span FFI 测试
// =============================================================================

#[test]
fn test_span_f64_ffi() {
    // 创建一个 Span
    let span: Span<f64> = Span::alloc::<32>(10).unwrap();
    let handle: SpanF64Handle = &span;

    unsafe {
        // 测试基本属性
        assert!(!span_f64_data(handle).is_null());
        assert_eq!(span_f64_len(handle), 10);
        assert!(span_f64_is_aligned(handle));
        assert!(span_f64_is_mutable(handle));
        assert!(!span_f64_is_view(handle));
    }
}

#[test]
fn test_span_f64_data_access() {
    let mut span: Span<f64> = Span::alloc::<32>(5).unwrap();

    // 写入数据
    for i in 0..5 {
        span[i] = (i + 1) as f64;
    }

    let handle: SpanF64Handle = &span;

    unsafe {
        let ptr = span_f64_data(handle);

        // 读取验证
        for i in 0..5 {
            assert_eq!(*ptr.add(i), (i + 1) as f64);
        }
    }
}

#[test]
fn test_span_f64_null_handle() {
    unsafe {
        // 空句柄应该返回安全的默认值
        assert!(span_f64_data(ptr::null()).is_null());
        assert_eq!(span_f64_len(ptr::null()), 0);
        assert!(!span_f64_is_aligned(ptr::null()));
        assert!(!span_f64_is_mutable(ptr::null()));
    }
}

#[test]
fn test_span_f32_ffi() {
    let span: Span<f32> = Span::alloc::<32>(20).unwrap();
    let handle: SpanF32Handle = &span;

    unsafe {
        assert!(!span_f32_data(handle).is_null());
        assert_eq!(span_f32_len(handle), 20);
        assert!(span_f32_is_aligned(handle));
    }
}

#[test]
fn test_span_i64_ffi() {
    let span: Span<i64> = Span::alloc::<32>(15).unwrap();
    let handle: SpanI64Handle = &span;

    unsafe {
        assert!(!span_i64_data(handle).is_null());
        assert_eq!(span_i64_len(handle), 15);
    }
}

#[test]
fn test_span_i32_ffi() {
    let span: Span<i32> = Span::alloc::<32>(8).unwrap();
    let handle: SpanI32Handle = &span;

    unsafe {
        assert!(!span_i32_data(handle).is_null());
        assert_eq!(span_i32_len(handle), 8);
    }
}

// =============================================================================
// 辅助函数：创建测试用 CSR
// =============================================================================

/// 使用 FFI 创建一个测试用 CSR 矩阵
///
/// 返回有效的句柄，调用者负责释放
unsafe fn create_test_csr_f64(
    rows: i64,
    cols: i64,
    data: &[f64],
    indices: &[i64],
    indptr: &[i64],
) -> CSRF64HandleMut {
    let mut handle: CSRF64HandleMut = ptr::null_mut();
    let result = csr_f64_from_scipy_copy(
        rows,
        cols,
        data.as_ptr(),
        data.len(),
        indices.as_ptr(),
        indices.len(),
        indptr.as_ptr(),
        indptr.len(),
        &mut handle,
    );
    assert_eq!(result, FfiResult::Ok, "创建 CSR 失败");
    assert!(!handle.is_null());
    handle
}

/// 使用 FFI 创建一个测试用 CSC 矩阵
unsafe fn create_test_csc_f64(
    rows: i64,
    cols: i64,
    data: &[f64],
    indices: &[i64],
    indptr: &[i64],
) -> CSCF64HandleMut {
    let mut handle: CSCF64HandleMut = ptr::null_mut();
    let result = csc_f64_from_scipy_copy(
        rows,
        cols,
        data.as_ptr(),
        data.len(),
        indices.as_ptr(),
        indices.len(),
        indptr.as_ptr(),
        indptr.len(),
        &mut handle,
    );
    assert_eq!(result, FfiResult::Ok, "创建 CSC 失败");
    assert!(!handle.is_null());
    handle
}

// =============================================================================
// CSR FFI 测试
// =============================================================================

#[test]
fn test_csr_f64_from_scipy_copy() {
    unsafe {
        // 创建 3x4 CSR 矩阵:
        // [1.0, 2.0, 0.0, 0.0]
        // [0.0, 0.0, 3.0, 0.0]
        // [4.0, 0.0, 0.0, 5.0]
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let indices = vec![0i64, 1, 2, 0, 3];
        let indptr = vec![0i64, 2, 3, 5];

        let handle = create_test_csr_f64(3, 4, &data, &indices, &indptr);

        // 验证维度
        assert_eq!(csr_f64_rows(handle), 3);
        assert_eq!(csr_f64_cols(handle), 4);
        assert_eq!(csr_f64_nnz(handle), 5);

        // 释放
        csr_f64_free(handle);
    }
}

#[test]
fn test_csr_f64_shape() {
    unsafe {
        let data = vec![1.0f64, 2.0];
        let indices = vec![0i64, 1];
        let indptr = vec![0i64, 1, 2];

        let handle = create_test_csr_f64(2, 5, &data, &indices, &indptr);

        let shape = csr_f64_shape(handle);
        assert_eq!(shape.rows, 2);
        assert_eq!(shape.cols, 5);

        csr_f64_free(handle);
    }
}

#[test]
fn test_csr_f64_null_handle() {
    unsafe {
        assert_eq!(csr_f64_rows(ptr::null()), 0);
        assert_eq!(csr_f64_cols(ptr::null()), 0);
        assert_eq!(csr_f64_nnz(ptr::null()), 0);

        // 空句柄的 shape 应该返回 (0, 0)
        let shape = csr_f64_shape(ptr::null());
        assert_eq!(shape.rows, 0);
        assert_eq!(shape.cols, 0);
    }
}

#[test]
fn test_csr_f64_is_empty_and_is_zero() {
    unsafe {
        // 有非零元素的矩阵
        let data = vec![1.0f64];
        let indices = vec![0i64];
        let indptr = vec![0i64, 1];

        let non_zero = create_test_csr_f64(1, 2, &data, &indices, &indptr);
        assert!(!csr_f64_is_empty(non_zero));
        assert!(!csr_f64_is_zero(non_zero));
        csr_f64_free(non_zero);

        // 没有非零元素的矩阵
        let empty_data: Vec<f64> = vec![];
        let empty_indices: Vec<i64> = vec![];
        let empty_indptr = vec![0i64, 0, 0];

        let zero = create_test_csr_f64(2, 3, &empty_data, &empty_indices, &empty_indptr);
        assert!(!csr_f64_is_empty(zero)); // 结构不为空
        assert!(csr_f64_is_zero(zero)); // 但没有非零元素
        csr_f64_free(zero);
    }
}

#[test]
fn test_csr_f64_clone() {
    unsafe {
        let data = vec![1.0f64, 2.0, 3.0];
        let indices = vec![0i64, 1, 2];
        let indptr = vec![0i64, 1, 2, 3];

        let original = create_test_csr_f64(3, 3, &data, &indices, &indptr);
        let cloned = csr_f64_clone(original);

        assert!(!cloned.is_null());
        assert_eq!(csr_f64_rows(cloned), csr_f64_rows(original));
        assert_eq!(csr_f64_cols(cloned), csr_f64_cols(original));
        assert_eq!(csr_f64_nnz(cloned), csr_f64_nnz(original));

        // 应该是不同的指针
        assert_ne!(original as usize, cloned as usize);

        csr_f64_free(original);
        csr_f64_free(cloned);
    }
}

// =============================================================================
// CSR F32 FFI 测试
// =============================================================================

#[test]
fn test_csr_f32_from_scipy_copy() {
    unsafe {
        let data = vec![1.0f32, 2.0, 3.0];
        let indices = vec![0i64, 1, 0];
        let indptr = vec![0i64, 2, 3];

        let mut handle: CSRF32HandleMut = ptr::null_mut();
        let result = csr_f32_from_scipy_copy(
            2,
            3,
            data.as_ptr(),
            data.len(),
            indices.as_ptr(),
            indices.len(),
            indptr.as_ptr(),
            indptr.len(),
            &mut handle,
        );

        assert_eq!(result, FfiResult::Ok);
        assert!(!handle.is_null());
        assert_eq!(csr_f32_rows(handle), 2);
        assert_eq!(csr_f32_cols(handle), 3);
        assert_eq!(csr_f32_nnz(handle), 3);

        csr_f32_free(handle);
    }
}

// =============================================================================
// CSC FFI 测试
// =============================================================================

#[test]
fn test_csc_f64_from_scipy_copy() {
    unsafe {
        // 创建 3x3 CSC 矩阵
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let indices = vec![0i64, 2, 0, 1];
        let indptr = vec![0i64, 2, 2, 4];

        let handle = create_test_csc_f64(3, 3, &data, &indices, &indptr);

        assert_eq!(csc_f64_rows(handle), 3);
        assert_eq!(csc_f64_cols(handle), 3);
        assert_eq!(csc_f64_nnz(handle), 4);

        csc_f64_free(handle);
    }
}

#[test]
fn test_csc_f64_shape() {
    unsafe {
        let data = vec![1.0f64, 2.0];
        let indices = vec![0i64, 1];
        let indptr = vec![0i64, 1, 2, 2];

        let handle = create_test_csc_f64(3, 3, &data, &indices, &indptr);

        let shape = csc_f64_shape(handle);
        assert_eq!(shape.rows, 3);
        assert_eq!(shape.cols, 3);

        csc_f64_free(handle);
    }
}

#[test]
fn test_csc_f64_null_handle() {
    unsafe {
        assert_eq!(csc_f64_rows(ptr::null()), 0);
        assert_eq!(csc_f64_cols(ptr::null()), 0);
        assert_eq!(csc_f64_nnz(ptr::null()), 0);
    }
}

#[test]
fn test_csc_f64_clone() {
    unsafe {
        let data = vec![1.0f64, 2.0];
        let indices = vec![0i64, 1];
        let indptr = vec![0i64, 1, 2];

        let original = create_test_csc_f64(2, 2, &data, &indices, &indptr);
        let cloned = csc_f64_clone(original);

        assert!(!cloned.is_null());
        assert_eq!(csc_f64_rows(cloned), csc_f64_rows(original));
        assert_eq!(csc_f64_cols(cloned), csc_f64_cols(original));

        csc_f64_free(original);
        csc_f64_free(cloned);
    }
}

#[test]
fn test_csc_f32_from_scipy_copy() {
    unsafe {
        let data = vec![1.0f32, 2.0];
        let indices = vec![0i64, 1];
        let indptr = vec![0i64, 1, 2];

        let mut handle: CSCF32HandleMut = ptr::null_mut();
        let result = csc_f32_from_scipy_copy(
            2,
            2,
            data.as_ptr(),
            data.len(),
            indices.as_ptr(),
            indices.len(),
            indptr.as_ptr(),
            indptr.len(),
            &mut handle,
        );

        assert_eq!(result, FfiResult::Ok);
        assert!(!handle.is_null());
        assert_eq!(csc_f32_rows(handle), 2);
        assert_eq!(csc_f32_cols(handle), 2);

        csc_f32_free(handle);
    }
}

// =============================================================================
// 内存安全测试
// =============================================================================

#[test]
fn test_null_free_safety() {
    // 释放空指针应该是安全的
    unsafe {
        csr_f64_free(ptr::null_mut());
        csc_f64_free(ptr::null_mut());
        csr_f32_free(ptr::null_mut());
        csc_f32_free(ptr::null_mut());
    }
}

// =============================================================================
// Shape 结构体测试
// =============================================================================

#[test]
fn test_shape_struct() {
    let shape = Shape { rows: 10, cols: 20 };
    assert_eq!(shape.rows, 10);
    assert_eq!(shape.cols, 20);
}

// =============================================================================
// 行/列访问 FFI 测试
// =============================================================================

#[test]
fn test_csr_f64_row_access() {
    let data = vec![1.0f64, 2.0, 3.0];
    let indices = vec![0i64, 1, 2];
    let indptr = vec![0i64, 1, 2, 3];

    unsafe {
        let handle = create_test_csr_f64(3, 3, &data, &indices, &indptr);

        // 获取每行的元素个数
        assert_eq!(csr_f64_row_len(handle, 0), 1);
        assert_eq!(csr_f64_row_len(handle, 1), 1);
        assert_eq!(csr_f64_row_len(handle, 2), 1);

        // 获取行指针
        let vals_ptr = csr_f64_row_values_ptr(handle, 0);
        let idxs_ptr = csr_f64_row_indices_ptr(handle, 0);

        assert!(!vals_ptr.is_null());
        assert!(!idxs_ptr.is_null());

        assert_eq!(*vals_ptr, 1.0);
        assert_eq!(*idxs_ptr, 0);

        csr_f64_free(handle);
    }
}

#[test]
fn test_csc_f64_col_access() {
    let data = vec![1.0f64, 2.0, 3.0];
    let indices = vec![0i64, 1, 2];
    let indptr = vec![0i64, 1, 2, 3];

    unsafe {
        let handle = create_test_csc_f64(3, 3, &data, &indices, &indptr);

        // 获取每列的元素个数
        assert_eq!(csc_f64_col_len(handle, 0), 1);
        assert_eq!(csc_f64_col_len(handle, 1), 1);
        assert_eq!(csc_f64_col_len(handle, 2), 1);

        // 获取列指针
        let vals_ptr = csc_f64_col_values_ptr(handle, 0);
        let idxs_ptr = csc_f64_col_indices_ptr(handle, 0);

        assert!(!vals_ptr.is_null());
        assert!(!idxs_ptr.is_null());

        assert_eq!(*vals_ptr, 1.0);
        assert_eq!(*idxs_ptr, 0);

        csc_f64_free(handle);
    }
}

// =============================================================================
// 验证和排序 FFI 测试
// =============================================================================

#[test]
fn test_csr_f64_validation() {
    let data = vec![1.0f64, 2.0];
    let indices = vec![0i64, 1];
    let indptr = vec![0i64, 1, 2];

    unsafe {
        let handle = create_test_csr_f64(2, 3, &data, &indices, &indptr);

        assert!(csr_f64_is_valid(handle));
        assert!(csr_f64_is_sorted(handle));
        assert!(csr_f64_validate(handle));
        assert!(csr_f64_indices_in_bounds(handle));

        csr_f64_free(handle);
    }
}

// =============================================================================
// 切片 FFI 测试
// =============================================================================

#[test]
fn test_csr_f64_slice_rows_ffi() {
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let indices = vec![0i64, 1, 0, 1, 2];
    let indptr = vec![0i64, 2, 4, 5];

    unsafe {
        let handle = create_test_csr_f64(3, 3, &data, &indices, &indptr);

        // 切片行 [0, 2)
        let mut sliced: CSRF64HandleMut = ptr::null_mut();
        let result = csr_f64_slice_rows(handle, 0, 2, &mut sliced);

        assert_eq!(result, FfiResult::Ok);
        assert!(!sliced.is_null());
        assert_eq!(csr_f64_rows(sliced), 2);
        assert_eq!(csr_f64_nnz(sliced), 4);

        csr_f64_free(sliced);
        csr_f64_free(handle);
    }
}

#[test]
fn test_csr_f64_slice_cols_ffi() {
    let data = vec![1.0f64, 2.0, 3.0];
    let indices = vec![0i64, 1, 2];
    let indptr = vec![0i64, 1, 2, 3];

    unsafe {
        let handle = create_test_csr_f64(3, 4, &data, &indices, &indptr);

        // 切片列 [0, 2)
        let mut sliced: CSRF64HandleMut = ptr::null_mut();
        let result = csr_f64_slice_cols(handle, 0, 2, &mut sliced);

        assert_eq!(result, FfiResult::Ok);
        assert!(!sliced.is_null());
        assert_eq!(csr_f64_cols(sliced), 2);

        csr_f64_free(sliced);
        csr_f64_free(handle);
    }
}

// =============================================================================
// 转换 FFI 测试
// =============================================================================

#[test]
fn test_csr_to_csc_ffi() {
    let data = vec![1.0f64, 2.0, 3.0];
    let indices = vec![0i64, 1, 0];
    let indptr = vec![0i64, 1, 2, 3];

    unsafe {
        let csr = create_test_csr_f64(3, 2, &data, &indices, &indptr);

        let mut csc: CSCF64HandleMut = ptr::null_mut();
        let result = csc_f64_from_csr(csr, &mut csc);

        assert_eq!(result, FfiResult::Ok);
        assert!(!csc.is_null());
        assert_eq!(csc_f64_rows(csc), 3);
        assert_eq!(csc_f64_cols(csc), 2);
        assert_eq!(csc_f64_nnz(csc), 3);

        csc_f64_free(csc);
        csr_f64_free(csr);
    }
}

#[test]
fn test_csc_to_csr_ffi() {
    let data = vec![1.0f64, 2.0, 3.0];
    let indices = vec![0i64, 1, 2];
    let indptr = vec![0i64, 2, 3];

    unsafe {
        let csc = create_test_csc_f64(3, 2, &data, &indices, &indptr);

        let mut csr: CSRF64HandleMut = ptr::null_mut();
        let result = csr_f64_from_csc(csc, &mut csr);

        assert_eq!(result, FfiResult::Ok);
        assert!(!csr.is_null());
        assert_eq!(csr_f64_rows(csr), 3);
        assert_eq!(csr_f64_cols(csr), 2);
        assert_eq!(csr_f64_nnz(csr), 3);

        csr_f64_free(csr);
        csc_f64_free(csc);
    }
}

// =============================================================================
// Stack FFI 测试
// =============================================================================

#[test]
fn test_csr_f64_vstack_ffi() {
    unsafe {
        // 创建两个 CSR 矩阵
        let data_a = vec![1.0f64, 2.0];
        let indices_a = vec![0i64, 1];
        let indptr_a = vec![0i64, 1, 2];
        let a = create_test_csr_f64(2, 3, &data_a, &indices_a, &indptr_a);

        let data_b = vec![3.0f64, 4.0, 5.0];
        let indices_b = vec![0i64, 1, 2];
        let indptr_b = vec![0i64, 1, 2, 3];
        let b = create_test_csr_f64(3, 3, &data_b, &indices_b, &indptr_b);

        let handles = [a as CSRF64Handle, b as CSRF64Handle];
        let mut result: CSRF64HandleMut = ptr::null_mut();
        let ffi_result = csr_f64_vstack(handles.as_ptr(), 2, &mut result);

        assert_eq!(ffi_result, FfiResult::Ok);
        assert!(!result.is_null());
        assert_eq!(csr_f64_rows(result), 5);
        assert_eq!(csr_f64_cols(result), 3);
        assert_eq!(csr_f64_nnz(result), 5);

        csr_f64_free(a);
        csr_f64_free(b);
        csr_f64_free(result);
    }
}

#[test]
fn test_csr_f64_hstack_ffi() {
    unsafe {
        // 创建两个 CSR 矩阵（相同行数）
        let data_a = vec![1.0f64, 2.0];
        let indices_a = vec![0i64, 0];
        let indptr_a = vec![0i64, 1, 2];
        let a = create_test_csr_f64(2, 2, &data_a, &indices_a, &indptr_a);

        let data_b = vec![3.0f64, 4.0];
        let indices_b = vec![0i64, 1];
        let indptr_b = vec![0i64, 1, 2];
        let b = create_test_csr_f64(2, 3, &data_b, &indices_b, &indptr_b);

        let handles = [a as CSRF64Handle, b as CSRF64Handle];
        let mut result: CSRF64HandleMut = ptr::null_mut();
        let ffi_result = csr_f64_hstack(handles.as_ptr(), 2, &mut result);

        assert_eq!(ffi_result, FfiResult::Ok);
        assert!(!result.is_null());
        assert_eq!(csr_f64_rows(result), 2);
        assert_eq!(csr_f64_cols(result), 5);
        assert_eq!(csr_f64_nnz(result), 4);

        csr_f64_free(a);
        csr_f64_free(b);
        csr_f64_free(result);
    }
}

// =============================================================================
// FfiResult 测试
// =============================================================================

#[test]
fn test_ffi_result_values() {
    // 验证 FfiResult 枚举值
    assert_eq!(FfiResult::Ok as i32, 0);
    assert_eq!(FfiResult::NullPointer as i32, -1);
    assert_eq!(FfiResult::DimensionMismatch as i32, -2);
    assert_eq!(FfiResult::LengthMismatch as i32, -3);
}

#[test]
fn test_ffi_error_handling() {
    unsafe {
        // 测试空指针返回正确的错误
        let mut handle: CSRF64HandleMut = ptr::null_mut();
        let result = csr_f64_from_scipy_copy(
            3,
            3,
            ptr::null(), // 空数据指针
            5,
            ptr::null(),
            5,
            ptr::null(),
            4,
            &mut handle,
        );

        assert_eq!(result, FfiResult::NullPointer);
        assert!(handle.is_null());
    }
}

// =============================================================================
// View 模式 FFI 测试
// =============================================================================

#[test]
fn test_csr_f64_from_scipy_view() {
    // 注意：View 模式需要数据在整个使用期间保持有效
    let data = vec![1.0f64, 2.0, 3.0];
    let indices = vec![0i64, 1, 2];
    let indptr = vec![0i64, 1, 2, 3];

    unsafe {
        let handle = csr_f64_from_scipy_view(
            3,
            3,
            data.as_ptr(),
            indices.as_ptr(),
            indptr.as_ptr(),
        );

        assert!(!handle.is_null());
        assert_eq!(csr_f64_rows(handle), 3);
        assert_eq!(csr_f64_cols(handle), 3);
        assert_eq!(csr_f64_nnz(handle), 3);

        // View 模式不拥有数据，但仍需释放句柄结构
        csr_f64_free(handle);
    }
    // 数据在这里仍然有效
}

#[test]
fn test_csc_f64_from_scipy_view() {
    let data = vec![1.0f64, 2.0];
    let indices = vec![0i64, 1];
    let indptr = vec![0i64, 1, 2];

    unsafe {
        let handle = csc_f64_from_scipy_view(
            2,
            2,
            data.as_ptr(),
            indices.as_ptr(),
            indptr.as_ptr(),
        );

        assert!(!handle.is_null());
        assert_eq!(csc_f64_rows(handle), 2);
        assert_eq!(csc_f64_cols(handle), 2);

        csc_f64_free(handle);
    }
}

// =============================================================================
// 综合测试
// =============================================================================

#[test]
fn test_roundtrip_csr_csc_csr() {
    let data = vec![1.0f64, 2.0, 3.0, 4.0];
    let indices = vec![0i64, 1, 0, 2];
    let indptr = vec![0i64, 2, 4];

    unsafe {
        // CSR -> CSC -> CSR
        let csr1 = create_test_csr_f64(2, 3, &data, &indices, &indptr);

        let mut csc: CSCF64HandleMut = ptr::null_mut();
        assert_eq!(csc_f64_from_csr(csr1, &mut csc), FfiResult::Ok);

        let mut csr2: CSRF64HandleMut = ptr::null_mut();
        assert_eq!(csr_f64_from_csc(csc, &mut csr2), FfiResult::Ok);

        // 验证往返后维度和 nnz 一致
        assert_eq!(csr_f64_rows(csr2), csr_f64_rows(csr1));
        assert_eq!(csr_f64_cols(csr2), csr_f64_cols(csr1));
        assert_eq!(csr_f64_nnz(csr2), csr_f64_nnz(csr1));

        csr_f64_free(csr1);
        csc_f64_free(csc);
        csr_f64_free(csr2);
    }
}
