//! Span 内存模型测试
//!
//! 测试内容：
//! - Span 内部结构和 flags
//! - Storage 生命周期和引用计数
//! - 线程安全和并发
//! - 内存对齐验证
//! - 移动语义测试
//! - 数据一致性测试
//! - 压力测试

use scl_core::span::{Span, SpanFlags};
use std::sync::Arc;
use std::thread;

// =============================================================================
// Span 内部结构验证测试
// =============================================================================

#[test]
fn test_span_internal_structure() {
    // 创建一个 Span 并验证其内部结构
    let span: Span<f64> = Span::alloc::<32>(100).unwrap();

    // 验证基本属性
    assert_eq!(span.len(), 100);
    assert!(span.has_storage());
    assert!(!span.is_view());
    assert!(span.is_mutable());
    assert!(span.is_aligned());

    // 验证对齐：指针应该是 32 字节对齐
    let ptr = span.as_ptr() as usize;
    assert_eq!(ptr % 32, 0, "Pointer should be 32-byte aligned");

    // 验证 flags
    let flags = span.flags();
    assert!(flags.contains(SpanFlags::ALIGNED));
    assert!(flags.contains(SpanFlags::MUTABLE));
    assert!(!flags.contains(SpanFlags::VIEW));
}

#[test]
fn test_span_view_mode_structure() {
    // 创建 View 模式的 Span
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let span = Span::from_slice(&data);

    // 验证 View 模式
    assert!(span.is_view());
    assert!(!span.has_storage());
    assert!(!span.is_mutable()); // 只读 View
    assert_eq!(span.storage_ref_count(), None); // View 没有 Storage

    // 验证指针指向原始数据
    assert_eq!(span.as_ptr(), data.as_ptr());
}

#[test]
fn test_span_mutable_view() {
    let mut data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let mut span = Span::from_slice_mut(&mut data);

    // 验证可变 View
    assert!(span.is_view());
    assert!(span.is_mutable());

    // 通过 Span 修改数据
    span[0] = 10.0;
    assert_eq!(data[0], 10.0); // 原始数据也被修改
}

#[test]
fn test_span_flags_operations() {
    // 测试 flags 的位操作
    let flags = SpanFlags::EMPTY;
    assert_eq!(flags.bits(), 0);

    // with 操作
    let flags = flags.with(SpanFlags::VIEW);
    assert!(flags.is_view());
    assert!(!flags.is_aligned());

    // 多个 flags
    let flags = flags.with(SpanFlags::ALIGNED).with(SpanFlags::MUTABLE);
    assert!(flags.is_view());
    assert!(flags.is_aligned());
    assert!(flags.is_mutable());

    // without 操作
    let flags = flags.without(SpanFlags::MUTABLE);
    assert!(!flags.is_mutable());
    assert!(flags.is_aligned()); // 其他 flags 不变

    // set 操作
    let flags = flags.set(SpanFlags::MUTABLE, true);
    assert!(flags.is_mutable());
    let flags = flags.set(SpanFlags::MUTABLE, false);
    assert!(!flags.is_mutable());

    // BitOr 操作
    let flags = SpanFlags::VIEW | SpanFlags::ALIGNED;
    assert!(flags.is_view());
    assert!(flags.is_aligned());
}

// =============================================================================
// Storage 生命周期和引用计数测试
// =============================================================================

#[test]
fn test_storage_reference_counting_basic() {
    // 单个 Span，引用计数应为 1
    let span1: Span<f64> = Span::alloc::<32>(100).unwrap();
    assert_eq!(span1.storage_ref_count(), Some(1));

    // clone 后引用计数增加
    let span2 = span1.clone();
    assert_eq!(span1.storage_ref_count(), Some(2));
    assert_eq!(span2.storage_ref_count(), Some(2));

    // 再次 clone
    let span3 = span1.clone();
    assert_eq!(span1.storage_ref_count(), Some(3));

    // drop 后引用计数减少
    drop(span3);
    assert_eq!(span1.storage_ref_count(), Some(2));

    drop(span2);
    assert_eq!(span1.storage_ref_count(), Some(1));
}

#[test]
fn test_storage_shared_among_alloc_n() {
    // alloc_n 创建的多个 Span 共享同一个 Storage
    let [s1, s2, s3] = Span::<f64>::alloc_n::<32, 3>([100, 200, 300]).unwrap();

    // 所有 Span 引用计数应为 3
    assert_eq!(s1.storage_ref_count(), Some(3));
    assert_eq!(s2.storage_ref_count(), Some(3));
    assert_eq!(s3.storage_ref_count(), Some(3));

    // 验证数据不重叠
    let ptr1 = s1.as_ptr() as usize;
    let ptr2 = s2.as_ptr() as usize;
    let ptr3 = s3.as_ptr() as usize;

    // s1 结束后应该是 s2 开始（考虑对齐）
    assert!(ptr2 >= ptr1 + s1.len() * std::mem::size_of::<f64>());
    assert!(ptr3 >= ptr2 + s2.len() * std::mem::size_of::<f64>());

    // drop 一个后引用计数减少
    drop(s3);
    assert_eq!(s1.storage_ref_count(), Some(2));
}

#[test]
fn test_storage_shared_among_alloc_slices() {
    // alloc_slices 创建的多个 Span 共享同一个 Storage
    let spans = Span::<f64>::alloc_slices::<32>(&[50, 100, 150, 200]).unwrap();

    assert_eq!(spans.len(), 4);

    // 所有 Span 引用计数应为 4
    for span in &spans {
        assert_eq!(span.storage_ref_count(), Some(4));
    }

    // 验证各自长度
    assert_eq!(spans[0].len(), 50);
    assert_eq!(spans[1].len(), 100);
    assert_eq!(spans[2].len(), 150);
    assert_eq!(spans[3].len(), 200);
}

#[test]
fn test_subspan_shares_storage() {
    let span: Span<f64> = Span::alloc::<32>(100).unwrap();
    assert_eq!(span.storage_ref_count(), Some(1));

    // 创建 subspan
    let sub1 = span.subspan(0, 50).unwrap();
    assert_eq!(span.storage_ref_count(), Some(2));
    assert_eq!(sub1.storage_ref_count(), Some(2));

    // 再创建一个 subspan
    let sub2 = span.subspan(50, 50).unwrap();
    assert_eq!(span.storage_ref_count(), Some(3));

    // 验证 subspan 指向正确位置
    let base_ptr = span.as_ptr() as usize;
    let sub1_ptr = sub1.as_ptr() as usize;
    let sub2_ptr = sub2.as_ptr() as usize;

    assert_eq!(sub1_ptr, base_ptr);
    assert_eq!(sub2_ptr, base_ptr + 50 * std::mem::size_of::<f64>());
}

#[test]
fn test_take_ownership_no_ref_count_change() {
    // 创建共享的 Span
    let [mut s1, s2] = Span::<f64>::alloc_n::<32, 2>([100, 100]).unwrap();
    assert_eq!(s1.storage_ref_count(), Some(2));

    // take_ownership 不应该改变引用计数
    let s3 = unsafe { s1.take_ownership() };

    // s1 变成空 View
    assert!(s1.is_empty());
    assert!(s1.is_view());
    assert!(!s1.has_storage());

    // s3 持有原来的所有权
    assert_eq!(s3.len(), 100);
    assert!(s3.has_storage());

    // 引用计数仍然是 2（s2 和 s3）
    assert_eq!(s2.storage_ref_count(), Some(2));
    assert_eq!(s3.storage_ref_count(), Some(2));
}

#[test]
fn test_deep_copy_creates_independent_storage() {
    // 创建原始 Span
    let mut span1: Span<i32> = Span::alloc::<32>(10).unwrap();
    for i in 0..10 {
        span1[i] = i as i32;
    }

    // 深拷贝
    let span2 = span1.deep_copy::<32>().unwrap();

    // 两个 Span 应该有独立的 Storage
    assert_eq!(span1.storage_ref_count(), Some(1));
    assert_eq!(span2.storage_ref_count(), Some(1));

    // 指针不同
    assert_ne!(span1.as_ptr(), span2.as_ptr());

    // 数据相同
    assert_eq!(span1.as_slice(), span2.as_slice());

    // 修改一个不影响另一个
    let mut span2 = span2;
    span2[0] = 999;
    assert_eq!(span1[0], 0);
    assert_eq!(span2[0], 999);
}

// =============================================================================
// 线程安全和并发测试
// =============================================================================

#[test]
fn test_span_send_to_thread() {
    // Span 可以发送到其他线程
    let mut span: Span<i32> = Span::alloc::<32>(100).unwrap();
    for i in 0..100 {
        span[i] = i as i32;
    }

    // 发送到新线程并读取
    let handle = thread::spawn(move || {
        let sum: i32 = span.iter().sum();
        sum
    });

    let result = handle.join().unwrap();
    assert_eq!(result, (0..100).sum::<i32>());
}

#[test]
fn test_span_shared_read_across_threads() {
    // 创建数据
    let mut span: Span<i32> = Span::alloc::<32>(1000).unwrap();
    for i in 0..1000 {
        span[i] = i as i32;
    }

    // 用 Arc 包装以在线程间共享
    let span = Arc::new(span);

    // 在多个线程中并发读取
    let mut handles = vec![];

    for _ in 0..4 {
        let span_clone = Arc::clone(&span);
        let handle = thread::spawn(move || {
            let sum: i32 = span_clone.iter().sum();
            sum
        });
        handles.push(handle);
    }

    // 收集结果
    let expected: i32 = (0..1000).sum();
    for handle in handles {
        let result = handle.join().unwrap();
        assert_eq!(result, expected);
    }
}

#[test]
fn test_storage_drop_in_different_thread() {
    // 创建共享 Storage 的多个 Span
    let [s1, s2, s3] = Span::<f64>::alloc_n::<32, 3>([100, 100, 100]).unwrap();
    assert_eq!(s1.storage_ref_count(), Some(3));

    // 在不同线程中 drop
    let h1 = thread::spawn(move || {
        drop(s1);
    });

    let h2 = thread::spawn(move || {
        drop(s2);
    });

    h1.join().unwrap();
    h2.join().unwrap();

    // s3 应该仍然有效
    assert_eq!(s3.storage_ref_count(), Some(1));
    assert_eq!(s3.len(), 100);
}

#[test]
fn test_concurrent_subspan_creation() {
    // 创建大 Span
    let span: Span<f64> = Span::alloc::<32>(1000).unwrap();
    let span = Arc::new(span);

    // 在多个线程中创建 subspan
    let mut handles = vec![];

    for i in 0..10 {
        let span_clone = Arc::clone(&span);
        let handle = thread::spawn(move || {
            let sub = unsafe { (*Arc::as_ptr(&span_clone)).subspan(i * 100, 100) };
            sub.unwrap().len()
        });
        handles.push(handle);
    }

    for handle in handles {
        let len = handle.join().unwrap();
        assert_eq!(len, 100);
    }
}

// =============================================================================
// 内存对齐验证测试
// =============================================================================

#[test]
fn test_alignment_verification_16() {
    for _ in 0..10 {
        let span: Span<f64> = Span::alloc::<16>(100).unwrap();
        let ptr = span.as_ptr() as usize;
        assert_eq!(ptr % 16, 0, "16-byte alignment failed");
    }
}

#[test]
fn test_alignment_verification_32() {
    for _ in 0..10 {
        let span: Span<f64> = Span::alloc::<32>(100).unwrap();
        let ptr = span.as_ptr() as usize;
        assert_eq!(ptr % 32, 0, "32-byte alignment failed");
    }
}

#[test]
fn test_alignment_verification_64() {
    for _ in 0..10 {
        let span: Span<f64> = Span::alloc::<64>(100).unwrap();
        let ptr = span.as_ptr() as usize;
        assert_eq!(ptr % 64, 0, "64-byte alignment failed");
    }
}

#[test]
fn test_alloc_n_alignment() {
    // alloc_n 中的每个 Span 都应该对齐
    let spans = Span::<f64>::alloc_slices::<32>(&[10, 20, 30, 40, 50]).unwrap();

    for (i, span) in spans.iter().enumerate() {
        let ptr = span.as_ptr() as usize;
        assert_eq!(ptr % 32, 0, "Span {} at {:x} is not 32-byte aligned", i, ptr);
    }
}

#[test]
fn test_different_alignments() {
    // 测试不同对齐值
    let span_16: Span<f64> = Span::alloc::<16>(100).unwrap();
    assert_eq!(span_16.as_ptr() as usize % 16, 0);

    let span_32: Span<f64> = Span::alloc::<32>(100).unwrap();
    assert_eq!(span_32.as_ptr() as usize % 32, 0);

    let span_64: Span<f64> = Span::alloc::<64>(100).unwrap();
    assert_eq!(span_64.as_ptr() as usize % 64, 0);
}

#[test]
fn test_small_allocations() {
    // 测试小分配
    let span_1: Span<f64> = Span::alloc::<32>(1).unwrap();
    assert_eq!(span_1.len(), 1);

    let span_2: Span<f64> = Span::alloc::<32>(2).unwrap();
    assert_eq!(span_2.len(), 2);
}

#[test]
fn test_span_copy_from_small() {
    // 测试从小切片复制
    let data = [1.0f64];
    let span: Span<f64> = Span::copy_from::<32>(&data).unwrap();
    assert_eq!(span.len(), 1);
    assert_eq!(span[0], 1.0);

    let data2 = [1.0f64, 2.0];
    let span2: Span<f64> = Span::copy_from::<32>(&data2).unwrap();
    assert_eq!(span2.len(), 2);
}

// =============================================================================
// 边界情况 - 移动语义测试
// =============================================================================

#[test]
fn test_take_from_empty_span() {
    let mut span: Span<f64> = Span::default();

    // 空 Span 的 take 应该返回 None
    assert!(span.take().is_none());
    assert!(span.is_empty());
}

#[test]
fn test_take_replace() {
    let mut span: Span<i32> = Span::alloc::<32>(50).unwrap();
    let original_len = span.len();

    let taken = span.take_replace();

    // 原 span 被替换为默认值
    assert!(span.is_empty());
    assert!(span.is_view());
    assert!(!span.has_storage());

    // taken 持有原来的值
    assert_eq!(taken.len(), original_len);
    assert!(taken.has_storage());
}

#[test]
fn test_drain_all() {
    let mut src = Span::<i32>::alloc_slices::<32>(&[10, 20, 30]).unwrap();
    let initial_ref_count = src[0].storage_ref_count();

    let mut dst = Vec::new();
    Span::drain_all(&mut src, &mut dst);

    assert!(src.is_empty());
    assert_eq!(dst.len(), 3);
    assert_eq!(dst[0].len(), 10);
    assert_eq!(dst[1].len(), 20);
    assert_eq!(dst[2].len(), 30);

    // 引用计数不变
    assert_eq!(dst[0].storage_ref_count(), initial_ref_count);
}

#[test]
fn test_take_at_checked() {
    let mut spans = Span::<i32>::alloc_slices::<32>(&[10, 20, 30]).unwrap();

    // 有效索引
    let taken = Span::take_at_checked(&mut spans, 1);
    assert!(taken.is_some());
    assert_eq!(taken.unwrap().len(), 20);
    assert!(spans[1].is_empty());

    // 无效索引
    let taken = Span::take_at_checked(&mut spans, 10);
    assert!(taken.is_none());
}

// =============================================================================
// 数据一致性测试
// =============================================================================

#[test]
fn test_copy_from_data_integrity() {
    // 测试各种类型的数据完整性
    let data_f64: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
    let span = Span::copy_from::<32>(&data_f64).unwrap();
    assert_eq!(span.as_slice(), &data_f64[..]);

    let data_i32: Vec<i32> = (-500..500).collect();
    let span = Span::copy_from::<32>(&data_i32).unwrap();
    assert_eq!(span.as_slice(), &data_i32[..]);

    let data_u8: Vec<u8> = (0..=255).collect();
    let span = Span::copy_from::<32>(&data_u8).unwrap();
    assert_eq!(span.as_slice(), &data_u8[..]);
}

#[test]
fn test_fill_and_verify() {
    let mut span: Span<f64> = Span::alloc::<32>(1000).unwrap();

    // 填充特定值
    span.fill(std::f64::consts::PI);

    // 验证所有元素
    for &val in span.iter() {
        assert_eq!(val, std::f64::consts::PI);
    }
}

#[test]
fn test_copy_from_span() {
    let src: Vec<i32> = (0..100).collect();
    let src_span = Span::copy_from::<32>(&src).unwrap();

    let mut dst_span: Span<i32> = Span::alloc::<32>(200).unwrap();
    dst_span.copy_from_span(&src_span);

    // 前 100 个元素应该与 src 相同
    assert_eq!(&dst_span.as_slice()[..100], &src[..]);
}

// =============================================================================
// 压力测试
// =============================================================================

#[test]
fn test_many_small_allocations() {
    // 大量小分配
    let mut spans: Vec<Span<f64>> = Vec::with_capacity(1000);

    for _ in 0..1000 {
        let span: Span<f64> = Span::alloc::<32>(10).unwrap();
        spans.push(span);
    }

    // 验证所有 Span 有效
    for (i, span) in spans.iter().enumerate() {
        assert_eq!(span.len(), 10, "Span {} has wrong length", i);
        assert!(span.has_storage());
    }
}

#[test]
fn test_large_allocation() {
    // 大分配
    let span: Span<f64> = Span::alloc::<32>(1_000_000).unwrap();

    assert_eq!(span.len(), 1_000_000);
    assert!(span.is_aligned());

    // 验证可以访问首尾
    let ptr = span.as_ptr();
    unsafe {
        let _ = *ptr; // 读取第一个元素
        let _ = *ptr.add(999_999); // 读取最后一个元素
    }
}

#[test]
fn test_repeated_clone_and_drop() {
    let original: Span<f64> = Span::alloc::<32>(100).unwrap();

    // 反复 clone 和 drop
    for _ in 0..100 {
        let cloned = original.clone();
        assert_eq!(original.storage_ref_count(), Some(2));
        drop(cloned);
        assert_eq!(original.storage_ref_count(), Some(1));
    }
}

#[test]
fn test_memory_alignment_general() {
    // 测试内存对齐
    let span: Span<f64> = Span::alloc::<32>(100).unwrap();

    // 验证对齐
    assert!(span.is_aligned());
    assert_eq!(span.as_ptr() as usize % 32, 0);
}
