//! Rust 模块测试入口
//!
//! 包含所有 Rust 核心模块的集成测试
//!
//! 测试模块划分：
//! - `convert_test`: 稀疏矩阵格式转换测试
//! - `ffi_test`: FFI 接口测试
//! - `slice_test`: 切片操作测试
//! - `sparse_test`: CSR/CSC 稀疏矩阵基础测试
//! - `span_test`: Span 内存模型测试
//! - `stack_test`: 矩阵堆叠操作测试
//! - `e2e_test`: 端到端集成测试

pub mod convert_test;
pub mod e2e_test;
pub mod ffi_test;
pub mod slice_test;
pub mod span_test;
pub mod sparse_test;
pub mod stack_test;
