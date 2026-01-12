//! BioSparse - 高性能数值计算核心库
//!
//! 提供用于生物信息学稀疏矩阵计算的基础设施。
//!
//! # 主要类型
//!
//! - [`Span`] - 类型化的连续内存视图，支持 Shared 和 View 两种模式
//! - [`SpanFlags`] - Span 的状态标志位
//! - [`CSR`] - 压缩稀疏行矩阵
//! - [`CSC`] - 压缩稀疏列矩阵
//!
//! # 模块
//!
//! - [`span`] - Span 相关类型和函数
//! - [`sparse`] - 稀疏矩阵（CSR/CSC）
//! - [`convert`] - 稀疏矩阵格式转换
//! - [`stack`] - 稀疏矩阵堆叠操作（vstack/hstack）
//! - [`tools`] - 编译器优化工具（assume、likely/unlikely 等）
//! - [`ffi`] - 外部函数接口（C ABI 句柄）
//! - `storage` - 引用计数的对齐内存管理（内部模块）

// 内部模块
pub(crate) mod storage;

// 公开模块
pub mod convert;
pub mod ffi;
pub mod slice;
pub mod span;
pub mod sparse;
pub mod stack;
pub mod tools;

// 重导出常用类型
pub use span::{Span, SpanFlags};
pub use sparse::{CSCf32, CSCf64, CSRf32, CSRf64, SparseIndex, CSC, CSR};
pub use storage::{AllocError, AllocResult, DEFAULT_ALIGN};

// 重导出 convert 类型
pub use convert::{AllocStrategy, ConvertError, DenseLayout};

// 重导出 stack 类型
pub use stack::StackError;

// 重导出优化工具
pub use tools::{likely, unlikely};
