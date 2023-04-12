use super::AbsKernelOp;
use crate::tensor_ops::cuda_kernels::cuda_unary;

#[cfg(feature = "f16")]
use half::{bf16, f16};

unsafe impl cudarc::driver::DeviceRepr for AbsKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/abs.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(AbsKernelOp, f16, PTX, "abs_fwd_f16", "abs_bwd_f16");
// #[cfg(feature = "f16")]
// cuda_unary!(AbsKernelOp, bf16, PTX, "abs_fwd_bf16", "abs_bwd_bf16");
cuda_unary!(AbsKernelOp, f32, PTX, "abs_fwd_f32", "abs_bwd_f32");
cuda_unary!(AbsKernelOp, f64, PTX, "abs_fwd_f64", "abs_bwd_f64");
