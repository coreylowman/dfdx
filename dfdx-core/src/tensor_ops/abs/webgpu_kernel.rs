use super::AbsKernelOp;
use crate::tensor_ops::webgpu_kernels::webgpu_unary;

const WGSL: &str = include_str!("abs.wgsl");

webgpu_unary!(AbsKernelOp, f32, WGSL, "abs_fwd_f32", "abs_bwd_f32");
