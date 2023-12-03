use super::TanhKernelOp;
use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &str = "TODO";

webgpu_unary!(TanhKernelOp, f32, WGSL, "tanh_fwd_f32", "tanh_bwd_f32");
