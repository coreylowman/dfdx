use super::TanhKernelOp;
use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &[u8] = b"TODO";

webgpu_unary!(TanhKernelOp, f32, WGSL, WGSL);
