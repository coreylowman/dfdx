use super::NansToKernelOp;
use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &[u8] = b"TODO";

webgpu_unary!(NansToKernelOp<f32>, f32, WGSL, WGSL);
