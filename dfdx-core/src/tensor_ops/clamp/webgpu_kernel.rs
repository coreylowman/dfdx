use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &[u8] = b"TODO";

webgpu_unary!(super::ClampKernelOp<f32>, f32, WGSL, WGSL);
