use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &str = "TODO";

webgpu_unary!(super::NegateKernelOp, f32, WGSL, "negate_fwd_f32", "negate_bwd_f32");
