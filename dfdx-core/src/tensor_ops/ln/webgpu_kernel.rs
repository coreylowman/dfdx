use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &str = "TODO";

webgpu_unary!(super::LnKernelOp, f32, WGSL, "ln_fwd_f32", "ln_bwd_f32");
