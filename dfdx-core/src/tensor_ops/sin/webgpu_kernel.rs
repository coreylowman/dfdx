use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &str = "TODO";

webgpu_unary!(super::SinKernelOp, f32, WGSL, "sin_fwd_f32", "sin_bwd_f32");
