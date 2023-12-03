use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &str = "TODO";

webgpu_unary!(
    super::AccurateGeLUKernelOp,
    f32,
    WGSL,
    "gelu_fwd_f32",
    "gelu_bwd_f32"
);
