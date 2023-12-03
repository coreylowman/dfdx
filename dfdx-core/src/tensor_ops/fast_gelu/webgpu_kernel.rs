use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &str = "TODO";

webgpu_unary!(
    super::FastGeLUKernelOp,
    f32,
    WGSL,
    "sigmoid_fwd_f32",
    "sigmoid_bwd_f32"
);
