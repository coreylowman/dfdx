use super::NansToKernelOp;
use crate::prelude::webgpu_kernels::webgpu_unary;

const WGSL: &str = "TODO";

webgpu_unary!(
    NansToKernelOp<f32>,
    f32,
    WGSL,
    "nans_to_fwd_f32",
    "nans_to_bwd_f32"
);
