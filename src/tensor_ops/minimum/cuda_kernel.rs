use super::MinimumKernelOp as Min;
use crate::tensor_ops::cuda_kernels::cuda_binary;

unsafe impl cudarc::driver::DeviceRepr for super::MinimumKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/minimum.ptx"));

#[cfg(feature = "f16")]
cuda_binary!(
    Min,
    half::f16,
    PTX,
    "minimum_fwd_f16",
    "minimum_bwd_lhs_f16",
    "minimum_bwd_rhs_f16"
);
cuda_binary!(
    Min,
    f32,
    PTX,
    "minimum_fwd_f32",
    "minimum_bwd_lhs_f32",
    "minimum_bwd_rhs_f32"
);
cuda_binary!(
    Min,
    f64,
    PTX,
    "minimum_fwd_f64",
    "minimum_bwd_lhs_f64",
    "minimum_bwd_rhs_f64"
);
