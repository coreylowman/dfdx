use super::MaximumKernelOp as Max;
#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::cuda_binary;

unsafe impl cudarc::driver::DeviceRepr for Max {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/maximum.ptx"));

#[cfg(feature = "f16")]
cuda_binary!(
    Max,
    f16,
    PTX,
    "maximum_fwd_f16",
    "maximum_bwd_lhs_f16",
    "maximum_bwd_rhs_f16"
);
#[cfg(feature = "f16")]
cuda_binary!(
    Max,
    AMP<f16>,
    PTX,
    "maximum_fwd_f16",
    "maximum_bwd_lhs_f16",
    "maximum_bwd_rhs_f16"
);
cuda_binary!(
    Max,
    f32,
    PTX,
    "maximum_fwd_f32",
    "maximum_bwd_lhs_f32",
    "maximum_bwd_rhs_f32"
);
cuda_binary!(
    Max,
    f64,
    PTX,
    "maximum_fwd_f64",
    "maximum_bwd_lhs_f64",
    "maximum_bwd_rhs_f64"
);
