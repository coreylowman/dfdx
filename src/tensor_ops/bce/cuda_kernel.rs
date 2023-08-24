use super::BCEKernelOp;
#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::cuda_binary;

unsafe impl cudarc::driver::DeviceRepr for BCEKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/bce.ptx"));

#[cfg(feature = "f16")]
cuda_binary!(
    BCEKernelOp,
    AMP<f16>,
    PTX,
    "bce_fwd_f16",
    "bce_bwd_lhs_f16",
    "bce_bwd_rhs_f16"
);
#[cfg(feature = "f16")]
cuda_binary!(
    BCEKernelOp,
    f16,
    PTX,
    "bce_fwd_f16",
    "bce_bwd_lhs_f16",
    "bce_bwd_rhs_f16"
);
cuda_binary!(
    BCEKernelOp,
    f32,
    PTX,
    "bce_fwd_f32",
    "bce_bwd_lhs_f32",
    "bce_bwd_rhs_f32"
);
cuda_binary!(
    BCEKernelOp,
    f64,
    PTX,
    "bce_fwd_f64",
    "bce_bwd_lhs_f64",
    "bce_bwd_rhs_f64"
);
