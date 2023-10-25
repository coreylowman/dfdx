use super::FastGeLUKernelOp;
#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for super::FastGeLUKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/fast_gelu.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(
    FastGeLUKernelOp,
    f16,
    PTX,
    "fast_gelu_fwd_f16",
    "fast_gelu_bwd_f16"
);
#[cfg(feature = "f16")]
cuda_unary!(
    FastGeLUKernelOp,
    AMP<f16>,
    PTX,
    "fast_gelu_fwd_f16",
    "fast_gelu_bwd_f16"
);
cuda_unary!(
    FastGeLUKernelOp,
    f32,
    PTX,
    "fast_gelu_fwd_f32",
    "fast_gelu_bwd_f32"
);
cuda_unary!(
    FastGeLUKernelOp,
    f64,
    PTX,
    "fast_gelu_fwd_f64",
    "fast_gelu_bwd_f64"
);
