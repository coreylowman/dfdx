use super::HuberErrorKernelOp as HuberError;
use crate::tensor_ops::cuda_kernels::cuda_binary;

#[cfg(feature = "f16")]
unsafe impl cudarc::driver::DeviceRepr for HuberError<half::f16> {}
unsafe impl cudarc::driver::DeviceRepr for HuberError<f32> {}
unsafe impl cudarc::driver::DeviceRepr for HuberError<f64> {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/huber_error.ptx"));

#[cfg(feature = "f16")]
cuda_binary!(
    HuberError<half::f16>,
    half::f16,
    PTX,
    "huber_fwd_f16",
    "huber_bwd_lhs_f16",
    "huber_bwd_rhs_f16"
);
cuda_binary!(
    HuberError<f32>,
    f32,
    PTX,
    "huber_fwd_f32",
    "huber_bwd_lhs_f32",
    "huber_bwd_rhs_f32"
);
cuda_binary!(
    HuberError<f64>,
    f64,
    PTX,
    "huber_fwd_f64",
    "huber_bwd_lhs_f64",
    "huber_bwd_rhs_f64"
);
