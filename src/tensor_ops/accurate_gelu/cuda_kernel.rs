use super::AccurateGeLUKernelOp;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for super::AccurateGeLUKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/accurate_gelu.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(
    AccurateGeLUKernelOp,
    half::f16,
    PTX,
    "accurate_gelu_fwd_f16",
    "accurate_gelu_bwd_f16"
);
cuda_unary!(
    AccurateGeLUKernelOp,
    f32,
    PTX,
    "accurate_gelu_fwd_f32",
    "accurate_gelu_bwd_f32"
);
cuda_unary!(
    AccurateGeLUKernelOp,
    f64,
    PTX,
    "accurate_gelu_fwd_f64",
    "accurate_gelu_bwd_f64"
);
