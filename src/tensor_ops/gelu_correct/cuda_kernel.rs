use super::GeLUCorrectKernelOp;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for super::GeLUKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/gelu_correct.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(
    GeLUCorrectKernelOp,
    half::f16,
    PTX,
    "gelu_correct_fwd_f16",
    "gelu_correct_bwd_f16"
);
cuda_unary!(
    GeLUCorrectKernelOp,
    f32,
    PTX,
    "gelu_correct_fwd_f32",
    "gelu_correct_bwd_f32"
);
cuda_unary!(
    GeLUCorrectKernelOp,
    f64,
    PTX,
    "gelu_correct_fwd_f64",
    "gelu_correct_bwd_f64"
);
