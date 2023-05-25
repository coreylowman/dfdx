use super::ReLU6KernelOp;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for ReLU6KernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/relu6.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(
    ReLU6KernelOp,
    half::f16,
    PTX,
    "relu6_fwd_f16",
    "relu6_bwd_f16"
);
cuda_unary!(ReLU6KernelOp, f32, PTX, "relu6_fwd_f32", "relu6_bwd_f32");
cuda_unary!(ReLU6KernelOp, f64, PTX, "relu6_fwd_f64", "relu6_bwd_f64");
