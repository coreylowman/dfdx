use super::GeLUKernelOp;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for super::GeLUKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/gelu.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(GeLUKernelOp, half::f16, PTX, "gelu_fwd_f16", "gelu_bwd_f16");
cuda_unary!(GeLUKernelOp, f32, PTX, "gelu_fwd_f32", "gelu_bwd_f32");
cuda_unary!(GeLUKernelOp, f64, PTX, "gelu_fwd_f64", "gelu_bwd_f64");
