use super::ReLUKernelOp;
#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for ReLUKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/relu.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(ReLUKernelOp, f16, PTX, "relu_fwd_f16", "relu_bwd_f16");
#[cfg(feature = "f16")]
cuda_unary!(ReLUKernelOp, AMP<f16>, PTX, "relu_fwd_f16", "relu_bwd_f16");
cuda_unary!(ReLUKernelOp, f32, PTX, "relu_fwd_f32", "relu_bwd_f32");
cuda_unary!(ReLUKernelOp, f64, PTX, "relu_fwd_f64", "relu_bwd_f64");
