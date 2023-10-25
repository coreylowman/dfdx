use super::SqrtKernelOp;
#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for SqrtKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/sqrt.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(df(f(x)) SqrtKernelOp, f16, PTX, "sqrt_fwd_f16", "sqrt_bwd_f16");
#[cfg(feature = "f16")]
cuda_unary!(df(f(x)) SqrtKernelOp, AMP<f16>, PTX, "sqrt_fwd_f16", "sqrt_bwd_f16");
cuda_unary!(df(f(x)) SqrtKernelOp, f32, PTX, "sqrt_fwd_f32", "sqrt_bwd_f32");
cuda_unary!(df(f(x)) SqrtKernelOp, f64, PTX, "sqrt_fwd_f64", "sqrt_bwd_f64");
