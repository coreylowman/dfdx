#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for super::ExpKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/exp.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(df(f(x)) super::ExpKernelOp, half::f16, PTX, "exp_fwd_f16", "exp_bwd_f16");
#[cfg(feature = "f16")]
cuda_unary!(df(f(x)) super::ExpKernelOp, AMP<f16>, PTX, "exp_fwd_f16", "exp_bwd_f16");
cuda_unary!(df(f(x)) super::ExpKernelOp, f32, PTX, "exp_fwd_f32", "exp_bwd_f32");
cuda_unary!(df(f(x)) super::ExpKernelOp, f64, PTX, "exp_fwd_f64", "exp_bwd_f64");
