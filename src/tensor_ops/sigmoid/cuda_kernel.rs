use super::SigmoidKernelOp as Sigmoid;
#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for Sigmoid {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/sigmoid.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(df(f(x)) Sigmoid, f16, PTX, "sigmoid_fwd_f16", "sigmoid_bwd_f16");
#[cfg(feature = "f16")]
cuda_unary!(df(f(x)) Sigmoid, AMP<f16>, PTX, "sigmoid_fwd_f16", "sigmoid_bwd_f16");
cuda_unary!(df(f(x)) Sigmoid, f32, PTX, "sigmoid_fwd_f32", "sigmoid_bwd_f32");
cuda_unary!(df(f(x)) Sigmoid, f64, PTX, "sigmoid_fwd_f64", "sigmoid_bwd_f64");
