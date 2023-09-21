use super::SigmoidrKernelOp as Sigmoidr;
#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for Sigmoidr {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/sigmoidr.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(df(f(x)) Sigmoidr, f16, PTX, "sigmoidr_fwd_f16", "sigmoidr_bwd_f16");
#[cfg(feature = "f16")]
cuda_unary!(df(f(x)) Sigmoidr, AMP<f16>, PTX, "sigmoidr_fwd_f16", "sigmoidr_bwd_f16");
cuda_unary!(df(f(x)) Sigmoidr, f32, PTX, "sigmoidr_fwd_f32", "sigmoidr_bwd_f32");
cuda_unary!(df(f(x)) Sigmoidr, f64, PTX, "sigmoidr_fwd_f64", "sigmoidr_bwd_f64");
