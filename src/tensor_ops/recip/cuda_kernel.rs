use super::RecipKernelOp;
#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for RecipKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/recip.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(df(f(x)) RecipKernelOp, f16, PTX, "recip_fwd_f16", "recip_bwd_f16");
#[cfg(feature = "f16")]
cuda_unary!(df(f(x)) RecipKernelOp, AMP<f16>, PTX, "recip_fwd_f16", "recip_bwd_f16");
cuda_unary!(df(f(x)) RecipKernelOp, f32, PTX, "recip_fwd_f32", "recip_bwd_f32");
cuda_unary!(df(f(x)) RecipKernelOp, f64, PTX, "recip_fwd_f64", "recip_bwd_f64");
