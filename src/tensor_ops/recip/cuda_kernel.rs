use super::RecipKernelOp;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for RecipKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/recip.ptx"));

cuda_unary!(df(f(x)) RecipKernelOp, f32, PTX, "recip_fwd_f32", "recip_bwd_f32");
cuda_unary!(df(f(x)) RecipKernelOp, f64, PTX, "recip_fwd_f64", "recip_bwd_f64");
