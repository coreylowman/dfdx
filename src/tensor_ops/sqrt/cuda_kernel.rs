use super::SqrtKernelOp;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for SqrtKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/sqrt.ptx"));

cuda_unary!(df(f(x)) SqrtKernelOp, f32, PTX, "sqrt_fwd_f32", "sqrt_bwd_f32");
cuda_unary!(df(f(x)) SqrtKernelOp, f64, PTX, "sqrt_fwd_f64", "sqrt_bwd_f64");
