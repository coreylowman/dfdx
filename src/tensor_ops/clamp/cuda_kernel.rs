use super::ClampKernelOp;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::AsKernelParam for ClampKernelOp<f32> {}
unsafe impl cudarc::driver::AsKernelParam for ClampKernelOp<f64> {}

const P: &str = include_str!(concat!(env!("OUT_DIR"), "/clamp.ptx"));

cuda_unary!(ClampKernelOp<f32>, f32, P, "clamp_fwd_f32", "clamp_bwd_f32");
cuda_unary!(ClampKernelOp<f64>, f64, P, "clamp_fwd_f64", "clamp_bwd_f64");
