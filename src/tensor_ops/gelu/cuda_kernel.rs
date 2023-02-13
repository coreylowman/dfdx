use super::GeLUKernelOp;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::AsKernelParam for super::GeLUKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/gelu.ptx"));

cuda_unary!(GeLUKernelOp, f32, PTX, "gelu_fwd_f32", "gelu_bwd_f32");
cuda_unary!(GeLUKernelOp, f64, PTX, "gelu_fwd_f64", "gelu_bwd_f64");
