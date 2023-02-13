use super::HuberErrorKernelOp as HuberError;
use crate::tensor_ops::cuda_kernels::cuda_binary;

unsafe impl cudarc::driver::AsKernelParam for HuberError<f32> {}
unsafe impl cudarc::driver::AsKernelParam for HuberError<f64> {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/huber_error.ptx"));

cuda_binary!(HuberError<f32>, f32, PTX, "huber_fwd_f32", "huber_bwd_f32");
cuda_binary!(HuberError<f64>, f64, PTX, "huber_fwd_f64", "huber_bwd_f64");
