use super::NansToKernelOp as NansTo;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::AsKernelParam for NansTo<f32> {}
unsafe impl cudarc::driver::AsKernelParam for NansTo<f64> {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/nans_to.ptx"));

cuda_unary!(NansTo<f32>, f32, PTX, "nans_to_fwd_f32", "nans_to_bwd_f32");
cuda_unary!(NansTo<f64>, f64, PTX, "nans_to_fwd_f64", "nans_to_bwd_f64");
