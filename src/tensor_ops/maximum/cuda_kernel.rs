use super::MaximumKernelOp as Max;
use crate::tensor_ops::cuda_kernels::cuda_binary;

unsafe impl cudarc::driver::AsKernelParam for Max {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/maximum.ptx"));

cuda_binary!(Max, f32, PTX, "maximum_fwd_f32", "maximum_bwd_f32");
cuda_binary!(Max, f64, PTX, "maximum_fwd_f64", "maximum_bwd_f64");
