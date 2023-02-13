use super::BCEKernelOp;
use crate::tensor_ops::cuda_kernels::cuda_binary;

unsafe impl cudarc::driver::AsKernelParam for BCEKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/bce.ptx"));

cuda_binary!(BCEKernelOp, f32, PTX, "bce_fwd_f32", "bce_bwd_f32");
cuda_binary!(BCEKernelOp, f64, PTX, "bce_fwd_f64", "bce_bwd_f64");
