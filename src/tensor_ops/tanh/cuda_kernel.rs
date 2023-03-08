use super::TanhKernelOp;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for TanhKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/tanh.ptx"));

cuda_unary!(TanhKernelOp, f32, PTX, "tanh_fwd_f32", "tanh_bwd_f32");
cuda_unary!(TanhKernelOp, f64, PTX, "tanh_fwd_f64", "tanh_bwd_f64");
