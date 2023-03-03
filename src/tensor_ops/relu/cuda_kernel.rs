use super::ReLUKernelOp;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for ReLUKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/relu.ptx"));

cuda_unary!(ReLUKernelOp, f32, PTX, "relu_fwd_f32", "relu_bwd_f32");
cuda_unary!(ReLUKernelOp, f64, PTX, "relu_fwd_f64", "relu_bwd_f64");
