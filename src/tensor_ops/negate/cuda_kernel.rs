use super::NegateKernelOp;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for NegateKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/negate.ptx"));

cuda_unary!(NegateKernelOp, f32, PTX, "negate_fwd_f32", "negate_bwd_f32");
cuda_unary!(NegateKernelOp, f64, PTX, "negate_fwd_f64", "negate_bwd_f64");
