use super::SquareKernelOp;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for SquareKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/square.ptx"));

cuda_unary!(SquareKernelOp, f32, PTX, "square_fwd_f32", "square_bwd_f32");
cuda_unary!(SquareKernelOp, f64, PTX, "square_fwd_f64", "square_bwd_f64");
