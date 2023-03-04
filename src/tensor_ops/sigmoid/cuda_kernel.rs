use super::SigmoidKernelOp as Sigmoid;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for Sigmoid {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/sigmoid.ptx"));

cuda_unary!(Sigmoid, f32, PTX, "sigmoid_fwd_f32", "sigmoid_bwd_f32");
cuda_unary!(Sigmoid, f64, PTX, "sigmoid_fwd_f64", "sigmoid_bwd_f64");
