use super::HardSigmoidKernelOp as HardSigmoid;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for HardSigmoid {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/hard_sigmoid.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(df(f(x)) HardSigmoid, half::f16, PTX, "hard_sigmoid_fwd_f16", "hard_sigmoid_bwd_f16");
cuda_unary!(df(f(x)) HardSigmoid, f32, PTX, "hard_sigmoid_fwd_f32", "hard_sigmoid_bwd_f32");
cuda_unary!(df(f(x)) HardSigmoid, f64, PTX, "hard_sigmoid_fwd_f64", "hard_sigmoid_bwd_f64");
