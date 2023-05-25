use super::HardSwishKernelOp as HardSwish;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for HardSwish {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/hard_swish.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(HardSwish, half::f16, PTX, "hard_swish_fwd_f16", "hard_swish_bwd_f16");
cuda_unary!(HardSwish, f32, PTX, "hard_swish_fwd_f32", "hard_swish_bwd_f32");
cuda_unary!(HardSwish, f64, PTX, "hard_swish_fwd_f64", "hard_swish_bwd_f64");
