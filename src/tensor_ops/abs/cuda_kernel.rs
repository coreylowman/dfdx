use super::AbsKernelOp;
#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for AbsKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/abs.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(AbsKernelOp, AMP<f16>, PTX, "abs_fwd_f16", "abs_bwd_f16");
#[cfg(feature = "f16")]
cuda_unary!(AbsKernelOp, f16, PTX, "abs_fwd_f16", "abs_bwd_f16");
cuda_unary!(AbsKernelOp, f32, PTX, "abs_fwd_f32", "abs_bwd_f32");
cuda_unary!(AbsKernelOp, f64, PTX, "abs_fwd_f64", "abs_bwd_f64");
