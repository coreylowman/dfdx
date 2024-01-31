use super::SiLUKernelOp;
#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for SiLUKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/silu.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(SiLUKernelOp, f16, PTX, "silu_fwd_f16", "silu_bwd_f16");
#[cfg(feature = "f16")]
cuda_unary!(SiLUKernelOp, AMP<f16>, PTX, "silu_fwd_f16", "silu_bwd_f16");
cuda_unary!(SiLUKernelOp, f32, PTX, "silu_fwd_f32", "silu_bwd_f32");
cuda_unary!(SiLUKernelOp, f64, PTX, "silu_fwd_f64", "silu_bwd_f64");
