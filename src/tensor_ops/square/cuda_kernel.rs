use super::SquareKernelOp;
#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for SquareKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/square.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(SquareKernelOp, f16, PTX, "square_fwd_f16", "square_bwd_f16");
#[cfg(feature = "f16")]
cuda_unary!(
    SquareKernelOp,
    AMP<f16>,
    PTX,
    "square_fwd_f16",
    "square_bwd_f16"
);
cuda_unary!(SquareKernelOp, f32, PTX, "square_fwd_f32", "square_bwd_f32");
cuda_unary!(SquareKernelOp, f64, PTX, "square_fwd_f64", "square_bwd_f64");
