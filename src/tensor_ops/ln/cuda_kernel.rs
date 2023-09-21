#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::cuda_unary;

unsafe impl cudarc::driver::DeviceRepr for super::LnKernelOp {}

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/ln.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(super::LnKernelOp, f16, PTX_SRC, "ln_fwd_f16", "ln_bwd_f16");
#[cfg(feature = "f16")]
cuda_unary!(
    super::LnKernelOp,
    AMP<f16>,
    PTX_SRC,
    "ln_fwd_f16",
    "ln_bwd_f16"
);
cuda_unary!(super::LnKernelOp, f32, PTX_SRC, "ln_fwd_f32", "ln_bwd_f32");
cuda_unary!(super::LnKernelOp, f64, PTX_SRC, "ln_fwd_f64", "ln_bwd_f64");
