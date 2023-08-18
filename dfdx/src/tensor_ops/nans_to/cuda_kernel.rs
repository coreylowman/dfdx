use super::NansToKernelOp as NansTo;
#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::cuda_unary;

#[cfg(feature = "f16")]
unsafe impl cudarc::driver::DeviceRepr for NansTo<f16> {}
#[cfg(feature = "f16")]
unsafe impl cudarc::driver::DeviceRepr for NansTo<AMP<f16>> {}
unsafe impl cudarc::driver::DeviceRepr for NansTo<f32> {}
unsafe impl cudarc::driver::DeviceRepr for NansTo<f64> {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/nans_to.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(NansTo<f16>, f16, PTX, "nans_to_fwd_f16", "nans_to_bwd_f16");
#[cfg(feature = "f16")]
cuda_unary!(
    NansTo<AMP<f16>>,
    AMP<f16>,
    PTX,
    "nans_to_fwd_f16",
    "nans_to_bwd_f16"
);
cuda_unary!(NansTo<f32>, f32, PTX, "nans_to_fwd_f32", "nans_to_bwd_f32");
cuda_unary!(NansTo<f64>, f64, PTX, "nans_to_fwd_f64", "nans_to_bwd_f64");
