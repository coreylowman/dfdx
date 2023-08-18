use super::{BinaryDivKernelOp as Binary, ScalarDivKernelOp as Scalar};
#[allow(unused_imports)]
use crate::dtypes::*;
use crate::tensor_ops::cuda_kernels::{cuda_binary, cuda_unary};

#[cfg(feature = "f16")]
unsafe impl cudarc::driver::DeviceRepr for Scalar<f16> {}
#[cfg(feature = "f16")]
unsafe impl cudarc::driver::DeviceRepr for Scalar<AMP<f16>> {}
unsafe impl cudarc::driver::DeviceRepr for Scalar<f32> {}
unsafe impl cudarc::driver::DeviceRepr for Scalar<f64> {}
unsafe impl cudarc::driver::DeviceRepr for Binary {}

const SCALAR_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/scalar_div.ptx"));
const BINARY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/binary_div.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(const_df() Scalar<f16>, f16, SCALAR_PTX, "sdiv_fwd_f16", "sdiv_bwd_f16");
#[cfg(feature = "f16")]
cuda_unary!(const_df() Scalar<AMP<f16>>, AMP<f16>, SCALAR_PTX, "sdiv_fwd_f16", "sdiv_bwd_f16");
cuda_unary!(const_df() Scalar<f32>, f32, SCALAR_PTX, "sdiv_fwd_f32", "sdiv_bwd_f32");
cuda_unary!(const_df() Scalar<f64>, f64, SCALAR_PTX, "sdiv_fwd_f64", "sdiv_bwd_f64");
#[cfg(feature = "f16")]
cuda_binary!(
    Binary,
    f16,
    BINARY_PTX,
    "bdiv_fwd_f16",
    "bdiv_bwd_lhs_f16",
    "bdiv_bwd_rhs_f16"
);
#[cfg(feature = "f16")]
cuda_binary!(
    Binary,
    AMP<f16>,
    BINARY_PTX,
    "bdiv_fwd_f16",
    "bdiv_bwd_lhs_f16",
    "bdiv_bwd_rhs_f16"
);
cuda_binary!(
    Binary,
    f32,
    BINARY_PTX,
    "bdiv_fwd_f32",
    "bdiv_bwd_lhs_f32",
    "bdiv_bwd_rhs_f32"
);
cuda_binary!(
    Binary,
    f64,
    BINARY_PTX,
    "bdiv_fwd_f64",
    "bdiv_bwd_lhs_f64",
    "bdiv_bwd_rhs_f64"
);
