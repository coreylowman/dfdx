use super::{BinaryAddKernelOp as Binary, ScalarAddKernelOp as Scalar};
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

const SCALAR_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/scalar_add.ptx"));
const BINARY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/binary_add.ptx"));

#[cfg(feature = "f16")]
cuda_unary!(const_df() Scalar<AMP<f16>>, AMP<f16>, SCALAR_PTX, "sadd_fwd_f16", "sadd_bwd_f16");
#[cfg(feature = "f16")]
cuda_unary!(const_df() Scalar<f16>, f16, SCALAR_PTX, "sadd_fwd_f16", "sadd_bwd_f16");
cuda_unary!(const_df() Scalar<f32>, f32, SCALAR_PTX, "sadd_fwd_f32", "sadd_bwd_f32");
cuda_unary!(const_df() Scalar<f64>, f64, SCALAR_PTX, "sadd_fwd_f64", "sadd_bwd_f64");
#[cfg(feature = "f16")]
cuda_binary!(
    const_df() Binary,
    AMP<f16>,
    BINARY_PTX,
    "badd_fwd_f16",
    "badd_bwd_lhs_f16",
    "badd_bwd_rhs_f16"
);
#[cfg(feature = "f16")]
cuda_binary!(
    const_df() Binary,
    f16,
    BINARY_PTX,
    "badd_fwd_f16",
    "badd_bwd_lhs_f16",
    "badd_bwd_rhs_f16"
);
cuda_binary!(
    const_df() Binary,
    f32,
    BINARY_PTX,
    "badd_fwd_f32",
    "badd_bwd_lhs_f32",
    "badd_bwd_rhs_f32"
);
cuda_binary!(
    const_df() Binary,
    f64,
    BINARY_PTX,
    "badd_fwd_f64",
    "badd_bwd_lhs_f64",
    "badd_bwd_rhs_f64"
);
