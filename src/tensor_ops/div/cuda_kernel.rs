use super::{BinaryDivKernelOp as Binary, ScalarDivKernelOp as Scalar};
use crate::tensor_ops::cuda_kernels::{cuda_binary, cuda_unary};

unsafe impl cudarc::driver::AsKernelParam for Scalar<f32> {}
unsafe impl cudarc::driver::AsKernelParam for Scalar<f64> {}
unsafe impl cudarc::driver::AsKernelParam for Binary {}

const SCALAR_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/scalar_div.ptx"));
const BINARY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/binary_div.ptx"));

cuda_unary!(Scalar<f32>, f32, SCALAR_PTX, "sdiv_fwd_f32", "sdiv_bwd_f32");
cuda_unary!(Scalar<f64>, f64, SCALAR_PTX, "sdiv_fwd_f64", "sdiv_bwd_f64");
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
