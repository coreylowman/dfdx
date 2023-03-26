use crate::tensor_ops::cuda_kernels::{cuda_binary, cuda_unary};

use super::{LeakyReLUKernelOp, PReLUKernelOp};

unsafe impl cudarc::driver::DeviceRepr for LeakyReLUKernelOp<f32> {}
unsafe impl cudarc::driver::DeviceRepr for LeakyReLUKernelOp<f64> {}
unsafe impl cudarc::driver::DeviceRepr for PReLUKernelOp {}

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/prelu.ptx"));

cuda_unary!(
    LeakyReLUKernelOp<f32>,
    f32,
    PTX,
    "lrelu_fwd_f32",
    "lrelu_bwd_f32"
);
cuda_unary!(
    LeakyReLUKernelOp<f64>,
    f64,
    PTX,
    "lrelu_fwd_f64",
    "lrelu_bwd_f64"
);
cuda_binary!(
    PReLUKernelOp,
    f32,
    PTX,
    "prelu_fwd_f32",
    "prelu_bwd_lhs_f32",
    "prelu_bwd_rhs_f32"
);
cuda_binary!(
    PReLUKernelOp,
    f64,
    PTX,
    "prelu_fwd_f64",
    "prelu_bwd_lhs_f64",
    "prelu_bwd_rhs_f64"
);
