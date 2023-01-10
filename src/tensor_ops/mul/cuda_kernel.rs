use crate::tensor_ops::cuda_kernels::{BinaryOpCudaKernel, UnaryOpCudaKernel};

unsafe impl cudarc::device::AsKernelParam for super::ScalarMulKernelOp<f32> {}
unsafe impl cudarc::device::AsKernelParam for super::BinaryMulKernelOp {}

impl UnaryOpCudaKernel for super::ScalarMulKernelOp<f32> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/scalar_mul.ptx"));
    const MODULE_NAME: &'static str = "scalar_mul";
    const FWD_FN_NAME: &'static str = "scalar_mul_forward";
    const BWD_FN_NAME: &'static str = "scalar_mul_backward";
}

impl BinaryOpCudaKernel for super::BinaryMulKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/binary_mul.ptx"));
    const MODULE_NAME: &'static str = "binary_mul";
    const FWD_FN_NAME: &'static str = "binary_mul_forward";
    const BWD_FN_NAME: &'static str = "binary_mul_backward";
}
