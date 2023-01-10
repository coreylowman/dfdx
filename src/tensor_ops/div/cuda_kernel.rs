use crate::tensor_ops::cuda_kernels::{BinaryOpCudaKernel, UnaryOpCudaKernel};

unsafe impl cudarc::device::AsKernelParam for super::ScalarDivKernelOp<f32> {}
unsafe impl cudarc::device::AsKernelParam for super::BinaryDivKernelOp {}

impl UnaryOpCudaKernel for super::ScalarDivKernelOp<f32> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/scalar_div.ptx"));
    const MODULE_NAME: &'static str = "scalar_div";
    const FWD_FN_NAME: &'static str = "scalar_div_forward";
    const BWD_FN_NAME: &'static str = "scalar_div_backward";
}

impl BinaryOpCudaKernel for super::BinaryDivKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/binary_div.ptx"));
    const MODULE_NAME: &'static str = "binary_div";
    const FWD_FN_NAME: &'static str = "binary_div_forward";
    const BWD_FN_NAME: &'static str = "binary_div_backward";
}
