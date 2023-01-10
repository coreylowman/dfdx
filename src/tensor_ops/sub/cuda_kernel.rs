use crate::tensor_ops::cuda_kernels::{BinaryOpCudaKernel, UnaryOpCudaKernel};

unsafe impl cudarc::device::AsKernelParam for super::ScalarSubKernelOp<f32> {}
unsafe impl cudarc::device::AsKernelParam for super::BinarySubKernelOp {}

impl UnaryOpCudaKernel for super::ScalarSubKernelOp<f32> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/scalar_sub.ptx"));
    const MODULE_NAME: &'static str = "scalar_sub";
    const FWD_FN_NAME: &'static str = "scalar_sub_forward";
    const BWD_FN_NAME: &'static str = "scalar_sub_backward";
}

impl BinaryOpCudaKernel for super::BinarySubKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/binary_sub.ptx"));
    const MODULE_NAME: &'static str = "binary_sub";
    const FWD_FN_NAME: &'static str = "binary_sub_forward";
    const BWD_FN_NAME: &'static str = "binary_sub_backward";
}
