use crate::tensor_ops::cuda_kernels::{BinaryOpCudaKernel, UnaryOpCudaKernel};

unsafe impl cudarc::driver::AsKernelParam for super::ScalarSubKernelOp<f32> {}
unsafe impl cudarc::driver::AsKernelParam for super::BinarySubKernelOp {}

impl UnaryOpCudaKernel<f32> for super::ScalarSubKernelOp<f32> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/scalar_sub.ptx"));
    const MODULE_NAME: &'static str = "scalar_sub";
    const FWD_FN_NAME: &'static str = "scalar_sub_forward_f32";
    const BWD_FN_NAME: &'static str = "scalar_sub_backward_f32";
}

impl BinaryOpCudaKernel<f32> for super::BinarySubKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/binary_sub.ptx"));
    const MODULE_NAME: &'static str = "binary_sub";
    const FWD_FN_NAME: &'static str = "binary_sub_forward_f32";
    const BWD_FN_NAME: &'static str = "binary_sub_backward_f32";
}
