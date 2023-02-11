use crate::tensor_ops::cuda_kernels::{BinaryOpCudaKernel, UnaryOpCudaKernel};

unsafe impl cudarc::driver::AsKernelParam for super::ScalarSubKernelOp<f32> {}
unsafe impl cudarc::driver::AsKernelParam for super::ScalarSubKernelOp<f64> {}
unsafe impl cudarc::driver::AsKernelParam for super::BinarySubKernelOp {}

const SCALAR_PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/scalar_sub.ptx"));
const BINARY_PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/binary_sub.ptx"));

impl UnaryOpCudaKernel<f32> for super::ScalarSubKernelOp<f32> {
    const PTX_SRC: &'static str = SCALAR_PTX_SRC;
    const MODULE_NAME: &'static str = "scalar_sub_f32";
    const FWD_FN_NAME: &'static str = "scalar_sub_forward_f32";
    const BWD_FN_NAME: &'static str = "scalar_sub_backward_f32";
}

impl UnaryOpCudaKernel<f64> for super::ScalarSubKernelOp<f64> {
    const PTX_SRC: &'static str = SCALAR_PTX_SRC;
    const MODULE_NAME: &'static str = "scalar_sub_f64";
    const FWD_FN_NAME: &'static str = "scalar_sub_forward_f64";
    const BWD_FN_NAME: &'static str = "scalar_sub_backward_f64";
}

impl BinaryOpCudaKernel<f32> for super::BinarySubKernelOp {
    const PTX_SRC: &'static str = BINARY_PTX_SRC;
    const MODULE_NAME: &'static str = "binary_sub_f32";
    const FWD_FN_NAME: &'static str = "binary_sub_forward_f32";
    const BWD_FN_NAME: &'static str = "binary_sub_backward_f32";
}

impl BinaryOpCudaKernel<f64> for super::BinarySubKernelOp {
    const PTX_SRC: &'static str = BINARY_PTX_SRC;
    const MODULE_NAME: &'static str = "binary_sub_f64";
    const FWD_FN_NAME: &'static str = "binary_sub_forward_f64";
    const BWD_FN_NAME: &'static str = "binary_sub_backward_f64";
}
