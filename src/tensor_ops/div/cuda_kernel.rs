use crate::tensor_ops::cuda_kernels::{BinaryOpCudaKernel, UnaryOpCudaKernel};

unsafe impl cudarc::driver::AsKernelParam for super::ScalarDivKernelOp<f32> {}
unsafe impl cudarc::driver::AsKernelParam for super::ScalarDivKernelOp<f64> {}
unsafe impl cudarc::driver::AsKernelParam for super::BinaryDivKernelOp {}

impl UnaryOpCudaKernel<f32> for super::ScalarDivKernelOp<f32> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/scalar_div.ptx"));
    const MODULE_NAME: &'static str = "scalar_div";
    const FWD_FN_NAME: &'static str = "scalar_div_forward_f32";
    const BWD_FN_NAME: &'static str = "scalar_div_backward_f32";
}

impl UnaryOpCudaKernel<f64> for super::ScalarDivKernelOp<f64> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/scalar_div.ptx"));
    const MODULE_NAME: &'static str = "scalar_div";
    const FWD_FN_NAME: &'static str = "scalar_div_forward_f64";
    const BWD_FN_NAME: &'static str = "scalar_div_backward_f64";
}

impl BinaryOpCudaKernel<f32> for super::BinaryDivKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/binary_div.ptx"));
    const MODULE_NAME: &'static str = "binary_div";
    const FWD_FN_NAME: &'static str = "binary_div_forward_f32";
    const BWD_FN_NAME: &'static str = "binary_div_backward_f32";
}

impl BinaryOpCudaKernel<f64> for super::BinaryDivKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/binary_div.ptx"));
    const MODULE_NAME: &'static str = "binary_div";
    const FWD_FN_NAME: &'static str = "binary_div_forward_f64";
    const BWD_FN_NAME: &'static str = "binary_div_backward_f64";
}
