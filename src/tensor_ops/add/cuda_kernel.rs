use crate::tensor_ops::cuda_kernels::{BinaryOpCudaKernel, UnaryOpCudaKernel};

unsafe impl cudarc::driver::AsKernelParam for super::ScalarAddKernelOp<f32> {}
unsafe impl cudarc::driver::AsKernelParam for super::ScalarAddKernelOp<f64> {}
unsafe impl cudarc::driver::AsKernelParam for super::BinaryAddKernelOp {}

const SCALAR_PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/scalar_add.ptx"));
const BINARY_PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/binary_add.ptx"));

impl UnaryOpCudaKernel<f32> for super::ScalarAddKernelOp<f32> {
    const PTX_SRC: &'static str = SCALAR_PTX_SRC;
    const MODULE_NAME: &'static str = "scalar_add_f32";
    const FWD_FN_NAME: &'static str = "scalar_add_forward_f32";
    const BWD_FN_NAME: &'static str = "scalar_add_backward_f32";
}

impl UnaryOpCudaKernel<f64> for super::ScalarAddKernelOp<f64> {
    const PTX_SRC: &'static str = SCALAR_PTX_SRC;
    const MODULE_NAME: &'static str = "scalar_add_f64";
    const FWD_FN_NAME: &'static str = "scalar_add_forward_f64";
    const BWD_FN_NAME: &'static str = "scalar_add_backward_f64";
}

impl BinaryOpCudaKernel<f32> for super::BinaryAddKernelOp {
    const PTX_SRC: &'static str = BINARY_PTX_SRC;
    const MODULE_NAME: &'static str = "binary_add_f32";
    const FWD_FN_NAME: &'static str = "binary_add_forward_f32";
    const BWD_FN_NAME: &'static str = "binary_add_backward_f32";
}

impl BinaryOpCudaKernel<f64> for super::BinaryAddKernelOp {
    const PTX_SRC: &'static str = BINARY_PTX_SRC;
    const MODULE_NAME: &'static str = "binary_add_f64";
    const FWD_FN_NAME: &'static str = "binary_add_forward_f64";
    const BWD_FN_NAME: &'static str = "binary_add_backward_f64";
}
