use crate::tensor_ops::cuda_kernels::{BinaryOpCudaKernel, UnaryOpCudaKernel};

unsafe impl cudarc::driver::AsKernelParam for super::ScalarAddKernelOp<f32> {}
unsafe impl cudarc::driver::AsKernelParam for super::BinaryAddKernelOp {}

impl UnaryOpCudaKernel<f32> for super::ScalarAddKernelOp<f32> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/scalar_add.ptx"));
    const MODULE_NAME: &'static str = "scalar_add";
    const FWD_FN_NAME: &'static str = "scalar_add_forward_f32";
    const BWD_FN_NAME: &'static str = "scalar_add_backward_f32";
}

impl BinaryOpCudaKernel<f32> for super::BinaryAddKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/binary_add.ptx"));
    const MODULE_NAME: &'static str = "binary_add";
    const FWD_FN_NAME: &'static str = "binary_add_forward_f32";
    const BWD_FN_NAME: &'static str = "binary_add_backward_f32";
}
