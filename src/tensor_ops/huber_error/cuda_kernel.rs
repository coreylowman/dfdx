use crate::tensor_ops::cuda_kernels::BinaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::HuberErrorKernelOp<f32> {}
unsafe impl cudarc::driver::AsKernelParam for super::HuberErrorKernelOp<f64> {}

const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/huber_error.ptx"));

impl BinaryOpCudaKernel<f32> for super::HuberErrorKernelOp<f32> {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "huber_error_f32";
    const FWD_FN_NAME: &'static str = "huber_error_forward_f32";
    const BWD_FN_NAME: &'static str = "huber_error_backward_f32";
}

impl BinaryOpCudaKernel<f64> for super::HuberErrorKernelOp<f64> {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "huber_error_f64";
    const FWD_FN_NAME: &'static str = "huber_error_forward_f64";
    const BWD_FN_NAME: &'static str = "huber_error_backward_f64";
}
