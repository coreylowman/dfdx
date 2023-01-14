use crate::tensor_ops::cuda_kernels::BinaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::HuberErrorKernelOp<f32> {}

impl BinaryOpCudaKernel for super::HuberErrorKernelOp<f32> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/huber_error.ptx"));
    const MODULE_NAME: &'static str = "huber_error";
    const FWD_FN_NAME: &'static str = "huber_error_forward";
    const BWD_FN_NAME: &'static str = "huber_error_backward";
}
