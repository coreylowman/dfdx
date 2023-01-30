use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::SinKernelOp {}

impl UnaryOpCudaKernel<f32> for super::SinKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/sin.ptx"));
    const MODULE_NAME: &'static str = "sin";
    const FWD_FN_NAME: &'static str = "sin_forward_f32";
    const BWD_FN_NAME: &'static str = "sin_backward_f32";
}

impl UnaryOpCudaKernel<f64> for super::SinKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/sin.ptx"));
    const MODULE_NAME: &'static str = "sin";
    const FWD_FN_NAME: &'static str = "sin_forward_f64";
    const BWD_FN_NAME: &'static str = "sin_backward_f64";
}
