use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::NansToKernelOp<f32> {}
unsafe impl cudarc::driver::AsKernelParam for super::NansToKernelOp<f64> {}

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/nans_to.ptx"));

impl UnaryOpCudaKernel<f32> for super::NansToKernelOp<f32> {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "nans_to_f32";
    const FWD_FN_NAME: &'static str = "nans_to_forward_f32";
    const BWD_FN_NAME: &'static str = "nans_to_backward_f32";
}

impl UnaryOpCudaKernel<f64> for super::NansToKernelOp<f64> {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "nans_to_f64";
    const FWD_FN_NAME: &'static str = "nans_to_forward_f64";
    const BWD_FN_NAME: &'static str = "nans_to_backward_f64";
}
