use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::NansToKernelOp<f32> {}

impl UnaryOpCudaKernel<f32> for super::NansToKernelOp<f32> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/nans_to.ptx"));
    const MODULE_NAME: &'static str = "nans_to";
    const FWD_FN_NAME: &'static str = "nans_to_forward_f32";
    const BWD_FN_NAME: &'static str = "nans_to_backward_f32";
}
