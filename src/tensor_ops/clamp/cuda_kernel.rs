use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::ClampKernelOp<f32> {}

impl UnaryOpCudaKernel<f32> for super::ClampKernelOp<f32> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/clamp.ptx"));
    const MODULE_NAME: &'static str = "clamp";
    const FWD_FN_NAME: &'static str = "clamp_forward_f32";
    const BWD_FN_NAME: &'static str = "clamp_backward_f32";
}
