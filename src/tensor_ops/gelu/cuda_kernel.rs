use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::GeLUKernelOp {}

impl UnaryOpCudaKernel<f32> for super::GeLUKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/gelu.ptx"));
    const MODULE_NAME: &'static str = "gelu";
    const FWD_FN_NAME: &'static str = "gelu_forward_f32";
    const BWD_FN_NAME: &'static str = "gelu_backward_f32";
}
