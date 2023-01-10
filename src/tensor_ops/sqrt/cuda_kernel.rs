use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::device::AsKernelParam for super::SqrtKernelOp {}

impl UnaryOpCudaKernel for super::SqrtKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/sqrt.ptx"));
    const MODULE_NAME: &'static str = "sqrt";
    const FWD_FN_NAME: &'static str = "sqrt_forward";
    const BWD_FN_NAME: &'static str = "sqrt_backward";
}
