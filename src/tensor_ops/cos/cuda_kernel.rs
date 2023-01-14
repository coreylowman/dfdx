use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::CosKernelOp {}

impl UnaryOpCudaKernel for super::CosKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/cos.ptx"));
    const MODULE_NAME: &'static str = "cos";
    const FWD_FN_NAME: &'static str = "cos_forward";
    const BWD_FN_NAME: &'static str = "cos_backward";
}
