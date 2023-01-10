use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::device::AsKernelParam for super::LnKernelOp {}

impl UnaryOpCudaKernel for super::LnKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/ln.ptx"));
    const MODULE_NAME: &'static str = "ln";
    const FWD_FN_NAME: &'static str = "ln_forward";
    const BWD_FN_NAME: &'static str = "ln_backward";
}
