use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::SinKernelOp {}

impl UnaryOpCudaKernel for super::SinKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/sin.ptx"));
    const MODULE_NAME: &'static str = "sin";
    const FWD_FN_NAME: &'static str = "sin_forward";
    const BWD_FN_NAME: &'static str = "sin_backward";
}
