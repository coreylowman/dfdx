use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::ExpKernelOp {}

impl UnaryOpCudaKernel for super::ExpKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/exp.ptx"));
    const MODULE_NAME: &'static str = "exp";
    const FWD_FN_NAME: &'static str = "exp_forward";
    const BWD_FN_NAME: &'static str = "exp_backward";
}
