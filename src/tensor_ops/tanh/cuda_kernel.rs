use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::TanhKernelOp {}

impl UnaryOpCudaKernel for super::TanhKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/tanh.ptx"));
    const MODULE_NAME: &'static str = "tanh";
    const FWD_FN_NAME: &'static str = "tanh_forward";
    const BWD_FN_NAME: &'static str = "tanh_backward";
}
