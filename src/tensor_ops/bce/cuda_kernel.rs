use crate::tensor_ops::cuda_kernels::BinaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::BCEKernelOp {}

impl BinaryOpCudaKernel for super::BCEKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/bce.ptx"));
    const MODULE_NAME: &'static str = "bce";
    const FWD_FN_NAME: &'static str = "bce_forward";
    const BWD_FN_NAME: &'static str = "bce_backward";
}
