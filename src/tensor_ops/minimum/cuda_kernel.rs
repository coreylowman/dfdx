use crate::tensor_ops::cuda_kernels::BinaryOpCudaKernel;

unsafe impl cudarc::device::AsKernelParam for super::MinimumKernelOp {}

impl BinaryOpCudaKernel for super::MinimumKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/minimum.ptx"));
    const MODULE_NAME: &'static str = "minimum";
    const FWD_FN_NAME: &'static str = "minimum_forward";
    const BWD_FN_NAME: &'static str = "minimum_backward";
}
