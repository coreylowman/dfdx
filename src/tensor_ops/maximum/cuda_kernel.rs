use crate::tensor_ops::cuda_kernels::BinaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::MaximumKernelOp {}

impl BinaryOpCudaKernel for super::MaximumKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/maximum.ptx"));
    const MODULE_NAME: &'static str = "maximum";
    const FWD_FN_NAME: &'static str = "maximum_forward";
    const BWD_FN_NAME: &'static str = "maximum_backward";
}
