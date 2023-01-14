use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::NegateKernelOp {}

impl UnaryOpCudaKernel for super::NegateKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/negate.ptx"));
    const MODULE_NAME: &'static str = "negate";
    const FWD_FN_NAME: &'static str = "negate_forward";
    const BWD_FN_NAME: &'static str = "negate_backward";
}
