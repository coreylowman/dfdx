use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::ReLUKernelOp {}

impl UnaryOpCudaKernel for super::ReLUKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/relu.ptx"));
    const MODULE_NAME: &'static str = "relu";
    const FWD_FN_NAME: &'static str = "relu_forward";
    const BWD_FN_NAME: &'static str = "relu_backward";
}
