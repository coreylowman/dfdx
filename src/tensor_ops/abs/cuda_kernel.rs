use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::AbsKernelOp {}

impl UnaryOpCudaKernel<f32> for super::AbsKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/abs.ptx"));
    const MODULE_NAME: &'static str = "abs";
    const FWD_FN_NAME: &'static str = "abs_forward_f32";
    const BWD_FN_NAME: &'static str = "abs_backward_f32";
}
