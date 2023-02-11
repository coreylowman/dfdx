use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::AbsKernelOp {}

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/abs.ptx"));

impl UnaryOpCudaKernel<f32> for super::AbsKernelOp {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "abs_f32";
    const FWD_FN_NAME: &'static str = "abs_forward_f32";
    const BWD_FN_NAME: &'static str = "abs_backward_f32";
}

impl UnaryOpCudaKernel<f64> for super::AbsKernelOp {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "abs_f64";
    const FWD_FN_NAME: &'static str = "abs_forward_f64";
    const BWD_FN_NAME: &'static str = "abs_backward_f64";
}
