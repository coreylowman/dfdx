use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::NegateKernelOp {}

const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/negate.ptx"));

impl UnaryOpCudaKernel<f32> for super::NegateKernelOp {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "negate_f32";
    const FWD_FN_NAME: &'static str = "negate_forward_f32";
    const BWD_FN_NAME: &'static str = "negate_backward_f32";
}

impl UnaryOpCudaKernel<f64> for super::NegateKernelOp {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "negate_f64";
    const FWD_FN_NAME: &'static str = "negate_forward_f64";
    const BWD_FN_NAME: &'static str = "negate_backward_f64";
}
