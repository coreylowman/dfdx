use crate::tensor_ops::cuda_kernels::BinaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::MinimumKernelOp {}

impl BinaryOpCudaKernel<f32> for super::MinimumKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/minimum.ptx"));
    const MODULE_NAME: &'static str = "minimum";
    const FWD_FN_NAME: &'static str = "minimum_forward_f32";
    const BWD_FN_NAME: &'static str = "minimum_backward_f32";
}

impl BinaryOpCudaKernel<f64> for super::MinimumKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/minimum.ptx"));
    const MODULE_NAME: &'static str = "minimum";
    const FWD_FN_NAME: &'static str = "minimum_forward_f64";
    const BWD_FN_NAME: &'static str = "minimum_backward_f64";
}
